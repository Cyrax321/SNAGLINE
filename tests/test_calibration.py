"""Tests for opt-in auto-calibration from a BaselineProfile (issue #101).

Honesty anchors:

* Threshold expectations are literals obtained by brute-force enumeration of
  all ``2**10`` error/no-error sequences per probability (independent
  arithmetic, quoted in the comments), never by calling the implementation
  under test.
* Detector-facing behavior is tested on BOTH sides: an injected-failure
  sequence that must fire the exact trigger name, plus a healthy sequence
  that must stay completely silent.
* Fail-open paths use real corrupt input files, not mocked loaders.
"""

from __future__ import annotations

import importlib.util
import json
import logging
import sys
from pathlib import Path
from typing import Any

import pytest

from snagline.baseline import (
    BaselineProfile,
    ToolBaseline,
    save_baseline,
)
from snagline.calibration import (
    build_plan,
    min_consecutive_threshold,
    min_window_threshold,
    observed_error_rate,
    resolve_baseline_profile,
)
from snagline.config import Config
from snagline.events import StepEvent, make_signature
from snagline.monitor import Monitor
from snagline.risk import FailureRisk

_REPO_ROOT = Path(__file__).resolve().parents[1]
_FIXTURES = _REPO_ROOT / "benchmarks" / "fixtures"
_HARNESS_PATH = _REPO_ROOT / "benchmarks" / "detection_accuracy.py"


def _load_harness():
    spec = importlib.util.spec_from_file_location("detection_accuracy", _HARNESS_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # Register before exec: slots dataclasses resolve their module while the
    # module body runs.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


class Collector:
    """AlertSink double recording every dispatched risk."""

    def __init__(self) -> None:
        self.risks: list[FailureRisk] = []

    def emit(self, risk: FailureRisk) -> None:
        self.risks.append(risk)


def _event(
    ep: str,
    i: int,
    *,
    tool: str = "search_web",
    error: bool = False,
    latency: float | None = None,
) -> StepEvent:
    return StepEvent(
        step_id=f"{ep}-s{i}",
        episode_id=ep,
        timestamp=1_700_000_000.0 + i,
        action_type="tool_call",
        # Distinct stable part per step: keeps the loop detector silent so
        # assertions stay pinned to the trigger under test.
        action_signature=make_signature("tool_call", tool, f"arg-{i}"),
        tool_name=tool,
        latency_ms=latency,
        error=error,
    )


def _monitor(cfg: Config) -> tuple[Monitor, Collector]:
    collector = Collector()
    return Monitor.default(config=cfg, sinks=[collector]), collector


def _rate_profile(errors: int, total: int, tool: str = "search_web") -> BaselineProfile:
    """Profile with one tool carrying ``errors`` failures over ``total`` calls."""
    profile = BaselineProfile()
    tb = ToolBaseline(tool_name=tool)
    for i in range(total):
        tb.add(100.0 + i, i < errors)
    profile.tools[tool] = tb
    return profile


def _latency_profile(
    latencies: list[float], tool: str = "search_web"
) -> BaselineProfile:
    profile = BaselineProfile()
    tb = ToolBaseline(tool_name=tool)
    for value in latencies:
        tb.add(value, False)
    profile.tools[tool] = tb
    return profile


def _detector_params(monitor: Monitor) -> list[tuple]:
    """Structural snapshot of tier-1 detector configuration."""
    params = []
    for d in monitor._detectors:
        params.append(
            (
                type(d).__name__,
                getattr(d, "window_size", None),
                getattr(d, "repeat_threshold", None),
                getattr(d, "error_threshold", None),
                getattr(d, "consecutive_threshold", None),
                getattr(d, "k", None),
                getattr(d, "h", None),
                getattr(d, "min_samples", None),
                getattr(d, "_baseline", None) is not None,
            )
        )
    return params


# --------------------------------------------------------------------------
# Derivation math: literals from brute-force enumeration of all 2**10
# sequences (binomial tails and run probabilities computed independently).
# --------------------------------------------------------------------------


class TestDerivationMath:
    def test_binomial_tail_known_values(self) -> None:
        # Brute force over all 1024 length-10 sequences:
        #   P(X>=2) at p=0.01  = 0.004266, P(X>=3) = 0.000114
        #   P(X>=2) at p=0.015521 (=7/451) = 0.009979, P(X>=3) = 0.000413
        #   P(X>=3) at p=0.05  = 0.011504, P(X>=5) <= 0.001
        assert min_window_threshold(10, 0.01, 0.001) == 3
        assert min_window_threshold(10, 7 / 451, 0.001) == 3
        assert min_window_threshold(10, 0.05, 0.001) == 5

    def test_run_probability_known_values(self) -> None:
        # Brute force over all 1024 length-10 sequences:
        #   P(run>=2) at p=0.005 = 0.000224, at p=7/451 = 0.002137
        #   P(run>=3) at p=7/451 = 0.000030
        assert min_consecutive_threshold(10, 0.005, 0.001) == 2
        assert min_consecutive_threshold(10, 7 / 451, 0.001) == 3

    def test_degenerate_probabilities(self) -> None:
        assert min_window_threshold(10, 0.0, 0.001) == 1
        assert min_consecutive_threshold(10, 0.0, 0.001) == 1
        assert min_window_threshold(10, 1.0, 0.001) == 10
        assert min_consecutive_threshold(10, 1.0, 0.001) == 10

    def test_observed_error_rate_pools_and_percentiles(self) -> None:
        # A(100 calls, 10 errors)=0.1, B(20, 0)=0.0, C(5, 5) excluded (< 20
        # samples). Well-sampled rates sorted [0.0, 0.1], nearest-rank p99 of
        # 2 items is the max: 0.1. Pooled = 15/125 = 0.12. Max wins: 0.12.
        profile = BaselineProfile()
        a = ToolBaseline(tool_name="a")
        for i in range(100):
            a.add(1.0, i < 10)
        b = ToolBaseline(tool_name="b")
        for i in range(20):
            b.add(1.0, False)
        c = ToolBaseline(tool_name="c")
        for i in range(5):
            c.add(1.0, True)
        profile.tools.update({"a": a, "b": b, "c": c})
        assert observed_error_rate(profile) == pytest.approx(0.12)

        # Percentile dominates when a well-sampled tool is unreliable:
        # A(100, 40)=0.4, B(100, 10)=0.1 -> p99 = 0.4 > pooled 50/200 = 0.25.
        profile2 = BaselineProfile()
        a2 = ToolBaseline(tool_name="a")
        for i in range(100):
            a2.add(1.0, i < 40)
        b2 = ToolBaseline(tool_name="b")
        for i in range(100):
            b2.add(1.0, i < 10)
        profile2.tools.update({"a": a2, "b": b2})
        assert observed_error_rate(profile2) == pytest.approx(0.4)

    def test_build_plan_clamps(self) -> None:
        cfg = Config()
        # Perfectly clean baseline: raw thresholds of 1 clamp up to the floor
        # of 2 (a single error is never a cascade).
        clean = build_plan(_rate_profile(0, 500), cfg)
        assert clean.cascade_error_threshold == 2
        assert clean.cascade_consecutive_threshold == 2

        # Corpus-like health (7/451): derivation lands exactly on the
        # hand-tuned defaults (3, 3).
        corpus = build_plan(_rate_profile(7, 451), cfg)
        assert corpus.cascade_error_threshold == 3
        assert corpus.cascade_consecutive_threshold == 3

        # Noisy deployment: raw (9, 7) clamps down to the hand-tuned ceiling;
        # auto can never become LESS sensitive than shipped behavior.
        noisy = build_plan(_rate_profile(3, 10), cfg)
        assert noisy.cascade_error_threshold == 3
        assert noisy.cascade_consecutive_threshold == 3

    def test_build_plan_monotone_and_bounded(self) -> None:
        cfg = Config()
        prev_w = prev_c = 2
        for errors in range(0, 61):
            plan = build_plan(_rate_profile(errors, 200), cfg)
            assert 2 <= plan.cascade_error_threshold <= cfg.cascade_error_threshold
            assert (
                2
                <= plan.cascade_consecutive_threshold
                <= cfg.cascade_consecutive_threshold
            )
            # Higher observed error rate must never yield tighter thresholds.
            assert plan.cascade_error_threshold >= prev_w
            assert plan.cascade_consecutive_threshold >= prev_c
            prev_w = plan.cascade_error_threshold
            prev_c = plan.cascade_consecutive_threshold


# --------------------------------------------------------------------------
# Config plumbing
# --------------------------------------------------------------------------


class TestConfigPlumbing:
    def test_defaults_are_unchanged(self) -> None:
        cfg = Config()
        assert cfg.calibration == "manual"
        assert cfg.calibration_alpha == 0.001
        assert cfg.calibration_baseline is None
        assert cfg.calibration_baseline_path is None

    def test_env_overrides_coerce(self) -> None:
        cfg = Config.from_env(
            {
                "SNAGLINE_CALIBRATION": "auto",
                "SNAGLINE_CALIBRATION_ALPHA": "0.005",
                "SNAGLINE_CALIBRATION_BASELINE_PATH": "/tmp/baseline.json",
            }
        )
        assert cfg.calibration == "auto"
        assert cfg.calibration_alpha == 0.005
        assert cfg.calibration_baseline_path == "/tmp/baseline.json"

    def test_resolve_precedence_object_then_path_then_none(
        self, tmp_path: Path
    ) -> None:
        profile = _rate_profile(1, 100)
        path = tmp_path / "baseline.json"
        save_baseline(profile, str(path))

        # Explicit object wins over a (worse) path.
        both = Config(calibration_baseline=profile, calibration_baseline_path="/nope")
        assert resolve_baseline_profile(both) is profile
        # Path loads a real file.
        assert (
            resolve_baseline_profile(Config(calibration_baseline_path=str(path)))  # type: ignore[union-attr]
            .tools["search_web"]
            .count
            == 100
        )
        # Neither configured: None, not an error.
        assert resolve_baseline_profile(Config()) is None

    def test_corrupt_path_raises_for_caller_to_handle(self, tmp_path: Path) -> None:
        bad = tmp_path / "bad.json"
        bad.write_text("{definitely not json", encoding="utf-8")
        with pytest.raises(json.JSONDecodeError):
            resolve_baseline_profile(Config(calibration_baseline_path=str(bad)))


# --------------------------------------------------------------------------
# Monitor wiring: both sides every time
# --------------------------------------------------------------------------


class TestMonitorWiring:
    def test_auto_without_baseline_matches_manual_exactly(self) -> None:
        manual, _ = _monitor(Config())
        auto, _ = _monitor(Config(calibration="auto"))
        assert _detector_params(auto) == _detector_params(manual)

    def test_unknown_calibration_value_behaves_as_manual(self) -> None:
        weird, _ = _monitor(Config(calibration="yes-please"))
        manual, _ = _monitor(Config())
        assert _detector_params(weird) == _detector_params(manual)

    def test_value_is_case_insensitive(self) -> None:
        # Whitespace/case tolerated; activation observable via the derived
        # (2, 2) cascade thresholds of a spotless 500-call baseline.
        cfg = Config(calibration="  Auto ", calibration_baseline=_rate_profile(0, 500))
        auto, _ = _monitor(cfg)
        cascade = next(
            d for d in auto._detectors if getattr(d, "name", "") == "error_cascade"
        )
        assert cascade.error_threshold == 2  # type: ignore[attr-defined]
        assert cascade.consecutive_threshold == 2  # type: ignore[attr-defined]

    def test_corrupt_baseline_file_fails_open(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        bad = tmp_path / "bad.json"
        bad.write_text("{definitely not json", encoding="utf-8")
        with caplog.at_level(logging.WARNING, logger="snagline"):
            monitor, collector = _monitor(
                Config(calibration="auto", calibration_baseline_path=str(bad))
            )
        assert any("hand-tuned" in r.message for r in caplog.records)
        manual, _ = _monitor(Config())
        assert _detector_params(monitor) == _detector_params(manual)
        # Ingest still works end to end.
        monitor.ingest(_event("ep", 0))
        assert collector.risks == []

    def test_derivation_failure_fails_open(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        # Force the pure-math collaborator to blow up: the unit under test is
        # the fail-open wiring in Monitor.default, not the math itself.
        def boom(profile: object, cfg: object) -> object:
            raise RuntimeError("synthetic derivation failure")

        monkeypatch.setattr("snagline.monitor.build_plan", boom)
        with caplog.at_level(logging.WARNING, logger="snagline"):
            monitor, _ = _monitor(
                Config(calibration="auto", calibration_baseline=_rate_profile(0, 100))
            )
        assert any("hand-tuned" in r.message for r in caplog.records)
        manual, _ = _monitor(Config())
        assert _detector_params(monitor) == _detector_params(manual)


class TestCalibratedCascade:
    CORPUS_LIKE = _rate_profile(7, 451)  # healthy-but-imperfect: derives (3, 3)

    def test_three_consecutive_errors_fire_and_interleaved_stay_silent(self) -> None:
        cfg = Config(calibration="auto", calibration_baseline=self.CORPUS_LIKE)
        monitor, collector = _monitor(cfg)
        for i in range(8):
            monitor.ingest(_event("fail-ep", i, tool="deploy", error=i in (3, 4, 5)))
        assert [r.trigger for r in collector.risks] == ["error_cascade"]

        # Healthy side: two isolated failures inside one window stay silent
        # under the calibrated monitor (density rule needs 3).
        monitor2, collector2 = _monitor(cfg)
        for i in range(12):
            monitor2.ingest(_event("ok-ep", i, tool="deploy", error=i in (1, 5)))
        assert collector2.risks == []

    def test_never_worse_than_manual_on_same_sequences(self) -> None:
        cfg_auto = Config(calibration="auto", calibration_baseline=self.CORPUS_LIKE)
        failing = [
            _event("f", i, tool="deploy", error=i in (3, 4, 5)) for i in range(8)
        ]
        healthy = [_event("h", i, tool="deploy", error=i in (1, 5)) for i in range(12)]
        for cfg in (Config(), cfg_auto):
            m1, c1 = _monitor(cfg)
            for ev in failing:
                m1.ingest(ev)
            m2, c2 = _monitor(cfg)
            for ev in healthy:
                m2.ingest(ev)
            assert [r.trigger for r in c1.risks] == ["error_cascade"]
            assert c2.risks == []

    def test_clean_baseline_tightens_consecutive_to_two(self) -> None:
        cfg = Config(calibration="auto", calibration_baseline=_rate_profile(0, 500))
        monitor, collector = _monitor(cfg)
        # Two consecutive failures: hand-tuned needs 3, calibrated fires at 2.
        for i in range(6):
            monitor.ingest(_event("fast-ep", i, tool="deploy", error=i in (2, 3)))
        assert [r.trigger for r in collector.risks] == ["error_cascade"]

        # Manual monitor on the identical sequence stays silent: the gain is
        # real, not assumed.
        manual, manual_collector = _monitor(Config())
        for i in range(6):
            manual.ingest(_event("fast-ep", i, tool="deploy", error=i in (2, 3)))
        assert manual_collector.risks == []

        # Healthy side under the tightened rule: ONE failure must stay silent
        # (floor of 2 holds even for a spotless baseline).
        monitor2, collector2 = _monitor(cfg)
        for i in range(6):
            monitor2.ingest(_event("calm-ep", i, tool="deploy", error=i == 2))
        assert collector2.risks == []


class TestSeededCusum:
    BASELINE = _latency_profile([400.0, 395.0, 405.0, 398.0, 402.0, 400.0])

    def test_spike_fires_without_warmup_and_short_healthy_stays_silent(self) -> None:
        cfg = Config(calibration="auto", calibration_baseline=self.BASELINE)
        monitor, collector = _monitor(cfg)
        # Only three samples: hand-tuned warm-up (5) could never alarm here.
        monitor.ingest(_event("spike-ep", 0, latency=395.0))
        monitor.ingest(_event("spike-ep", 1, latency=405.0))
        monitor.ingest(_event("spike-ep", 2, latency=2500.0))
        assert [r.trigger for r in collector.risks] == ["latency_anomaly"]

        # Healthy side: same length, sane latencies, completely silent.
        monitor2, collector2 = _monitor(cfg)
        for i, ms in enumerate((395.0, 405.0, 420.0)):
            monitor2.ingest(_event("short-ok-ep", i, latency=ms))
        assert collector2.risks == []

    def test_manual_stays_silent_on_same_short_sequence(self) -> None:
        manual, collector = _monitor(Config())
        for i, ms in enumerate((395.0, 405.0, 2500.0)):
            manual.ingest(_event("spike-ep", i, latency=ms))
        assert collector.risks == []

    def test_thin_baseline_entry_keeps_warmup_behavior(self) -> None:
        thin = _latency_profile([300.0, 900.0])  # count 2 < min_samples 5
        cfg = Config(calibration="auto", calibration_baseline=thin)
        monitor, collector = _monitor(cfg)
        monitor.ingest(_event("cold-ep", 0, latency=2500.0))
        # Not seeded: the lone spike feeds the warm-up instead of alarming,
        # exactly like today's hand-tuned behavior.
        assert collector.risks == []

    def test_unknown_tool_still_detected_via_online_learning(self) -> None:
        cfg = Config(calibration="auto", calibration_baseline=self.BASELINE)
        monitor, collector = _monitor(cfg)
        for i in range(5):
            monitor.ingest(_event("new-ep", i, tool="brand_new_tool", latency=400.0))
        monitor.ingest(_event("new-ep", 5, tool="brand_new_tool", latency=2500.0))
        assert "latency_anomaly" in [r.trigger for r in collector.risks]


# --------------------------------------------------------------------------
# Fixture-corpus honesty gate: auto must not reduce F1 vs hand-tuned
# --------------------------------------------------------------------------


def _fit_baseline_from_controls(path: Path) -> BaselineProfile:
    """Fit a BaselineProfile from the envelope-format healthy-controls file."""
    harness = _load_harness()
    profile = BaselineProfile()
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        record = json.loads(line)
        for raw in record["events"]:
            profile.add_event(harness._parse_event(raw, record["episode_id"]))
    return profile


def _replay(harness: Any, episode: Any, cfg: Config) -> Any:
    collector = Collector()
    monitor = Monitor.default(config=cfg, sinks=[collector])
    for ev in episode.events:
        monitor.ingest(ev)
    monitor.end_episode(episode.episode_id)
    outcome = harness.EpisodeOutcome(episode.episode_id, episode.label_triggers)
    for risk in collector.risks:
        outcome.predicted.add(harness.risk_to_label_trigger(risk))
    return outcome


@pytest.fixture(scope="module")
def harness():
    return _load_harness()


@pytest.fixture(scope="module")
def episodes(harness):
    return list(harness.iter_fixtures(_FIXTURES))


class TestFixtureCorpusGate:
    def test_auto_is_never_worse_than_manual(self, harness, episodes) -> None:
        baseline = _fit_baseline_from_controls(_FIXTURES / "healthy_controls.jsonl")
        assert baseline.tools, "baseline fitting from healthy controls failed"

        cfg_manual = harness.harness_config()
        cfg_auto = harness.harness_config()
        cfg_auto.calibration = "auto"
        cfg_auto.calibration_baseline = baseline

        report_manual = harness.score(
            [_replay(harness, ep, cfg_manual) for ep in episodes]
        )
        report_auto = harness.score([_replay(harness, ep, cfg_auto) for ep in episodes])

        # Both modes must keep every healthy control silent.
        assert report_manual.healthy_fired == 0
        assert report_auto.healthy_fired == 0
        # THE GATE: calibrated macro-F1 must not regress.
        assert report_auto.macro_f1 >= report_manual.macro_f1

        # Per-trigger: recall never drops and precision never drops on the
        # triggers calibration touches.
        for trig in ("loop", "error_cascade", "latency_anomaly"):
            m = report_manual.per_trigger[trig]
            a = report_auto.per_trigger[trig]
            assert a.tp >= m.tp, trig
            assert a.fp <= m.fp, trig

    def test_calibrated_cascade_thresholds_equal_defaults_here(self, harness) -> None:
        # The fitted corpus baseline yields (3, 3) on the old 36-control
        # corpus and (3, 2) on the expanded 40-control corpus with longer
        # windows (issue #181): extra healthy steps lower the pooled error
        # rate from ~0.012 to ~0.009, so the consecutive threshold tightens by
        # one while staying within the [2, default] clamp. Identical firing
        # sets remain (both modes keep every healthy silent, tested above).
        baseline = _fit_baseline_from_controls(_FIXTURES / "healthy_controls.jsonl")
        plan = build_plan(baseline, harness.harness_config())
        assert plan.cascade_error_threshold == 3
        assert plan.cascade_consecutive_threshold in (2, 3)
