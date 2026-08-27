"""Tests for the optional ESN + CUSUM ensemble detector (issue #80).

These tests need the ``ml`` extra (numpy). When numpy is absent the whole
module skips gracefully; CI runs a dedicated leg with the extra installed
so they are always exercised somewhere. The import-guard behavior for
numpy-less environments lives in test_ml_extra_guard.py.
"""

from __future__ import annotations

import logging

import pytest

np = pytest.importorskip("numpy", reason="ml extra (numpy) not installed")

from snagline.baseline import BaselineProfile, ToolBaseline  # noqa: E402
from snagline.config import Config  # noqa: E402
from snagline.detectors.ml_ensemble import MLOrchestrator  # noqa: E402
from snagline.events import StepEvent  # noqa: E402
from snagline.ml.esn_ensemble import EsnCusumDetector  # noqa: E402
from snagline.monitor import Monitor  # noqa: E402
from snagline.risk import FailureRisk  # noqa: E402


def _fast(**overrides) -> EsnCusumDetector:
    """A detector with short warm-up and a low CUSUM threshold for tests."""
    params = dict(warmup_steps=5, cusum_h=1.0)
    params.update(overrides)
    return EsnCusumDetector(**params)  # type: ignore[arg-type]


def _ev(
    i: int,
    episode: str = "ep",
    *,
    tool: str = "t",
    latency: float = 100.0,
    error: bool = False,
    signature: str = "sig",
) -> StepEvent:
    return StepEvent(
        step_id=f"s{i}",
        episode_id=episode,
        timestamp=float(i),
        action_type="tool_call",
        action_signature=signature,
        tool_name=tool,
        latency_ms=latency,
        error=error,
    )


def _healthy(n: int, start: int = 0) -> list[StepEvent]:
    """Steady, predictable steps: stable latency, cycling signatures."""
    return [
        _ev(i, signature=f"a{i % 4}", latency=100.0 + (i % 3))
        for i in range(start, start + n)
    ]


def _unhealthy(n: int, start: int) -> list[StepEvent]:
    """Failing steps: novel signatures, huge latency, errors."""
    return [
        _ev(i, signature=f"x{i}", latency=9000.0 + i, error=True)
        for i in range(start, start + n)
    ]


class _Collector:
    """Sink that records every emitted risk."""

    def __init__(self) -> None:
        self.risks: list[FailureRisk] = []

    def emit(self, risk: FailureRisk) -> None:
        self.risks.append(risk)


class _Stub:
    """Detector stub with a fixed score, used to exercise noisy-OR fusion."""

    def __init__(self, score: float) -> None:
        self._score = score

    def observe(self, event: StepEvent) -> FailureRisk | None:
        return FailureRisk(
            event.episode_id,
            event.step_id,
            self._score,
            "loop",
            "stub",
            event.timestamp,
        )

    def reset(self, episode_id: str) -> None:
        pass


# --- Both sides: healthy stays silent ---------------------------------------


def test_healthy_stream_never_emits():
    det = _fast()
    for event in _healthy(40):
        assert det.observe(event) is None


def test_healthy_stream_with_baseline_stays_silent():
    tb = ToolBaseline(tool_name="t")
    for _ in range(50):
        tb.add(100.0, False)
    profile = BaselineProfile(tools={"t": tb}, total_steps=50)
    det = _fast(baseline=profile)
    for event in _healthy(40):
        assert det.observe(event) is None


def test_warmup_phase_is_silent_even_for_faulty_input():
    det = _fast(warmup_steps=30)
    for event in _unhealthy(25, start=0):
        assert det.observe(event) is None


def test_monitor_default_ml_flag_healthy_run_emits_nothing():
    cfg = Config()
    cfg.ml_ensemble_enabled = True
    sink = _Collector()
    mon = Monitor.default(config=cfg, sinks=[sink])
    # Unique signatures keep the deterministic loop detector quiet; stable
    # latency/error keep cascade, latency-CUSUM, and the ESN quiet too.
    for i in range(60):
        mon.ingest(_ev(i, signature=f"u{i}", latency=100.0 + (i % 3)))
    assert sink.risks == []


# --- Both sides: injected failure fires with the exact trigger --------------


def test_unhealthy_stream_fires_ml_ensemble_trigger():
    det = _fast()
    fired: list[FailureRisk] = []
    stream = _healthy(10) + _unhealthy(12, start=10)
    for event in stream:
        risk = det.observe(event)
        if risk is not None:
            fired.append(risk)
    assert len(fired) >= 1
    assert all(r.trigger == "ml_ensemble" for r in fired)
    assert all(0.5 <= r.score <= 1.0 for r in fired)


def test_cusum_rearms_after_firing():
    det = _fast()
    stream = _healthy(10) + _unhealthy(40, start=10)
    fired = [r for e in stream if (r := det.observe(e)) is not None]
    # A sustained fault must alarm more than once across 40 bad steps.
    assert len(fired) >= 2


def test_esn_score_feeds_noisy_or_inside_ml_orchestrator():
    params = dict(warmup_steps=5, cusum_h=1.0)
    cfg = Config()
    cfg.ml_ensemble_score_threshold = 0.5
    orch = MLOrchestrator([_Stub(0.3), _fast(**params)], config=cfg)
    direct = _fast(**params)
    stream = _healthy(6) + _unhealthy(12, start=6)
    saw_combined = False
    for event in stream:
        solo = direct.observe(event)
        combined = orch.observe(event)
        if solo is not None:
            # Independent hand computation of noisy-OR over 0.3 and solo.
            expected = 1.0 - (1.0 - 0.3) * (1.0 - solo.score)
            assert combined is not None
            assert abs(combined.score - expected) < 1e-9
            assert combined.trigger == "ml_ensemble"
            saw_combined = True
    assert saw_combined


def test_monitor_default_ml_flag_unhealthy_run_emits_ml_ensemble():
    cfg = Config()
    cfg.ml_ensemble_enabled = True
    sink = _Collector()
    mon = Monitor.default(config=cfg, sinks=[sink])
    for i in range(10):
        mon.ingest(_ev(i, signature=f"u{i}", latency=100.0 + (i % 3)))
    for j in range(15):
        mon.ingest(_ev(10 + j, signature=f"x{j}", latency=9000.0 + j, error=True))
    assert len(sink.risks) >= 1
    assert all(r.trigger == "ml_ensemble" for r in sink.risks)


# --- Mahalanobis baseline scoring: exact hand-computed values ---------------


def _profile(mean: float, std_ms: float, n: int = 50) -> BaselineProfile:
    tb = ToolBaseline(tool_name="t")
    if std_ms == 0.0:
        for _ in range(n):
            tb.add(mean, False)
    else:
        for i in range(n):
            tb.add(mean + (std_ms if i % 2 else -std_ms), False)
    return BaselineProfile(tools={"t": tb}, total_steps=n)


def test_mahalanobis_exact_value_two_sigma_latency_and_error():
    # mean=100, alternating +/-20 latency, every 5th step an error.
    tb = ToolBaseline(tool_name="t")
    for i in range(50):
        tb.add(120.0 if i % 2 else 80.0, error=(i % 5 == 0))
    profile = BaselineProfile(tools={"t": tb}, total_steps=50)
    det = _fast(baseline=profile)
    # Closed-form expectations from the documented diagonal formula, computed
    # here by hand rather than by calling the detector under test:
    # sample std uses the n-1 denominator: var = (sum_sq - sum^2/n)/(n-1).
    sum_sq = 25 * 120.0**2 + 25 * 80.0**2
    std = ((sum_sq - 50 * 100.0**2) / 49) ** 0.5
    p = 10 / 50
    sigma_e = max((p * (1 - p)) ** 0.5, 0.05)
    z_lat_wild = (140.0 - 100.0) / std
    z_err_wild = (1.0 - p) / sigma_e
    d2_wild = z_lat_wild**2 + z_err_wild**2
    expected_wild = d2_wild / (d2_wild + 2)
    wild = _ev(1, latency=140.0, error=True)
    assert det._mahalanobis_score(wild) == pytest.approx(expected_wild, abs=1e-9)
    # Calm event at the healthy mean: latency term is exactly 0; the error
    # term contributes ((0-p)/sigma_e)^2 = 0.25, dof=2 -> 0.25/2.25 = 1/9.
    calm = _ev(0, latency=100.0, error=False)
    assert det._mahalanobis_score(calm) == pytest.approx(1.0 / 9.0, abs=1e-9)


def test_mahalanobis_unknown_tool_scores_constant():
    profile = _profile(100.0, 0.0)
    det = _fast(baseline=profile)
    assert det._mahalanobis_score(_ev(0, tool="never_seen")) == pytest.approx(0.6)


def test_mahalanobis_disabled_without_baseline():
    det = _fast()
    assert det._mahalanobis_score(_ev(0, latency=9999.0, error=True)) == 0.0


# --- Fail-open guarantees ----------------------------------------------------


def test_fail_open_internal_exception_is_swallowed_and_logged(caplog):
    det = _fast()

    def boom(self_event: StepEvent) -> object:
        raise RuntimeError("feature extraction exploded")

    det._features = boom  # type: ignore[assignment]
    with caplog.at_level(logging.ERROR, logger="snagline"):
        assert det.observe(_ev(0)) is None
    assert any("fail-open" in rec.getMessage() for rec in caplog.records)
    # Removing the shadowing attribute restores the class method: the
    # detector keeps working after the fault clears.
    del det._features  # type: ignore[attr-defined]
    assert det.observe(_ev(1)) is None


class _ObserveBoom(EsnCusumDetector):
    """Overrides observe() itself, bypassing the detector's own guard, so
    the MONITOR's fail-open layer is what must catch the fault."""

    def observe(self, event: StepEvent) -> FailureRisk | None:
        raise RuntimeError("ml path exploded")


def test_fail_open_monitor_survives_crashing_ml_detector():
    sink = _Collector()
    mon = Monitor([_ObserveBoom(warmup_steps=2)], [sink], fail_open=True)
    for i in range(18):
        mon.ingest(
            _ev(i, signature=f"u{i}")
            if i < 10
            else _ev(i, signature=f"x{i}", latency=9000.0, error=True)
        )  # must never raise into the host
    assert sink.risks == []
    assert mon.metrics()["detector_errors"] >= 1


# --- Determinism, reset, fit, memory bounds ---------------------------------


def test_same_seed_produces_identical_outputs():
    stream = _healthy(8) + _unhealthy(14, start=8)
    det_a, det_b = _fast(), _fast()
    scores_a = [r.score if (r := det_a.observe(e)) else None for e in stream]
    scores_b = [r.score if (r := det_b.observe(e)) else None for e in stream]
    assert scores_a == scores_b
    assert any(s is not None for s in scores_a)


def test_reset_restores_fresh_behavior():
    stream = _healthy(8) + _unhealthy(14, start=8)
    fresh = _fast()
    expected = [fresh.observe(e) for e in stream]
    reused = _fast()
    for event in stream[:9]:
        reused.observe(event)
    reused.reset("ep")
    again = [reused.observe(e) for e in stream]
    assert [(r.score if r else None) for r in expected] == [
        (r.score if r else None) for r in again
    ]


def test_fit_on_healthy_trajectory_enables_immediate_scoring():
    det = _fast(warmup_steps=5)
    det.fit(_healthy(30))
    # New episodes skip warm-up entirely: scoring starts at once.
    for event in _healthy(15):
        assert det.observe(event) is None
    fired = [det.observe(e) for e in _unhealthy(10, start=15)]
    assert any(r is not None and r.trigger == "ml_ensemble" for r in fired)


def test_fit_rejects_too_short_trajectories():
    det = _fast()
    with pytest.raises(ValueError):
        det.fit(_healthy(2))


def test_feature_vector_is_structural_and_bounded():
    """Features read only structure (error, latency, tokens), never content,
    and stay within [0, 1] even for extreme inputs."""
    det = _fast()
    extreme = StepEvent(
        step_id="s",
        episode_id="ep",
        timestamp=0.0,
        action_type="tool_call",
        action_signature="sig",
        tool_name="t",
        latency_ms=10_000_000.0,
        error=True,
        tokens_in=10_000_000,
        tokens_out=10_000_000,
        metadata={"prompt": "SECRET content that must not be read"},
    )
    x = det._features(extreme)
    assert x.shape == (4,)
    assert all(0.0 <= v <= 1.0 for v in x)


def test_noisy_but_healthy_stream_stays_silent():
    """A fixed-seed realistic stream: gaussian latency, sparse errors,
    varying token counts. Sustained-drift gating must keep it quiet."""
    import random

    rng = random.Random(7)
    det = _fast(warmup_steps=12, cusum_h=3.0)
    for i in range(80):
        event = _ev(
            i,
            signature=f"a{i % 6}",
            latency=max(1.0, rng.gauss(100, 25)),
            error=rng.random() < 0.03,
        )
        assert det.observe(event) is None


# --- Privacy ------------------------------------------------------------------


def test_features_ignore_metadata_content():
    secret = StepEvent(
        step_id="s0",
        episode_id="ep",
        timestamp=0.0,
        action_type="tool_call",
        action_signature="sig",
        tool_name="t",
        latency_ms=120.0,
        metadata={"prompt": "SECRET instructions do not leak"},
    )
    other = StepEvent(
        step_id="s0",
        episode_id="ep",
        timestamp=0.0,
        action_type="tool_call",
        action_signature="sig",
        tool_name="t",
        latency_ms=120.0,
        metadata={"prompt": "totally different content"},
    )
    stream_a = [secret] + _unhealthy(10, start=1)
    stream_b = [other] + _unhealthy(10, start=1)
    det_a, det_b = _fast(), _fast()
    scores_a = [r.score if (r := det_a.observe(e)) else None for e in stream_a]
    scores_b = [r.score if (r := det_b.observe(e)) else None for e in stream_b]
    assert scores_a == scores_b
