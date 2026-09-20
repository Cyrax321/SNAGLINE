"""Regression tests for the unvalidated detector knob surface (issues #331-#333).

#331  cusum_h / token_cusum_h -- the CUSUM alarm bar is the denominator of the
       risk score, so 0 raised ZeroDivisionError that fail-open swallowed (a
       permanently dead detector) and any negative value made the alarm
       trivially true on every step.
#332  BaselineStore max_versions <= 0 made _prune delete the version the same
       save() call had just written, leaving an empty store behind a save that
       reported success.
#333  *_min_samples / *_window_size of 0 either made a detector's fire condition
       unreachable (silent disable) or let it score an empty window and emit a
       fabricated alert on step 0.

Each bad value must be rejected at every config layer (constructor, env,
``resolve``) and at the store/CLI boundary, matching the _validated_stagnation
precedent (issue #132): broken monitoring config aborts startup loudly instead
of running silently mis-configured.
"""

from __future__ import annotations

import json

import pytest

from snagline.baseline_store import BaselineStore
from snagline.config import Config
from snagline.events import StepEvent


def _ev(latency_ms: float, tool_name: str = "search") -> StepEvent:
    return StepEvent(
        step_id="s",
        episode_id="ep",
        timestamp=1.0,
        action_type="tool_call",
        action_signature="SIG",
        tool_name=tool_name,
        latency_ms=latency_ms,
    )


def _env(name: str, value) -> dict[str, str]:
    return {f"SNAGLINE_{name.upper()}": str(value)}


# --- issue #331: the CUSUM alarm bars --------------------------------------


@pytest.mark.parametrize("name", ["cusum_h", "token_cusum_h"])
@pytest.mark.parametrize("bad", [0.0, -0.001, -5.0])
def test_cusum_alarm_bar_rejects_zero_and_negative(name, bad):
    with pytest.raises(ValueError, match=name):
        Config(**{name: bad})
    with pytest.raises(ValueError, match=name):
        Config.from_env(environ=_env(name, bad))
    with pytest.raises(ValueError, match=name):
        Config.resolve(environ=_env(name, bad))


@pytest.mark.parametrize("name", ["cusum_h", "token_cusum_h"])
@pytest.mark.parametrize("good", [0.001, 1.0, 5.0, 100.0])
def test_cusum_alarm_bar_accepts_positive(name, good):
    assert getattr(Config(**{name: good}), name) == good
    assert getattr(Config.from_env(environ=_env(name, good)), name) == good


def test_cusum_h_zero_no_longer_deadens_the_latency_detector(capsys):
    """The reported end-to-end symptom: with h=0 the latency detector raised
    ZeroDivisionError on every anomalous step, fail-open swallowed it, and a
    100x latency anomaly never paged. Validation now stops that config at
    startup instead."""
    from snagline.monitor import Monitor

    with pytest.raises(ValueError, match="cusum_h"):
        Monitor.default(config=Config(cusum_h=0.0, cusum_min_samples=3))

    mon = Monitor.default(config=Config(cusum_min_samples=3))
    for _ in range(6):
        mon.ingest(_ev(50.0))
    for _ in range(24):
        mon.ingest(_ev(5000.0))
    m = mon.metrics()
    assert m["detector_errors"] == 0, "the latency detector must not raise"
    assert m["risks_emitted"] > 0, "a 100x latency anomaly must page"
    assert "latency_anomaly" in capsys.readouterr().err


# --- issue #333: sample counts and window sizes ----------------------------


COUNT_AND_WINDOW_KNOBS = (
    "cusum_min_samples",
    "token_min_samples",
    "loop_window_size",
    "cascade_window_size",
    "loop_cycle_window_size",
    "meltdown_window_size",
    "meltdown_rearm_steps",
)


@pytest.mark.parametrize("name", COUNT_AND_WINDOW_KNOBS)
@pytest.mark.parametrize("bad", [0, -1, -100])
def test_count_and_window_knobs_reject_below_one(name, bad):
    with pytest.raises(ValueError, match=name):
        Config(**{name: bad})
    with pytest.raises(ValueError, match=name):
        Config.from_env(environ=_env(name, bad))
    with pytest.raises(ValueError, match=name):
        Config.resolve(environ=_env(name, bad))


@pytest.mark.parametrize(
    "name", [n for n in COUNT_AND_WINDOW_KNOBS if n != "meltdown_window_size"]
)
@pytest.mark.parametrize("good", [1, 5, 50])
def test_count_and_window_knobs_accept_one_and_above(name, good):
    assert getattr(Config(**{name: good}), name) == good
    assert getattr(Config.from_env(environ=_env(name, good)), name) == good


@pytest.mark.parametrize("bad", [1, 2, 0, -1])
def test_meltdown_window_size_rejects_below_three(bad):
    """Issue #346: the >= 1 check from #333 is not enough. A one-item window
    scores 0.0 bits by construction and a two-item window only 0.0 or 1.0, so
    both page on an ordinary one-tool episode. The floor is the smallest window
    that can hold a distribution."""
    with pytest.raises(ValueError, match="meltdown_window_size"):
        Config(meltdown_enabled=True, meltdown_window_size=bad)
    with pytest.raises(ValueError, match="meltdown_window_size"):
        Config.from_env(environ=_env("meltdown_window_size", bad))
    with pytest.raises(ValueError, match="meltdown_window_size"):
        Config.resolve(environ=_env("meltdown_window_size", bad))


@pytest.mark.parametrize("good", [3, 4, 20])
def test_meltdown_window_size_accepts_three_and_above(good):
    assert Config(meltdown_window_size=good).meltdown_window_size == good


def test_zero_is_the_only_rejected_boundary_for_counts_and_windows():
    """Knobs whose documented zero means "disabled" must stay valid: the
    neighbouring families already range-checked by _validated_horizon and
    _validated_stagnation are untouched by the new checks."""
    Config(stagnation_window_size=1, stagnation_patience=1)
    Config(max_window=1, window_scale_steps=0, cusum_refit_every=0)
    Config()  # stock config must never raise


def test_meltdown_window_size_zero_cannot_fabricate_a_step_zero_alert():
    """The concrete fabricated-alert symptom: deque(maxlen=0) is always empty
    but ``len(window) < target`` is ``0 < 0`` == False, so the detector scored
    an empty window and emitted a score-0.7 risk on the first step. The
    detector's own guard (issue #346) also closes the direct-constructor path
    in the issue's repro, which bypasses Config."""
    from snagline.detectors.meltdown import MeltdownDetector

    with pytest.raises(ValueError, match="meltdown_window_size"):
        MeltdownDetector(config=Config(meltdown_enabled=True, meltdown_window_size=0))
    for bad in (1, 2):
        with pytest.raises(ValueError, match="window_size"):
            MeltdownDetector(window_size=bad)
    det = MeltdownDetector(config=Config(meltdown_enabled=True))
    assert det.observe(_ev(50.0)) is None, "a single step must not page"


def test_meltdown_still_fires_on_a_genuine_one_tool_collapse():
    """The floor must not deaden the detector (#346): an episode that collapses
    onto a single tool still pages once the window fills -- the alert is just
    no longer reachable from an unrepresentative one- or two-step window."""
    from snagline.detectors.meltdown import MeltdownDetector

    det = MeltdownDetector(config=Config(meltdown_enabled=True, meltdown_window_size=3))
    # Alternating tools: any 3-step window is a 2:1 split at ~0.92 bits, above
    # the low bar, so this is the quiet zone the floor exists to protect.
    healthy = [
        det.observe(_ev(50.0, "search" if i % 2 == 0 else "fetch")) for i in range(6)
    ]
    assert all(r is None for r in healthy), "a mixed episode must stay quiet"
    # Collapse onto one tool only.
    collapse = [det.observe(_ev(50.0, "search")) for _ in range(6)]
    assert any(r is not None for r in collapse), "a real collapse must page"


# --- issue #332: baseline retention ----------------------------------------


def _profile():
    from snagline.baseline import BaselineProfile

    return BaselineProfile()


@pytest.mark.parametrize("bad", [0, -1, -3])
def test_baseline_store_rejects_non_positive_retention(tmp_path, bad):
    with pytest.raises(ValueError, match="max_versions"):
        BaselineStore(str(tmp_path / "store"), max_versions=bad)


def test_baseline_store_save_rejects_non_positive_per_call_retention(tmp_path):
    """The reported repro: save(max_versions=0) on a well-configured store.
    The check must fire before any file is touched, so a failed save leaves
    the store exactly as it was rather than half-applied."""
    store = BaselineStore(str(tmp_path / "store"))
    with pytest.raises(ValueError, match="max_versions"):
        store.save(_profile(), version="v1", max_versions=0)
    with pytest.raises(ValueError, match="max_versions"):
        store.save(_profile(), version="v1", max_versions=-1)
    assert store.list_versions() == []
    assert store.load() is None
    assert not (tmp_path / "store").exists()


def test_baseline_store_retention_one_keeps_the_pointer_and_history(tmp_path):
    """max_versions=1 is the legitimate "keep only the latest" reading and must
    stay supported: latest.json resolves and the just-written version survives."""
    store = BaselineStore(str(tmp_path / "store"), max_versions=1)
    v = store.save(_profile(), version="v1")
    assert store.list_versions() == ["v1"]
    assert store.load() is not None
    assert store.load_version("default", "default", v) is not None


def _window(path, tool: str, latencies: list[float]) -> str:
    rows = [
        {
            "step_id": str(i),
            "episode_id": "ep",
            "timestamp": 1.0 + i,
            "action_type": "tool_call",
            "action_signature": "SIG",
            "tool_name": tool,
            "latency_ms": latency,
        }
        for i, latency in enumerate(latencies)
    ]
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return str(path)


@pytest.mark.parametrize("bad", ["0", "-1"])
def test_cli_rejects_non_positive_max_versions(tmp_path, capsys, bad):
    """``args.max_versions or 10`` treated an explicit 0 as "unset" and silently
    substituted 10, while a negative value reached the store and destroyed the
    history. Both commands now treat < 1 as a usage error (exit 2)."""
    from snagline.cli import main

    win = _window(tmp_path / "w.jsonl", "search", [100.0, 102.0, 104.0])
    store_dir = str(tmp_path / "store")

    for argv in (
        ["baseline", "retrain", "--store-dir", store_dir, "--jsonl", win],
        ["baseline", win, "--store-dir", store_dir],
    ):
        rc = main([*argv, "--max-versions", bad])
        assert rc == 2, argv
        assert "--max-versions must be >= 1" in capsys.readouterr().err
        assert not (tmp_path / "store").exists(), "no partial store written"


def test_cli_accepts_explicit_max_versions_one(tmp_path):
    from snagline.cli import main

    win = _window(tmp_path / "w.jsonl", "search", [100.0, 102.0, 104.0])
    store_dir = str(tmp_path / "store")
    assert main(["baseline", win, "--store-dir", store_dir, "--max-versions", "1"]) == 0
    store = BaselineStore(store_dir)
    assert store.load() is not None
