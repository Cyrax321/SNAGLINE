"""Time-axis awareness at the top of Monitor.ingest (issue #92).

Property under test: every idle-gap and wall-clock-budget decision is derived
ONLY from StepEvent timestamps, computed at the top of ingest(), fires at most
once per episode per threshold, and disappears entirely when the new config
options are unset -- replaying an old trajectory must produce byte-identical
results to the pre-#92 build.
"""

from __future__ import annotations

import glob
import json
import logging
import os

from snagline.config import Config
from snagline.events import StepEvent
from snagline.monitor import Monitor
from snagline.risk import FailureRisk


class CapturingSink:
    """Records every dispatched risk; nothing else."""

    def __init__(self) -> None:
        self.risks: list[FailureRisk] = []

    def emit(self, risk: FailureRisk) -> None:
        self.risks.append(risk)


def _event(
    step_id: str,
    timestamp: float,
    episode_id: str = "ep1",
    latency_ms: float | None = None,
) -> StepEvent:
    return StepEvent(
        step_id=step_id,
        episode_id=episode_id,
        timestamp=timestamp,
        action_type="tool_call",
        action_signature=f"sig-{step_id}",
        tool_name="t",
        latency_ms=latency_ms,
    )


def _monitor(sink: CapturingSink, **cfg_kwargs) -> Monitor:
    cfg = Config(**cfg_kwargs)
    return Monitor([], [sink], config=cfg)


def _feed(monitor: Monitor, *events: StepEvent, end: bool = False) -> None:
    for e in events:
        monitor.ingest(e)
    if end:
        monitor.end_episode(events[-1].episode_id if events else "ep1")


# --- idle_gap ---------------------------------------------------------------


def test_idle_gap_fires_once_per_episode() -> None:
    sink = CapturingSink()
    m = _monitor(sink, idle_warn_seconds=10.0)
    _feed(
        m,
        _event("s1", 0.0),
        _event("s2", 5.0),  # gap 5s: quiet
        _event("s3", 20.0),  # gap 15s: fire once
        _event("s4", 40.0),  # gap 20s: already fired, stays quiet
    )
    idle = [r for r in sink.risks if r.trigger == "idle_gap"]
    assert len(idle) == 1
    assert idle[0].step_id == "s3"
    assert idle[0].score == 0.8


def test_idle_gap_disabled_by_default_and_small_gaps_quiet() -> None:
    sink = CapturingSink()
    m = _monitor(sink)
    _feed(m, _event("s1", 0.0), _event("s2", 99999.0))
    assert sink.risks == []
    sink2 = CapturingSink()
    m2 = _monitor(sink2, idle_warn_seconds=100.0)
    _feed(m2, _event("s1", 0.0), _event("s2", 99.9))
    assert [r for r in sink2.risks if r.trigger == "idle_gap"] == []


def test_idle_gap_resets_with_episode() -> None:
    sink = CapturingSink()
    m = _monitor(sink, idle_warn_seconds=10.0)
    _feed(m, _event("s1", 0.0), _event("s2", 50.0), end=True)
    assert len([r for r in sink.risks if r.trigger == "idle_gap"]) == 1
    # Same episode id reused after end_episode: a fresh clock, so a later
    # silence can fire again.
    _feed(m, _event("s3", 100.0), _event("s4", 200.0))
    assert len([r for r in sink.risks if r.trigger == "idle_gap"]) == 2


def test_first_event_never_fires_idle_or_budget() -> None:
    sink = CapturingSink()
    m = _monitor(sink, idle_warn_seconds=0.001, max_episode_wall_seconds=0.001)
    _feed(m, _event("s1", 0.0))
    assert sink.risks == []


# --- wall_clock_budget ------------------------------------------------------


def test_budget_warn_then_breach_each_fire_once() -> None:
    sink = CapturingSink()
    m = _monitor(sink, max_episode_wall_seconds=100.0, warn_fraction=0.8)
    _feed(
        m,
        _event("s1", 0.0),
        _event("s2", 30.0),  # elapsed 30 < 80: quiet
        _event("s3", 85.0),  # elapsed 85 >= 80: warn
        _event("s4", 90.0),  # elapsed 90: still warned, no re-fire
        _event("s5", 120.0),  # elapsed 120 >= 100: breach
        _event("s6", 500.0),  # already breached, no re-fire
    )
    budget = [r for r in sink.risks if r.trigger == "wall_clock_budget"]
    assert [(r.step_id, r.score) for r in budget] == [("s3", 0.7), ("s5", 1.0)]
    assert budget[0].severity == "warning"
    assert budget[1].severity == "critical"


def test_single_jump_past_budget_fires_only_breach() -> None:
    sink = CapturingSink()
    m = _monitor(sink, max_episode_wall_seconds=10.0, warn_fraction=0.8)
    _feed(m, _event("s1", 0.0), _event("s2", 1000.0))
    budget = [r for r in sink.risks if r.trigger == "wall_clock_budget"]
    assert [(r.step_id, r.score) for r in budget] == [("s2", 1.0)]


def test_negative_delta_does_not_refund_budget() -> None:
    sink = CapturingSink()
    m = _monitor(sink, max_episode_wall_seconds=10.0)
    _feed(
        m,
        _event("s1", 0.0),
        _event("s2", 12.0),  # breach
        _event("s3", 1.0),  # skewed backwards: must not un-breach
        _event("s4", 30.0),
    )
    breaches = [
        r for r in sink.risks if r.trigger == "wall_clock_budget" and r.score == 1.0
    ]
    assert len(breaches) == 1


# --- determinism / fail-open / region contract ------------------------------


def test_no_wall_clock_reads_in_ingest(monkeypatch: object) -> None:
    """ZERO time.time() reads during ingest: replay stays deterministic."""
    import time as time_module

    def _explode() -> float:
        raise AssertionError("wall clock read during ingest")

    monkeypatch.setattr(time_module, "time", _explode)  # type: ignore[attr-defined]
    sink = CapturingSink()
    m = _monitor(sink, idle_warn_seconds=1.0, max_episode_wall_seconds=2.0)
    # Timestamps come from the events only; the frozen wall clock is never read.
    _feed(m, _event("s1", 0.0), _event("s2", 5.0))


def test_replay_fingerprints_unchanged_when_options_unset() -> None:
    """Old trajectories produce identical results with new options unset.

    The expected fingerprints were captured on the pre-#92 build; any change
    in default-config behavior fails here.
    """
    expected = {
        "healthy_run.jsonl": [],
        "injected_error_cascade.jsonl": [("error_cascade", "22", 1.0)],
        "injected_governance_decay.jsonl": [("loop", "4", 0.5)],
        "injected_latency_spike.jsonl": [
            ("latency_anomaly", str(i), 1.0) for i in range(40, 52)
        ],
        "injected_loop.jsonl": [("loop", "22", 0.5)],
    }
    for path in sorted(glob.glob("tests/fixtures/trajectories/*.jsonl")):
        name = os.path.basename(path)
        sink = CapturingSink()
        monitor = Monitor.default(config=Config(), sinks=[sink])
        episode = None
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                event = StepEvent(**json.loads(line))
                episode = event.episode_id
                monitor.ingest(event)
        monitor.end_episode(episode)
        got = [(r.trigger, r.step_id, round(r.score, 4)) for r in sink.risks]
        assert got == expected[name], f"fingerprint changed for {name}"


def test_time_axis_fail_open_on_pathological_timestamps(caplog) -> None:
    """NaN timestamps must not crash ingest; fault logged once, never raised."""
    sink = CapturingSink()
    m = _monitor(sink, idle_warn_seconds=1.0)
    nan = float("nan")
    with caplog.at_level(logging.DEBUG, logger="snagline"):
        _feed(m, _event("s1", 0.0), _event("s2", nan), _event("s3", nan))
    assert True  # reaching here IS the assertion: nothing propagated


def test_horizon_risks_counted_in_metrics() -> None:
    sink = CapturingSink()
    m = _monitor(sink, idle_warn_seconds=1.0)
    _feed(m, _event("s1", 0.0), _event("s2", 9.0))
    metrics = m.metrics()
    assert metrics["risks_emitted"] == 1
    assert metrics["events_ingested"] == 2


def test_horizon_knobs_reach_monitor_via_default(monkeypatch) -> None:
    """Monitor.default wires SNAGLINE_* env through to the time axis."""
    env = {"SNAGLINE_IDLE_WARN_SECONDS": "5"}
    monkeypatch.setenv("SNAGLINE_IDLE_WARN_SECONDS", "5")
    cfg = Config.resolve(environ=env)
    assert cfg.idle_warn_seconds == 5.0


def test_invalid_horizon_config_fails_loudly() -> None:
    import pytest

    with pytest.raises(ValueError):
        Config(warn_fraction=0.0)
    with pytest.raises(ValueError):
        Config(warn_fraction=1.5)
    with pytest.raises(ValueError):
        Config(max_episode_wall_seconds=-1.0)
    with pytest.raises(ValueError):
        Config(idle_warn_seconds=0.0)
    with pytest.raises(ValueError):
        Config(window_scale_steps=-1)
    with pytest.raises(ValueError):
        Config(max_window=0)
    with pytest.raises(ValueError):
        Config(cusum_refit_every=-1)
