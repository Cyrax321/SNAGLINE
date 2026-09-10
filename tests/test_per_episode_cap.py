"""Per-episode retention cap (issue #184).

The Monitor must not grow without bound when hosts never call end_episode.
A bounded LRU of live episode ids evicts the least-recently-seen id once
max_live_episodes is exceeded. Explicit end_episode still frees immediately
and the cap is only the safety net. Clocks are not retained when the horizon
feature is off, and the retained count is observable via metrics.
"""

from __future__ import annotations

from snagline import Config, Monitor, StepEvent, make_signature


def _event(episode_id: str, step_id: str = "0", ts: float = 0.0) -> StepEvent:
    return StepEvent(
        step_id=step_id,
        episode_id=episode_id,
        timestamp=ts,
        action_type="tool_call",
        action_signature=make_signature("tool_call", "t", "x"),
        tool_name="t",
    )


def test_cap_evicts_lru_and_frees_detector_state() -> None:
    mon = Monitor.default(config=Config(max_live_episodes=3), sinks=[])
    for eid in ["a", "b", "c"]:
        mon.ingest(_event(eid, ts=0))
    assert mon.retained_episodes == 3
    # Touch a so it becomes most recent; b becomes LRU.
    mon.ingest(_event("a", step_id="1", ts=1))
    assert list(mon._live_episodes.keys()) == ["b", "c", "a"]
    # Next new episode evicts b, not a.
    mon.ingest(_event("d", ts=2))
    assert mon.retained_episodes == 3
    assert "b" not in mon._live_episodes
    assert "a" in mon._live_episodes
    # Detector state for evicted id is gone.
    for det in mon._detectors:
        if hasattr(det, "_windows"):
            assert "b" not in det._windows  # type: ignore[attr-defined]
        if hasattr(det, "_stats"):
            # Latency detector keyed by (episode, tool)
            assert not any(k[0] == "b" for k in det._stats)  # type: ignore[attr-defined]


def test_cap_gauge_drops_and_detector_freed() -> None:
    mon = Monitor.default(config=Config(max_live_episodes=2), sinks=[])
    mon.ingest(_event("x", ts=0))
    mon.ingest(_event("y", ts=1))
    assert mon.metrics()["retained_episodes"] == 2
    mon.ingest(_event("z", ts=2))
    assert mon.metrics()["retained_episodes"] == 2
    assert mon.metrics()["live_episodes"] == 2


def test_explicit_end_still_frees_immediately() -> None:
    mon = Monitor.default(config=Config(max_live_episodes=10), sinks=[])
    mon.ingest(_event("ep1", ts=0))
    mon.ingest(_event("ep2", ts=1))
    assert mon.retained_episodes == 2
    mon.end_episode("ep1")
    assert mon.retained_episodes == 1
    assert "ep1" not in mon._live_episodes
    # Re-ingesting a freed id works and creates fresh state.
    mon.ingest(_event("ep1", ts=2))
    assert mon.retained_episodes == 2


def test_clocks_not_retained_when_horizon_off() -> None:
    mon = Monitor.default(
        config=Config(
            max_live_episodes=100, max_episode_wall_seconds=None, idle_warn_seconds=None
        ),
        sinks=[],
    )
    for i in range(50):
        mon.ingest(_event(f"ep-{i}", ts=float(i)))
    assert len(mon._clocks) == 0
    assert mon.retained_episodes == 50


def test_clocks_bounded_with_cap_when_horizon_on() -> None:
    mon = Monitor.default(
        config=Config(
            max_live_episodes=10, max_episode_wall_seconds=1000, idle_warn_seconds=10
        ),
        sinks=[],
    )
    for i in range(20):
        mon.ingest(_event(f"ep-{i}", ts=float(i)))
    assert len(mon._clocks) == 10
    assert mon.retained_episodes == 10


def test_retained_metric_exposed_and_prometheus_renders() -> None:
    from snagline.server.http_server import SidecarMetricsCollector

    mon = Monitor.default(config=Config(max_live_episodes=5), sinks=[])
    for i in range(7):
        mon.ingest(_event(f"ep-{i}", ts=float(i)))
    metrics = mon.metrics()
    assert metrics["retained_episodes"] == 5
    col = SidecarMetricsCollector()
    body = col.render_prometheus(metrics)
    assert "snagline_monitor_retained_episodes 5" in body


def test_eviction_is_silent_no_finalize_risk() -> None:
    class Sink:
        def __init__(self) -> None:
            self.risks: list = []

        def emit(self, risk) -> None:
            self.risks.append(risk)

    sink = Sink()
    mon = Monitor.default(
        config=Config(max_live_episodes=1, silent_abort_enabled=True), sinks=[sink]
    )
    mon.ingest(_event("ep1", ts=0))
    # ep1 would be silent_abort if explicitly ended
    mon.ingest(_event("ep2", ts=1))  # evicts ep1 silently
    assert len(sink.risks) == 0
    mon.end_episode("ep2")
    assert len(sink.risks) == 1
    assert sink.risks[0].trigger == "silent_abort"


def test_bounded_memory_repro() -> None:
    # Repro from issue: 10k distinct episodes without end_episode must stay bounded.
    mon = Monitor.default(config=Config(max_live_episodes=10000), sinks=[])
    for e in range(10_000):
        mon.ingest(
            StepEvent(
                step_id=f"{e}-0",
                episode_id=f"ep-{e}",
                timestamp=1000.0 + e,
                action_type="tool_call",
                action_signature=make_signature("tool_call", "search", str(e)),
                tool_name="search",
                latency_ms=100.0,
                error=False,
            )
        )
    assert mon.retained_episodes == 10000
    assert len(mon._live_episodes) == 10000
    # One more should not grow
    mon.ingest(
        StepEvent(
            step_id="extra",
            episode_id="ep-extra",
            timestamp=20000.0,
            action_type="tool_call",
            action_signature=make_signature("tool_call", "search", "extra"),
            tool_name="search",
            latency_ms=100.0,
        )
    )
    assert mon.retained_episodes == 10000


# --- Restore and the cap (issue #273) ----------------------------------------
#
# restore_dict used to apply every snapshot episode's detector state without
# registering ids in the live LRU (and its time-axis loop, the only part that
# did register, merely popped _clocks without the detector reset). So
# restored episodes' state was invisible to the LRU: it survived every future
# eviction forever while retained_episodes reported the cap.


def _event_full(episode_id: str, step: int, sig: str, ts: float) -> StepEvent:
    return StepEvent(
        step_id=f"{episode_id}-{step}",
        episode_id=episode_id,
        timestamp=ts,
        action_type="tool_call",
        action_signature=make_signature("tool_call", "t", sig),
        tool_name="t",
        latency_ms=10.0,
    )


def _restore_source_and_target(cap: int, horizon: bool, tmp_path):
    """Snapshot a 5-episode monitor; restore into one capped at ``cap``."""

    cfg_src = Config(max_live_episodes=100)
    cfg_dst = Config(max_live_episodes=cap)
    if horizon:
        cfg_src = Config(max_live_episodes=100, max_episode_wall_seconds=1000)
        cfg_dst = Config(max_live_episodes=cap, max_episode_wall_seconds=1000)
    src = Monitor.default(config=cfg_src, sinks=[])
    for ep in "ABCDE":
        for i in range(5):
            src.ingest(_event_full(ep, i, f"{ep}-sig-{i}", float(i)))
    path = str(tmp_path / "snap.json")
    src.snapshot(path)
    dst = Monitor.default(config=cfg_dst, sinks=[])
    dst.restore(path)
    return dst


def _loop_windows(mon: Monitor) -> set[str]:
    for det in mon._detectors:
        if getattr(det, "name", "") == "loop":
            return set(det._windows.keys())  # type: ignore[attr-defined]
    raise AssertionError("loop detector not found")


def test_restore_applies_cap_to_restored_state(tmp_path) -> None:
    """Restored detector state must respect the target monitor's cap, in
    both horizon configurations: the over-cap ids' windows are gone."""
    for horizon in (True, False):
        dst = _restore_source_and_target(cap=2, horizon=horizon, tmp_path=tmp_path)
        assert dst.metrics()["retained_episodes"] == 2
        assert _loop_windows(dst) == {"D", "E"}, f"horizon={horizon}"


def test_restore_eviction_is_full_teardown_not_clock_pop(tmp_path) -> None:
    """Over-cap restored ids must run the real eviction path: detector
    state reset (not just the clock dropped) and the LRU honest. The
    pre-fix code only popped the clock, leaving detector windows for the
    evicted ids live."""
    dst = _restore_source_and_target(cap=2, horizon=True, tmp_path=tmp_path)
    # Most-recent clock survives; evicted ids' clocks are gone too...
    assert set(dst._clocks.keys()) == {"D", "E"}
    # ...and so is their DETECTOR state -- the part the pre-fix time-axis
    # eviction never touched.
    assert _loop_windows(dst) == {"D", "E"}
    # Detector and clock state for the evicted ids is gone, proving the full
    # teardown path ran rather than only popping clocks.


def test_restored_state_survives_no_future_eviction_leak(tmp_path) -> None:
    """The issue's permanent-leak shape: feeding many new episodes used to
    evict only the new ids, so restored over-cap state (A/B/C) survived
    forever. With the fix they were torn down at restore time."""
    dst = _restore_source_and_target(cap=2, horizon=False, tmp_path=tmp_path)
    for n in range(10):
        for i in range(5):
            dst.ingest(_event_full(f"new{n}", i, f"n{n}-{i}", float(i)))
    survivors = _loop_windows(dst) & {"A", "B", "C"}
    assert survivors == set(), "restored over-cap ids must not outlive restore"


def test_restore_under_cap_keeps_all_state(tmp_path) -> None:
    """Sanity: restoring below the cap changes nothing vs pre-fix behavior
    (the round-trip parity tests pin that separately)."""
    dst = _restore_source_and_target(cap=10, horizon=False, tmp_path=tmp_path)
    assert dst.metrics()["retained_episodes"] == 5
    assert _loop_windows(dst) == {"A", "B", "C", "D", "E"}
