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


def test_cap_enforced_across_restore_and_detector_freed(tmp_path) -> None:
    """Issue #273: restore() must enforce retention cap and evict detector state."""
    path = str(tmp_path / "snapshot.json")

    # Source monitor with high cap ingests 5 episodes A..E
    source = Monitor.default(
        config=Config(max_live_episodes=100, max_episode_wall_seconds=1000), sinks=[]
    )
    for i, ep in enumerate(["A", "B", "C", "D", "E"]):
        source.ingest(_event(ep, ts=float(i + 1)))
    source.snapshot(path)

    # Destination monitor has cap of 2
    target = Monitor.default(
        config=Config(max_live_episodes=2, max_episode_wall_seconds=1000), sinks=[]
    )
    target.restore(path)

    # Live episodes and retained count must be capped at 2, preserving the newest ('D', 'E')
    assert target.retained_episodes == 2
    assert target.metrics()["retained_episodes"] == 2
    assert list(target._live_episodes.keys()) == ["D", "E"]

    # Over-cap episodes' detector state ('A', 'B', 'C') must be completely evicted
    for det in target._detectors:
        if hasattr(det, "_windows"):
            windows = det._windows
            assert "A" not in windows
            assert "B" not in windows
            assert "C" not in windows
            assert "D" in windows
            assert "E" in windows

    # Clocks must also reflect only 'D' and 'E'
    assert set(target._clocks.keys()) == {"D", "E"}


def test_cap_enforced_across_restore_without_horizon_knobs(tmp_path) -> None:
    """Issue #273: restore() enforces cap even when horizon time-axis is inactive."""
    path = str(tmp_path / "snapshot_nohorizon.json")

    source = Monitor.default(
        config=Config(max_live_episodes=100, max_episode_wall_seconds=None), sinks=[]
    )
    for i, ep in enumerate(["A", "B", "C", "D", "E"]):
        source.ingest(_event(ep, ts=float(i + 1)))
    source.snapshot(path)

    target = Monitor.default(
        config=Config(max_live_episodes=2, max_episode_wall_seconds=None), sinks=[]
    )
    target.restore(path)

    assert target.retained_episodes == 2
    for det in target._detectors:
        if hasattr(det, "_windows"):
            assert len(det._windows) == 2
