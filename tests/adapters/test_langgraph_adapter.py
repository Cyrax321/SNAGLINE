"""Tests for the LangGraph adapter (project.md §6.3).

Uses hand-built update items with the same shape LangGraph's
``stream_mode="updates"`` yields, so no langgraph install is needed.
A separate integration demo (examples/) exercises a real LangGraph-based
``create_agent`` graph.
"""

from __future__ import annotations

from snagline.adapters.langgraph_adapter import watch_graph
from snagline.events import StepEvent
from snagline.risk import FailureRisk


class _RecordingMonitor:
    """Duck-typed Monitor: records events, never dispatches risks."""

    def __init__(self):
        self.events: list[StepEvent] = []

    def ingest(self, event: StepEvent) -> None:
        self.events.append(event)

    def end_episode(self, episode_id: str) -> None:
        pass


class _RiskMonitor(_RecordingMonitor):
    def __init__(self):
        super().__init__()
        self.risks: list[FailureRisk] = []

    def ingest(self, event: StepEvent) -> None:
        super().ingest(event)
        # Mirror the real loop detector trigger so the adapter test can assert
        # that events it emits are loop-detectable at all.
        from collections import deque

        self._window = getattr(self, "_window", deque(maxlen=4))
        self._window.append(event.action_signature)
        sigs = list(self._window)
        if len(sigs) >= 4 and sigs[-1] in sigs[:-1] and sigs.count(sigs[-1]) >= 3:
            self.risks.append(FailureRisk("test", event.step_id, 0.5, "loop", "", 0.0))


def test_stream_passes_through_unchanged():
    monitor = _RecordingMonitor()
    stream = [{"a": {"x": 1}}, {"b": {"y": 2}}]
    out = list(watch_graph(monitor, "ep-1", iter(stream)))
    assert out == stream


def test_one_event_per_node_update():
    monitor = _RecordingMonitor()
    stream = [{"n1": {"x": 1}, "n2": {"y": 2}}, {"n1": {"x": 3}}]
    list(watch_graph(monitor, "ep-1", iter(stream)))
    assert len(monitor.events) == 3
    assert [e.tool_name for e in monitor.events] == ["n1", "n2", "n1"]
    assert all(e.action_type == "node_run" for e in monitor.events)
    assert all(e.episode_id == "ep-1" for e in monitor.events)
    # Same node + same update shape -> same signature (loop-detectable);
    # different shape -> different signature.
    assert monitor.events[0].action_signature == monitor.events[2].action_signature
    assert monitor.events[1].action_signature != monitor.events[0].action_signature


def test_error_key_and_exception_updates_set_error_flag():
    monitor = _RecordingMonitor()
    stream = [
        {"ok": {"x": 1}},
        {"boom": {"x": 1, "error": ValueError("node failed")}},
        {"crash": RuntimeError("unhandled")},
    ]
    list(watch_graph(monitor, "ep-1", iter(stream)))
    assert monitor.events[0].error is False
    assert monitor.events[1].error is True
    assert monitor.events[1].error_type == "ValueError"
    assert monitor.events[2].error is True
    assert monitor.events[2].error_type == "RuntimeError"


def test_latency_is_measured_between_yields():
    monitor = _RecordingMonitor()
    ticks = iter([0.0, 0.5, 0.75])

    def fake_stream():
        yield {"a": {"x": 1}}
        yield {"b": {"y": 2}}

    list(watch_graph(monitor, "ep-1", fake_stream(), clock=lambda: next(ticks)))
    # First yield: 0.0 -> 0.5 = 500 ms; second: 0.5 -> 0.75 = 250 ms.
    assert monitor.events[0].latency_ms == 500.0
    assert monitor.events[1].latency_ms == 250.0


def test_emitted_events_are_loop_detectable():
    monitor = _RiskMonitor()
    stream = [{"retry": {"x": 1, "error": "fail"}} for _ in range(4)]
    list(watch_graph(monitor, "ep-1", iter(stream)))
    assert monitor.risks, "identical repeated node updates should be loop-detectable"
