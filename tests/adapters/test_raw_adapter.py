"""Tests for the raw adapter's ``watch()`` context manager (project.md §6.1)."""

from __future__ import annotations

from snagline.adapters.raw import watch
from snagline.monitor import Monitor


class RecordingSink:
    def __init__(self) -> None:
        self.risks: list = []

    def emit(self, risk) -> None:
        self.risks.append(risk)


class RecordingDetector:
    name = "rec"

    def __init__(self) -> None:
        self.events: list = []

    def observe(self, event):
        self.events.append(event)
        return None

    def reset(self, episode_id: str) -> None:
        pass


def test_watch_builds_and_ingests_events():
    rec = RecordingDetector()
    mon = Monitor([rec], [RecordingSink()])
    with watch(mon, "ep1") as step:
        step("tool_call", tool_name="search", args="q=cat", latency_ms=120, error=False)
        step("tool_call", tool_name="search", args="q=cat", latency_ms=130, error=False)

    assert len(rec.events) == 2
    assert rec.events[0].episode_id == "ep1"
    assert rec.events[0].step_id == "0"
    assert rec.events[1].step_id == "1"
    # same args -> same signature (loop detection depends on this)
    assert rec.events[0].action_signature == rec.events[1].action_signature


def test_watch_triggers_loop_detector():
    sink = RecordingSink()
    mon = Monitor.default(sinks=[sink])  # loop + error-cascade + this sink
    with watch(mon, "ep3") as step:
        for _ in range(5):
            step("tool_call", tool_name="retry", args="same", error=False)

    assert any(r.trigger == "loop" for r in sink.risks)


def test_watch_does_not_leak_metadata_into_risk():
    # metadata is accepted but never reaches a FailureRisk (privacy invariant)
    sink = RecordingSink()
    mon = Monitor.default(sinks=[sink])
    with watch(mon, "ep4") as step:
        step("tool_call", tool_name="retry", args="x", metadata={"secret": "nope"})
        step("tool_call", tool_name="retry", args="x")
        step("tool_call", tool_name="retry", args="x")
    for r in sink.risks:
        assert not hasattr(r, "metadata")
