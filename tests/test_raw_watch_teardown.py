"""Tests that the raw ``watch`` adapter correctly tears down episode state.

Covers issue #1: ``watch`` must call ``Monitor.end_episode`` on exit (normal
and exceptional) so per-episode detector state does not leak across runs.
"""

from __future__ import annotations

from snagline import Monitor, watch
from snagline.detectors.loop import LoopDetector


class RecordingSink:
    def __init__(self) -> None:
        self.risks: list = []

    def emit(self, risk) -> None:
        self.risks.append(risk)


def test_watch_calls_end_episode_on_normal_exit():
    calls: list = []

    class FakeMonitor:
        def ingest(self, event) -> None:
            calls.append(("ingest", event))

        def end_episode(self, episode_id: str) -> None:
            calls.append(("end", episode_id))

    with watch(FakeMonitor(), "ep-x") as step:
        step("tool_call", tool_name="search", args="a")
    assert ("end", "ep-x") in calls


def test_watch_calls_end_episode_on_exception():
    calls: list = []

    class FakeMonitor:
        def ingest(self, event) -> None:
            calls.append(("ingest", event))

        def end_episode(self, episode_id: str) -> None:
            calls.append(("end", episode_id))

    try:
        with watch(FakeMonitor(), "ep-y") as step:
            step("tool_call", tool_name="search", args="a")
            raise ValueError("boom")
    except ValueError:
        pass
    assert ("end", "ep-y") in calls


def test_watch_clears_state_across_runs():
    # Without teardown, loop counts from a prior run would carry into a later
    # run with the same episode_id and cause false positives.
    mon = Monitor([LoopDetector()], [RecordingSink()])
    with watch(mon, "same-ep") as step:
        step("tool_call", tool_name="retry", args="x")
        step("tool_call", tool_name="retry", args="x")
    # second run, same episode id, fresh window expected (so 1 step != loop)
    with watch(mon, "same-ep") as step:
        step("tool_call", tool_name="retry", args="x")
    assert not any(r.trigger == "loop" for r in mon._sinks[0].risks)
