"""Replay tests against the fixture trajectories (project.md §9/§10)."""

from __future__ import annotations

import os

from snagline.cli import replay
from snagline.monitor import Monitor

FIX = os.path.join(os.path.dirname(__file__), "fixtures", "trajectories")


class RecordingSink:
    def __init__(self) -> None:
        self.risks: list = []

    def emit(self, risk) -> None:
        self.risks.append(risk)


def _monitor() -> Monitor:
    return Monitor.default(sinks=[RecordingSink()])


def test_replay_healthy_no_false_positive():
    mon = _monitor()
    n = replay(os.path.join(FIX, "healthy_run.jsonl"), monitor=mon)
    assert n == 24
    assert mon._sinks[0].risks == [], f"false positives: {mon._sinks[0].risks}"


def test_replay_injected_loop_detected():
    mon = _monitor()
    replay(os.path.join(FIX, "injected_loop.jsonl"), monitor=mon)
    assert any(r.trigger == "loop" for r in mon._sinks[0].risks)


def test_replay_injected_cascade_detected():
    mon = _monitor()
    replay(os.path.join(FIX, "injected_error_cascade.jsonl"), monitor=mon)
    assert any(r.trigger == "error_cascade" for r in mon._sinks[0].risks)


def test_replay_injected_latency_detected():
    mon = _monitor()
    replay(os.path.join(FIX, "injected_latency_spike.jsonl"), monitor=mon)
    assert any(r.trigger == "latency_anomaly" for r in mon._sinks[0].risks)


def test_replay_healthy_no_latency_false_positive():
    mon = _monitor()
    replay(os.path.join(FIX, "healthy_run.jsonl"), monitor=mon)
    assert not any(r.trigger == "latency_anomaly" for r in mon._sinks[0].risks)


def test_replay_clears_episode_state_across_calls():
    # Issue #18: replay() must call end_episode() so detector state from one
    # trajectory does not leak into a later replay() on the same monitor.
    import tempfile

    from snagline.cli import _CountingSink

    def _write(lines):
        f = tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False)
        f.write("\n".join(lines) + "\n")
        f.close()
        return f.name

    file_a = _write(
        [
            '{"step_id":"0","episode_id":"ep-1","timestamp":1.0,'
            '"action_type":"tool_call","action_signature":"SIG"}',
            '{"step_id":"1","episode_id":"ep-1","timestamp":1.0,'
            '"action_type":"tool_call","action_signature":"SIG"}',
            '{"step_id":"2","episode_id":"ep-1","timestamp":1.0,'
            '"action_type":"tool_call","action_signature":"SIG"}',
        ]
    )
    file_b = _write(
        [
            '{"step_id":"3","episode_id":"ep-1","timestamp":1.0,'
            '"action_type":"tool_call","action_signature":"SIG"}',
            '{"step_id":"4","episode_id":"ep-1","timestamp":1.0,'
            '"action_type":"tool_call","action_signature":"SIG"}',
            '{"step_id":"5","episode_id":"ep-1","timestamp":1.0,'
            '"action_type":"tool_call","action_signature":"UNIQUE"}',
        ]
    )

    counter = _CountingSink()
    mon = Monitor.default(sinks=[counter])
    replay(file_a, monitor=mon)
    assert counter.count == 1  # loop fired on file A
    counter.count = 0
    counter.risks = []
    replay(file_b, monitor=mon)
    # No leak: file B alone (2x SIG + 1 unique) is below the repeat threshold.
    assert counter.count == 0

    os.unlink(file_a)
    os.unlink(file_b)
