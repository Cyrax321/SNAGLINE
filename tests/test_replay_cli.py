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
