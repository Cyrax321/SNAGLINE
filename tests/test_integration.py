"""End-to-end integration test of the whole SNAGLINE architecture with NO framework.

This is the test that answers "does the architecture actually work as a system?":
a simulated agent loop is driven through the real ``raw`` adapter, with
deliberately injected failures, and we assert the Monitor -> detectors -> sink
pipeline emits the correct ``FailureRisk`` for each, while a clean run stays
silent. Runs in CI with zero dependencies.
"""

from __future__ import annotations

from snagline import Monitor, watch
from snagline.detectors.error_cascade import ErrorCascadeDetector
from snagline.detectors.latency_anomaly import LatencyAnomalyDetector
from snagline.detectors.loop import LoopDetector


class RecordingSink:
    def __init__(self) -> None:
        self.risks: list = []

    def emit(self, risk) -> None:
        self.risks.append(risk)


class EventRecorder:
    name = "rec"

    def __init__(self) -> None:
        self.events: list = []

    def observe(self, event):
        self.events.append(event)
        return None

    def reset(self, episode_id: str) -> None:
        pass


def _monitor() -> Monitor:
    rec = EventRecorder()
    sink = RecordingSink()
    mon = Monitor(
        [LoopDetector(), ErrorCascadeDetector(), LatencyAnomalyDetector(), rec],
        [sink],
    )
    mon._recorder = rec
    return mon


def test_full_agent_run_triggers_all_three_detectors():
    mon = _monitor()
    with watch(mon, "agent-ep") as step:
        # 1) healthy baseline: unique calls, stable latency (no false positives)
        for i in range(20):
            step("tool_call", tool_name="search", args=f"q-{i}", latency_ms=80.0)
        # 2) injected loop: identical retries
        for _ in range(4):
            step("tool_call", tool_name="retry", args="same", latency_ms=80.0)
        # 3) injected error cascade: 3 consecutive errors (varied args so the
        #    loop detector doesn't also fire)
        for j in range(3):
            step(
                "tool_call",
                tool_name="call",
                args=f"err-{j}",
                latency_ms=80.0,
                error=True,
                error_type="TimeoutError",
            )
        # 4) injected latency spike: sustained shift (varied args so loop stays quiet)
        for j in range(6):
            step("tool_call", tool_name="search", args=f"heavy-{j}", latency_ms=400.0)

    triggers = {r.trigger for r in mon._sinks[0].risks}
    assert "loop" in triggers, triggers
    assert "error_cascade" in triggers, triggers
    assert "latency_anomaly" in triggers, triggers
    # events actually flowed through the adapter into every detector
    assert len(mon._recorder.events) == 20 + 4 + 3 + 6


def test_clean_run_stays_silent():
    mon = _monitor()
    with watch(mon, "clean-ep") as step:
        for i in range(25):
            step("tool_call", tool_name="search", args=f"q-{i}", latency_ms=80.0)
    assert mon._sinks[0].risks == [], mon._sinks[0].risks
    assert len(mon._recorder.events) == 25
