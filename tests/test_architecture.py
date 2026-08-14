"""Architecture smoke tests: prove the extension points and invariants hold.

These are deliberately framework-free. They verify that SNAGLINE's core is a
sound foundation to build adapters/detectors/sinks on top of (project.md
§1.3 / §2): third parties can plug in custom detectors and sinks without
touching core, per-episode state is isolated and persists until ``end_episode``,
and the fail-open guarantee holds for custom code too.
"""

from __future__ import annotations

from snagline import Monitor
from snagline.detectors.base import Detector
from snagline.events import StepEvent
from snagline.risk import FailureRisk
from snagline.sinks.base import AlertSink


class CustomDetector:
    """A from-scratch detector (no core subclassing) -- must still work."""

    name = "custom_flag"

    def __init__(self) -> None:
        self._seen: set = set()

    def observe(self, event: StepEvent) -> FailureRisk | None:
        if event.step_id in self._seen:
            return FailureRisk(
                event.episode_id,
                event.step_id,
                0.9,
                "custom_flag",
                "duplicate step_id seen",
                event.timestamp,
            )
        self._seen.add(event.step_id)
        return None

    def reset(self, episode_id: str) -> None:
        self._seen = set()


class RecordingSink:
    """A from-scratch sink -- must receive every risk."""

    def __init__(self) -> None:
        self.risks: list = []

    def emit(self, risk: FailureRisk) -> None:
        self.risks.append(risk)


def _event(step_id: str, episode: str, sig: str) -> StepEvent:
    return StepEvent(
        step_id=step_id,
        episode_id=episode,
        timestamp=1.0,
        action_type="tool_call",
        action_signature=sig,
    )


def test_custom_detector_and_sink_plug_in_without_core_changes():
    sink = RecordingSink()
    mon = Monitor([CustomDetector()], [sink])
    mon.ingest(_event("s1", "ep", "a"))
    mon.ingest(_event("s1", "ep", "a"))  # duplicate step_id -> custom detector fires
    assert len(sink.risks) == 1
    assert sink.risks[0].trigger == "custom_flag"


def test_per_episode_state_is_isolated_and_persists():
    from snagline.detectors.loop import LoopDetector

    mon = Monitor([LoopDetector(window_size=4, repeat_threshold=3)], [RecordingSink()])
    # Episode A: two repeats -- not yet a loop
    mon.ingest(_event("1", "A", "x"))
    mon.ingest(_event("2", "A", "x"))
    # Episode B: many unique signatures -- must never trigger A's state
    for i in range(10):
        mon.ingest(_event(str(i), "B", f"b{i}"))
    # Back to A: one more repeat completes the loop for A only
    mon.ingest(_event("3", "A", "x"))
    # The single risk must belong to episode A, not B
    risks = mon._sinks[0].risks
    assert any(r.episode_id == "A" and r.trigger == "loop" for r in risks)
    assert not any(r.episode_id == "B" for r in risks)


def test_end_episode_clears_per_episode_state():
    from snagline.detectors.loop import LoopDetector

    mon = Monitor([LoopDetector(window_size=4, repeat_threshold=3)], [RecordingSink()])
    mon.ingest(_event("1", "A", "x"))
    mon.ingest(_event("2", "A", "x"))
    mon.end_episode("A")  # should wipe A's window
    # After reset, the same signature must start fresh -- no loop yet
    mon.ingest(_event("3", "A", "x"))
    assert not mon._sinks[0].risks


def test_fail_open_holds_for_custom_detector():
    class BoomDetector:
        name = "boom"

        def observe(self, event: StepEvent) -> FailureRisk | None:
            raise RuntimeError("custom boom")

        def reset(self, episode_id: str) -> None:
            raise RuntimeError("custom boom reset")

    # Must not raise under default fail-open.
    mon = Monitor([BoomDetector()], [RecordingSink()])
    mon.ingest(_event("1", "ep", "a"))
    mon.end_episode("ep")  # reset is also fail-open


def test_protocols_are_satisfied_structurally():
    # Detector/AlertSink are (non-runtime-checkable) protocols; conformance is
    # structural. Verify the required surface exists and is callable -- which is
    # all Monitor relies on.
    d = CustomDetector()
    s = RecordingSink()
    for obj, callables, attrs in [
        (d, ("observe", "reset"), ("name",)),
        (s, ("emit",), ()),
    ]:
        for m in callables:
            assert hasattr(obj, m) and callable(getattr(obj, m)), m
        for a in attrs:
            assert hasattr(obj, a), a
