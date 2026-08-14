"""The single most important test file in the project (project.md §10 / §13).

Property under test: the fail-open guarantee. A monitoring library that can
crash or stall the thing it monitors is a non-starter for adoption, so:

  * With ``fail_open=True`` (the default), an exception raised inside a
    detector's ``observe`` or a sink's ``emit`` MUST NOT propagate out of
    ``Monitor.ingest``. It is logged and swallowed.
  * With ``fail_open=False``, those same exceptions MUST propagate, so a
    caller can opt into strict behavior (tests, debugging).

Written before any real detector exists -- it uses deliberately-broken
stub detectors/sinks, exercising only ``events.py`` + ``monitor.py``.
"""

from __future__ import annotations

import logging

import pytest

from snagline.events import StepEvent
from snagline.monitor import Monitor
from snagline.risk import FailureRisk


def _event(step_id: str = "s1", episode_id: str = "ep1") -> StepEvent:
    return StepEvent(
        step_id=step_id,
        episode_id=episode_id,
        timestamp=1.0,
        action_type="tool_call",
        action_signature="deadbeef",
    )


class RaisingDetector:
    """A detector that raises on every observe() call."""

    name = "raising_detector"

    def observe(self, event: StepEvent) -> FailureRisk | None:
        raise RuntimeError("boom in detector")

    def reset(self, episode_id: str) -> None:
        raise RuntimeError("boom in reset")


class RaisingSink:
    """A sink that raises on every emit() call."""

    def emit(self, risk: FailureRisk) -> None:
        raise RuntimeError("boom in sink")


class QuietDetector:
    """A detector that returns a fixed risk on the first call only."""

    name = "quiet_detector"

    def __init__(self) -> None:
        self._fired = False

    def observe(self, event: StepEvent) -> FailureRisk | None:
        if self._fired:
            return None
        self._fired = True
        return FailureRisk(
            event.episode_id,
            event.step_id,
            0.5,
            "loop",
            "synthetic risk",
            event.timestamp,
        )

    def reset(self, episode_id: str) -> None:
        self._fired = False


class RecordingSink:
    """A sink that records everything it receives."""

    def __init__(self) -> None:
        self.received: list[FailureRisk] = []

    def emit(self, risk: FailureRisk) -> None:
        self.received.append(risk)


def test_fail_open_default_swallows_detector_exception():
    monitor = Monitor(detectors=[RaisingDetector()], sinks=[RecordingSink()])
    # Must NOT raise.
    monitor.ingest(_event())
    monitor.ingest(_event("s2"))


def test_fail_open_default_swallows_sink_exception():
    monitor = Monitor(detectors=[QuietDetector()], sinks=[RaisingSink()])
    # QuietDetector produces a risk that RaisingSink chokes on; ingest must
    # still not raise.
    monitor.ingest(_event())


def test_fail_open_true_explicit_swallows_detector_exception():
    monitor = Monitor(
        detectors=[RaisingDetector()], sinks=[RecordingSink()], fail_open=True
    )
    monitor.ingest(_event())


def test_fail_open_false_propagates_detector_exception():
    monitor = Monitor(
        detectors=[RaisingDetector()], sinks=[RecordingSink()], fail_open=False
    )
    with pytest.raises(RuntimeError):
        monitor.ingest(_event())


def test_fail_open_false_propagates_sink_exception():
    monitor = Monitor(
        detectors=[QuietDetector()], sinks=[RaisingSink()], fail_open=False
    )
    with pytest.raises(RuntimeError):
        monitor.ingest(_event())


def test_fail_open_one_bad_detector_does_not_block_others():
    good = RecordingSink()
    monitor = Monitor(
        detectors=[RaisingDetector(), QuietDetector()],
        sinks=[good],
        fail_open=True,
    )
    monitor.ingest(_event())
    # The good detector's risk still reached the good sink despite the bad
    # detector raising.
    assert len(good.received) == 1


def test_fail_open_logs_the_exception(caplog):
    monitor = Monitor(detectors=[RaisingDetector()], sinks=[RecordingSink()])
    with caplog.at_level(logging.ERROR, logger="snagline"):
        monitor.ingest(_event())
    assert any("raising_detector" in r.message for r in caplog.records)
    assert any("fail-open" in r.message for r in caplog.records)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
