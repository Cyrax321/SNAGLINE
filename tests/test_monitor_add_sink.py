"""Public Monitor.add_sink() and remove_sink() (issue #122).

The sidecar used to reach into the private ``_sinks`` list because Monitor
had no public way to attach a late sink. These tests pin the contract of the
public methods: a late sink receives subsequent dispatches, removal stops
dispatch, construction-time sinks are untouched, and fail-open semantics are
identical for late sinks.

Both sides are covered: an injected-failure stream that must dispatch to the
late sink (exact trigger asserted) and a healthy stream that must stay
silent.
"""

from __future__ import annotations

import pytest

from snagline.events import StepEvent
from snagline.monitor import Monitor
from snagline.risk import FailureRisk


def _event(step_id: str = "s1", episode_id: str = "ep-add") -> StepEvent:
    return StepEvent(
        step_id=step_id,
        episode_id=episode_id,
        timestamp=1.0,
        action_type="tool_call",
        action_signature="deadbeef",
    )


def _loop_risk(step_id: str = "s1") -> FailureRisk:
    return FailureRisk(
        episode_id="ep-add",
        step_id=step_id,
        score=0.9,
        trigger="loop",
        detail="repeated action",
        timestamp=1.0,
    )


class _FixedRiskDetector:
    """Returns one fixed verdict for every event: a risk, or None."""

    name = "fixed_risk"

    def __init__(self, risk: FailureRisk | None) -> None:
        self._risk = risk

    def observe(self, event: StepEvent) -> FailureRisk | None:
        return self._risk

    def reset(self, episode_id: str) -> None:
        return None


class _RecordingSink:
    def __init__(self) -> None:
        self.risks: list[FailureRisk] = []

    def emit(self, risk: FailureRisk) -> None:
        self.risks.append(risk)


class _RaisingSink:
    def emit(self, risk: FailureRisk) -> None:
        raise RuntimeError("boom in late sink")


def test_add_sink_dispatches_to_new_sink_on_subsequent_ingests():
    monitor = Monitor([_FixedRiskDetector(_loop_risk())], [])
    late = _RecordingSink()
    monitor.add_sink(late)
    monitor.ingest(_event())
    assert [r.trigger for r in late.risks] == ["loop"]
    assert late.risks[0].step_id == "s1"


def test_healthy_stream_stays_silent_after_add_sink():
    monitor = Monitor([_FixedRiskDetector(None)], [])
    late = _RecordingSink()
    monitor.add_sink(late)
    for step in range(10):
        monitor.ingest(_event(step_id=f"s{step}"))
    assert late.risks == []


def test_construction_and_late_sinks_both_receive_the_same_risk():
    construction = _RecordingSink()
    late = _RecordingSink()
    monitor = Monitor([_FixedRiskDetector(_loop_risk())], [construction])
    monitor.add_sink(late)
    monitor.ingest(_event())
    assert len(construction.risks) == 1
    assert len(late.risks) == 1
    assert construction.risks[0] == late.risks[0]


def test_remove_sink_stops_dispatch_and_reports_unknown():
    monitor = Monitor([_FixedRiskDetector(_loop_risk())], [])
    late = _RecordingSink()
    monitor.add_sink(late)
    assert monitor.remove_sink(late) is True
    monitor.ingest(_event())
    assert late.risks == []
    assert monitor.remove_sink(late) is False


def test_added_raising_sink_is_fail_open():
    good = _RecordingSink()
    monitor = Monitor([_FixedRiskDetector(_loop_risk())], [good])
    monitor.add_sink(_RaisingSink())
    monitor.ingest(_event())  # must not raise
    assert len(good.risks) == 1
    assert monitor.metrics()["sink_errors"] == 1


def test_added_raising_sink_propagates_with_fail_open_disabled():
    monitor = Monitor([_FixedRiskDetector(_loop_risk())], [], fail_open=False)
    monitor.add_sink(_RaisingSink())
    with pytest.raises(RuntimeError, match="boom in late sink"):
        monitor.ingest(_event())
