"""Tests for the silent-abort completion check (issue #86)."""

from __future__ import annotations

import pytest

from snagline.detectors.silent_abort import SilentAbortDetector
from snagline.events import StepEvent
from snagline.monitor import Monitor


def _event(
    step_id: int, action_type: str = "tool_call", error: bool = False
) -> StepEvent:
    return StepEvent(
        step_id=str(step_id),
        episode_id="ep",
        timestamp=float(step_id),
        action_type=action_type,
        action_signature=f"s{step_id}",
        error=error,
    )


class ListSink:
    def __init__(self) -> None:
        self.risks: list = []

    def emit(self, risk) -> None:
        self.risks.append(risk)


def test_fires_when_episode_ends_on_tool_call():
    d = SilentAbortDetector()
    assert d.observe(_event(0)) is None
    risk = d.finalize("ep")
    assert risk is not None
    assert risk.trigger == "silent_abort"
    assert risk.detail == "episode ended on 'tool_call', not an output step"


def test_silent_on_output_step():
    for action in ("message", "plan_step"):
        d = SilentAbortDetector()
        d.observe(_event(0, action_type=action))
        assert d.finalize("ep") is None, action


def test_errored_final_step_not_flagged():
    d = SilentAbortDetector()
    d.observe(_event(0, error=True))
    assert d.finalize("ep") is None, "error-cascade owns error signals"


def test_finalize_is_consumed_once() -> None:
    d = SilentAbortDetector()
    d.observe(_event(0))
    assert d.finalize("ep") is not None
    assert d.finalize("ep") is None, "finalize pops its state"
    d.reset("ep")  # reset safe on empty state


def test_unknown_episode_finalizes_to_none():
    assert SilentAbortDetector().finalize("missing") is None


def test_monitor_end_episode_dispatches_finalize_risk():
    sink = ListSink()
    m = Monitor([SilentAbortDetector()], [sink])
    m.ingest(_event(0))
    assert sink.risks == [], "nothing may fire before end_episode"
    m.end_episode("ep")
    assert len(sink.risks) == 1
    assert sink.risks[0].trigger == "silent_abort"
    # State was consumed: a duplicate teardown must stay silent.
    m.end_episode("ep")
    assert len(sink.risks) == 1


def test_fail_open_finalize_never_propagates():
    class Boom(SilentAbortDetector):
        def finalize(self, episode_id):
            raise RuntimeError("boom")

    boom = Boom()
    boom.observe(_event(0))
    m_ok = Monitor([boom], [])
    m_ok.end_episode("ep")  # fail_open=True default: swallowed

    boom2 = Boom()
    boom2.observe(_event(0))
    with pytest.raises(RuntimeError):
        Monitor([boom2], [], fail_open=False).end_episode("ep")


def test_state_round_trip():
    d1 = SilentAbortDetector()
    d1.observe(_event(7))
    d2 = SilentAbortDetector()
    d2.load_state(d1.dump_state())
    r1, r2 = d1.finalize("ep"), d2.finalize("ep")
    assert r1 is not None and r2 is not None
    assert (r1.trigger, r1.step_id, r1.detail) == (r2.trigger, r2.step_id, r2.detail)
