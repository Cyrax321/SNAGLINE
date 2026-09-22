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


def test_load_state_does_not_overwrite_output_action_types():
    """Issue #347: ``output_action_types`` is operator configuration, not
    per-episode state. A snapshot written by a stock-configured host carried
    the shipped default, and restoring it silently replaced a host's custom
    types -- so a real silent abort could be missed (the snapshot's types
    don't include this host's output action) or a clean ending flagged."""
    d = SilentAbortDetector(output_action_types={"assistant_message"})
    d.load_state({"output_action_types": ["message", "plan_step"], "last": {}})
    assert d.output_action_types == frozenset({"assistant_message"}), (
        "restoring a snapshot must not change which steps count as output"
    )

    # The live config then decides the restored episode's outcome.
    d.observe(_event(9))
    risk = d.finalize("ep")
    assert risk is not None and risk.trigger == "silent_abort"


def test_restored_last_event_is_scored_under_the_live_config():
    """Issue #347: the counterpart failure. A snapshot whose writer considered
    ``tool_call`` an output type must not suppress a silent abort on a host
    that does not -- the snapshot's judgment follows the snapshot, and the
    live detector's configuration follows the operator."""
    d = SilentAbortDetector(output_action_types={"message"})
    d.load_state(
        {
            "output_action_types": ["tool_call"],
            "last": {
                "ep": {
                    "step_id": "9",
                    "timestamp": 1.0,
                    "action_type": "tool_call",
                    "error": False,
                }
            },
        }
    )
    assert d.finalize("ep") is not None, "live types govern, not the snapshot's"


def test_dump_state_still_records_the_types_for_diagnostics():
    """Issue #347 regression guard: the fix lives in ``load_state`` only. The
    field stays in the snapshot (mirroring MeltdownDetector's ``window_size``)
    so existing readers keep working and the config is inspectable."""
    d = SilentAbortDetector(output_action_types={"assistant_message"})
    dumped = d.dump_state()
    assert dumped["output_action_types"] == ["assistant_message"]
    # A round trip through a stock-configured detector must not pick it up.
    stock = SilentAbortDetector()
    stock.load_state(dumped)
    assert stock.output_action_types == frozenset({"message", "plan_step"}), (
        "stock config must survive reading a custom-types snapshot"
    )
    stock.observe(_event(0, action_type="assistant_message"))
    assert stock.finalize("ep") is not None, (
        "stock config treats assistant_message as just another tool call"
    )
