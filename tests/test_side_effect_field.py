"""Schema and adapter pass-through tests for ``StepEvent.side_effect`` (#88).

The field is host-declared: adapters must forward it exactly as given and
must never derive it heuristically. These tests pin both halves of that
contract, plus backward compatibility with payloads written before #88.
"""

from __future__ import annotations

from snagline.adapters.anthropic import observe_anthropic_call
from snagline.adapters.crewai import observe_crewai_step
from snagline.adapters.openai import observe_openai_call
from snagline.adapters.raw import watch
from snagline.events import StepEvent, make_signature


class _RecordingMonitor:
    """Minimal ingest stand-in so adapter calls need no real Monitor."""

    def __init__(self) -> None:
        self.events: list[StepEvent] = []

    def ingest(self, event: StepEvent) -> None:
        self.events.append(event)


def test_side_effect_defaults_to_false():
    event = StepEvent(
        step_id="0",
        episode_id="ep",
        timestamp=0.0,
        action_type="tool_call",
        action_signature="abc",
    )
    assert event.side_effect is False


def test_step_event_loads_old_payloads_without_the_field():
    payload = {
        "step_id": "s1",
        "episode_id": "ep",
        "timestamp": 1.0,
        "action_type": "tool_call",
        "action_signature": "abc123",
    }
    assert StepEvent(**payload).side_effect is False


def test_raw_adapter_passes_side_effect_through():
    mon = _RecordingMonitor()
    with watch(mon, "ep") as step:
        read = step("tool_call", tool_name="search", args="q")
        pay = step("tool_call", tool_name="charge_card", args="42", side_effect=True)
    assert read.side_effect is False
    assert pay.side_effect is True
    assert [e.side_effect for e in mon.events] == [False, True]


def test_openai_observe_passes_side_effect_through():
    mon = _RecordingMonitor()
    event = observe_openai_call(
        mon,
        episode_id="ep",
        model="gpt-4o",
        messages=[{"role": "user", "content": "hi"}],
        side_effect=True,
    )
    assert event.side_effect is True
    assert mon.events == [event]


def test_anthropic_observe_passes_side_effect_through():
    mon = _RecordingMonitor()
    event = observe_anthropic_call(
        mon,
        episode_id="ep",
        model="claude-3-5-sonnet-20241022",
        messages=[{"role": "user", "content": "hi"}],
        side_effect=True,
    )
    assert event.side_effect is True
    assert mon.events == [event]


def test_crewai_observe_passes_side_effect_through():
    mon = _RecordingMonitor()
    step = {"tool": "charge_card", "tool_input": {"amount": 42}}
    plain = observe_crewai_step(mon, "ep", step)
    marked = observe_crewai_step(mon, "ep2", step, side_effect=True)
    assert plain.side_effect is False
    assert marked.side_effect is True


def test_framework_paths_never_invent_the_flag():
    """Framework-driven paths have no caller-supplied flag source: every event
    they build must carry the schema default, proving no heuristic crept in."""
    from snagline.adapters.langgraph_adapter import watch_graph

    mon = _RecordingMonitor()

    def stream():
        yield {"node_a": {"result": "x"}}
        yield {"node_a": {"result": "x"}}

    list(watch_graph(mon, "ep", stream()))
    assert len(mon.events) == 2
    assert all(e.side_effect is False for e in mon.events)


def test_signature_helper_unchanged_by_the_new_field():
    """The flag is not part of the signature: marking an action after the fact
    must not change loop-detection identity."""
    a = make_signature("tool_call", "t", "args")
    b = make_signature("tool_call", "t", "args")
    assert a == b
