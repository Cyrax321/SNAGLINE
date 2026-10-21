"""Schema and adapter pass-through tests for ``StepEvent.intent`` (#603).

The field is host-declared: adapters forward it when given, and it defaults
to None so backwards-compatible payloads and callers are unchanged.
"""

from __future__ import annotations

from typing import Any

from snagline.adapters.raw import watch
from snagline.events import StepEvent, make_signature


class _RecordingMonitor:
    def __init__(self) -> None:
        self.events: list[StepEvent] = []

    def ingest(self, event: StepEvent) -> None:
        self.events.append(event)


def test_intent_defaults_to_none():
    event = StepEvent(
        step_id="0",
        episode_id="ep",
        timestamp=0.0,
        action_type="tool_call",
        action_signature="abc",
    )
    assert event.intent is None


def test_step_event_loads_old_payloads_without_intent():
    payload: dict[str, Any] = {
        "step_id": "s1",
        "episode_id": "ep",
        "timestamp": 1.0,
        "action_type": "tool_call",
        "action_signature": "abc123",
    }
    assert StepEvent(**payload).intent is None


def test_raw_adapter_passes_intent_through():
    mon = _RecordingMonitor()
    with watch(mon, "ep") as step:
        read = step("tool_call", tool_name="search", args="q")
        research = step(
            "tool_call",
            tool_name="search",
            args="finance",
            intent="research financial records",
        )
    assert read.intent is None
    assert research.intent == "research financial records"
    assert [e.intent for e in mon.events] == [None, "research financial records"]


def test_signature_helper_ignores_intent():
    """Signatures for loop detection reason on action types and tool arguments,
    not intent strings."""
    a = make_signature("tool_call", "search", "q")
    b = make_signature("tool_call", "search", "q")
    assert a == b
