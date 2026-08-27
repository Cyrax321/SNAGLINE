"""Host-declared side_effect allowlist for callback adapters (issue #150)."""

from __future__ import annotations

from snagline.adapters.autogen import SnaglineAutogenHandler
from snagline.adapters.crewai import snagline_step_callback
from snagline.adapters.langchain_adapter import SnaglineCallbackHandler
from snagline.monitor import Monitor


class RecMonitor(Monitor):
    def __init__(self):
        super().__init__([], [], fail_open=True)
        self.events = []

    def ingest(self, event):
        self.events.append(event)
        super().ingest(event)


def _mon() -> RecMonitor:
    m = RecMonitor()
    return m


def test_langchain_allowlist_marks_side_effect_true():
    mon = _mon()
    h = SnaglineCallbackHandler(mon, "ep", side_effect_tools={"charge_card"})
    # Direct funnel
    e1 = h._emit("tool_call", "charge_card", args="amt=10")
    e2 = h._emit("tool_call", "search", args="q=cat")
    assert e1.side_effect is True
    assert e2.side_effect is False
    # plan_step with same name must not be marked (only tool_call)
    e3 = h._emit("plan_step", "charge_card", args="x")
    assert e3.side_effect is False


def test_langchain_allowlist_via_callback_hooks():
    mon = _mon()
    h = SnaglineCallbackHandler(mon, "ep", side_effect_tools=["charge_card"])
    # on_tool_start / on_tool_end path uses _emit internally
    h.on_tool_start({"name": "charge_card"}, "amt=10", run_id="r1")
    h.on_tool_end("ok", run_id="r1")
    assert mon.events[-1].side_effect is True
    assert mon.events[-1].tool_name == "charge_card"
    h.on_tool_start({"name": "search"}, "q=cat", run_id="r2")
    h.on_tool_end("ok", run_id="r2")
    assert mon.events[-1].side_effect is False


def test_langchain_nothing_inferred_from_payloads():
    mon = _mon()
    h = SnaglineCallbackHandler(mon, "ep", side_effect_tools={"charge_card"})
    # Args containing the substring should not trigger
    e = h._emit("tool_call", "search", args="side_effect=True")
    assert e.side_effect is False
    # Metadata containing side_effect should not be read (rejected in #88)
    # Simulate a tool whose serialized payload has metadata side_effect
    h.on_tool_start(
        {"name": "search", "metadata": {"side_effect": True}},
        "x",
        run_id="r3",
        metadata={"side_effect": True},
    )
    h.on_tool_end("ok", run_id="r3")
    assert mon.events[-1].side_effect is False


def test_langchain_default_without_allowlist_is_false():
    mon = _mon()
    h = SnaglineCallbackHandler(mon, "ep")
    e = h._emit("tool_call", "charge_card", args="x")
    assert e.side_effect is False
    h2 = SnaglineCallbackHandler(mon, "ep", side_effect_tools=set())
    e2 = h2._emit("tool_call", "charge_card", args="x")
    assert e2.side_effect is False


def test_autogen_allowlist_marks_side_effect():
    mon = _mon()
    h = SnaglineAutogenHandler(mon, "ep", side_effect_tools={"charge_card"})
    # ToolCallRequest shape
    h.observe(
        {
            "type": "ToolCallRequest",
            "content": [{"name": "charge_card", "arguments": "{}"}],
        }
    )
    assert mon.events[-1].side_effect is True
    h.observe(
        {"type": "ToolCallRequest", "content": [{"name": "search", "arguments": "{}"}]}
    )
    assert mon.events[-1].side_effect is False
    # ToolCallExecution shape also
    h.observe(
        {
            "type": "ToolCallExecution",
            "content": [{"name": "charge_card", "is_error": False}],
        }
    )
    assert mon.events[-1].side_effect is True


def test_autogen_nothing_inferred_and_default_false():
    mon = _mon()
    h = SnaglineAutogenHandler(mon, "ep", side_effect_tools={"charge_card"})
    # payload with side_effect key should be ignored
    h.observe(
        {
            "type": "ToolCallRequest",
            "content": [{"name": "search", "arguments": "side_effect"}],
            "metadata": {"side_effect": True},
        }
    )
    assert mon.events[-1].side_effect is False
    # default without allowlist
    mon2 = _mon()
    h2 = SnaglineAutogenHandler(mon2, "ep")
    h2.observe(
        {
            "type": "ToolCallRequest",
            "content": [{"name": "charge_card", "arguments": "{}"}],
        }
    )
    assert mon2.events[-1].side_effect is False


def test_crewai_callback_allowlist_marks_side_effect():
    mon = _mon()
    cb = snagline_step_callback(mon, "ep", side_effect_tools={"charge_card"})
    cb({"tool": "charge_card", "tool_input": {"amt": 10}, "text": "ok"})
    assert mon.events[-1].side_effect is True
    assert mon.events[-1].tool_name == "charge_card"
    cb({"tool": "search", "tool_input": {"q": "cat"}, "text": "ok"})
    assert mon.events[-1].side_effect is False
    # action wrapper
    cb({"action": {"tool": "charge_card", "tool_input": {"amt": 5}}})
    assert mon.events[-1].side_effect is True
    # agent_step (no tool) must stay False even if allowlist has similar string
    cb({"text": "thinking"})
    assert mon.events[-1].side_effect is False
    assert mon.events[-1].action_type == "agent_step"


def test_crewai_nothing_inferred_from_payload():
    mon = _mon()
    cb = snagline_step_callback(mon, "ep", side_effect_tools={"charge_card"})
    cb(
        {
            "tool": "search",
            "tool_input": {"side_effect": True},
            "metadata": {"side_effect": True},
        }
    )
    assert mon.events[-1].side_effect is False


def test_langgraph_and_claude_code_keep_false():
    # Pure payload adapters have no allowlist source; they stay False.
    from snagline.adapters.claude_code import payload_to_event
    from snagline.adapters.langgraph_adapter import watch_graph

    mon = _mon()
    # langgraph watch_graph
    stream = [{"node1": {"state": "x"}}, {"node2": {"error": "boom"}}]
    list(watch_graph(mon, "ep", iter(stream)))
    for e in mon.events:
        assert e.side_effect is False

    payload = {
        "hook_event_name": "PostToolUse",
        "session_id": "sess",
        "tool_use_id": "u1",
        "tool_name": "charge_card",
        "tool_input": {"amt": 10},
    }
    evt = payload_to_event(payload)
    assert evt is not None
    assert evt.side_effect is False
