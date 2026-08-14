"""Tests for the LangChain callback adapter (project.md §6.2 / §10).

These drive ``SnaglineCallbackHandler`` directly with stub callback arguments,
so they run in CI with ``--no-deps`` (no LangChain installed). The handler
module guards its LangChain import for exactly this reason.
"""

from __future__ import annotations

from snagline.adapters.langchain_adapter import SnaglineCallbackHandler
from snagline.monitor import Monitor


class RecordingSink:
    def __init__(self) -> None:
        self.risks: list = []
        self.events: list = []

    def emit(self, risk) -> None:
        self.risks.append(risk)


class RecMonitor(Monitor):
    """Monitor that also records every ingested event for assertions."""

    def __init__(self, detectors, sinks, fail_open: bool = True):
        super().__init__(detectors, sinks, fail_open=fail_open)
        self.events: list = []

    def ingest(self, event):
        self.events.append(event)
        super().ingest(event)


def _monitor() -> RecMonitor:
    return RecMonitor.default(sinks=[RecordingSink()])


def test_tool_call_emits_event_with_latency():
    mon = _monitor()
    h = SnaglineCallbackHandler(mon, "ep1")
    h.on_tool_start({"name": "search"}, "query=cat", run_id="r1")
    h.on_tool_end("result", run_id="r1")
    assert len(mon.events) == 1
    e = mon.events[0]
    assert e.action_type == "tool_call"
    assert e.tool_name == "search"
    assert e.latency_ms is not None and e.latency_ms >= 0
    assert e.error is False


def test_tool_error_emits_error_event():
    mon = _monitor()
    h = SnaglineCallbackHandler(mon, "ep1")
    h.on_tool_start({"name": "search"}, "query=cat", run_id="r2")
    h.on_tool_error(RuntimeError("boom"), run_id="r2")
    e = mon.events[-1]
    assert e.error is True
    assert e.error_type == "RuntimeError"


def test_agent_action_and_finish_emit_plan_steps():
    mon = _monitor()
    h = SnaglineCallbackHandler(mon, "ep1")
    h.on_agent_action(type("A", (), {"tool": "lookup", "tool_input": {"x": 1}})(), run_id="ra")
    h.on_agent_finish(type("F", (), {"return_values": {"out": 2}})(), run_id="rf")
    types = [e.action_type for e in mon.events]
    assert types == ["plan_step", "plan_step"]
    assert mon.events[0].tool_name == "lookup"


def test_llm_end_emits_message_with_tokens():
    mon = _monitor()
    h = SnaglineCallbackHandler(mon, "ep1")
    h.on_llm_start({"name": "llm"}, ["prompt"], run_id="r3")
    h.on_llm_end(type("R", (), {"llm_output": {"token_usage": {"prompt_tokens": 10, "completion_tokens": 20}}})(), run_id="r3")
    e = mon.events[-1]
    assert e.action_type == "message"
    assert e.tokens_in == 10 and e.tokens_out == 20


def test_repeated_tool_calls_trigger_loop_detector():
    mon = _monitor()
    h = SnaglineCallbackHandler(mon, "ep-loop")
    for i in range(4):
        h.on_tool_start({"name": "retry"}, "same-args", run_id=f"loop-{i}")
        h.on_tool_end("out", run_id=f"loop-{i}")
    assert any(r.trigger == "loop" for r in mon._sinks[0].risks)


def test_close_clears_episode_state():
    mon = _monitor()
    h = SnaglineCallbackHandler(mon, "ep-clear")
    h.on_tool_start({"name": "retry"}, "same-args", run_id="c1")
    h.on_tool_end("out", run_id="c1")
    h.on_tool_start({"name": "retry"}, "same-args", run_id="c2")
    h.close()  # should clear, so a fresh repeat does not immediately loop
    h.on_tool_start({"name": "retry"}, "same-args", run_id="c3")
    h.on_tool_end("out", run_id="c3")
    assert not mon._sinks[0].risks


def test_llm_error_emits_error_event():
    # LLM / chat-model failures route through on_llm_error (or
    # on_chat_model_error); the adapter must capture them as error events so
    # error_cascade can fire. Previously only on_tool_error existed.
    mon = _monitor()
    h = SnaglineCallbackHandler(mon, "ep1")
    h.on_chat_model_start({"name": "chat"}, [["user", "hi"]], run_id="rl")
    h.on_llm_error(RuntimeError("model down"), run_id="rl")
    e = mon.events[-1]
    assert e.error is True
    assert e.error_type == "RuntimeError"
    assert e.action_type == "message"


def test_chain_error_emits_error_event():
    mon = _monitor()
    h = SnaglineCallbackHandler(mon, "ep1")
    h.on_chain_start({"name": "planner"}, {"input": 1}, run_id="rc")
    h.on_chain_error(ValueError("bad plan"), run_id="rc")
    e = mon.events[-1]
    assert e.error is True
    assert e.error_type == "ValueError"
    assert e.action_type == "plan_step"
