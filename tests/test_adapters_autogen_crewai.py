"""Tests for the autogen and crewai adapters (next phase, step 4)."""

from __future__ import annotations

import asyncio

from snagline.adapters.autogen import SnaglineAutogenHandler, run_and_monitor
from snagline.adapters.crewai import observe_crewai_step, snagline_step_callback
from snagline.detectors.loop import LoopDetector
from snagline.monitor import Monitor


class _Collector:
    """A Monitor stand-in that records ingested events instead of detecting."""

    def __init__(self):
        self.events = []
        self.ended = []

    def ingest(self, event):
        self.events.append(event)

    def end_episode(self, episode_id):
        self.ended.append(episode_id)


def test_autogen_handler_maps_tool_call_request():
    mon = _Collector()
    h = SnaglineAutogenHandler(mon, "ep-1")  # noqa: F821
    events = h.observe(
        {
            "type": "ToolCallRequestEvent",
            "content": [{"name": "search", "arguments": "q=cat"}],
        }
    )
    assert len(events) == 1
    ev = events[0]
    assert ev.action_type == "tool_call"
    assert ev.tool_name == "search"
    assert ev.error is False
    assert mon.events[0] is ev


def test_autogen_handler_maps_tool_execution_error():
    mon = _Collector()
    h = SnaglineAutogenHandler(mon, "ep-1")  # noqa: F821
    events = h.observe(
        {
            "type": "ToolCallExecutionEvent",
            "content": [{"name": "search", "is_error": True, "result": "boom"}],
        }
    )
    assert events[0].error is True


def test_autogen_handler_falls_back_to_agent_step():
    mon = _Collector()
    h = SnaglineAutogenHandler(mon, "ep-1")  # noqa: F821
    events = h.observe({"type": "TextMessage", "content": "thinking..."})
    assert events[0].action_type == "agent_step"
    assert events[0].tool_name is None


def test_autogen_handler_close_ends_episode():
    mon = _Collector()
    h = SnaglineAutogenHandler(mon, "ep-9")  # noqa: F821
    h.close()
    assert mon.ended == ["ep-9"]


def test_crewai_callback_maps_tool_call():
    mon = _Collector()
    cb = snagline_step_callback(mon, "ep-1")  # noqa: F821
    cb({"action": {"tool": "calculator"}, "text": "12 + 30"})
    assert len(mon.events) == 1
    ev = mon.events[0]
    assert ev.action_type == "tool_call"
    assert ev.tool_name == "calculator"


def test_crewai_callback_maps_agent_step_without_tool():
    mon = _Collector()
    cb = snagline_step_callback(mon, "ep-1")  # noqa: F821
    cb({"text": "I will plan now"})
    ev = mon.events[0]
    assert ev.action_type == "agent_step"
    assert ev.tool_name is None


def test_crewai_callback_captures_error_and_latency():
    mon = _Collector()
    cb = snagline_step_callback(mon, "ep-1")  # noqa: F821
    cb({"action": {"tool": "search"}, "error": True, "latency_ms": 42.0})
    ev = mon.events[0]
    assert ev.error is True
    assert ev.latency_ms == 42.0


def test_crewai_observe_manual_mapping():
    mon = _Collector()
    ev = observe_crewai_step(mon, "ep-2", {"action": {"tool": "wiki"}, "text": "x"})  # noqa: F821
    assert ev.episode_id == "ep-2"
    assert ev.tool_name == "wiki"


def test_run_and_monitor_streams_autogen_events():
    class _FakeStream:
        def __init__(self, events):
            self._events = list(events)

        def __aiter__(self):
            return self

        async def __anext__(self):
            if self._events:
                return self._events.pop(0)
            raise StopAsyncIteration

    class _FakeAgent:
        def __init__(self, events):
            self._events = events

        async def run_stream(self, task):  # noqa: ARG002 - task unused in stub
            return _FakeStream(self._events)

    mon = _Collector()
    agent = _FakeAgent(
        [
            {"type": "TextMessage", "content": "hi"},
            {
                "type": "ToolCallRequestEvent",
                "content": [{"name": "t", "arguments": "x"}],
            },
        ]
    )
    result = asyncio.run(run_and_monitor(agent, "task", monitor=mon, episode_id="ep-1"))
    assert mon.ended == ["ep-1"]
    assert len(mon.events) == 2
    assert result["type"] == "ToolCallRequestEvent"


def test_run_and_monitor_raises_for_agent_with_no_stream_or_run():
    class _Bare:
        pass

    mon = _Collector()
    try:
        asyncio.run(run_and_monitor(_Bare(), "task", monitor=mon, episode_id="ep-1"))
        raise AssertionError("expected TypeError")
    except TypeError as exc:
        assert "run_stream" in str(exc)


def test_run_and_monitor_raises_when_stream_is_none():
    class _BrokenAgent:
        async def run_stream(self, task):  # noqa: ARG002 - task unused in stub
            return None

    mon = _Collector()
    try:
        asyncio.run(
            run_and_monitor(_BrokenAgent(), "task", monitor=mon, episode_id="ep-1")
        )
        raise AssertionError("expected TypeError")
    except TypeError as exc:
        assert "returned None" in str(exc)


def test_run_and_monitor_falls_back_to_run():
    class _LegacyAgent:
        async def run(self, task):
            return {"type": "TaskResult", "content": "done"}

    mon = _Collector()
    result = asyncio.run(
        run_and_monitor(_LegacyAgent(), "task", monitor=mon, episode_id="ep-1")
    )
    assert result["type"] == "TaskResult"
    assert mon.ended == ["ep-1"]
    assert len(mon.events) == 1


def test_run_and_monitor_ends_episode_when_stream_raises():
    class _ExplodingStream:
        def __aiter__(self):
            return self

        async def __anext__(self):
            raise RuntimeError("boom mid-stream")

    class _FakeAgent:
        async def run_stream(self, task):  # noqa: ARG002 - task unused in stub
            return _ExplodingStream()

    mon = _Collector()
    try:
        asyncio.run(
            run_and_monitor(_FakeAgent(), "task", monitor=mon, episode_id="ep-1")
        )
        raise AssertionError("expected RuntimeError")
    except RuntimeError:
        pass
    assert mon.ended == ["ep-1"]


def test_crewai_observe_reads_action_level_latency():
    mon = _Collector()
    ev = observe_crewai_step(  # noqa: F821
        mon, "ep-2", {"action": {"tool": "search", "latency_ms": 42.0}, "text": "x"}
    )
    assert ev.latency_ms == 42.0


def test_crewai_callback_close_ends_episode():
    mon = _Collector()
    cb = snagline_step_callback(mon, "ep-7")  # noqa: F821
    cb({"text": "hello"})
    cb.close()
    assert mon.ended == ["ep-7"]
    assert len(mon.events) == 1


def _crewai_step(tool_input, text, tool="search"):
    # Shape of crewai.agents.parser.AgentAction: the raw LLM block lands in
    # ``text``, the argument payload in ``tool_input``.
    return {"tool": tool, "tool_input": tool_input, "text": text}


def test_crewai_signature_is_stable_across_reworded_prose():
    # Issue #61: the signature must come from the tool arguments, not the step
    # output. A stuck agent retries the identical call while the model re-words
    # its thought each attempt; if that prose feeds the hash, every retry looks
    # unique and the loop detector never fires.
    mon = _Collector()
    args = '{"query": "Q3 revenue"}'
    first = observe_crewai_step(  # noqa: F821
        mon, "ep", _crewai_step(args, "Thought: let me search for that")
    )
    second = observe_crewai_step(  # noqa: F821
        mon, "ep", _crewai_step(args, "Thought: the search failed, retrying")
    )
    assert first.action_signature == second.action_signature
    # The nested ``action`` dict shape must resolve to the same signature.
    nested = observe_crewai_step(  # noqa: F821
        mon,
        "ep",
        {"action": {"tool": "search", "tool_input": args}, "text": "unrelated prose"},
    )
    assert nested.action_signature == first.action_signature


def test_crewai_different_tool_inputs_have_different_signatures():
    # Issue #61, other direction: leaving the arguments out entirely collapses a
    # legitimate iteration over distinct inputs into one signature, which the
    # loop detector then reports as a loop. Same prose, different arguments.
    mon = _Collector()
    text = "Thought: look up the next city"
    a = observe_crewai_step(mon, "ep", _crewai_step('{"q": "Paris"}', text))  # noqa: F821
    b = observe_crewai_step(mon, "ep", _crewai_step('{"q": "Berlin"}', text))  # noqa: F821
    assert a.action_signature != b.action_signature


def test_crewai_stuck_tool_loop_escalates_end_to_end():
    # The user-visible consequence, through a real Monitor and LoopDetector: an
    # agent that keeps issuing the same call with the same arguments must
    # escalate exactly once (the dedupe from issue #4 keeps it to one).
    risks = []

    class _Sink:
        def emit(self, risk):
            risks.append(risk)

    monitor = Monitor([LoopDetector()], [_Sink()])
    cb = snagline_step_callback(monitor, "ep-stuck")  # noqa: F821
    for i in range(5):
        cb(_crewai_step('{"query": "Q3 revenue"}', f"Thought: attempt {i}"))
    assert [r.trigger for r in risks] == ["loop"]


def test_crewai_exotic_tool_input_does_not_raise_into_the_host():
    # Mapping runs in the framework's thread, before Monitor.ingest and so
    # outside its fail-open guard. A payload JSON cannot canonicalize (unorderable
    # key types, a reference cycle, an arbitrary object) must fall back to the
    # previous stable part rather than surface inside the user's agent.
    mon = _Collector()
    cb = snagline_step_callback(mon, "ep-odd")  # noqa: F821
    cycle: dict = {"a": None}
    cycle["a"] = cycle
    for payload in ({1: "a", "b": 2}, cycle, {"x", "y"}, object()):
        cb({"tool": "search", "tool_input": payload, "text": "prose"})
    assert len(mon.events) == 4
    assert all(ev.action_signature for ev in mon.events)
