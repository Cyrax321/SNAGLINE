"""Tests for the autogen and crewai adapters (next phase, step 4)."""

from __future__ import annotations

import asyncio

from snagline.adapters.autogen import SnaglineAutogenHandler, run_and_monitor
from snagline.adapters.crewai import observe_crewai_step, snagline_step_callback


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
