"""Integration test: the LangChain adapter driven by a REAL LangChain runnable.

This is not a mocked callback invocation -- a genuine ``langchain_core`` model is
invoked with the ``SnaglineCallbackHandler`` attached, proving the callback
wiring works against the live library. Auto-skipped when LangChain isn't
installed, so it's safe in CI (which runs --no-deps).
"""

from __future__ import annotations

import pytest

pytest.importorskip("langchain_core")

from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, HumanMessage

from snagline import Monitor
from snagline.adapters.langchain_adapter import SnaglineCallbackHandler
from snagline.detectors.loop import LoopDetector
from snagline.sinks.console import ConsoleSink


class RecordingSink:
    def __init__(self) -> None:
        self.risks: list = []

    def emit(self, risk) -> None:
        self.risks.append(risk)


def test_real_langchain_run_produces_snakeline_events():
    rec = _Recorder()
    handler = SnaglineCallbackHandler(rec, "lc-ep")
    model = FakeMessagesListChatModel(responses=[AIMessage(content="hello")])

    model.invoke([HumanMessage(content="hi")], config={"callbacks": [handler]})

    # A real chat-model invocation must have produced at least a message event
    # via on_chat_model_start + on_llm_end.
    assert rec.events, "no events captured from a real LangChain run"
    assert any(e.action_type == "message" for e in rec.events)
    msg_event = next(e for e in rec.events if e.action_type == "message")
    assert msg_event.tool_name in ("llm", "chat")


def test_real_langchain_repeated_prompt_triggers_loop():
    sink = RecordingSink()
    mon = Monitor([LoopDetector()], [sink])
    handler = SnaglineCallbackHandler(mon, "lc-loop-ep")
    model = FakeMessagesListChatModel(
        responses=[AIMessage("a"), AIMessage("b"), AIMessage("c"), AIMessage("d")]
    )
    # Four identical prompts -> four identical message signatures -> loop.
    for _ in range(4):
        model.invoke([HumanMessage(content="repeat")], config={"callbacks": [handler]})

    assert any(r.trigger == "loop" for r in sink.risks)


class _Recorder:
    """Minimal monitor stand-in that just captures ingested events."""

    def __init__(self) -> None:
        self.events: list = []

    def ingest(self, event) -> None:
        self.events.append(event)

    def end_episode(self, episode_id: str) -> None:
        pass
