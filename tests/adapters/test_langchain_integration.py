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
from langchain_core.outputs import ChatGeneration, LLMResult

from snagline import Monitor
from snagline.adapters.langchain_adapter import SnaglineCallbackHandler
from snagline.detectors.loop import LoopDetector
from snagline.detectors.token_runaway import TokenRunawayDetector
from snagline.risk import FailureRisk


class RecordingSink:
    def __init__(self) -> None:
        self.risks: list[FailureRisk] = []

    def emit(self, risk: FailureRisk) -> None:
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


def test_real_langchain_chat_model_usage_metadata_is_captured():
    # Issue #515: real langchain-core (>= 0.2) chat models report per-message
    # usage on AIMessage.usage_metadata and leave LLMResult.llm_output as None
    # on that path. Verified against a genuine LLMResult/ChatGeneration/AIMessage
    # built from the installed library -- no stubs -- so the duck-typed read is
    # proven against the real attribute layout this bug was reported against.
    rec = _Recorder()
    handler = SnaglineCallbackHandler(rec, "lc-tok-ep")
    handler.on_chat_model_start({}, [[]], run_id="tok-1")
    handler.on_llm_end(
        LLMResult(
            generations=[
                [
                    ChatGeneration(
                        message=AIMessage(
                            content="hi",
                            usage_metadata={
                                "input_tokens": 11,
                                "output_tokens": 7,
                                "total_tokens": 18,
                            },
                        )
                    )
                ]
            ],
            llm_output=None,
        ),
        run_id="tok-1",
    )
    msg_event = next(e for e in rec.events if e.action_type == "message")
    assert msg_event.tokens_in == 11
    assert msg_event.tokens_out == 7


def test_real_langchain_chat_usage_metadata_feeds_token_runaway():
    # End-to-end against the real library: chat-model usage_metadata must reach
    # the token-runaway detector, not vanish with llm_output=None.
    sink = RecordingSink()
    mon = Monitor([TokenRunawayDetector(budget_total_tokens=40)], [sink])
    handler = SnaglineCallbackHandler(mon, "lc-runaway-ep")
    for i in range(3):
        rid = f"tok-{i}"
        handler.on_chat_model_start({}, [[]], run_id=rid)
        handler.on_llm_end(
            LLMResult(
                generations=[
                    [
                        ChatGeneration(
                            message=AIMessage(
                                content="hi",
                                usage_metadata={
                                    "input_tokens": 8,
                                    "output_tokens": 8,
                                    "total_tokens": 16,
                                },
                            )
                        )
                    ]
                ],
                llm_output=None,
            ),
            run_id=rid,
        )
    assert any(r.trigger == "token_runaway" for r in sink.risks)


class _Recorder:
    """Minimal monitor stand-in that just captures ingested events."""

    def __init__(self) -> None:
        self.events: list = []

    def ingest(self, event) -> None:
        self.events.append(event)

    def end_episode(self, episode_id: str) -> None:
        pass
