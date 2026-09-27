"""Tests for the LangChain callback adapter (project.md §6.2 / §10).

These drive ``SnaglineCallbackHandler`` directly with stub callback arguments,
so they run in CI with ``--no-deps`` (no LangChain installed). The handler
module guards its LangChain import for exactly this reason.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, cast

from snagline.adapters.langchain_adapter import SnaglineCallbackHandler
from snagline.events import StepEvent
from snagline.monitor import Monitor
from snagline.risk import FailureRisk


class RecordingSink:
    def __init__(self) -> None:
        self.risks: list[FailureRisk] = []
        self.events: list[StepEvent] = []

    def emit(self, risk: FailureRisk) -> None:
        self.risks.append(risk)


class RecMonitor(Monitor):
    """Monitor that also records every ingested event for assertions."""

    def __init__(
        self,
        detectors: list[Any],
        sinks: list[Any],
        fail_open: bool = True,
    ) -> None:
        super().__init__(detectors, sinks, fail_open=fail_open)
        self.events: list[StepEvent] = []

    def ingest(self, event: StepEvent) -> None:
        self.events.append(event)
        super().ingest(event)


def _monitor() -> RecMonitor:
    return cast(RecMonitor, RecMonitor.default(sinks=[RecordingSink()]))


class _StubMessage:
    """Duck-types ``langchain_core.messages.AIMessage`` for the token path."""

    def __init__(self, usage_metadata: dict[str, int] | None) -> None:
        self.usage_metadata = usage_metadata


class _StubGeneration:
    """Duck-types ``langchain_core.outputs.ChatGeneration``."""

    def __init__(self, message: _StubMessage) -> None:
        self.message = message


class _StubLLMResult:
    """Duck-types ``langchain_core.outputs.LLMResult``."""

    def __init__(
        self,
        generations: Sequence[Sequence[_StubGeneration]] = (),
        llm_output: dict[str, Any] | None = None,
    ) -> None:
        self.generations = [list(batch) for batch in generations]
        self.llm_output = llm_output


def test_tool_call_emits_event_with_latency() -> None:
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


def test_tool_error_emits_error_event() -> None:
    mon = _monitor()
    h = SnaglineCallbackHandler(mon, "ep1")
    h.on_tool_start({"name": "search"}, "query=cat", run_id="r2")
    h.on_tool_error(RuntimeError("boom"), run_id="r2")
    e = mon.events[-1]
    assert e.error is True
    assert e.error_type == "RuntimeError"


def test_agent_action_and_finish_emit_plan_steps() -> None:
    mon = _monitor()
    h = SnaglineCallbackHandler(mon, "ep1")
    h.on_agent_action(
        type("A", (), {"tool": "lookup", "tool_input": {"x": 1}})(), run_id="ra"
    )
    h.on_agent_finish(type("F", (), {"return_values": {"out": 2}})(), run_id="rf")
    types = [e.action_type for e in mon.events]
    assert types == ["plan_step", "plan_step"]
    assert mon.events[0].tool_name == "lookup"


def test_llm_end_emits_message_with_tokens() -> None:
    mon = _monitor()
    h = SnaglineCallbackHandler(mon, "ep1")
    h.on_llm_start({"name": "llm"}, ["prompt"], run_id="r3")
    h.on_llm_end(
        type(
            "R",
            (),
            {
                "llm_output": {
                    "token_usage": {"prompt_tokens": 10, "completion_tokens": 20}
                }
            },
        )(),
        run_id="r3",
    )
    e = mon.events[-1]
    assert e.action_type == "message"
    assert e.tokens_in == 10 and e.tokens_out == 20


def test_llm_end_reads_usage_metadata_when_llm_output_is_none() -> None:
    # Issue #515: since langchain-core 0.2 a chat model reports per-message
    # usage on AIMessage.usage_metadata and leaves LLMResult.llm_output as
    # None on that path. The old code read only llm_output, so a chat model --
    # the create_agent / LangGraph default -- reported no tokens at all and
    # the token-runaway detector was silently starved for the whole run.
    mon = _monitor()
    h = SnaglineCallbackHandler(mon, "ep1")
    h.on_chat_model_start({"name": "chat"}, [["user", "hi"]], run_id="r4")
    h.on_llm_end(
        _StubLLMResult(
            generations=[
                [
                    _StubGeneration(
                        _StubMessage(
                            {"input_tokens": 11, "output_tokens": 7, "total_tokens": 18}
                        )
                    )
                ]
            ],
            llm_output=None,
        ),
        run_id="r4",
    )
    e = mon.events[-1]
    assert e.action_type == "message"
    assert e.tokens_in == 11 and e.tokens_out == 7


def test_llm_end_sums_usage_metadata_across_generations() -> None:
    # A batched LLMResult carries one message per generation; the fallback
    # must aggregate rather than take the first.
    mon = _monitor()
    h = SnaglineCallbackHandler(mon, "ep1")
    h.on_chat_model_start({"name": "chat"}, [["user", "hi"]], run_id="r5")
    h.on_llm_end(
        _StubLLMResult(
            generations=[
                [
                    _StubGeneration(
                        _StubMessage({"input_tokens": 4, "output_tokens": 1})
                    )
                ],
                [
                    _StubGeneration(
                        _StubMessage({"input_tokens": 6, "output_tokens": 2})
                    )
                ],
            ],
            llm_output=None,
        ),
        run_id="r5",
    )
    e = mon.events[-1]
    assert e.tokens_in == 10 and e.tokens_out == 3


def test_llm_end_prefers_legacy_llm_output_when_both_present() -> None:
    # The legacy aggregate is the tighter, already-summed number, so it wins
    # when both shapes are available; usage_metadata is only a fallback.
    mon = _monitor()
    h = SnaglineCallbackHandler(mon, "ep1")
    h.on_llm_start({"name": "llm"}, ["prompt"], run_id="r6")
    h.on_llm_end(
        _StubLLMResult(
            generations=[
                [
                    _StubGeneration(
                        _StubMessage({"input_tokens": 99, "output_tokens": 99})
                    )
                ]
            ],
            llm_output={"token_usage": {"prompt_tokens": 10, "completion_tokens": 20}},
        ),
        run_id="r6",
    )
    e = mon.events[-1]
    assert e.tokens_in == 10 and e.tokens_out == 20


def test_llm_end_without_any_usage_reports_no_tokens() -> None:
    # Neither shape present -> still None, None. Nothing is fabricated and the
    # detector keeps ignoring the step, matching documented behaviour.
    mon = _monitor()
    h = SnaglineCallbackHandler(mon, "ep1")
    h.on_llm_start({"name": "llm"}, ["prompt"], run_id="r7")
    h.on_llm_end(
        _StubLLMResult(generations=[[_StubGeneration(_StubMessage(None))]]),
        run_id="r7",
    )
    e = mon.events[-1]
    assert e.tokens_in is None and e.tokens_out is None


def test_chat_model_usage_metadata_feeds_token_runaway_detector() -> None:
    # End-to-end proof that the fix removes the starvation: a chat-model run
    # whose usage_metadata crosses a budget envelope now raises a risk where
    # before it emitted None tokens and the detector stayed permanently quiet.
    from snagline.detectors.token_runaway import TokenRunawayDetector

    sink = RecordingSink()
    mon = RecMonitor([TokenRunawayDetector(budget_total_tokens=50)], [sink])
    h = SnaglineCallbackHandler(cast(Monitor, mon), "ep-runaway")
    for i in range(4):
        rid = f"run-{i}"
        h.on_chat_model_start({"name": "chat"}, [["user", "hi"]], run_id=rid)
        h.on_llm_end(
            _StubLLMResult(
                generations=[
                    [
                        _StubGeneration(
                            _StubMessage({"input_tokens": 20, "output_tokens": 20})
                        )
                    ]
                ],
                llm_output=None,
            ),
            run_id=rid,
        )
    assert any(r.trigger == "token_runaway" for r in sink.risks)
    assert any(r.trigger == "budget_breach" for r in sink.risks)


def test_repeated_tool_calls_trigger_loop_detector() -> None:
    mon = _monitor()
    h = SnaglineCallbackHandler(mon, "ep-loop")
    for i in range(4):
        h.on_tool_start({"name": "retry"}, "same-args", run_id=f"loop-{i}")
        h.on_tool_end("out", run_id=f"loop-{i}")
    sink = cast(RecordingSink, mon._sinks[0])
    assert any(r.trigger == "loop" for r in sink.risks)


def test_close_clears_episode_state() -> None:
    mon = _monitor()
    h = SnaglineCallbackHandler(mon, "ep-clear")
    h.on_tool_start({"name": "retry"}, "same-args", run_id="c1")
    h.on_tool_end("out", run_id="c1")
    h.on_tool_start({"name": "retry"}, "same-args", run_id="c2")
    h.close()  # should clear, so a fresh repeat does not immediately loop
    h.on_tool_start({"name": "retry"}, "same-args", run_id="c3")
    h.on_tool_end("out", run_id="c3")
    assert not cast(RecordingSink, mon._sinks[0]).risks


def test_llm_error_emits_error_event() -> None:
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


def test_chain_error_emits_error_event() -> None:
    mon = _monitor()
    h = SnaglineCallbackHandler(mon, "ep1")
    h.on_chain_start({"name": "planner"}, {"input": 1}, run_id="rc")
    h.on_chain_error(ValueError("bad plan"), run_id="rc")
    e = mon.events[-1]
    assert e.error is True
    assert e.error_type == "ValueError"
    assert e.action_type == "plan_step"


def test_error_callbacks_carry_latency_ms() -> None:
    # Issue #17: error callbacks (tool/llm/chain) must compute latency_ms from
    # the captured start time, just like the success callbacks, so the CUSUM
    # detector can analyze latency for failed operations too.
    #
    # A scripted fake clock keeps this deterministic on every platform:
    # time.monotonic ticks at roughly 15.6 ms on Windows, so the previous
    # 10 ms sleep measured as latency 0.0 and failed the positive-latency
    # assertion intermittently there (first windows-latest CI leg). Read
    # order per operation: on_*_start captures one reading, the error
    # callback reads twice more (once in _latency_from, once for the event
    # timestamp), so nine scripted readings cover all three operations; any
    # unexpected extra read raises StopIteration and fails loudly instead
    # of drifting.
    reads = iter(
        [
            10.0,
            10.5,
            11.0,  # tool pair: 500 ms
            20.0,
            20.25,
            21.0,  # chat model pair: 250 ms
            30.0,
            35.0,
            36.0,  # chain pair: 5000 ms
        ]
    )

    mon = _monitor()
    h = SnaglineCallbackHandler(mon, "ep-lat", clock=lambda: next(reads))

    h.on_tool_start({"name": "search"}, "query=cat", run_id="e1")
    h.on_tool_error(RuntimeError("timeout"), run_id="e1")
    h.on_chat_model_start({"name": "chat"}, [["user", "hi"]], run_id="e2")
    h.on_llm_error(RuntimeError("model down"), run_id="e2")
    h.on_chain_start({"name": "planner"}, {"input": 1}, run_id="e3")
    h.on_chain_error(ValueError("bad plan"), run_id="e3")

    errors = [e for e in mon.events if e.error]
    assert len(errors) == 3
    # Exact deltas from the scripted clock (values chosen to be binary
    # exact), stronger than the former > 0.
    assert [e.latency_ms for e in errors] == [500.0, 250.0, 5000.0]
