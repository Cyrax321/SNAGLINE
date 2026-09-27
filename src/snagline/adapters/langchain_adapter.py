"""LangChain adapter (optional extra: ``pip install snagline[langchain]``).

Turns a LangChain run into ``StepEvent``s and feeds them to a ``Monitor``. This
is the highest-leverage framework integration (project.md §13 step 5) -- most
adopting projects already run LangChain.

The handler subclasses LangChain's ``BaseCallbackHandler`` and overrides the
callback hooks. Signatures were verified against the installed
``langchain-core`` (1.5.x) at build time; they have changed across versions, so
the import is guarded: this module imports fine without LangChain installed
(useful for CI), and only fails at *use* time if LangChain is missing.

No raw prompt/response content is retained -- only the one-way
``action_signature`` hash, tool name, latency, error flag, and token counts.
"""

from __future__ import annotations

import itertools
import time
from collections.abc import Callable
from typing import Any

from snagline.events import StepEvent, make_signature

try:  # optional dependency
    from langchain_core.callbacks import BaseCallbackHandler
except Exception:  # pragma: no cover - exercised only without LangChain
    BaseCallbackHandler = object  # type: ignore[assignment,misc]


class SnaglineCallbackHandler(BaseCallbackHandler):
    """Drop-in LangChain callback handler that monitors an agent run.

    Usage::

        from snagline import Monitor
        from snagline.adapters.langchain_adapter import SnaglineCallbackHandler

        monitor = Monitor.default()
        handler = SnaglineCallbackHandler(monitor, "ep-1")
        chain.invoke(question, config={"callbacks": [handler]})
        handler.close()  # optional: clears per-episode detector state
    """

    def __init__(
        self,
        monitor: Any,
        episode_id: str,
        agent_name: str | None = None,
        clock: Callable[[], float] | None = None,
        side_effect_tools: set[str] | list[str] | tuple[str, ...] | None = None,
    ) -> None:
        """Wire the handler to ``monitor`` for one ``episode_id``.

        ``clock`` defaults to :func:`time.perf_counter`: monotonic and
        high-resolution on every platform. The previous default,
        :func:`time.time`, advances in ~15.6 ms ticks on Windows, which
        quantized sub-tick latencies to zero and starved the CUSUM latency
        detector of usable samples (issue #155). Note that ``perf_counter``
        has no meaningful epoch: event timestamps produced with it are only
        comparable within one process. Detectors consume them solely as
        in-process latency differences, which is exactly what it guarantees.

        ``side_effect_tools`` is the host-declared allowlist for
        ``SideEffectGuardDetector`` (issue #150): tool names that the host
        knows are non-idempotent. When an emitted ``tool_call`` step has a
        ``tool_name`` in this set, the adapter marks ``side_effect=True``;
        nothing is ever inferred from payloads and ``_emit(side_effect=False)``
        remains the default for all other cases.
        """
        super().__init__()
        self._monitor = monitor
        self._episode_id = episode_id
        self._agent_name = agent_name
        self._clock = clock or time.perf_counter
        self._counter = itertools.count()
        self._runs: dict[str, dict] = {}
        self._side_effect_tools: set[str] = set(side_effect_tools or [])

    def _emit(
        self,
        action_type: str,
        tool_name: str | None,
        *,
        args: Any = "",
        latency_ms: float | None = None,
        error: bool = False,
        error_type: str | None = None,
        tokens_in: int | None = None,
        tokens_out: int | None = None,
        side_effect: bool = False,
    ) -> StepEvent:
        # Host-declared allowlist (issue #150): only a tool_call whose name
        # the host put in side_effect_tools becomes side_effect=True. Never
        # read metadata and never guess from args or payload content.
        if (
            not side_effect
            and action_type == "tool_call"
            and tool_name is not None
            and tool_name in self._side_effect_tools
        ):
            side_effect = True
        sig = make_signature(action_type, tool_name, str(args))
        event = StepEvent(
            step_id=str(next(self._counter)),
            episode_id=self._episode_id,
            timestamp=self._clock(),
            action_type=action_type,
            action_signature=sig,
            tool_name=tool_name,
            latency_ms=latency_ms,
            error=error,
            error_type=error_type,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            metadata={},
            side_effect=side_effect,
        )
        self._monitor.ingest(event)
        return event

    def _latency_from(self, info: dict) -> float | None:
        """Derive ``latency_ms`` from a run-tracking dict if a start time was
        captured. Both success and error callbacks share this so failed
        operations still carry latency for the CUSUM detector (issue #17).
        """
        start = info.get("start")
        if start is None:
            return None
        return (self._clock() - start) * 1000.0

    def _tokens_from_response(self, response: Any) -> tuple[int | None, int | None]:
        """Read ``tokens_in`` / ``tokens_out`` from an ``LLMResult``.

        Two report shapes exist, and both must be honoured or the
        token-runaway detector is silently starved (issue #515):

        * **Legacy**: completions-style models (``langchain-openai``) put a
          single aggregate dict on ``LLMResult.llm_output["token_usage"]`` with
          ``prompt_tokens`` / ``completion_tokens``.
        * **Chat models**: since langchain-core 0.2, a chat model reports
          per-message usage on ``AIMessage.usage_metadata``
          (``input_tokens`` / ``output_tokens``) and leaves ``llm_output`` as
          ``None`` on that path. A chat model is the ``create_agent`` /
          LangGraph default, so this is the common case.

        The legacy read wins when present (it is the tighter, already-aggregated
        number); otherwise we fall back to summing ``usage_metadata`` across
        ``response.generations``. Everything is duck-typed, so no LangChain
        import is needed and this stays testable under ``--no-deps``.
        """
        llm_output = getattr(response, "llm_output", None)
        if isinstance(llm_output, dict):
            tu = llm_output.get("token_usage") or {}
            tokens_in = tu.get("prompt_tokens")
            tokens_out = tu.get("completion_tokens")
            if tokens_in is not None or tokens_out is not None:
                return tokens_in, tokens_out

        generations = getattr(response, "generations", None) or []
        tokens_in = tokens_out = 0
        seen = False
        for batch in generations:
            # LLMResult.generations is List[List[Generation]] but stay tolerant
            # of a flat list, which some hand-built stubs pass.
            entries = batch if isinstance(batch, (list, tuple)) else (batch,)
            for generation in entries:
                # ChatGeneration carries the AIMessage in .message; accept a
                # bare message too so this works with either stub shape.
                message = getattr(generation, "message", None) or generation
                usage = getattr(message, "usage_metadata", None)
                if not isinstance(usage, dict):
                    continue
                i = usage.get("input_tokens")
                o = usage.get("output_tokens")
                if i is None and o is None:
                    continue
                tokens_in += i or 0
                tokens_out += o or 0
                seen = True
        if not seen:
            return None, None
        return tokens_in, tokens_out

    # -- tool calls ---------------------------------------------------------
    def on_tool_start(
        self,
        serialized: dict,
        input_str: str,
        *,
        run_id: Any,
        parent_run_id: Any = None,
        tags: list | None = None,
        metadata: dict | None = None,
        inputs: dict | None = None,
        **kwargs: Any,
    ) -> None:
        name = (serialized or {}).get("name") or (serialized or {}).get("id") or "tool"
        self._runs[str(run_id)] = {
            "start": self._clock(),
            "type": "tool_call",
            "tool": name,
            "args": input_str or ("" if inputs is None else str(inputs)),
        }

    def on_tool_end(
        self, output: Any, *, run_id: Any, parent_run_id: Any = None, **kwargs: Any
    ) -> None:
        info = self._runs.pop(str(run_id), None)
        if info is None:
            return
        latency = (self._clock() - info["start"]) * 1000.0
        self._emit("tool_call", info["tool"], args=info["args"], latency_ms=latency)

    def on_tool_error(
        self,
        error: BaseException,
        *,
        run_id: Any,
        parent_run_id: Any = None,
        **kwargs: Any,
    ) -> None:
        info = self._runs.pop(str(run_id), None) or {
            "tool": "tool",
            "args": "",
            "start": None,
        }
        latency_ms = self._latency_from(info)
        self._emit(
            "tool_call",
            info["tool"],
            args=info["args"],
            latency_ms=latency_ms,
            error=True,
            error_type=type(error).__name__,
        )

    # -- errors (LLM / chat / chain) -----------------------------------------
    def on_llm_error(
        self,
        error: BaseException,
        *,
        run_id: Any,
        parent_run_id: Any = None,
        **kwargs: Any,
    ) -> None:
        info = self._runs.pop(str(run_id), None) or {
            "tool": "llm",
            "args": "",
            "start": None,
        }
        latency_ms = self._latency_from(info)
        self._emit(
            "message",
            info["tool"],
            args=info["args"],
            latency_ms=latency_ms,
            error=True,
            error_type=type(error).__name__,
        )

    def on_chat_model_error(
        self,
        error: BaseException,
        *,
        run_id: Any,
        parent_run_id: Any = None,
        **kwargs: Any,
    ) -> None:
        # In langchain-core chat-model errors route through on_llm_error. Delegate
        # so exactly one error event is emitted regardless of which hook fires.
        self.on_llm_error(error, run_id=run_id, parent_run_id=parent_run_id, **kwargs)

    def on_chain_error(
        self,
        error: BaseException,
        *,
        run_id: Any,
        parent_run_id: Any = None,
        **kwargs: Any,
    ) -> None:
        info = self._runs.pop(str(run_id), None) or {
            "tool": "chain",
            "args": "",
            "start": None,
        }
        latency_ms = self._latency_from(info)
        self._emit(
            "plan_step",
            info["tool"],
            args=info["args"],
            latency_ms=latency_ms,
            error=True,
            error_type=type(error).__name__,
        )

    # -- agent decisions ----------------------------------------------------
    def on_agent_action(
        self, action: Any, *, run_id: Any, parent_run_id: Any = None, **kwargs: Any
    ) -> None:
        tool = getattr(action, "tool", None)
        tool_input = getattr(action, "tool_input", None)
        self._emit(
            "plan_step", tool, args="" if tool_input is None else str(tool_input)
        )

    def on_agent_finish(
        self, finish: Any, *, run_id: Any, parent_run_id: Any = None, **kwargs: Any
    ) -> None:
        rv = getattr(finish, "return_values", None)
        self._emit("plan_step", "agent_finish", args="" if rv is None else str(rv))

    # -- LLM / chat model calls (latency + token counts) --------------------
    def on_llm_start(
        self,
        serialized: dict,
        prompts: list,
        *,
        run_id: Any,
        parent_run_id: Any = None,
        **kwargs: Any,
    ) -> None:
        # Hash the real prompt text so distinct prompts get distinct signatures
        # (a constant args would make every LLM call look identical -> false loops).
        self._runs.setdefault(
            str(run_id),
            {
                "start": self._clock(),
                "type": "message",
                "tool": "llm",
                "args": str(prompts),
            },
        )

    def on_chat_model_start(
        self,
        serialized: dict,
        messages: list,
        *,
        run_id: Any,
        parent_run_id: Any = None,
        **kwargs: Any,
    ) -> None:
        self._runs.setdefault(
            str(run_id),
            {
                "start": self._clock(),
                "type": "message",
                "tool": "chat",
                "args": str(messages),
            },
        )

    def on_llm_end(
        self, response: Any, *, run_id: Any, parent_run_id: Any = None, **kwargs: Any
    ) -> None:
        info = self._runs.pop(str(run_id), None)
        if info is None:
            return
        latency = (self._clock() - info["start"]) * 1000.0
        tokens_in, tokens_out = self._tokens_from_response(response)
        self._emit(
            "message",
            info["tool"],
            args=info["args"],
            latency_ms=latency,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
        )

    # -- generic chains -----------------------------------------------------
    def on_chain_start(
        self,
        serialized: dict,
        inputs: dict,
        *,
        run_id: Any,
        parent_run_id: Any = None,
        **kwargs: Any,
    ) -> None:
        name = (serialized or {}).get("name") or (serialized or {}).get("id") or "chain"
        self._runs[str(run_id)] = {
            "start": self._clock(),
            "type": "plan_step",
            "tool": name,
            "args": str(inputs),
        }

    def on_chain_end(
        self, outputs: dict, *, run_id: Any, parent_run_id: Any = None, **kwargs: Any
    ) -> None:
        info = self._runs.pop(str(run_id), None)
        if info is None:
            return
        # Emit the planning step WITHOUT a latency sample: a chain's duration is
        # the whole reasoning turn (often tens of seconds for a real LLM), not a
        # single tool call. Feeding it to the latency/CUSUM detector would flag
        # every nested LangChain/LangGraph run as an anomaly (issue #10). The
        # latency detector also refuses non-``tool_call`` action types as a
        # belt-and-suspenders guard.
        self._emit("plan_step", info["tool"], args=info["args"])

    def close(self) -> None:
        """Clear per-episode detector state via the Monitor (project.md §2)."""
        self._monitor.end_episode(self._episode_id)
        self._runs.clear()
