"""LangChain adapter (optional extra: ``pip install snagline-agent[langchain]``).

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
from typing import Any, Callable, Optional

from snagline.events import StepEvent, make_signature

try:  # optional dependency
    from langchain_core.callbacks import BaseCallbackHandler
except Exception:  # pragma: no cover - exercised only without LangChain
    BaseCallbackHandler = object


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
        agent_name: Optional[str] = None,
        clock: Optional[Callable[[], float]] = None,
    ) -> None:
        super().__init__()
        self._monitor = monitor
        self._episode_id = episode_id
        self._agent_name = agent_name
        self._clock = clock or time.time
        self._counter = itertools.count()
        self._runs: dict[str, dict] = {}

    def _emit(
        self,
        action_type: str,
        tool_name: Optional[str],
        *,
        args: Any = "",
        latency_ms: Optional[float] = None,
        error: bool = False,
        error_type: Optional[str] = None,
        tokens_in: Optional[int] = None,
        tokens_out: Optional[int] = None,
    ) -> StepEvent:
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
        )
        self._monitor.ingest(event)
        return event

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
        self, error: BaseException, *, run_id: Any, parent_run_id: Any = None, **kwargs: Any
    ) -> None:
        info = self._runs.pop(str(run_id), None) or {"tool": "tool", "args": ""}
        self._emit(
            "tool_call",
            info["tool"],
            args=info["args"],
            error=True,
            error_type=type(error).__name__,
        )

    # -- agent decisions ----------------------------------------------------
    def on_agent_action(self, action: Any, *, run_id: Any, parent_run_id: Any = None, **kwargs: Any) -> None:
        tool = getattr(action, "tool", None)
        tool_input = getattr(action, "tool_input", None)
        self._emit("plan_step", tool, args="" if tool_input is None else str(tool_input))

    def on_agent_finish(self, finish: Any, *, run_id: Any, parent_run_id: Any = None, **kwargs: Any) -> None:
        rv = getattr(finish, "return_values", None)
        self._emit("plan_step", "agent_finish", args="" if rv is None else str(rv))

    # -- LLM / chat model calls (latency + token counts) --------------------
    def on_llm_start(
        self, serialized: dict, prompts: list, *, run_id: Any, parent_run_id: Any = None, **kwargs: Any
    ) -> None:
        # Hash the real prompt text so distinct prompts get distinct signatures
        # (a constant args would make every LLM call look identical -> false loops).
        self._runs.setdefault(str(run_id), {"start": self._clock(), "type": "message", "tool": "llm", "args": str(prompts)})

    def on_chat_model_start(
        self, serialized: dict, messages: list, *, run_id: Any, parent_run_id: Any = None, **kwargs: Any
    ) -> None:
        self._runs.setdefault(str(run_id), {"start": self._clock(), "type": "message", "tool": "chat", "args": str(messages)})

    def on_llm_end(
        self, response: Any, *, run_id: Any, parent_run_id: Any = None, **kwargs: Any
    ) -> None:
        info = self._runs.pop(str(run_id), None)
        if info is None:
            return
        latency = (self._clock() - info["start"]) * 1000.0
        tokens_in = tokens_out = None
        llm_output = getattr(response, "llm_output", None)
        if isinstance(llm_output, dict):
            tu = llm_output.get("token_usage") or {}
            tokens_in = tu.get("prompt_tokens")
            tokens_out = tu.get("completion_tokens")
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
        self, serialized: dict, inputs: dict, *, run_id: Any, parent_run_id: Any = None, **kwargs: Any
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
        latency = (self._clock() - info["start"]) * 1000.0
        self._emit("plan_step", info["tool"], args=info["args"], latency_ms=latency)

    def close(self) -> None:
        """Clear per-episode detector state via the Monitor (project.md §2)."""
        self._monitor.end_episode(self._episode_id)
        self._runs.clear()
