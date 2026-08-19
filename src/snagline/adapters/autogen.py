"""Autogen adapter (optional extra: ``pip install snagline-agent[autogen]``).

Turns Autogen agent events into ``StepEvent``s and feeds them to a ``Monitor``.
Autogen is async-first; the most stable integration point is the event stream
emitted by ``agent.run_stream(task)``. This adapter provides a handler you can
feed events to manually, plus ``run_and_monitor`` which wraps ``run_stream``
and streams every event through the handler.

The adapter is duck-typed: it reads Autogen event objects via ``model_dump()``
with fallbacks, so it imports fine without Autogen installed (safe in CI) and
never hard-couples to a specific Autogen release. No raw prompt/response content
is retained -- only the one-way ``action_signature`` hash, tool name, latency
(when available), and error flag.

Autogen event objects are pydantic models; we read them via ``model_dump()``
when present and fall back to attribute access / ``__dict__`` so the adapter
works across Autogen versions without hard coupling.
"""

from __future__ import annotations

import itertools
import time
from collections.abc import Callable
from typing import Any

from snagline.events import StepEvent, make_signature


def _to_dict(event: Any) -> dict[str, Any]:
    if isinstance(event, dict):
        return event
    model_dump = getattr(event, "model_dump", None)
    if callable(model_dump):
        try:
            return model_dump()
        except Exception:
            pass
    if hasattr(event, "__dict__"):
        return dict(vars(event))
    return {}


class SnaglineAutogenHandler:
    """Observer that maps Autogen events to ``StepEvent``s and ingests them.

    Usage (manual wiring into an event stream)::

        handler = SnaglineAutogenHandler(monitor, "ep-1")
        async for event in agent.run_stream(task):
            handler.observe(event)
        handler.close()

    Or, more simply, use :func:`run_and_monitor`::

        result = await run_and_monitor(agent, task, monitor=monitor, episode_id="ep-1")
    """

    def __init__(
        self,
        monitor: Any,
        episode_id: str,
        agent_name: str | None = None,
        clock: Callable[[], float] | None = None,
    ) -> None:
        self._monitor = monitor
        self._episode_id = episode_id
        self._agent_name = agent_name
        self._clock = clock or time.time
        self._counter = itertools.count()

    def observe(self, event: Any) -> list[StepEvent]:
        d = _to_dict(event)
        kind = str(d.get("type") or d.get("event_type") or "")
        emitted: list[StepEvent] = []

        # Tool call requests: content is a list of FunctionCall(name, arguments).
        if "ToolCallRequest" in kind or "tool_call" in kind.lower():
            calls = d.get("content") or []
            if not isinstance(calls, list):
                calls = [calls]
            for call in calls:
                cd = _to_dict(call)
                name = cd.get("name") or cd.get("tool") or "tool"
                args = cd.get("arguments") if "arguments" in cd else str(call)
                emitted.append(
                    self._emit("tool_call", name, args=str(args), error=False)
                )
            return emitted

        # Tool call execution results: content is a list of FunctionExecutionResult.
        if "ToolCallExecution" in kind or "tool_result" in kind.lower():
            results = d.get("content") or []
            if not isinstance(results, list):
                results = [results]
            for res in results:
                rd = _to_dict(res)
                name = rd.get("name") or rd.get("tool") or "tool"
                is_error = bool(rd.get("is_error") or rd.get("error"))
                emitted.append(
                    self._emit("tool_call", name, args=str(rd), error=is_error)
                )
            return emitted

        # Anything else (TextMessage, Response, TaskResult, ...) is an agent step.
        content = d.get("content") or d.get("text") or d.get("thought") or ""
        emitted.append(self._emit("agent_step", None, args=str(content), error=False))
        return emitted

    def _emit(
        self,
        action_type: str,
        tool_name: str | None,
        *,
        args: str,
        error: bool,
        latency_ms: float | None = None,
    ) -> StepEvent:
        sig = make_signature(action_type, tool_name, args)
        event = StepEvent(
            step_id=str(next(self._counter)),
            episode_id=self._episode_id,
            timestamp=self._clock(),
            action_type=action_type,
            action_signature=sig,
            tool_name=tool_name,
            latency_ms=latency_ms,
            error=error,
            metadata={"agent_name": self._agent_name, "adapter": "autogen"},
        )
        self._monitor.ingest(event)
        return event

    def close(self) -> None:
        """Optional: clears per-episode detector state."""
        self._monitor.end_episode(self._episode_id)


async def run_and_monitor(
    agent: Any,
    task: Any,
    *,
    monitor: Any,
    episode_id: str,
    agent_name: str | None = None,
    clock: Callable[[], float] | None = None,
) -> Any:
    """Run an Autogen agent while streaming its events through a handler.

    Wraps ``await agent.run_stream(task)`` (falling back to ``agent.run``) and
    feeds every emitted event to a :class:`SnaglineAutogenHandler`. Returns the
    agent's final result. The agent must expose an async ``run_stream`` (or
    ``run``) method; if it exposes neither, raise a clear error at call time.
    """
    handler = SnaglineAutogenHandler(
        monitor, episode_id, agent_name=agent_name, clock=clock
    )
    if not hasattr(agent, "run_stream"):
        if not hasattr(agent, "run"):  # pragma: no cover - caller bug
            raise TypeError(
                "agent must expose an async 'run_stream' (or 'run') method, "
                f"got {type(agent).__name__!r}"
            )
        try:
            result = await agent.run(task)  # type: ignore[attr-defined]
            handler.observe(result)
            return result
        finally:
            handler.close()

    try:
        stream = await agent.run_stream(task)
        if stream is None:
            raise TypeError(
                f"agent.run_stream({task!r}) returned None; expected an async iterator"
            )
        final = None
        async for event in stream:
            handler.observe(event)
            final = event
        return final
    finally:
        handler.close()
