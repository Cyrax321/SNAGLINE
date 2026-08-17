"""CrewAI adapter (optional extra: ``pip install snagline-agent[crewai]``).

Turns CrewAI agent steps into ``StepEvent``s and feeds them to a ``Monitor``.
CrewAI agents accept a ``step_callback`` that receives each agent step as it is
produced. :func:`snagline_step_callback` returns a callback compatible with
that hook, so wiring is a one-liner::

    from snagline import Monitor
    from snagline.adapters.crewai import snagline_step_callback

    monitor = Monitor.default()
    agent = Agent(..., step_callback=snagline_step_callback(monitor, "ep-1"))

The adapter is duck-typed: it reads CrewAI step objects via ``model_dump()``
with fallbacks, so it imports fine without CrewAI installed (safe in CI) and
never hard-couples to a specific CrewAI release. No raw prompt/response content
is retained -- only the one-way ``action_signature`` hash, tool name, latency
(when available), and error flag.

CrewAI step objects vary by version; we read common attributes (``tool``,
``tool_input``, ``output``/``text``, ``error``) and fall back to a dict view, so
the adapter stays loosely coupled to a specific CrewAI release.
"""

from __future__ import annotations

import itertools
import time
from collections.abc import Callable
from typing import Any

from snagline.events import StepEvent, make_signature


def _to_dict(obj: Any) -> dict[str, Any]:
    if isinstance(obj, dict):
        return obj
    model_dump = getattr(obj, "model_dump", None)
    if callable(model_dump):
        try:
            return model_dump()
        except Exception:
            pass
    if hasattr(obj, "__dict__"):
        return dict(vars(obj))
    return {}


def _extract_tool_name(step: dict[str, Any]) -> str | None:
    # CrewAI surfaces the tool under action.tool, tool, or name.
    action = step.get("action") if isinstance(step.get("action"), dict) else None
    if action:
        name = action.get("tool")
        if name:
            return str(name)
    for key in ("tool", "name"):
        if step.get(key):
            return str(step[key])
    return None


def _extract_text(step: dict[str, Any]) -> str:
    for key in ("text", "thought", "output", "content", "result"):
        val = step.get(key)
        if val:
            return str(val)
    return ""


def snagline_step_callback(
    monitor: Any,
    episode_id: str,
    agent_name: str | None = None,
    clock: Callable[[], float] | None = None,
) -> Callable[[Any], None]:
    """Build a CrewAI ``step_callback`` that monitors each agent step.

    The returned callable accepts a CrewAI step object (or dict) and ingests a
    corresponding ``StepEvent``. Returns the callback for assignment to
    ``Agent(step_callback=...)``.
    """
    counter = itertools.count()
    current_clock = clock or time.time

    def callback(step: Any) -> None:
        d = _to_dict(step)
        tool_name = _extract_tool_name(d)
        action_type = "tool_call" if tool_name else "agent_step"
        text = _extract_text(d)
        error = bool(d.get("error") or d.get("is_error"))
        # Some CrewAI versions attach a duration on the step or action.
        latency = d.get("latency_ms") or d.get("duration_ms")
        raw_action = d.get("action")
        action: dict[str, Any] = raw_action if isinstance(raw_action, dict) else {}
        if latency is None:
            latency = action.get("latency_ms")
        sig = make_signature(action_type, tool_name, text or str(d))
        event = StepEvent(
            step_id=str(next(counter)),
            episode_id=episode_id,
            timestamp=current_clock(),
            action_type=action_type,
            action_signature=sig,
            tool_name=tool_name,
            latency_ms=latency,
            error=error,
            metadata={"agent_name": agent_name, "adapter": "crewai"},
        )
        monitor.ingest(event)

    return callback


def observe_crewai_step(
    monitor: Any,
    episode_id: str,
    step: Any,
    *,
    agent_name: str | None = None,
    clock: Callable[[], float] | None = None,
    step_id: str | None = None,
) -> StepEvent:
    """Manually map a single CrewAI step object to a ``StepEvent`` and ingest it.

    Useful when you manage the agent loop yourself and want the same mapping the
    callback uses, without registering ``step_callback``.
    """
    d = _to_dict(step)
    tool_name = _extract_tool_name(d)
    action_type = "tool_call" if tool_name else "agent_step"
    text = _extract_text(d)
    error = bool(d.get("error") or d.get("is_error"))
    latency = d.get("latency_ms") or d.get("duration_ms")
    sig = make_signature(action_type, tool_name, text or str(d))
    event = StepEvent(
        step_id=step_id or "manual",
        episode_id=episode_id,
        timestamp=(clock or time.time)(),
        action_type=action_type,
        action_signature=sig,
        tool_name=tool_name,
        latency_ms=latency,
        error=error,
        metadata={"agent_name": agent_name, "adapter": "crewai"},
    )
    monitor.ingest(event)
    return event
