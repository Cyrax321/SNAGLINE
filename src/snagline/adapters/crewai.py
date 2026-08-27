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

The optional ``clock=`` defaults to :func:`time.perf_counter`: monotonic and
high-resolution on every platform. ``time.time`` advances in ~15.6 ms ticks on
Windows and quantized sub-tick latencies to zero (issue #155). Note that
``perf_counter`` has no meaningful epoch: event timestamps produced with it
are only comparable within one process; detectors consume them solely as
in-process latency differences.
"""

from __future__ import annotations

import itertools
import json
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


def _extract_tool_input(step: dict[str, Any]) -> str | None:
    """Canonical form of the tool arguments, or ``None`` when unavailable.

    The arguments -- not the model's prose -- are what make two tool calls "the
    same attempt" (see ``make_signature`` and docs/ADAPTER_GUIDE.md). CrewAI
    exposes them on the step itself or nested under ``action``, so mirror the
    dual lookup :func:`_extract_tool_name` already does. ``sort_keys`` matches
    the canonicalization the Claude Code adapter uses, so a dict payload hashes
    the same regardless of key order.

    A payload JSON cannot represent (unorderable key types, a reference cycle,
    an arbitrary object) yields ``None`` so the caller keeps its previous stable
    part. Serializing is deliberately strict rather than falling back to
    ``str``: the default repr of an object embeds its memory address, which
    would vary per attempt and defeat loop detection all over again.
    """
    action = step.get("action") if isinstance(step.get("action"), dict) else None
    for src in (action, step):
        if src is None:
            continue
        val = src.get("tool_input")
        if val is None:
            continue
        try:
            return json.dumps(val, sort_keys=True)
        except Exception:
            # _map_step runs in the framework's thread, before Monitor.ingest
            # and therefore outside its fail-open guard, so an exotic payload
            # must never propagate into the host agent.
            return None
    return None


def _map_step(
    step: Any,
    *,
    episode_id: str,
    step_id: str,
    agent_name: str | None,
    clock: Callable[[], float],
    side_effect: bool = False,
) -> StepEvent:
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
    # Key the signature on the *attempt* -- tool plus arguments -- never on the
    # step's output. Model prose is re-worded on every retry, so hashing it made
    # a stuck loop look like a stream of unique actions, while omitting the
    # arguments collapsed a legitimate iteration into one signature (issue #61).
    # Non-tool steps carry no argument payload, so they keep hashing the text.
    tool_args = _extract_tool_input(d)
    if action_type == "tool_call" and tool_args is not None:
        stable = tool_args
    else:
        stable = text or str(d)
    sig = make_signature(action_type, tool_name, stable)
    return StepEvent(
        step_id=step_id,
        episode_id=episode_id,
        timestamp=clock(),
        action_type=action_type,
        action_signature=sig,
        tool_name=tool_name,
        latency_ms=latency,
        error=error,
        metadata={"agent_name": agent_name, "adapter": "crewai"},
        side_effect=side_effect,
    )


class _CrewAIStepCallback:
    """Callable CrewAI ``step_callback`` with an optional ``close()`` teardown.

    CrewAI only calls ``__call__``; ``close()`` mirrors
    :meth:`SnaglineAutogenHandler.close` and clears per-episode detector state.
    ``side_effect_tools`` is the host-declared allowlist for
    ``SideEffectGuardDetector`` (issue #150): matching ``tool_call`` steps get
    ``side_effect=True``; nothing is inferred from payloads.
    """

    def __init__(
        self,
        monitor: Any,
        episode_id: str,
        agent_name: str | None = None,
        clock: Callable[[], float] | None = None,
        side_effect_tools: set[str] | list[str] | tuple[str, ...] | None = None,
    ) -> None:
        self._monitor = monitor
        self._episode_id = episode_id
        self._agent_name = agent_name
        # perf_counter, not time.time: time.time ticks at ~15.6 ms on Windows,
        # quantizing sub-tick latencies to zero (issue #155). perf_counter has
        # no meaningful epoch; detectors only consume in-process differences.
        self._clock = clock or time.perf_counter
        self._counter = itertools.count()
        self._side_effect_tools: set[str] = set(side_effect_tools or [])

    def __call__(self, step: Any) -> None:
        # Host-declared allowlist (issue #150): only a tool_call whose name
        # the host put in side_effect_tools becomes side_effect=True. Never
        # read metadata and never guess from args or payload content.
        d = _to_dict(step)
        tool_name = _extract_tool_name(d)
        side_effect = tool_name is not None and tool_name in self._side_effect_tools
        self._monitor.ingest(
            _map_step(
                step,
                episode_id=self._episode_id,
                step_id=str(next(self._counter)),
                agent_name=self._agent_name,
                clock=self._clock,
                side_effect=side_effect,
            )
        )

    def close(self) -> None:
        self._monitor.end_episode(self._episode_id)


def snagline_step_callback(
    monitor: Any,
    episode_id: str,
    agent_name: str | None = None,
    clock: Callable[[], float] | None = None,
    side_effect_tools: set[str] | list[str] | tuple[str, ...] | None = None,
) -> Callable[[Any], None]:
    """Build a CrewAI ``step_callback`` that monitors each agent step.

    The returned callable accepts a CrewAI step object (or dict) and ingests a
    corresponding ``StepEvent``. Returns the callback for assignment to
    ``Agent(step_callback=...)``. After the episode finishes, call
    ``callback.close()`` to clear per-episode detector state.

    ``side_effect_tools`` is the host-declared allowlist for
    ``SideEffectGuardDetector`` (issue #150); matching ``tool_call`` steps get
    ``side_effect=True`` and nothing is inferred from payloads.
    """
    return _CrewAIStepCallback(
        monitor,
        episode_id,
        agent_name=agent_name,
        clock=clock,
        side_effect_tools=side_effect_tools,
    )


def observe_crewai_step(
    monitor: Any,
    episode_id: str,
    step: Any,
    *,
    agent_name: str | None = None,
    clock: Callable[[], float] | None = None,
    step_id: str | None = None,
    side_effect: bool = False,
) -> StepEvent:
    """Manually map a single CrewAI step object to a ``StepEvent`` and ingest it.

    Useful when you manage the agent loop yourself and want the same mapping the
    callback uses, without registering ``step_callback``. ``side_effect=True``
    marks the step as a non-idempotent action (issue #88); set it only from
    your own knowledge of the tool, never heuristically.
    """
    event = _map_step(
        step,
        episode_id=episode_id,
        step_id=step_id or "manual",
        agent_name=agent_name,
        # Same rationale as SnaglineAutogenHandler: sub-tick latency fidelity
        # on Windows requires a monotonic high-resolution default (issue #155).
        clock=clock or time.perf_counter,
        side_effect=side_effect,
    )
    monitor.ingest(event)
    return event
