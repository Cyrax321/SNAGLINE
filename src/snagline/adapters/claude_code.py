"""Claude Code adapter (project.md §6.7) - hooks bridge for an external process.

Claude Code is not a Python library; it is a separate process that can run a
shell command ("command" hook) or make an HTTP call ("http" hook) on
lifecycle events. This module maps Claude Code's native hook JSON payloads
(payload fields verified against the live docs at
https://code.claude.com/docs/en/hooks) into canonical ``StepEvent``s:

  payload field           ->  StepEvent field
  ------------------------------------------------
  session_id              ->  episode_id
  tool_use_id             ->  step_id (falls back to a uuid)
  tool_name               ->  tool_name
  tool_input              ->  hashed into action_signature (never stored raw)
  hook_event_name         ->  action_type / error mapping (see _EVENT_MAP)

Only tool-level and failure-level events are mapped; noisy lifecycle events
(FileChanged, Notification, ...) return ``None`` so callers drop them.

``HookTracker`` pairs ``PreToolUse`` with ``PostToolUse`` via ``tool_use_id``
to derive ``latency_ms`` for the CUSUM detector - a single hook payload has
no duration, but the two events share the id.

Privacy: tool_input / prompt content enters only the one-way SHA-256
signature, never ``metadata``.
"""

from __future__ import annotations

import contextlib
import json
import time
import uuid
from typing import Any

from snagline.events import StepEvent, make_signature

_ERROR_EVENTS = {"PostToolUseFailure", "StopFailure"}

# hook_event_name -> (action_type, subkind); None means "drop this event".
_EVENT_MAP: dict[str, tuple[str, str] | None] = {
    "PreToolUse": ("tool_call", "start"),
    "PostToolUse": ("tool_call", "end"),
    "PostToolUseFailure": ("tool_call", "end"),
    "UserPromptSubmit": ("message", "prompt"),
    "StopFailure": ("message", "turn"),
    # Everything else (SessionStart, Stop, Notification, FileChanged, ...) is
    # lifecycle noise with no detection value; drop it.
}


def is_claude_code_payload(obj: Any) -> bool:
    """Heuristic: Claude Code hook payloads always carry ``hook_event_name``."""
    return isinstance(obj, dict) and "hook_event_name" in obj


class HookTracker:
    """Pairs PreToolUse/PostToolUse by ``tool_use_id`` to derive latency.

    Stateful companion for a long-lived bridge (the sidecar server). Kept tiny
    and defensive: malformed payloads never raise out of ``note``/``latency``.
    """

    def __init__(
        self,
        clock=None,
        ttl_seconds: float = 600.0,
        max_pending: int = 1024,
    ) -> None:
        self._clock = clock or time.monotonic
        self._starts: dict[str, float] = {}
        self._ttl = ttl_seconds
        self._max_pending = max_pending

    def note(self, payload: dict) -> None:
        """Record a ``PreToolUse`` start, or retire a paired start.

        Call this *after* the payload has been mapped: it pops the start, so
        reading the latency afterwards would always see ``None`` (issue #64).
        """
        event = payload.get("hook_event_name")
        tool_use_id = payload.get("tool_use_id")
        if not isinstance(tool_use_id, str):
            return
        now = self._clock()
        if event == "PreToolUse":
            self._starts[tool_use_id] = now
        else:
            self._starts.pop(tool_use_id, None)
        if len(self._starts) > self._max_pending:
            self._evict(now)

    def _evict(self, now: float) -> None:
        """Bound memory by age, so a tool still running keeps its start.

        A ``PreToolUse`` whose ``PostToolUse`` never arrives (crashed tool,
        dropped hook) would otherwise leak. Drop those by age first; only if
        nothing is actually stale do we shed the oldest half, which keeps the
        newest -- still pairable -- starts.
        """
        cutoff = now - self._ttl
        for key in [k for k, started in self._starts.items() if started < cutoff]:
            del self._starts[key]
        if len(self._starts) > self._max_pending:
            by_age = sorted(self._starts.items(), key=lambda kv: kv[1])
            self._starts = dict(by_age[len(by_age) // 2 :])

    def latency_ms(self, payload: dict) -> float | None:
        tool_use_id = payload.get("tool_use_id")
        if not isinstance(tool_use_id, str):
            return None
        start = self._starts.get(tool_use_id)
        if start is None:
            return None
        return (self._clock() - start) * 1000.0


def payload_to_event(
    payload: dict,
    tracker: HookTracker | None = None,
    timestamp: float | None = None,
) -> StepEvent | None:
    """Map one Claude Code hook payload to a ``StepEvent`` (or ``None``).

    ``tracker`` is consulted for latency: map the payload with this function
    first, then call ``tracker.note(payload)``. ``note`` retires the paired
    ``PreToolUse`` start, so noting before mapping loses the latency entirely
    (issue #64). ``latency_ms`` is filled only on the events that *end* a tool
    call (``PostToolUse`` / ``PostToolUseFailure``); a ``PreToolUse`` has not
    run yet and therefore has no duration.
    """
    mapped = _EVENT_MAP.get(str(payload.get("hook_event_name")))
    if mapped is None:
        return None
    action_type, subkind = mapped

    episode_id = str(payload.get("session_id") or "claude-code")
    step_id = str(
        payload.get("tool_use_id") or payload.get("prompt_id") or uuid.uuid4().hex[:12]
    )
    tool_name = payload.get("tool_name")
    is_tool = action_type == "tool_call"

    # Stable, structural parts only: the same logical attempt (same tool,
    # same input) must hash identically for loop detection to see a retry.
    if is_tool:
        stable = json.dumps(payload.get("tool_input") or {}, sort_keys=True)
        tool_for_sig = str(tool_name or "tool")
    else:
        stable = str(payload.get("prompt") or payload.get("error") or subkind)
        # prompt_id is volatile (unique per turn): excluding it from the
        # signature lets a repeated identical prompt be loop-detectable.
        tool_for_sig = "claude-code"

    error = payload.get("hook_event_name") in _ERROR_EVENTS
    error_type = None
    if error:
        err = payload.get("error")
        error_type = (
            type(err).__name__
            if isinstance(err, BaseException)
            else (str(err) if err else payload.get("hook_event_name"))
        )

    # Only the event that *ends* a tool call has a duration; PreToolUse fires
    # before the tool runs, so asking the tracker there would read the start it
    # just wrote and report a bogus 0.0 ms (issue #64).
    latency_ms = (
        tracker.latency_ms(payload)
        if tracker is not None and is_tool and subkind == "end"
        else None
    )

    return StepEvent(
        step_id=step_id,
        episode_id=episode_id,
        timestamp=timestamp if timestamp is not None else time.time(),
        action_type=action_type,
        action_signature=make_signature(action_type, tool_for_sig, stable),
        tool_name=str(tool_name) if tool_name is not None else None,
        latency_ms=latency_ms,
        error=error,
        error_type=error_type,
        tokens_in=None,
        tokens_out=None,
        metadata={},
    )


def ingest_payload(
    monitor: Any, payload: dict, tracker: HookTracker | None = None
) -> StepEvent | None:
    """Convenience: map the payload, ingest it, then note it on ``tracker``.

    Returns the event that was ingested, or ``None`` for unmapped events.
    Never raises: mapping/ingest failures are the Monitor's fail-open concern,
    and a bridge must not break the host process.

    Order matters. ``tracker.note`` retires the ``PreToolUse`` start that
    ``payload_to_event`` needs to compute ``latency_ms``, so mapping happens
    first and the tracker is updated afterwards -- in a ``finally`` so an
    unmapped or malformed payload still advances the tracker (issue #64).
    """
    try:
        event = payload_to_event(payload, tracker=tracker)
        if event is not None:
            monitor.ingest(event)
        return event
    except Exception:
        return None
    finally:
        if tracker is not None:
            with contextlib.suppress(Exception):
                tracker.note(payload)
