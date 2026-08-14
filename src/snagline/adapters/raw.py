"""Raw adapter -- for anyone with a plain loop and no framework (project.md §6.1).

This is likely the single most-used adapter: most real agent code today is a
custom loop, not a framework. Its only job is to turn a call in the host loop
into a ``StepEvent`` and call ``monitor.ingest``. It contains no detection
logic.

Usage::

    from snagline import Monitor
    from snagline.adapters.raw import watch

    monitor = Monitor.default()
    with watch(monitor, "ep-1") as step:
        ...                       # host loop body
        step("tool_call", tool_name="search", latency_ms=120, error=False)
"""

from __future__ import annotations

import itertools
import time
from contextlib import contextmanager
from typing import Any, Callable, Dict, Iterator, Optional

from snagline.events import EpisodeMeta, StepEvent, make_signature


def _build_signature(action_type: str, tool_name: Optional[str], args: Any) -> str:
    # Volatile fields (timestamps, nonces, retry counters) must NOT enter the
    # signature or every retry would look unique and defeat loop detection.
    return make_signature(action_type, tool_name, str(args))


@contextmanager
def watch(
    monitor: Any,
    episode_id: str,
    agent_name: Optional[str] = None,
) -> Iterator[Callable[..., StepEvent]]:
    """Context manager yielding a ``step`` callable bound to ``monitor``/``episode_id``.

    ``step(action_type, tool_name=None, *, args="", latency_ms=None,
    error=False, error_type=None, tokens_in=None, tokens_out=None,
    metadata=None, **extra)`` builds a ``StepEvent`` (with an auto-incrementing
    ``step_id``), ingests it, and returns it. Any extra kwargs are folded into
    ``metadata`` (which detectors never read).
    """
    counter = itertools.count()
    EpisodeMeta(episode_id, agent_name=agent_name, started_at=time.time())

    def step(
        action_type: str,
        tool_name: Optional[str] = None,
        *,
        args: Any = "",
        latency_ms: Optional[float] = None,
        error: bool = False,
        error_type: Optional[str] = None,
        tokens_in: Optional[int] = None,
        tokens_out: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **extra: Any,
    ) -> StepEvent:
        sig = _build_signature(action_type, tool_name, args)
        md: Dict[str, Any] = dict(metadata or {})
        md.update(extra)
        event = StepEvent(
            step_id=str(next(counter)),
            episode_id=episode_id,
            timestamp=time.time(),
            action_type=action_type,
            action_signature=sig,
            tool_name=tool_name,
            latency_ms=latency_ms,
            error=error,
            error_type=error_type,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            metadata=md,
        )
        monitor.ingest(event)
        return event

    try:
        yield step
    finally:
        pass
