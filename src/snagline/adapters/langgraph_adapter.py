"""LangGraph adapter (optional extra: ``pip install snagline-agent[langgraph]``).

Wraps ``graph.stream(...)`` -- LangGraph's public streaming API -- and turns
each node update into a ``StepEvent`` for the Monitor (project.md §6.3).
Works with any framework that yields ``{node_name: state_update}`` items
(LangGraph's default ``stream_mode="updates"``), so the module has no hard
langgraph import at all; if you already have a stream of updates, you can
pass it directly.

Mapping rules:
  * one yielded update item  -> one StepEvent per node key in that item
  * ``tool_name``            -> the node name
  * ``latency_ms``           -> wall time between the previous yield and this
    one (a superstep boundary); this is a coarse per-node latency, which is
    exactly what the CUSUM detector needs -- a sustained deviation, not an
    exact per-invocation timing
  * ``error``                -> the node's update contains a truthy ``error``
    key, or the update value is an Exception (LangGraph signals node errors
    this way in updates mode)
  * ``action_signature``     -> hash of (node name, sorted top-level update
    keys). Deliberately structural, not content: a node repeatedly producing
    the same *shape* of failed update is a loop; changing content is not.

Usage::

    from snagline import Monitor
    from snagline.adapters.langgraph_adapter import watch_graph

    monitor = Monitor.default()
    for update in watch_graph(monitor, "ep-1", graph.stream(inputs)):
        ...  # your normal stream consumption, unchanged
"""

from __future__ import annotations

import itertools
import logging
import time
from collections.abc import Callable, Iterator
from typing import Any

from snagline.events import StepEvent, make_signature

logger = logging.getLogger("snagline")


def _node_error(update: Any) -> tuple[bool, str | None]:
    if isinstance(update, BaseException):
        return True, type(update).__name__
    if isinstance(update, dict):
        err = update.get("error")
        if err:
            if isinstance(err, BaseException):
                return True, type(err).__name__
            return True, str(err)
    return False, None


def watch_graph(
    monitor: Any,
    episode_id: str,
    stream: Iterator[dict],
    clock: Callable[[], float] | None = None,
) -> Iterator[dict]:
    """Pass-through iterator over a LangGraph update stream that monitors it.

    Yields every item from ``stream`` unchanged, so wrapping ``graph.stream``
    is a one-line change to the host code. Call ``monitor.end_episode`` (or use
    the raw-style teardown in your own code) when the run completes; this
    function wraps the stream, so it cannot know when the caller is done with
    the Monitor.
    """
    clk = clock or time.monotonic
    counter = itertools.count()
    last = clk()

    for item in stream:
        now = clk()
        latency_ms = (now - last) * 1000.0
        last = now
        if isinstance(item, dict):
            for node, update in item.items():
                error, error_type = _node_error(update)
                shape = (
                    ",".join(sorted(update.keys()))
                    if isinstance(update, dict)
                    else type(update).__name__
                )
                event = StepEvent(
                    step_id=str(next(counter)),
                    episode_id=episode_id,
                    timestamp=time.time(),
                    action_type="node_run",
                    action_signature=make_signature("node_run", str(node), shape),
                    tool_name=str(node),
                    latency_ms=latency_ms,
                    error=error,
                    error_type=error_type,
                    metadata={},
                )
                monitor.ingest(event)
        yield item
