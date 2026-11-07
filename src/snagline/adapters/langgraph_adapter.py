# Copyright 2026 SNAGLINE contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""LangGraph adapter (optional extra: ``pip install snagline[langgraph]``).

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
    one (a superstep boundary), recorded on the event as coarse per-node
    latency. Note it is NOT consumed by the CUSUM latency-anomaly detector:
    that detector scores only ``tool_call`` steps (``action_type !=
    "tool_call"`` is skipped) and deliberately ignores aggregate ``node_run``
    supersteps, since a superstep spans a whole node (often a nested subgraph
    or reasoning turn), and scoring it would flag every nested LangGraph run
    as an anomaly (issue #10). The field is still carried for hosts and sinks
    that want raw per-node timing; wrap tool leaves with a ``tool_call``
    adapter if you want their latency CUSUM-monitored.
  * ``error``                -> the node's update contains a truthy ``error``
    key, or the update value is an Exception (LangGraph signals node errors
    this way in updates mode). A node that *raises* never arrives as an
    update: LangGraph propagates the exception out of ``graph.stream()``.
    ``watch_graph`` catches it, emits one error ``StepEvent`` so the failure
    still reaches the Monitor, and re-raises so your own code sees the
    exception unchanged.
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
    clk = clock or time.perf_counter
    counter = itertools.count()
    last = clk()
    iterator = iter(stream)

    while True:
        try:
            item = next(iterator)
        except StopIteration:
            return
        except Exception as exc:
            # A real LangGraph node error is not delivered as a yielded update:
            # graph.stream() raises it out of the iterator (verified against
            # langgraph 1.x). Emit the failure ourselves so the node crash --
            # the very signal the error detectors exist to catch -- still
            # reaches the Monitor, then re-raise so the caller's own handling
            # of the exception is unchanged.
            now = clk()
            monitor.ingest(
                StepEvent(
                    step_id=str(next(counter)),
                    episode_id=episode_id,
                    timestamp=now,
                    action_type="node_run",
                    action_signature=make_signature(
                        "node_run", None, type(exc).__name__
                    ),
                    tool_name=None,
                    latency_ms=(now - last) * 1000.0,
                    error=True,
                    error_type=type(exc).__name__,
                    metadata={},
                )
            )
            raise
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
                    timestamp=now,
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
