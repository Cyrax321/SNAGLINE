"""Canonical event schema for SNAGLINE.

This is the only wire format detectors and sinks ever see. Framework-specific
adapters translate their host runtime's events into ``StepEvent`` instances and
pass them to ``Monitor.ingest``. No core code imports a framework.

Design constraints honored here (see project.md §1):
  * Zero third-party dependencies (stdlib only).
  * No raw content retention: detectors reason on hashes, timings, counts,
    and booleans. ``action_signature`` is a one-way SHA-256 digest.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class StepEvent:
    """A single observed step in an agent's execution stream.

    Only five fields are load-bearing for tier-1 detection: ``step_id``,
    ``episode_id``, ``timestamp``, ``action_signature``, and ``error``.
    Everything else is optional and improves detector quality but nothing in
    core requires it. ``metadata`` is explicitly NEVER read by detectors and
    must not be forwarded by sinks (see project.md §11).
    """

    step_id: str
    episode_id: str
    timestamp: float  # unix epoch seconds, float for sub-second precision
    action_type: str  # "tool_call" | "message" | "plan_step" | "observation" | adapter-defined
    action_signature: str  # normalized hash -- see make_signature()

    tool_name: str | None = None
    latency_ms: float | None = None
    error: bool = False
    error_type: str | None = None
    tokens_in: int | None = None
    tokens_out: int | None = None
    metadata: dict = field(default_factory=dict)  # adapter-specific; detectors never read this


@dataclass(frozen=True, slots=True)
class EpisodeMeta:
    """Lightweight descriptor for a run/episode, not used in detection math."""

    episode_id: str
    agent_name: str | None = None
    started_at: float | None = None
    tags: dict = field(default_factory=dict)


def make_signature(action_type: str, tool_name: str | None, *stable_parts: str) -> str:
    """Build a loop-detectable, one-way signature for an action.

    Rules for adapter authors:
      * Include the logical action (tool name, target element, endpoint) --
        the things that make two actions "the same attempt."
      * EXCLUDE volatile fields: timestamps, request/session ids, nonces,
        retry counters. Including these defeats loop detection by making
        every retry look unique.
      * Hashing already-sensitive values is fine (SHA-256 is one-way), but
        prefer hashing only the minimum needed to detect repetition, not the
        full payload, to keep signatures meaningful and short.

    Returns the first 16 hex chars of the SHA-256 digest of the joined
    stable parts.
    """
    raw = "||".join([action_type, tool_name or "", *stable_parts])
    return hashlib.sha256(raw.encode()).hexdigest()[:16]
