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
import time
from dataclasses import dataclass, field
from functools import lru_cache


@dataclass(frozen=True, slots=True)
class StepEvent:
    """A single observed step in an agent's execution stream.

    Only five fields are load-bearing for tier-1 detection: ``step_id``,
    ``episode_id``, ``timestamp``, ``action_signature``, and ``error``.
    Everything else is optional and improves detector quality but nothing in
    core requires it. ``metadata`` is explicitly NEVER read by detectors and
    must not be forwarded by sinks (see project.md §11).

    ``side_effect`` (issue #88) marks a step whose action is known to be
    non-idempotent (a payment, a send, a deploy). It is a plain boolean the
    host sets deliberately: adapters only forward it and never derive it,
    because guessing wrong either hides duplicate charges or flags harmless
    reads. Detectors may read this boolean (it is a flag, not content).
    """

    step_id: str
    episode_id: str
    # Monotonic, process-local seconds from :func:`step_clock` -- the default
    # and what every auto-instrumented adapter stamps (issue #155:
    # ``time.time`` ticks at ~15.6 ms on Windows and quantizes sub-tick
    # intervals away). The time-axis detectors consume this as a *delta*,
    # which is only meaningful inside one clock domain (issue #532).
    #
    # Not universal: the CONTINUUM adapter deliberately stamps unix epoch,
    # because it pairs a claim's observed time with its terminal record to
    # compute a real claim-to-terminal duration, and that pairing is only
    # meaningful in one shared domain (ledger times on both sides). Mixing
    # that with a perf_counter is exactly the mismatch
    # :func:`snagline.monitor.Monitor` re-anchors around. Hosts that replay a
    # timeline across processes should rebase onto one clock rather than
    # interleaving domains.
    timestamp: float
    action_type: (
        str  # "tool_call" | "message" | "plan_step" | "observation" | adapter-defined
    )
    action_signature: str  # normalized hash -- see make_signature()

    tool_name: str | None = None
    latency_ms: float | None = None
    error: bool = False
    error_type: str | None = None
    tokens_in: int | None = None
    tokens_out: int | None = None
    metadata: dict = field(
        default_factory=dict
    )  # adapter-specific; detectors never read this
    side_effect: bool = False  # host-declared non-idempotent action (issue #88)
    intent: str | None = (
        None  # host-declared task/intent or high-level purpose (issue #603)
    )


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
        full payload, to keep signatures meaningful.

    The parts are serialized as a canonical JSON array before hashing. JSON
    array encoding is unambiguous (``["a", "b||c"]`` can never be confused
    with ``["a||b", "c"]``), and the full 64-character SHA-256 digest is
    returned -- truncating to 16 hex chars invited collisions between
    distinct actions (issue #15).

    The function is memoized (issue #313): real traffic repeats a small
    palette of signatures heavily -- the same few tools over a long episode --
    so the JSON+SHA-256 work is skipped on repeats. The cache is keyed on the
    argument tuple and bounded, so the output is unchanged (a pure function of
    its inputs) and memory cannot grow without limit on a hostile stream. A
    cache miss still pays the full canonical-JSON cost, so a cold or
    high-cardinality stream sees the original ~1.2 us/step and no
    compatibility break: the wire format is untouched.
    """
    return _make_signature(action_type, tool_name, *stable_parts)


@lru_cache(maxsize=1024)
def _make_signature(action_type: str, tool_name: str | None, *stable_parts: str) -> str:
    import json

    parts = [action_type, tool_name or "", *stable_parts]
    raw = json.dumps(parts, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _clear_signature_cache() -> None:
    """Drop every memoized signature (test hook; not part of the public API)."""
    _make_signature.cache_clear()


def step_clock() -> float:
    """The canonical clock for ``StepEvent.timestamp`` (issue #532).

    The default for live instrumentation: read the timestamp field from this
    helper rather than ``time.time()``, so a host's own events stay in the
    same clock domain as the shipped adapters. ``time.time`` is wall-clock
    epoch (~1.79e9) while this is monotonic and process-local (~1e3-1e5), so
    mixing the two makes a single mixed-adapter step report a
    ~1.79-billion-second gap and latches a fabricated, unrecoverable
    ``wall_clock_budget`` breach plus a bogus ``idle_gap``.

    An adapter that derives times from an external record (the CONTINUUM
    ledger, a replayed file) may legitimately stamp its own domain instead;
    that is a real measurement rather than a host clock read. The monitor
    re-anchors rather than scoring a span past ``_IMPLAUSIBLE_STEP_SECONDS``,
    so such a mix degrades to a dropped interval instead of a false alarm.

    Monotonic (not epoch) because the detectors consume *deltas*, never
    absolute times, and monotonic time cannot jump backwards under NTP or a
    clock adjustment (issue #155).
    """
    return time.perf_counter()
