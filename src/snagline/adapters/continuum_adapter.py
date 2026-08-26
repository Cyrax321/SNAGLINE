"""CONTINUUM ledger adapter (optional extra: ``pip install snagline-agent[continuum]``).

"Free telemetry": reads a CONTINUUM run's hash-chained event log through its
existing public ``Storage`` API and translates entries into SNAGLINE
``StepEvent``s. Zero new instrumentation on the CONTINUUM side.

Verified against the current CONTINUUM source (``continuum.storage.base.Storage``
and ``continuum.events``); project.md section 6.6 described a
``read_by_sequence`` method, but the real public read-by-sequence surface is::

    Storage.read_events(run_id, *, after_sequence: int = 0,
                        upto: int | None = None) -> Sequence[Event]
    Storage.last_sequence(run_id) -> int

Polling therefore advances ``after_sequence``; this adapter codes only against
that verified pair of methods plus the ``Event`` fields ``sequence``, ``type``,
``timestamp`` and ``payload``. The delta is recorded here and in the PR.

Ledger entries translated (project.md section 6.6 mapping):

=========================  =====================  ==================================
CONTINUUM entry            StepEvent.action_type  Notes
=========================  =====================  ==================================
PERCEPTION_OBSERVED        "observation"          trust_level/verifier/hash kept;
                                                  raw_claim NEVER copied (privacy)
BRANCH_RESOLVED            "plan_step"            requires_review flag kept
ACTION_RECORDED            "tool_call"            claim/perform/complete lifecycle:
                                                  status started -> completed/failed;
                                                  latency paired from observed times
ACTION_RECONCILED          "action_reconciled"
ACTION_COMPENSATED         "action_compensated"
=========================  =====================  ==================================

Other entry types (RUN_STARTED, TASK_UPDATED, checkpoints, ...) are skipped:
they carry no agent-step semantics for SNAGLINE's detectors.

Two consumption modes:

* **Poll**: :meth:`ContinuumAdapter.poll` reads new entries by sequence and
  ingests them. Call it periodically from your own loop (or use
  :meth:`ContinuumAdapter.poll_forever`).
* **Push**: :meth:`ContinuumAdapter.push` translates one already-observed
  ledger entry, for callers that tail the log live via their own callback.

The adapter is duck-typed like every other adapter: it never imports
``continuum``, so this module is importable without any extra installed and
never hard-couples to a release. Fail-open (project.md principle 2): storage
errors are caught, logged once and ignored; a poisoned entry is skipped
without stopping the rest; nothing here can crash or block the host agent.

Privacy (project.md principle 4): hashes, ids, counts, statuses, timestamps.
Payload text such as ``raw_claim`` is dropped at translation time and never
reaches a ``StepEvent``.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable, Mapping
from datetime import datetime
from typing import Any

from snagline.events import StepEvent, make_signature

logger = logging.getLogger("snagline")

__all__ = ["ContinuumAdapter"]

#: Entry types that map onto an agent step; everything else is skipped.
_PERCEPTION = "PERCEPTION_OBSERVED"
_BRANCH = "BRANCH_RESOLVED"
_ACTION_RECORDED = "ACTION_RECORDED"
_ACTION_RECONCILED = "ACTION_RECONCILED"
_ACTION_COMPENSATED = "ACTION_COMPENSATED"


def _to_epoch(timestamp: Any) -> float:
    """Accept a datetime (CONTINUUM's Event.timestamp) or a numeric epoch."""
    if isinstance(timestamp, datetime):
        return timestamp.timestamp()
    try:
        return float(timestamp)
    except (TypeError, ValueError):
        return time.time()


def _as_dict(payload: Any) -> Mapping[str, Any]:
    return payload if isinstance(payload, Mapping) else {}


def _str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text or None


class ContinuumAdapter:
    """Translate CONTINUUM ledger entries into ``StepEvent``s for a Monitor.

    Args:
        monitor: a :class:`snagline.monitor.Monitor` (duck-typed: anything
            with ``ingest``).
        storage: a CONTINUUM ``Storage`` (duck-typed: only ``read_events`` and
            ``last_sequence`` are called).
        run_id: the CONTINUUM run to follow; becomes the SNAGLINE episode id.
        start_at_tail: when True (default), the first :meth:`poll` starts at
            the log's current tail so you monitor live activity instead of
            replaying history. Set False to replay the whole run from
            sequence 1.
    """

    def __init__(
        self,
        monitor: Any,
        storage: Any,
        run_id: str,
        *,
        start_at_tail: bool = True,
        clock: Callable[[], float] | None = None,
    ) -> None:
        self._monitor = monitor
        self._storage = storage
        self._run_id = run_id
        self._clock = clock or time.time
        self._cursor = -1 if start_at_tail else 0
        self._started = False
        #: action key -> epoch seconds of the observed claim, for latency pairing
        self._pending_claims: dict[str, float] = {}
        self._stop = threading.Event()

    # -- public API -------------------------------------------------------- #

    def poll(self) -> int:
        """Read and ingest every entry after the cursor. Returns count ingested.

        Fail-open: any storage error is logged and swallowed; the caller's loop
        keeps running and the next poll retries.
        """
        if not self._started:
            self._seek_initial_cursor()
            self._started = True
        try:
            entries = list(
                self._storage.read_events(self._run_id, after_sequence=self._cursor)
            )
        except Exception:
            logger.warning(
                "snagline continuum adapter: read failed for run %s; will retry",
                self._run_id,
                exc_info=True,
            )
            return 0
        count = 0
        for entry in entries:
            if self.push(entry):
                count += 1
        return count

    def poll_forever(
        self, interval: float = 1.0, stop_event: threading.Event | None = None
    ) -> None:
        """Poll on a fixed interval until ``stop_event`` is set (or Ctrl-C).

        Deliberately boring: stdlib sleep loop, no threads spawned, no network.
        """
        stop = stop_event or self._stop
        while not stop.is_set():
            self.poll()
            stop.wait(max(0.05, interval))

    def stop(self) -> None:
        """Signal a :meth:`poll_forever` loop using its default stop event."""
        self._stop.set()

    def push(self, entry: Any) -> bool:
        """Translate one live-tailed ledger entry and ingest it.

        Push mode: for callers that observe entries as they are appended and
        prefer handing them over directly. Returns True when an event was
        ingested. Never raises.
        """
        sequence = getattr(entry, "sequence", None)
        try:
            event = self.translate(entry)
        except Exception:
            logger.warning(
                "snagline continuum adapter: failed to translate entry; skipped",
                exc_info=True,
            )
            self._advance(sequence)
            return False
        if event is None:
            self._advance(sequence)
            return False
        try:
            self._monitor.ingest(event)
        except Exception:
            logger.warning(
                "snagline continuum adapter: monitor.ingest raised; event dropped",
                exc_info=True,
            )
            self._advance(sequence)
            return False
        self._advance(sequence)
        return True

    def _advance(self, sequence: Any) -> None:
        """Move the poll cursor past an entry, whatever happened to it."""
        if isinstance(sequence, int) and not isinstance(sequence, bool):
            self._cursor = max(self._cursor, sequence)

    def translate(self, entry: Any) -> StepEvent | None:
        """Map one ledger entry to a ``StepEvent``; None for irrelevant types."""
        sequence = getattr(entry, "sequence", None)
        entry_type = getattr(entry, "type", None)
        payload = _as_dict(getattr(entry, "payload", None))
        name = str(entry_type).rsplit(".", 1)[-1] if entry_type is not None else ""

        if name == _PERCEPTION:
            event = self._perception(sequence, payload)
        elif name == _BRANCH:
            event = self._branch(sequence, payload)
        elif name in (_ACTION_RECORDED, _ACTION_RECONCILED, _ACTION_COMPENSATED):
            event = self._action(name, sequence, payload)
        else:
            return None
        if event is not None and sequence is not None:
            event.metadata["sequence"] = sequence
        return event

    # -- translation internals ---------------------------------------------- #

    def _seek_initial_cursor(self) -> None:
        try:
            tail = int(self._storage.last_sequence(self._run_id))
        except Exception:
            logger.warning(
                "snagline continuum adapter: last_sequence failed for run %s; "
                "starting from sequence 0",
                self._run_id,
                exc_info=True,
            )
            self._cursor = 0
            return
        self._cursor = max(tail, 0) if self._cursor < 0 else 0

    def _perception(self, sequence: Any, payload: Mapping[str, Any]) -> StepEvent:
        source = _str(payload.get("source"))
        observation_id = _str(payload.get("observation_id")) or ""
        trust = _str(payload.get("trust_level")) or ""
        verifier = _str(payload.get("verifier")) or ""
        content_hash = _str(payload.get("content_hash")) or ""
        sig = make_signature("observation", source, observation_id, content_hash)
        return StepEvent(
            step_id=str(sequence),
            episode_id=self._run_id,
            timestamp=self._clock(),
            action_type="observation",
            action_signature=sig,
            tool_name=source,
            metadata={
                "adapter": "continuum",
                "event_type": "PERCEPTION_OBSERVED",
                "observation_id": observation_id,
                "trust_level": trust,
                "verifier": verifier,
            },
        )

    def _branch(self, sequence: Any, payload: Mapping[str, Any]) -> StepEvent:
        requires_review = bool(payload.get("requires_review", False))
        branch = _as_dict(payload.get("branch"))
        branch_name = _str(branch.get("name")) or _str(branch.get("branch_id")) or ""
        sig = make_signature(
            "plan_step",
            branch_name or None,
            str(branch_name),
            "review" if requires_review else "auto",
        )
        return StepEvent(
            step_id=str(sequence),
            episode_id=self._run_id,
            timestamp=self._clock(),
            action_type="plan_step",
            action_signature=sig,
            tool_name=_str(branch_name),
            error=False,
            metadata={
                "adapter": "continuum",
                "event_type": "BRANCH_RESOLVED",
                "requires_review": requires_review,
            },
        )

    def _action(
        self, name: str, sequence: Any, payload: Mapping[str, Any]
    ) -> StepEvent:
        status = _str(payload.get("status")) or ""
        tool_name = _str(payload.get("action_type"))
        key = _str(payload.get("key")) or _str(payload.get("action_id")) or ""
        now = self._clock()
        latency: float | None = None
        if status == "started":
            # A fresh claim opens the latency window; a repeat claim for a key
            # we already track keeps the original start (retry, not restart).
            self._pending_claims.setdefault(key, now)
        elif key in self._pending_claims:
            latency = max(0.0, (now - self._pending_claims.pop(key)) * 1000.0)
        action_type = {
            _ACTION_RECONCILED: "action_reconciled",
            _ACTION_COMPENSATED: "action_compensated",
        }.get(name, "tool_call")
        sig = make_signature(action_type, tool_name, key, status)
        return StepEvent(
            step_id=str(sequence),
            episode_id=self._run_id,
            timestamp=now,
            action_type=action_type,
            action_signature=sig,
            tool_name=tool_name,
            latency_ms=latency,
            error=status == "failed",
            metadata={
                "adapter": "continuum",
                "event_type": name,
                "status": status,
                "action_key_hash": make_signature("key", None, key)[:16] if key else "",
                "external_id": _str(payload.get("external_id")),
            },
        )
