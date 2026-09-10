"""Compaction tripwire: governance-decay detection across compactions.

Motivation (issue #90): arXiv:2606.22528 studies "governance decay": when a
host agent compacts its context (summarization, truncation, eviction), the
governance constraints stated earlier in that context can silently drop out,
and policy-violation rates rise sharply afterwards. Their finding is cited as
motivation only; SNAGLINE claims no benchmark numbers of its own for this
detector.

Contract (adapter-defined action types; core treats them opaquely per
project.md §4.1):

    step("compaction", metadata={"pinned": ["<sha256>", ...]})  # just compacted
    step("constraint_present", metadata={"pin": "<sha256>"})     # pin re-seen

After a ``compaction`` event the host has ``grace_steps`` subsequent events to
re-confirm every pinned constraint hash with a ``constraint_present`` event.
If any pin is still unconfirmed once that deadline is reached, exactly one
``FailureRisk(score=0.9, trigger="governance_decay")`` fires, naming the
16-hex prefixes of the missing pins. A later ``compaction`` replaces the
pending set and restarts the grace window (the issue's "new compaction resets
pending set" rule). Confirmations that arrive after the risk fired are noted
but do not retract or re-fire anything; only a new compaction opens a new
window.

Privacy posture (project.md §1.4 / §11): hashes only. Constraint text never
reaches snagline; the adapter hashes canonical constraint text itself and
sends only SHA-256 hex digests. This module is also the ONE documented
exception to "detectors never read ``metadata``": it reads exactly two keys,
``pinned`` on ``compaction`` events and ``pin`` on ``constraint_present``
events, and nothing else from anywhere else on the event. Risk details carry
at most 16-hex pin prefixes, never full digests of foreign input and never
constraint text.

Inert by design: a host with no compaction hooks never emits these action
types, so the detector sees ordinary steps and stays silent forever. The
adapter and bridge docs state this honestly instead of promising coverage
where no visibility exists (ADAPTER_GUIDE.md, FRAMEWORK_BRIDGES.md).

Performance: O(1) amortized per event. One dict lookup, one integer bump,
one or two string comparisons, at most one set discard per step. The only
superlinear work (sorting missing pins for the detail line) runs once per
fired risk, not per step, and is bounded by the number of pins a harness
pins (small by construction). Memory: one small state object per episode;
``reset(episode_id)`` releases it.
"""

from __future__ import annotations

from typing import Any

from snagline.config import Config
from snagline.detectors.base import snapshot_items
from snagline.events import StepEvent
from snagline.risk import FailureRisk

_COMPACTION = "compaction"
_CONSTRAINT_PRESENT = "constraint_present"

# Risks carry at most this many leading hex characters of any pin, so alert
# channels can never become a digest-oracle for arbitrary content (issue #90:
# only 16-hex prefixes ever appear in risks/details).
_PIN_PREFIX_LEN = 16

# Upper bound on prefixes named in one detail line so a pathological
# many-pin compaction cannot produce an enormous alert payload.
_MAX_NAMED_PINS = 8


class _PendingPins:
    """Pins awaiting re-confirmation within one grace window."""

    __slots__ = ("deadline_ordinal", "fired", "pins")

    def __init__(self, pins: set[str], deadline_ordinal: int) -> None:
        self.pins = pins
        self.deadline_ordinal = deadline_ordinal
        self.fired = False


class _EpisodeState:
    """Per-episode bookkeeping: an event ordinal plus any pending window.

    Deadlines are counted against an internal ordinal (events observed for
    the episode), not ``step_id``, because ``step_id`` is an opaque string
    with no guaranteed monotonic numeric form across adapters.
    """

    __slots__ = ("ordinal", "pending")

    def __init__(self) -> None:
        self.ordinal = 0
        self.pending: _PendingPins | None = None


class CompactionTripwireDetector:
    """Fires once per compaction whose pinned constraints go unconfirmed.

    Opt-in via ``Config.compaction_tripwire_enabled``; see module docstring
    for the full contract and the privacy rationale.
    """

    name = "governance_decay"

    def __init__(
        self,
        grace_steps: int | None = None,
        config: Config | None = None,
    ) -> None:
        cfg = config or Config()
        self.grace_steps = (
            grace_steps
            if grace_steps is not None
            else cfg.compaction_tripwire_grace_steps
        )
        if self.grace_steps < 1:
            raise ValueError("grace_steps must be >= 1")
        self._episodes: dict[str, _EpisodeState] = {}

    def observe(self, event: StepEvent) -> FailureRisk | None:
        st = self._episodes.get(event.episode_id)
        if st is None:
            st = _EpisodeState()
            self._episodes[event.episode_id] = st
        st.ordinal += 1
        n = st.ordinal

        if event.action_type == _COMPACTION:
            # A new compaction replaces any previous window wholesale and
            # restarts the grace countdown (issue #90 acceptance criteria).
            st.pending = self._open_window(event.metadata, n)
        elif event.action_type == _CONSTRAINT_PRESENT:
            pending = st.pending
            if (
                pending is not None
                and not pending.fired
                and isinstance(event.metadata, dict)
            ):
                pin = event.metadata.get("pin")
                if isinstance(pin, str) and pin:
                    pending.pins.discard(pin)

        pending = st.pending
        if (
            pending is not None
            and not pending.fired
            and n >= pending.deadline_ordinal
            and pending.pins
        ):
            pending.fired = True
            return self._risk(event, sorted(pending.pins))
        return None

    def reset(self, episode_id: str) -> None:
        """Drop the event ordinal and any pending window for the episode."""
        self._episodes.pop(episode_id, None)

    def dump_state(self) -> dict[str, Any]:
        """Serialize per-episode ordinals and pending windows (#91/#149).

        All primitives are JSON-compatible; pin sets become sorted lists (raw
        sets are not JSON-serializable) and ``load_state`` rebuilds the set.
        Pins are hashes by contract, so serialization carries no content risk.
        A restart between a compaction and its confirmations must not let the
        grace deadline vanish silently: the pending set, its deadline ordinal,
        and the fired latch all survive.

        The walk goes through ``snapshot_items``: a concurrent ingest meeting a
        new episode must not change the key set mid-loop (issue #231).
        """
        episodes: dict[str, Any] = {}
        for ep, st in snapshot_items(self._episodes):
            entry: dict[str, Any] = {"ordinal": st.ordinal, "pending": None}
            if st.pending is not None:
                entry["pending"] = {
                    "pins": sorted(st.pending.pins),
                    "deadline_ordinal": st.pending.deadline_ordinal,
                    "fired": st.pending.fired,
                }
            episodes[ep] = entry
        return {"episodes": episodes}

    def load_state(self, state: dict[str, Any]) -> None:
        self._episodes = {}
        for ep, entry in state.get("episodes", {}).items():
            st = _EpisodeState()
            st.ordinal = int(entry.get("ordinal", 0))
            raw_pending = entry.get("pending")
            if raw_pending is not None:
                pending = _PendingPins(
                    set(raw_pending.get("pins", [])),
                    int(raw_pending.get("deadline_ordinal", 0)),
                )
                pending.fired = bool(raw_pending.get("fired", False))
                st.pending = pending
            self._episodes[str(ep)] = st

    def _open_window(self, metadata: dict, ordinal: int) -> _PendingPins | None:
        """Build the pending window for a compaction event, or None when the
        event carries no usable pins (in which case nothing can decay)."""
        raw = metadata.get("pinned") if isinstance(metadata, dict) else None
        if not isinstance(raw, (list, tuple, set, frozenset)):
            return None
        pins = {p for p in raw if isinstance(p, str) and p}
        if not pins:
            return None
        return _PendingPins(pins, ordinal + self.grace_steps)

    @staticmethod
    def _risk(event: StepEvent, missing: list[str]) -> FailureRisk:
        named = [p[:_PIN_PREFIX_LEN] for p in missing[:_MAX_NAMED_PINS]]
        extra = len(missing) - len(named)
        detail = "unconfirmed governance pins after compaction: " + ", ".join(named)
        if extra > 0:
            detail += f" (+{extra} more)"
        return FailureRisk(
            event.episode_id,
            event.step_id,
            0.9,
            "governance_decay",
            detail,
            event.timestamp,
        )
