"""Extension point: the Detector protocol.

Detectors operate ONLY on ``StepEvent``. They must be O(1) amortized per step
(project.md §1.5) and must never raise into the host agent -- the Monitor wraps
every ``observe`` call in a fail-open guard. ``reset`` is called when an
episode ends so per-episode state does not leak across runs.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol, TypeVar

from snagline.events import StepEvent
from snagline.risk import FailureRisk

_K = TypeVar("_K")
_V = TypeVar("_V")


class Detector(Protocol):
    """Protocol every detector (core or third-party) must satisfy."""

    name: str

    def observe(self, event: StepEvent) -> FailureRisk | None:
        """Inspect one event against the detector's internal state.

        Return a ``FailureRisk`` if a signal is present, else ``None``.
        """
        ...

    def reset(self, episode_id: str) -> None:
        """Drop any per-episode state for ``episode_id``."""
        ...


class EpisodeFinalizer(Protocol):
    """Optional extension point (issue #86).

    A detector that can only judge an episode once it is *finished* (the
    completion check: did the run end on a bare tool call instead of an
    output?) implements ``finalize``. ``Monitor.end_episode`` discovers it by
    duck typing, so ordinary detectors are unaffected and third-party
    detectors never need to grow this method.
    """

    def finalize(self, episode_id: str) -> FailureRisk | None:
        """Judge a finished episode; return a risk or ``None``.

        Called exactly once per ``end_episode`` before ``reset``. Must be
        cheap and side-effect free apart from dropping that episode's state.
        """
        ...


class StatefulDetector(Protocol):
    """Optional extension point (issue #91).

    Detectors implementing these two methods participate in
    ``Monitor.snapshot()`` / ``Monitor.restore()`` so monitoring survives
    process restarts on long-running episodes. State must be plain,
    JSON-compatible data -- snapshots are stdlib JSON, never pickle.
    """

    def dump_state(self) -> dict[str, Any] | None:
        """Return JSON-compatible internal state, or ``None`` if stateless."""
        ...

    def load_state(self, state: dict[str, Any]) -> None:
        """Restore internal state previously produced by ``dump_state``."""
        ...


def snapshot_items(state: Mapping[_K, _V]) -> list[tuple[_K, _V]]:
    """Atomically copy a per-episode state dict before serializing it (#231).

    ``dump_state`` runs on whichever thread called ``Monitor.snapshot`` and
    takes no lock, while an ingest on another thread may be meeting a new
    episode and inserting its first-sight key. Detector state is partitioned by
    ``episode_id``, so the *values* are safe -- but the key set is shared, and
    the Monitor's per-episode lock cannot serialize a walk of it against an
    insert for a different episode.

    A Python-level comprehension over ``state.items()`` yields between
    bytecodes, so that insert lands mid-walk and ``RuntimeError: dictionary
    changed size during iteration`` escapes ``Monitor.snapshot`` -- a public API
    with no fail-open guard, because restore-time errors are deliberately loud.
    ``list(state.items())`` builds the copy inside a single C call with no
    Python executing in between, so it cannot be interrupted; callers then
    build their JSON payload from the returned copy.

    Cross-dict atomicity is deliberately *not* provided: a detector holding
    several dicts may snapshot episode X in one and miss it in another. Every
    ``load_state`` already reads each dict independently and defaults what is
    absent, and a snapshot taken during live ingest is a point-in-time smear
    either way. Guaranteeing more would mean holding one lock across the whole
    of ``observe``, which would serialize ingest across episodes and cost the
    per-episode concurrency the Monitor is built around.
    """
    return list(state.items())
