"""Extension point: the Detector protocol.

Detectors operate ONLY on ``StepEvent``. They must be O(1) amortized per step
(project.md §1.5) and must never raise into the host agent -- the Monitor wraps
every ``observe`` call in a fail-open guard. ``reset`` is called when an
episode ends so per-episode state does not leak across runs.
"""

from __future__ import annotations

from typing import Any, Protocol

from snagline.events import StepEvent
from snagline.risk import FailureRisk


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
