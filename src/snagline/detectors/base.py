"""Extension point: the Detector protocol.

Detectors operate ONLY on ``StepEvent``. They must be O(1) amortized per step
(project.md §1.5) and must never raise into the host agent -- the Monitor wraps
every ``observe`` call in a fail-open guard. ``reset`` is called when an
episode ends so per-episode state does not leak across runs.
"""

from __future__ import annotations

from typing import Protocol

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
