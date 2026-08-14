"""Error-cascade detector (tier-1, deterministic, O(1) amortized).

Same sliding-window shape as the loop detector but tracks the ``error``
boolean per episode instead of signatures. Fires on either of two conditions:

  * Consecutive: ``cascade_consecutive_threshold`` (default 3) errors in a row
    -- catches fast cascades immediately.
  * Windowed: ``cascade_error_threshold`` (default 3) errors anywhere in the
    last ``cascade_window_size`` (default 10) steps -- catches slow-burn
    degradations where errors are interleaved with occasional successes.

Only the boolean ``error`` flag is consulted; no content is read
(project.md §1.4).
"""

from __future__ import annotations

from collections import deque
from typing import Dict

from snagline.config import Config
from snagline.events import StepEvent
from snagline.risk import FailureRisk


class ErrorCascadeDetector:
    name = "error_cascade"

    def __init__(
        self,
        window_size: int | None = None,
        error_threshold: int | None = None,
        consecutive_threshold: int | None = None,
        config: Config | None = None,
    ) -> None:
        cfg = config or Config()
        self.window_size = (
            window_size if window_size is not None else cfg.cascade_window_size
        )
        self.error_threshold = (
            error_threshold if error_threshold is not None else cfg.cascade_error_threshold
        )
        self.consecutive_threshold = (
            consecutive_threshold
            if consecutive_threshold is not None
            else cfg.cascade_consecutive_threshold
        )
        self._windows: Dict[str, deque] = {}
        self._consecutive: Dict[str, int] = {}

    def observe(self, event: StepEvent) -> FailureRisk | None:
        w = self._windows.setdefault(event.episode_id, deque(maxlen=self.window_size))
        w.append(event.error)
        self._consecutive[event.episode_id] = (
            self._consecutive.get(event.episode_id, 0) + 1 if event.error else 0
        )

        if self._consecutive[event.episode_id] >= self.consecutive_threshold:
            return FailureRisk(
                event.episode_id,
                event.step_id,
                0.8,
                "error_cascade",
                f"{self._consecutive[event.episode_id]} consecutive errors",
                event.timestamp,
            )

        if sum(w) >= self.error_threshold and len(w) >= self.error_threshold:
            return FailureRisk(
                event.episode_id,
                event.step_id,
                0.6,
                "error_cascade",
                f"{sum(w)} errors in last {len(w)} steps",
                event.timestamp,
            )

        return None

    def reset(self, episode_id: str) -> None:
        self._windows.pop(episode_id, None)
        self._consecutive.pop(episode_id, None)
