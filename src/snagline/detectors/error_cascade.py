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
from typing import Any

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
            error_threshold
            if error_threshold is not None
            else cfg.cascade_error_threshold
        )
        self.consecutive_threshold = (
            consecutive_threshold
            if consecutive_threshold is not None
            else cfg.cascade_consecutive_threshold
        )
        # A tool failure and an LLM/chain error are different signals. By
        # default we only escalate *tool* failures as a cascade; flip
        # ``cascade_count_non_tool_errors`` to widen to every error step
        # (issue #16).
        self._count_non_tool = bool(
            getattr(cfg, "cascade_count_non_tool_errors", False)
        )
        self._windows: dict[str, deque] = {}
        self._consecutive: dict[str, int] = {}
        # Dedupe: emit at most once per cascade, then stay quiet until the alarm
        # condition clears and re-arms (issue #4).
        self._fired: dict[str, bool] = {}

    def _is_error(self, event: StepEvent) -> bool:
        if not event.error:
            return False
        if event.action_type == "tool_call":
            return True
        return self._count_non_tool

    def observe(self, event: StepEvent) -> FailureRisk | None:
        counted = self._is_error(event)
        # Only *counted* errors feed the cascade window; an LLM/chain error
        # (when excluded) is treated as a clean step so it cannot inflate the
        # cascade signal.
        w = self._windows.setdefault(event.episode_id, deque(maxlen=self.window_size))
        w.append(counted)
        if counted:
            self._consecutive[event.episode_id] = (
                self._consecutive.get(event.episode_id, 0) + 1
            )
        else:
            self._consecutive[event.episode_id] = 0

        consecutive = self._consecutive[event.episode_id]
        consecutive_alarm = consecutive >= self.consecutive_threshold
        density_alarm = (
            sum(w) >= self.error_threshold and len(w) >= self.error_threshold
        )

        if not (consecutive_alarm or density_alarm):
            # The cascade cleared: re-arm so a later, independent cascade in the
            # same episode escalates again (mirrors ``LoopDetector``). Without
            # this the flag latches for the life of the episode, and a long-lived
            # episode -- a user session, or a sidecar episode that never calls
            # ``end_episode`` -- alerts exactly once, ever. A *sustained* cascade
            # still emits only once (issue #4): the flag clears only when neither
            # rule holds any more.
            self._fired[event.episode_id] = False
            return None
        # Already escalated this cascade -- suppress until it clears.
        if self._fired.get(event.episode_id, False):
            return None

        self._fired[event.episode_id] = True
        if consecutive_alarm:
            score = min(1.0, consecutive / self.consecutive_threshold)
            detail = f"{consecutive} consecutive errors"
        else:
            score = min(1.0, sum(w) / self.error_threshold)
            detail = f"{sum(w)} errors in last {len(w)} steps"
        return FailureRisk(
            event.episode_id,
            event.step_id,
            score,
            "error_cascade",
            detail,
            event.timestamp,
        )

    def reset(self, episode_id: str) -> None:
        self._windows.pop(episode_id, None)
        self._consecutive.pop(episode_id, None)
        self._fired.pop(episode_id, None)

    def dump_state(self) -> dict[str, Any]:
        return {
            "windows": {ep: list(w) for ep, w in self._windows.items()},
            "consecutive": dict(self._consecutive),
            "fired": dict(self._fired),
        }

    def load_state(self, state: dict[str, Any]) -> None:
        self._windows = {
            ep: deque(flags, maxlen=self.window_size)
            for ep, flags in state.get("windows", {}).items()
        }
        self._consecutive = {
            ep: int(v) for ep, v in state.get("consecutive", {}).items()
        }
        self._fired = {ep: bool(v) for ep, v in state.get("fired", {}).items()}
