"""Loop detector (tier-1, deterministic, O(1) amortized).

Per-episode sliding window (``collections.deque``) of recent
``action_signature`` values. If the same signature appears
``repeat_threshold`` times within ``window_size`` steps, emit a risk.

No raw content is read -- only the one-way ``action_signature`` hash, so a
loop of identical retry attempts is caught without ever seeing the prompt or
response text (project.md §1.4).
"""

from __future__ import annotations

from collections import deque
from typing import Any

from snagline.config import Config
from snagline.events import StepEvent
from snagline.risk import FailureRisk


class LoopDetector:
    name = "loop"

    def __init__(
        self,
        window_size: int | None = None,
        repeat_threshold: int | None = None,
        config: Config | None = None,
    ) -> None:
        cfg = config or Config()
        self.window_size = (
            window_size if window_size is not None else cfg.loop_window_size
        )
        self.repeat_threshold = (
            repeat_threshold
            if repeat_threshold is not None
            else cfg.loop_repeat_threshold
        )
        self._windows: dict[str, deque] = {}
        # Dedupe: emit once per repetition episode (issue #4). Without this the
        # detector re-fires on every step while the same action keeps repeating,
        # which is alert spam.
        self._fired: dict[str, bool] = {}

    def observe(self, event: StepEvent) -> FailureRisk | None:
        w = self._windows.setdefault(event.episode_id, deque(maxlen=self.window_size))
        w.append(event.action_signature)
        count = w.count(event.action_signature)
        if count < self.repeat_threshold:
            # The repeated signature has dropped out of the window: allow a
            # future repetition to re-escalate.
            self._fired[event.episode_id] = False
            return None
        if self._fired.get(event.episode_id, False):
            return None
        self._fired[event.episode_id] = True
        score = min(1.0, count / self.repeat_threshold * 0.5)
        return FailureRisk(
            event.episode_id,
            event.step_id,
            score,
            "loop",
            f"action repeated {count}x in last {len(w)} steps",
            event.timestamp,
        )

    def reset(self, episode_id: str) -> None:
        self._windows.pop(episode_id, None)
        self._fired.pop(episode_id, None)

    def dump_state(self) -> dict[str, Any]:
        return {
            "windows": {ep: list(w) for ep, w in self._windows.items()},
            "fired": dict(self._fired),
        }

    def load_state(self, state: dict[str, Any]) -> None:
        self._windows = {
            ep: deque(sigs, maxlen=self.window_size)
            for ep, sigs in state.get("windows", {}).items()
        }
        self._fired = {ep: bool(v) for ep, v in state.get("fired", {}).items()}
