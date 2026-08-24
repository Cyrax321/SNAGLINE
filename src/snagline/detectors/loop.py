"""Loop detector (tier-1, deterministic, O(1) amortized).

Per-episode sliding window (``collections.deque``) of recent
``action_signature`` values. If the same signature appears
``repeat_threshold`` times within ``window_size`` steps, emit a risk.

Each looping signature escalates once and then stays quiet until that loop
actually clears, so a long repetition alerts once rather than on every step.

No raw content is read -- only the one-way ``action_signature`` hash, so a
loop of identical retry attempts is caught without ever seeing the prompt or
response text (project.md §1.4).
"""

from __future__ import annotations

from collections import deque

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
        #
        # Keyed by signature, not by episode alone: two different actions looping
        # in one episode are two distinct findings, and one of them going quiet
        # must not re-arm the other.
        self._fired: dict[str, set[str]] = {}

    def observe(self, event: StepEvent) -> FailureRisk | None:
        w = self._windows.setdefault(event.episode_id, deque(maxlen=self.window_size))
        w.append(event.action_signature)
        count = w.count(event.action_signature)
        fired = self._fired.get(event.episode_id)
        if fired:
            # Re-arm any signature whose loop has ended -- it fell below the
            # threshold, or aged out of the window entirely -- so a later
            # repetition of it escalates again.
            #
            # Re-arming is decided here, from the window, and *not* from the
            # current step's own count. A step that is not part of the loop says
            # nothing about whether the loop is still running, so reading it as an
            # all-clear re-armed the flag and re-fired on the very next repeat:
            # one loop interleaved with distinct steps -- a retry between
            # reasoning turns -- then alerted on every repetition, which is the
            # spam this flag exists to prevent.
            #
            # Only the escalated signatures are re-checked, and the window holds
            # at most one per ``repeat_threshold`` slots. On the common path
            # nothing is looping, this dict has no entry, and the step costs
            # exactly what it did before.
            for sig in tuple(fired):
                seen = count if sig == event.action_signature else w.count(sig)
                if seen < self.repeat_threshold:
                    fired.discard(sig)
        if count < self.repeat_threshold:
            return None
        if fired is None:
            fired = self._fired.setdefault(event.episode_id, set())
        elif event.action_signature in fired:
            return None
        fired.add(event.action_signature)
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
