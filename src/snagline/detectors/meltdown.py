"""Meltdown detector (opt-in): entropy collapse / thrash detection.

Long-horizon agents exhibit a characteristic transition called *meltdown*
(arXiv:2603.29231, which detects it via sliding-window entropy over tool-call
sequences): coherent behavior degrades into either rote repetition or chaotic
churn. This detector computes exactly that statistic per episode.

Per episode, maintain a sliding window of the last ``meltdown_window_size``
**tool-call identities** (``tool_name``, falling back to a signature prefix).
Each full window yields Shannon entropy H over the identity distribution:

* ``H < meltdown_low_entropy``  -> the window collapsed onto essentially one
  repeated tool. Rote looping that exact-signature matching misses whenever
  args vary slightly (timestamps, page numbers, attempt counters).
* ``H > meltdown_high_entropy`` -> pathological churn across many unrelated
  tools: the "exploration spiral" shape. The paper's practical note is that
  these are ambition mismatches where a context reset recovers value.

Thresholds are in **bits**, tuned against fixtures so healthy purposeful work
stays quiet: uniform alternation across eight tools (~2.97 bits at window 20)
is below the default high threshold of 3.4 (about 10.5 tools uniform,
2**3.4); collapse onto 1-2 tools (<0.4 bits) and churn across 12+
(~3.52 bits) fire. An eight-tool ReAct agent used evenly is therefore in
the quiet zone while 12 distinct tools in one window still triggers.
Non-tool steps (message /
plan_step / observation) do not feed the window -- they dilute the tool-call
distribution the statistic reasons about.

Re-arm discipline mirrors the other detectors: emit once per crossing, stay
quiet while the condition persists, and re-arm only after ``rearm_steps``
consecutive in-band steps so a second independent collapse still alerts.
"""

from __future__ import annotations

import math
from collections import Counter, deque
from typing import Any

from snagline.config import Config
from snagline.detectors.base import snapshot_items
from snagline.detectors.windowing import effective_window_size
from snagline.events import StepEvent
from snagline.risk import FailureRisk


class _EpisodeWindow:
    """Sliding window + incremental counts for one episode."""

    __slots__ = ("window", "counts")

    def __init__(self) -> None:
        self.window: deque[str] = deque()
        self.counts: Counter[str] = Counter()

    def push(self, key: str, maxlen: int) -> None:
        self.window.append(key)
        self.counts[key] += 1
        if len(self.window) > maxlen:
            old = self.window.popleft()
            self.counts[old] -= 1
            if self.counts[old] <= 0:
                del self.counts[old]

    def entropy(self) -> float:
        total = len(self.window)
        if total == 0:
            return 0.0
        h = 0.0
        for c in self.counts.values():
            p = c / total
            h -= p * math.log2(p)
        return h


class MeltdownDetector:
    name = "meltdown"

    def __init__(
        self,
        window_size: int | None = None,
        low_entropy: float | None = None,
        high_entropy: float | None = None,
        rearm_steps: int | None = None,
        config: Config | None = None,
    ) -> None:
        cfg = config or Config()
        self.window_size = (
            window_size if window_size is not None else cfg.meltdown_window_size
        )
        self.low_entropy = (
            low_entropy if low_entropy is not None else cfg.meltdown_low_entropy
        )
        self.high_entropy = (
            high_entropy if high_entropy is not None else cfg.meltdown_high_entropy
        )
        self.rearm_steps = (
            rearm_steps if rearm_steps is not None else cfg.meltdown_rearm_steps
        )
        if not (self.low_entropy < self.high_entropy):
            raise ValueError("meltdown_low_entropy must be < meltdown_high_entropy")
        self._eps: dict[str, _EpisodeWindow] = {}
        self._fired: dict[str, bool] = {}
        self._clear_streak: dict[str, int] = {}
        # Window auto-scaling (issue #92): inert unless cfg.window_scale_steps
        # > 0; the effective size is recomputed per step and handed to push(),
        # which retains more items as the cap grows.
        self._scale_steps = cfg.window_scale_steps
        self._max_window = cfg.max_window
        self._counts: dict[str, int] = {}

    @staticmethod
    def _identity(event: StepEvent) -> str:
        return event.tool_name or event.action_signature[:16]

    def observe(self, event: StepEvent) -> FailureRisk | None:
        if event.action_type != "tool_call":
            return None
        ep = event.episode_id
        n = self._counts.get(ep, 0) + 1
        self._counts[ep] = n
        target = effective_window_size(
            self.window_size, n, self._scale_steps, self._max_window
        )
        w = self._eps.setdefault(ep, _EpisodeWindow())
        w.push(self._identity(event), target)

        if len(w.window) < target:
            return None

        h = w.entropy()
        in_band = self.low_entropy <= h <= self.high_entropy
        if in_band:
            streak = self._clear_streak.get(ep, 0) + 1
            self._clear_streak[ep] = streak
            if streak >= self.rearm_steps:
                self._fired[ep] = False
            return None

        self._clear_streak[ep] = 0
        if self._fired.get(ep, False):
            return None
        self._fired[ep] = True
        distinct = len(w.counts)
        if h < self.low_entropy:
            score = 0.7
            detail = (
                f"tool-choice entropy collapsed to {h:.2f} bits "
                f"({distinct} distinct in last {len(w.window)} steps)"
            )
        else:
            score = 0.6
            detail = (
                f"tool-choice entropy spiked to {h:.2f} bits "
                f"({distinct} distinct in last {len(w.window)} steps)"
            )
        return FailureRisk(
            ep, event.step_id, score, "meltdown", detail, event.timestamp
        )

    def reset(self, episode_id: str) -> None:
        self._eps.pop(episode_id, None)
        self._counts.pop(episode_id, None)
        self._fired.pop(episode_id, None)
        self._clear_streak.pop(episode_id, None)

    def dump_state(self) -> dict[str, Any]:
        # snapshot_items: a concurrent ingest meeting a new episode must not
        # change the key set mid-comprehension (issue #231).
        return {
            "window_size": self.window_size,
            "windows": {ep: list(w.window) for ep, w in snapshot_items(self._eps)},
            "counts": dict(self._counts),
            "fired": dict(self._fired),
            "clear_streak": dict(self._clear_streak),
        }

    def load_state(self, state: dict[str, Any]) -> None:
        self._eps = {}
        for ep, keys in state.get("windows", {}).items():
            w = _EpisodeWindow()
            for key in keys:
                w.push(key, self.window_size)
            self._eps[ep] = w
        # Tolerant .get(): pre-#92 snapshots carry no scaler positions.
        self._counts = {ep: int(n) for ep, n in state.get("counts", {}).items()}
        self._fired = {ep: bool(v) for ep, v in state.get("fired", {}).items()}
        self._clear_streak = {
            ep: int(v) for ep, v in state.get("clear_streak", {}).items()
        }
