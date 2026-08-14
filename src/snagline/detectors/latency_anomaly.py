"""Latency / CUSUM anomaly detector (tier-1, deterministic, O(1) amortized).

Stdlib-only. Per ``(episode_id, tool_name)`` it maintains a running mean and
variance via Welford's algorithm (no numpy) and feeds deviations into a
CUSUM-with-alarms statistic::

    cusum = max(0, cusum + (x - mean) / std - k);  alarm when cusum > h

This is the CUSUM approach the source paper uses, minus their echo-state-network
layer (that is the optional ``ml`` extra, not tier-1). Only ``latency_ms`` and
the ``tool_name`` are read -- no content (project.md §1.4).

A short warm-up (``cusum_min_samples``) learns the baseline before any alarm can
fire, so the early variability of a run does not produce false positives.
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

from snagline.config import Config
from snagline.events import StepEvent
from snagline.risk import FailureRisk


class _WelfordCUSUM:
    """Online mean/variance (Welford) plus a CUSUM alarm."""

    def __init__(self, k: float, h: float) -> None:
        self.k = k
        self.h = h
        self.n = 0
        self.mean = 0.0
        self._m2 = 0.0
        self.cusum = 0.0

    def learn_only(self, x: float) -> None:
        """Update the running statistics without touching the CUSUM (warm-up)."""
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        delta2 = x - self.mean
        self._m2 += delta * delta2

    def update(self, x: float) -> bool:
        """Update statistics, then advance the CUSUM. Returns True if it alarms."""
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        delta2 = x - self.mean
        self._m2 += delta * delta2
        if self.n < 2:
            return False
        variance = self._m2 / (self.n - 1)
        std = math.sqrt(variance)
        if std == 0.0:
            return False
        self.cusum = max(0.0, self.cusum + (x - self.mean) / std - self.k)
        return self.cusum > self.h


class LatencyAnomalyDetector:
    name = "latency_anomaly"

    def __init__(
        self,
        k: Optional[float] = None,
        h: Optional[float] = None,
        min_samples: Optional[int] = None,
        config: Optional[Config] = None,
    ) -> None:
        cfg = config or Config()
        self.k = k if k is not None else cfg.cusum_k
        self.h = h if h is not None else cfg.cusum_h
        self.min_samples = (
            min_samples if min_samples is not None else cfg.cusum_min_samples
        )
        self._states: Dict[Tuple[str, str], _WelfordCUSUM] = {}

    def observe(self, event: StepEvent) -> Optional[FailureRisk]:
        if event.latency_ms is None:
            return None
        key = (event.episode_id, event.tool_name or "default")
        state = self._states.get(key)
        if state is None:
            state = _WelfordCUSUM(self.k, self.h)
            self._states[key] = state

        if state.n < self.min_samples:
            state.learn_only(event.latency_ms)
            return None

        alarmed = state.update(event.latency_ms)
        if alarmed:
            score = min(1.0, 0.6 + 0.1 * max(0.0, state.cusum / self.h - 1.0))
            return FailureRisk(
                event.episode_id,
                event.step_id,
                score,
                "latency_anomaly",
                f"latency {event.latency_ms:.0f}ms deviates from baseline "
                f"(mean {state.mean:.0f}ms)",
                event.timestamp,
            )
        return None

    def reset(self, episode_id: str) -> None:
        for key in [k for k in self._states if k[0] == episode_id]:
            self._states.pop(key, None)
