"""Latency / CUSUM anomaly detector (tier-1, deterministic, O(1) amortized).

Stdlib-only. Per ``(episode_id, tool_name)`` it learns a baseline mean/variance
via Welford's algorithm during a short warm-up, *freezes* that baseline, and
then feeds standardized deviations into a CUSUM-with-alarms statistic::

    cusum = max(0, cusum + (x - mu0) / sigma0 - k);  alarm when cusum > h

The baseline is frozen at the end of warm-up (it is NOT updated with later
samples). This is deliberate: updating the baseline with the anomalies lets a
sustained shift be "learned away" so the alert stops, which is exactly the
failure mode the old implementation had. With a frozen mu0/sigma0, a sustained
regression keeps the CUSUM elevated (and therefore keeps alerting) until latency
actually returns to the baseline.

A non-zero reference sigma0 is required: a perfectly stable baseline has sample
std 0, which would make every deviation infinite (and force a div-by-zero guard
that never fires). We therefore floor sigma0 at ``max(std, abs_floor,
rel_floor * |mu0|)`` so that even a constant baseline has a meaningful deviation
scale. This makes a single large spike alarm immediately instead of requiring
several sustained spikes. Only ``latency_ms`` and ``tool_name`` are read -- no
content (project.md §1.4).
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

from snagline.config import Config
from snagline.events import StepEvent
from snagline.risk import FailureRisk


class _WelfordCUSUM:
    """Online mean/variance (Welford) during warm-up, then a frozen CUSUM."""

    def __init__(
        self,
        k: float,
        h: float,
        sigma_floor_abs: float = 1.0,
        sigma_floor_rel: float = 0.05,
    ) -> None:
        self.k = k
        self.h = h
        self.sigma_floor_abs = sigma_floor_abs
        self.sigma_floor_rel = sigma_floor_rel
        self.n = 0
        self.mean = 0.0
        self._m2 = 0.0
        self.cusum = 0.0
        self.mu0: Optional[float] = None
        self.sigma0: float = 0.0
        self.frozen = False

    def learn_only(self, x: float) -> None:
        """Welford update of the running baseline statistics (no CUSUM)."""
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        delta2 = x - self.mean
        self._m2 += delta * delta2

    def freeze(self) -> None:
        """Snapshot the baseline mean/variance as the fixed CUSUM target."""
        var = self._m2 / (self.n - 1) if self.n >= 2 else 0.0
        std = math.sqrt(var)
        self.mu0 = self.mean
        if self.mean != 0.0:
            floor = max(self.sigma_floor_abs, self.sigma_floor_rel * abs(self.mean))
        else:
            floor = self.sigma_floor_abs
        self.sigma0 = max(std, floor)
        self.frozen = True

    def update(self, x: float) -> bool:
        """Advance the CUSUM against the frozen baseline. Returns True if it alarms."""
        assert self.mu0 is not None
        self.cusum = max(0.0, self.cusum + (x - self.mu0) / self.sigma0 - self.k)
        return self.cusum > self.h


class LatencyAnomalyDetector:
    name = "latency_anomaly"

    def __init__(
        self,
        k: Optional[float] = None,
        h: Optional[float] = None,
        min_samples: Optional[int] = None,
        sigma_floor_abs: Optional[float] = None,
        sigma_floor_rel: Optional[float] = None,
        config: Optional[Config] = None,
    ) -> None:
        cfg = config or Config()
        self.k = k if k is not None else cfg.cusum_k
        self.h = h if h is not None else cfg.cusum_h
        self.min_samples = min_samples if min_samples is not None else cfg.cusum_min_samples
        self.sigma_floor_abs = (
            sigma_floor_abs if sigma_floor_abs is not None else cfg.cusum_sigma_floor_abs
        )
        self.sigma_floor_rel = (
            sigma_floor_rel if sigma_floor_rel is not None else cfg.cusum_sigma_floor_rel
        )
        self._states: Dict[Tuple[str, str], _WelfordCUSUM] = {}

    def observe(self, event: StepEvent) -> Optional[FailureRisk]:
        if event.latency_ms is None:
            return None
        # Only leaf tool calls carry a meaningful per-tool latency. A planning
        # chain (``plan_step``) or other aggregate step reports the duration of
        # a whole reasoning turn, not a single tool -- counting it would flag
        # every nested LangChain/LangGraph run as a latency anomaly (issue #10).
        if event.action_type != "tool_call":
            return None
        key = (event.episode_id, event.tool_name or "default")
        state = self._states.get(key)
        if state is None:
            state = _WelfordCUSUM(self.k, self.h, self.sigma_floor_abs, self.sigma_floor_rel)
            self._states[key] = state

        if not state.frozen:
            state.learn_only(event.latency_ms)
            if state.n >= self.min_samples:
                state.freeze()
            return None

        if state.update(event.latency_ms):
            score = min(1.0, 0.6 + 0.1 * max(0.0, state.cusum / self.h - 1.0))
            return FailureRisk(
                event.episode_id,
                event.step_id,
                score,
                "latency_anomaly",
                f"latency {event.latency_ms:.0f}ms deviates from baseline "
                f"(mean {state.mu0:.0f}ms)",
                event.timestamp,
            )
        return None

    def reset(self, episode_id: str) -> None:
        for key in [k for k in self._states if k[0] == episode_id]:
            self._states.pop(key, None)
