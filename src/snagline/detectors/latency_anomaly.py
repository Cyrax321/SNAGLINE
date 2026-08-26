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

Calibrated start (issue #101): passing a fitted ``BaselineProfile`` replaces
the live warm-up for tools it describes well enough (``count >=
min_samples``): the CUSUM starts frozen at the profile's healthy mean and
floored spread instead of learning from early live samples. Episodes shorter
than the warm-up are therefore monitorable from their first step, with no new
knobs: k/h stay as configured.
"""

from __future__ import annotations

import math
from typing import Any

from snagline.baseline import BaselineProfile, ToolBaseline
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
        self.mu0: float | None = None
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
        std = math.sqrt(max(0.0, var))
        self.mu0 = self.mean
        self.sigma0 = self._floored_sigma(self.mean, std)
        self.frozen = True

    def seed(self, mean: float, std: float) -> None:
        """Start from an externally fitted healthy baseline (no warm-up).

        Used by auto-calibration (issue #101): the profile's healthy mean and
        spread become the fixed CUSUM target immediately.
        """
        self.mu0 = mean
        self.sigma0 = self._floored_sigma(mean, std)
        self.frozen = True

    def _floored_sigma(self, mean: float, std: float) -> float:
        """Reference spread: observed std floored per the configured floors."""
        if mean != 0.0:
            return max(std, self.sigma_floor_abs, self.sigma_floor_rel * abs(mean))
        return max(std, self.sigma_floor_abs)

    def update(self, x: float) -> bool:
        """Advance the CUSUM against the frozen baseline. Returns True if it alarms."""
        assert self.mu0 is not None
        self.cusum = max(0.0, self.cusum + (x - self.mu0) / self.sigma0 - self.k)
        return self.cusum > self.h


class LatencyAnomalyDetector:
    name = "latency_anomaly"

    def __init__(
        self,
        k: float | None = None,
        h: float | None = None,
        min_samples: int | None = None,
        sigma_floor_abs: float | None = None,
        sigma_floor_rel: float | None = None,
        baseline: BaselineProfile | None = None,
        config: Config | None = None,
    ) -> None:
        cfg = config or Config()
        self.k = k if k is not None else cfg.cusum_k
        self.h = h if h is not None else cfg.cusum_h
        self.min_samples = (
            min_samples if min_samples is not None else cfg.cusum_min_samples
        )
        self.sigma_floor_abs = (
            sigma_floor_abs
            if sigma_floor_abs is not None
            else cfg.cusum_sigma_floor_abs
        )
        self.sigma_floor_rel = (
            sigma_floor_rel
            if sigma_floor_rel is not None
            else cfg.cusum_sigma_floor_rel
        )
        # Healthy reference for calibrated starts (issue #101). Aggregate
        # per-tool stats only; no content is retained or consulted.
        self._baseline = baseline
        self._states: dict[tuple[str, str], _WelfordCUSUM] = {}

    def observe(self, event: StepEvent) -> FailureRisk | None:
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
            state = _WelfordCUSUM(
                self.k, self.h, self.sigma_floor_abs, self.sigma_floor_rel
            )
            # Calibrated start (issue #101): when a healthy profile describes
            # this tool well enough, skip warm-up and freeze onto its stats.
            # Tools without a sufficient entry keep today's learn-then-freeze
            # behavior.
            seeded: ToolBaseline | None = None
            if self._baseline is not None:
                candidate = self._baseline.tools.get(key[1])
                if candidate is not None and candidate.count >= self.min_samples:
                    seeded = candidate
            if seeded is not None:
                state.seed(seeded.mean_latency, seeded.std_latency)
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

    @staticmethod
    def _state_to_dict(s: _WelfordCUSUM) -> dict[str, Any]:
        return {
            "n": s.n,
            "mean": s.mean,
            "m2": s._m2,
            "cusum": s.cusum,
            "mu0": s.mu0,
            "sigma0": s.sigma0,
            "frozen": s.frozen,
        }

    @classmethod
    def _state_from_dict(cls, s: _WelfordCUSUM, raw: dict[str, Any]) -> None:
        s.n = raw["n"]
        s.mean = raw["mean"]
        s._m2 = raw["m2"]
        s.cusum = raw["cusum"]
        s.mu0 = raw["mu0"]
        s.sigma0 = raw["sigma0"]
        s.frozen = raw["frozen"]

    def dump_state(self) -> dict[str, Any]:
        # Keys are (episode_id, tool_name) tuples; JSON dict keys must be
        # strings, so serialize as [key-pair, state] entries.
        return {
            "states": [
                [[ep, tool], self._state_to_dict(s)]
                for (ep, tool), s in sorted(self._states.items())
            ]
        }

    def load_state(self, state: dict[str, Any]) -> None:
        self._states = {}
        for key_pair, raw in state.get("states", []):
            ep, tool = key_pair[0], key_pair[1]
            s = _WelfordCUSUM(
                self.k, self.h, self.sigma_floor_abs, self.sigma_floor_rel
            )
            self._state_from_dict(s, raw)
            self._states[(ep, tool)] = s
