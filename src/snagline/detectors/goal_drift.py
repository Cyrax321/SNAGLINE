"""Goal-drift detector (next phase, step 2).

Compares a live run's per-tool behavior against a *persisted healthy*
``BaselineProfile`` (see ``snagline.baseline``) and flags meaningful drift: a
rising error rate, latency blowing past the healthy mean by several sigmas, or
tools that never appeared in the healthy baseline.

The detector is dependency-free. An optional ``embedder`` callable
(``StepEvent -> Sequence[float]``) is accepted for future content-level
semantic drift, but the shipped behavior is the structural comparison above,
which needs no ML dependency and stays safe for the zero-dep default.

The detector is a no-op until a baseline is supplied, so it can be wired into
``Monitor.default()`` behind ``Config.goal_drift_enabled`` without changing
default behavior.
"""

from __future__ import annotations

from typing import Any

from snagline.baseline import BaselineProfile
from snagline.config import Config
from snagline.events import StepEvent
from snagline.risk import FailureRisk


class GoalDriftDetector:
    name = "goal_drift"

    def __init__(
        self,
        baseline: BaselineProfile | None = None,
        config: Config | None = None,
        embedder: Any = None,
    ) -> None:
        self._baseline = baseline
        self._cfg = config or Config()
        self._embedder = embedder
        # Per-episode live accumulation reuses BaselineProfile's accumulators.
        self._live: dict[str, BaselineProfile] = {}
        self._fired: dict[str, bool] = {}

    def observe(self, event: StepEvent) -> FailureRisk | None:
        if self._baseline is None:
            return None
        baseline = self._baseline
        live = self._live.setdefault(event.episode_id, BaselineProfile())
        live.add_event(event)
        if live.total_steps < self._cfg.goal_drift_min_samples:
            return None
        score = self._drift_score(event.episode_id, baseline)
        if score < self._cfg.goal_drift_score_threshold:
            return None
        if self._fired.get(event.episode_id, False):
            return None
        self._fired[event.episode_id] = True
        return FailureRisk(
            event.episode_id,
            event.step_id,
            min(1.0, score),
            "goal_drift",
            f"behavior diverged from healthy baseline (score {score:.2f})",
            event.timestamp,
        )

    def _drift_score(self, episode_id: str, baseline: BaselineProfile) -> float:
        live = self._live[episode_id]
        cfg = self._cfg
        contributions: list[float] = []
        for name, tb in live.tools.items():
            ref = baseline.tools.get(name)
            if ref is None:
                # A tool that never ran in the healthy baseline is suspicious.
                contributions.append(0.6)
                continue
            err_drift = max(
                0.0, tb.error_rate - (ref.error_rate + cfg.goal_drift_error_tolerance)
            )
            if err_drift > 0:
                contributions.append(min(1.0, err_drift * 2.0))
            # Latency drift: compare live mean to the healthy mean. Floor the
            # reference spread so a zero-variance baseline (all identical healthy
            # latencies) does not treat any tiny deviation as infinite z.
            if tb.mean_latency > 0 and ref.mean_latency > 0:
                spread = max(ref.std_latency, 0.05 * ref.mean_latency, 1.0)
                z = (tb.mean_latency - ref.mean_latency) / spread
                if z > cfg.goal_drift_latency_k:
                    contributions.append(
                        min(1.0, (z - cfg.goal_drift_latency_k) / 10.0)
                    )
        if not contributions:
            return 0.0
        return min(1.0, sum(contributions))

    def reset(self, episode_id: str) -> None:
        self._live.pop(episode_id, None)
        self._fired.pop(episode_id, None)

    def dump_state(self) -> dict[str, Any]:
        return {
            "live": {ep: p.to_dict() for ep, p in self._live.items()},
            "fired": dict(self._fired),
        }

    def load_state(self, state: dict[str, Any]) -> None:
        self._live = {
            ep: BaselineProfile.from_dict(raw)
            for ep, raw in state.get("live", {}).items()
        }
        self._fired = {ep: bool(v) for ep, v in state.get("fired", {}).items()}
