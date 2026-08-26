"""Token-runaway detector (opt-in, deterministic, O(1) amortized).

Two signals in one detector (issue #84):

* **Sustained-burn CUSUM** over per-step token volume (``tokens_in +
  tokens_out``), using the same Welford-warmup -> frozen-baseline -> CUSUM
  machinery as the latency detector. Catches a run whose appetite quietly
  shifts upward (RetryGuard, arXiv:2511.23278: local retry amplification
  shows up first as a sustained token-volume shift).
* **Budget envelope**: one ``token_runaway`` warning when cumulative episode
  tokens cross ``token_budget_warn_fraction`` of ``episode_token_budget``,
  and a single critical ``budget_breach`` at 100%. Each envelope threshold
  emits at most once per episode no matter how long the run continues.

Only token *counts* are read -- never content (project.md §1.4). Events
carrying neither token field are ignored entirely. Like the latency detector,
the CUSUM signal re-fires while elevated; wrap in ``DedupSink`` if per-episode
quiet is preferred.
"""

from __future__ import annotations

from typing import Any

from snagline.config import Config
from snagline.detectors.latency_anomaly import _WelfordCUSUM
from snagline.events import StepEvent
from snagline.risk import FailureRisk


class TokenRunawayDetector:
    name = "token_runaway"

    def __init__(
        self,
        k: float | None = None,
        h: float | None = None,
        min_samples: int | None = None,
        budget_total_tokens: int | None = None,
        warn_fraction: float | None = None,
        config: Config | None = None,
    ) -> None:
        cfg = config or Config()
        self.k = k if k is not None else cfg.token_cusum_k
        self.h = h if h is not None else cfg.token_cusum_h
        self.min_samples = (
            min_samples if min_samples is not None else cfg.token_min_samples
        )
        self.budget = (
            budget_total_tokens
            if budget_total_tokens is not None
            else cfg.episode_token_budget
        )
        self.warn_fraction = (
            warn_fraction
            if warn_fraction is not None
            else cfg.token_budget_warn_fraction
        )
        self._states: dict[str, _WelfordCUSUM] = {}
        self._totals: dict[str, int] = {}
        self._warned: dict[str, bool] = {}
        self._breached: dict[str, bool] = {}

    def observe(self, event: StepEvent) -> FailureRisk | None:
        if event.tokens_in is None and event.tokens_out is None:
            return None
        step_tokens = int((event.tokens_in or 0) + (event.tokens_out or 0))
        ep = event.episode_id

        # Deterministic envelope first: it needs no warm-up and carries the
        # hardest signal (a breach must be reported even mid-CUSUM-warmup).
        if self.budget is not None:
            total = self._totals.get(ep, 0) + step_tokens
            self._totals[ep] = total
            if not self._breached.get(ep, False):
                if total >= self.budget:
                    self._breached[ep] = True
                    return FailureRisk(
                        ep,
                        event.step_id,
                        1.0,
                        "budget_breach",
                        f"episode exceeded its {self.budget}-token budget "
                        f"({total} observed)",
                        event.timestamp,
                    )
                threshold = self.budget * self.warn_fraction
                if not self._warned.get(ep, False) and total >= threshold:
                    self._warned[ep] = True
                    return FailureRisk(
                        ep,
                        event.step_id,
                        0.8,
                        "token_runaway",
                        f"episode at {total / self.budget:.0%} of its "
                        f"{self.budget}-token budget",
                        event.timestamp,
                    )

        # Sustained-shift CUSUM over per-step volume.
        state = self._states.get(ep)
        if state is None:
            state = _WelfordCUSUM(self.k, self.h)
            self._states[ep] = state
        if not state.frozen:
            state.learn_only(float(step_tokens))
            if state.n >= self.min_samples:
                state.freeze()
            return None
        if state.update(float(step_tokens)):
            score = min(1.0, 0.6 + 0.1 * max(0.0, state.cusum / self.h - 1.0))
            return FailureRisk(
                ep,
                event.step_id,
                score,
                "token_runaway",
                f"per-step token burn deviates from baseline "
                f"(mean {state.mu0:.0f}/step)",
                event.timestamp,
            )
        return None

    def reset(self, episode_id: str) -> None:
        self._states.pop(episode_id, None)
        self._totals.pop(episode_id, None)
        self._warned.pop(episode_id, None)
        self._breached.pop(episode_id, None)

    def dump_state(self) -> dict[str, Any]:
        return {
            "states": {
                ep: {
                    "n": s.n,
                    "mean": s.mean,
                    "m2": s._m2,
                    "cusum": s.cusum,
                    "mu0": s.mu0,
                    "sigma0": s.sigma0,
                    "frozen": s.frozen,
                }
                for ep, s in self._states.items()
            },
            "totals": dict(self._totals),
            "warned": dict(self._warned),
            "breached": dict(self._breached),
        }

    def load_state(self, state: dict[str, Any]) -> None:
        self._states = {}
        for ep, raw in state.get("states", {}).items():
            s = _WelfordCUSUM(self.k, self.h)
            s.n = raw["n"]
            s.mean = raw["mean"]
            s._m2 = raw["m2"]
            s.cusum = raw["cusum"]
            s.mu0 = raw["mu0"]
            s.sigma0 = raw["sigma0"]
            s.frozen = raw["frozen"]
            self._states[ep] = s
        self._totals = {ep: int(v) for ep, v in state.get("totals", {}).items()}
        self._warned = {ep: bool(v) for ep, v in state.get("warned", {}).items()}
        self._breached = {ep: bool(v) for ep, v in state.get("breached", {}).items()}
