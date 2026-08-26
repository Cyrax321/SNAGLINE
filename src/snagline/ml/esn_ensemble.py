"""Optional echo-state-network + CUSUM ensemble detector (the ``ml`` extra).

Part of the optional ``ml`` extra: ``pip install snagline-agent[ml]`` (numpy
and scikit-learn). Core code never imports this module unconditionally:
``import snagline`` works with nothing but the standard library, and
``Monitor.default()`` only attempts the guarded import when
``Config.ml_ensemble_enabled`` is set. If numpy is missing or broken, the
zero-dependency noisy-OR ``MLOrchestrator`` runs exactly as before
(fail-open; project.md section 1.2, issue #80).

Signals combined per step (structure only: hashes, counts, timings,
booleans; never prompt or response content):

* Mahalanobis distance with a diagonal covariance approximation between the
  current step's latency/error pair and the matching per-tool statistics of
  a healthy-run ``BaselineProfile``, when one is configured via
  ``Config.goal_drift_baseline``. The diagonal form is used because the
  persisted profile stores marginal per-tool statistics only.
* A one-class echo-state-network residual: the reservoir predicts the next
  feature vector from the previous reservoir state, and large prediction
  error means the step looks unlike what the readout was trained on.
  Training data comes from ``fit()`` (a healthy trajectory), or, as a
  fallback, from the first ``warmup_steps`` steps of each live episode
  (assumed healthy; nothing emits during that warm-up).
* A CUSUM accumulator over the combined anomaly signal so that only
  sustained drift alarms; it fires once, then re-arms.

The detector is designed to run inside ``MLOrchestrator`` (its score feeds
the noisy-OR combination and the emitted trigger is ``ml_ensemble``), so its
standalone emissions carry the same ``ml_ensemble`` trigger owned by the ML
signal namespace.

Performance: fixed O(reservoir_size^2) floating point work per step, O(1)
amortized bookkeeping, bounded per-episode memory (the warm-up buffers hold
at most ``warmup_steps`` vectors). One small feature vector is allocated per
step; this path is opt-in and never runs in the zero-dependency preset.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Iterable

try:
    import numpy as np
except ImportError as _exc:  # pragma: no cover - exercised via guard tests
    raise ImportError(
        "snagline.ml.esn_ensemble requires the optional 'ml' extra: "
        "pip install snagline-agent[ml]"
    ) from _exc

from snagline.baseline import BaselineProfile
from snagline.events import StepEvent
from snagline.risk import FailureRisk

logger = logging.getLogger("snagline")

# Feature layout (all values clamped to [0, 1], structure only):
# [error flag, scaled latency_ms, scaled tokens_in, scaled tokens_out]
# Repetition patterns are deliberately excluded here: the zero-dependency
# loop detector owns them, and a one-class model cannot predict a free-
# running novelty channel without false alarms.
_FEATURE_DIM = 4
_LATENCY_SCALE_MS = 10_000.0
_TOKEN_SCALE = math.log1p(50_000.0)
# Sigma floors so a perfectly stable baseline cannot make every deviation
# infinite (mirrors the zero-dep latency detector's floors).
_SIGMA_FLOOR_ABS_MS = 1.0
_SIGMA_FLOOR_REL = 0.05
_ERROR_SIGMA_FLOOR = 0.05
# Score contribution for a tool absent from the healthy profile (mirrors the
# goal_drift detector's convention for unseen tools).
_UNKNOWN_TOOL_SCORE = 0.6
# The readout's own training residuals underestimate future residuals
# (in-sample optimism), so widen the band before calling a residual odd.
_RESIDUAL_SIGMA_INFLATION = 3.0
_ESN_ANOMALY_FLOOR = 0.05
_RIDGE_ALPHA = 1e-3


class _EpisodeState:
    """Per-episode reservoir state, warm-up buffers, and CUSUM accumulator."""

    __slots__ = (
        "beta",
        "context_prev",
        "cusum",
        "gram",
        "res_m2",
        "res_mu",
        "res_n",
        "rhs",
        "state",
        "warm_ctx",
        "warm_n",
        "warm_tgt",
    )

    def __init__(self, feature_dim: int, reservoir_size: int) -> None:
        self.state = np.zeros(reservoir_size)
        self.context_prev: np.ndarray | None = None
        size = reservoir_size + 1  # augmented with a bias term
        self.gram = np.zeros((size, size))
        self.rhs = np.zeros((size, feature_dim))
        self.warm_n = 0
        self.beta: np.ndarray | None = None
        self.res_mu = 0.0
        self.res_m2 = 0.0
        self.res_n = 0
        self.cusum = 0.0
        # Warm-up pair buffers, bounded by warmup_steps, used to seed the
        # residual statistics the moment the readout is solved.
        self.warm_ctx: list[np.ndarray] = []
        self.warm_tgt: list[np.ndarray] = []


class EsnCusumDetector:
    """One-class ESN + CUSUM ensemble emitting through ``MLOrchestrator``.

    Deterministic: reservoir weights come from a fixed-seed generator, so two
    instances built with the same arguments produce identical outputs on the
    same stream. Any exception inside :meth:`observe` is logged and swallowed
    (fail-open); it must never reach the host agent.
    """

    name = "esn_cusum"

    def __init__(
        self,
        baseline: BaselineProfile | None = None,
        reservoir_size: int = 32,
        spectral_radius: float = 0.9,
        input_scaling: float = 0.5,
        leak_rate: float = 0.3,
        seed: int = 1337,
        warmup_steps: int = 20,
        cusum_k: float = 0.25,
        cusum_h: float = 3.0,
    ) -> None:
        self._baseline = baseline
        self._warmup_steps = max(1, warmup_steps)
        self._k = cusum_k
        self._h = cusum_h
        rng = np.random.Generator(np.random.PCG64(seed))
        w = rng.uniform(-1.0, 1.0, (reservoir_size, reservoir_size))
        radius = float(np.max(np.abs(np.linalg.eigvals(w))))
        self._w: np.ndarray = w * (spectral_radius / radius)
        self._w_in: np.ndarray = rng.uniform(
            -input_scaling, input_scaling, (reservoir_size, _FEATURE_DIM)
        )
        self._leak = leak_rate
        # Readout fitted on healthy data via fit(); new episodes copy it.
        self._fitted_beta: np.ndarray | None = None
        self._fitted_res_mu = 0.0
        self._fitted_res_sigma = 0.0
        self._fitted_res_n = 0
        self._episodes: dict[str, _EpisodeState] = {}

    def observe(self, event: StepEvent) -> FailureRisk | None:
        """Score one step; fail-open wrapper around the numeric path.

        Never raises: an internal error is logged once per occurrence by the
        Monitor and here as well, then ignored (project.md section 1.2).
        """
        try:
            return self._observe(event)
        except Exception:
            logger.exception("snagline: esn_cusum raised; ignoring (fail-open)")
            return None

    def fit(self, events: Iterable[StepEvent]) -> None:
        """Fit the readout and residual statistics on a healthy trajectory.

        Call this with a known-healthy run (for example replayed from JSONL)
        before monitoring. Afterwards new episodes start scoring immediately
        against the healthy model instead of learning their own dynamics
        during warm-up. Raises ``ValueError`` on too few steps; this is an
        explicit setup-time call, not a hot-path one.
        """
        contexts: list[np.ndarray] = []
        targets: list[np.ndarray] = []
        state = np.zeros(self._w.shape[0])
        prev_context: np.ndarray | None = None
        for event in events:
            x = self._features(event)
            if prev_context is not None:
                contexts.append(prev_context)
                targets.append(x)
            state = self._advance(state, x)
            prev_context = np.concatenate((np.ones(1), state))
        if len(contexts) < 3:
            raise ValueError("need at least 3 healthy steps to fit the ESN")
        x_mat = np.vstack(contexts)
        y_mat = np.vstack(targets)
        beta = self._solve_ridge(x_mat.T @ x_mat, x_mat.T @ y_mat)
        residuals = np.linalg.norm(x_mat @ beta - y_mat, axis=1) / math.sqrt(
            _FEATURE_DIM
        )
        n = int(residuals.shape[0])
        mu = float(np.mean(residuals))
        sigma = float(np.std(residuals, ddof=1)) if n > 1 else 0.0
        self._fitted_beta = beta
        self._fitted_res_mu = mu
        self._fitted_res_sigma = sigma
        self._fitted_res_n = n

    def reset(self, episode_id: str) -> None:
        """Drop all per-episode state (reservoir, warm-up, CUSUM)."""
        self._episodes.pop(episode_id, None)

    # --- internals ---------------------------------------------------------

    def _observe(self, event: StepEvent) -> FailureRisk | None:
        st = self._episodes.get(event.episode_id)
        if st is None:
            st = self._new_state()
            self._episodes[event.episode_id] = st
        x = self._features(event)
        anomaly = self._esn_anomaly(st, x)
        signal = max(anomaly, self._mahalanobis_score(event))
        st.cusum = max(0.0, st.cusum + signal - self._k)
        if st.cusum >= self._h:
            score = min(1.0, 0.5 + 0.5 * (st.cusum - self._h) / self._h)
            st.cusum = 0.0  # re-arm so a persistent fault re-alarms later
            return FailureRisk(
                event.episode_id,
                event.step_id,
                score,
                "ml_ensemble",
                "sustained ensemble anomaly crossed the CUSUM threshold",
                event.timestamp,
            )
        return None

    def _new_state(self) -> _EpisodeState:
        st = _EpisodeState(_FEATURE_DIM, self._w.shape[0])
        if self._fitted_beta is not None:
            st.beta = self._fitted_beta
            st.res_mu = self._fitted_res_mu
            variance = self._fitted_res_sigma**2 * max(0, self._fitted_res_n - 1)
            st.res_m2 = max(0.0, variance)
            st.res_n = self._fitted_res_n
        return st

    def _features(self, event: StepEvent) -> np.ndarray:
        """Build the structural feature vector (no content, no novelty)."""
        f_lat = 0.0
        if event.latency_ms is not None and event.latency_ms > 0.0:
            f_lat = min(1.0, event.latency_ms / _LATENCY_SCALE_MS)
        tin = 0.0
        if event.tokens_in is not None and event.tokens_in > 0:
            tin = min(1.0, math.log1p(event.tokens_in) / _TOKEN_SCALE)
        tout = 0.0
        if event.tokens_out is not None and event.tokens_out > 0:
            tout = min(1.0, math.log1p(event.tokens_out) / _TOKEN_SCALE)
        err = 1.0 if event.error else 0.0
        return np.array([err, f_lat, tin, tout])

    def _advance(self, state: np.ndarray, x: np.ndarray) -> np.ndarray:
        pre = self._w_in @ x + self._w @ state
        return (1.0 - self._leak) * state + self._leak * np.tanh(pre)

    def _esn_anomaly(self, st: _EpisodeState, x: np.ndarray) -> float:
        """Next-step prediction anomaly of the ESN, in [0, 1]."""
        st.state = self._advance(st.state, x)
        context = np.concatenate((np.ones(1), st.state))
        if st.beta is None:
            # Warm-up: learn this episode's own (assumed healthy) dynamics
            # from previous-step context to current features. Silent phase.
            if st.context_prev is not None:
                st.gram += np.outer(st.context_prev, st.context_prev)
                st.rhs += np.outer(st.context_prev, x)
                st.warm_ctx.append(st.context_prev)
                st.warm_tgt.append(x)
                st.warm_n += 1
                if st.warm_n >= self._warmup_steps:
                    st.beta = self._solve_ridge(st.gram, st.rhs)
                    self._seed_residual_stats(st)
            st.context_prev = context
            return 0.0
        predicted = context @ st.beta
        residual = float(np.linalg.norm(predicted - x)) / math.sqrt(_FEATURE_DIM)
        excess = max(0.0, residual - st.res_mu) / (
            _RESIDUAL_SIGMA_INFLATION
            * math.sqrt(max(0.0, st.res_m2 / max(1, st.res_n - 1)))
            + _ESN_ANOMALY_FLOOR
        )
        anomaly = min(1.0, excess)
        if anomaly < 1.0:
            # Update running residual statistics only with healthy-ish
            # residuals so sustained faults cannot normalize themselves.
            st.res_n += 1
            delta = residual - st.res_mu
            st.res_mu += delta / st.res_n
            st.res_m2 += delta * (residual - st.res_mu)
        return anomaly

    def _mahalanobis_score(self, event: StepEvent) -> float:
        """Diagonal-covariance Mahalanobis score vs BaselineProfile, [0, 1).

        Computed over the standardized latency and error terms of the
        event's tool; mapped through d^2 / (d^2 + dof) so the value stays in
        range and grows monotonically with distance.
        """
        if self._baseline is None:
            return 0.0
        key = event.tool_name or "default"
        tb = self._baseline.tools.get(key)
        if tb is None:
            return _UNKNOWN_TOOL_SCORE
        if tb.count < 2:
            return 0.0
        terms = 0.0
        dof = 0
        p = tb.error_rate
        sigma_e = max(math.sqrt(max(0.0, p * (1.0 - p))), _ERROR_SIGMA_FLOOR)
        z_e = ((1.0 if event.error else 0.0) - p) / sigma_e
        terms += z_e * z_e
        dof += 1
        lat = event.latency_ms
        if lat is not None and lat > 0.0:
            sigma_l = max(
                tb.std_latency,
                _SIGMA_FLOOR_ABS_MS,
                _SIGMA_FLOOR_REL * tb.mean_latency,
            )
            z_l = (lat - tb.mean_latency) / sigma_l
            terms += z_l * z_l
            dof += 1
        if dof == 0:
            return 0.0
        return terms / (terms + dof)

    def _seed_residual_stats(self, st: _EpisodeState) -> None:
        """Seed running residual statistics from the warm-up pairs."""
        ctx = np.vstack(st.warm_ctx)
        tgt = np.vstack(st.warm_tgt)
        assert st.beta is not None
        residuals = np.linalg.norm(ctx @ st.beta - tgt, axis=1) / math.sqrt(
            _FEATURE_DIM
        )
        n = int(residuals.shape[0])
        st.res_mu = float(np.mean(residuals))
        var = float(np.var(residuals, ddof=1)) if n > 1 else 0.0
        st.res_m2 = max(0.0, var * (n - 1))
        st.res_n = n
        # The buffers have served their purpose; drop them.
        st.warm_ctx = []
        st.warm_tgt = []

    @staticmethod
    def _solve_ridge(gram: np.ndarray, rhs: np.ndarray) -> np.ndarray:
        reg = gram.copy()
        size = reg.shape[0]
        reg.flat[:: size + 1] += _RIDGE_ALPHA
        return np.linalg.solve(reg, rhs)
