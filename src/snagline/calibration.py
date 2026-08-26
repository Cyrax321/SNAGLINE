"""Opt-in threshold auto-calibration from a fitted BaselineProfile (issue #101).

With ``Config(calibration="auto")`` and a healthy-run profile loaded (the
output of ``snagline baseline``), ``Monitor.default()`` replaces two
hand-tuned inputs with values derived from the deployment's own observed
behavior:

* Error-cascade thresholds: the smallest windowed and consecutive error
  counts whose exceedance probability under the healthy per-step error rate
  stays within ``calibration_alpha``.
* Latency/CUSUM reference: the latency anomaly detector is seeded with the
  profile's healthy mean and spread per tool (the "mean + k*sigma" drift
  line), so monitoring starts immediately instead of after a live warm-up.

The math runs once per Monitor construction; nothing extra happens on the
ingest hot path (project.md section 1.5).

Model
-----
Each counted tool failure is treated as a Bernoulli(p) trial. The planning
rate is::

    p = max(pooled error rate, 99th-percentile per-tool error rate)

where per-tool rates from tools with fewer than ``MIN_TOOL_SAMPLES``
observations are excluded from the percentile: a tool used twice with one
failure would otherwise claim a 50% rate on almost no evidence, while the
pooled rate still represents it. Windowed exceedance uses the Binomial(window,
p) tail; consecutive exceedance uses exact run probabilities over the same
window.

Safety rails ("never worse than today", issue #101):

* Tighten-only clamp: derived counts get a floor of 2 and are clamped down to
  never rise above the hand-tuned defaults. The shipped defaults encode the
  product's agreed false-positive/recall tradeoff; one fitted profile
  justifies becoming more sensitive, never less.
* Fallback: ``calibration="auto"`` without a usable BaselineProfile keeps
  every hand-tuned default unchanged.
* Fail-open: baseline resolution and derivation run inside try/except in
  ``Monitor.default()``; any failure logs a warning and falls back to the
  hand-tuned defaults.

Two detectors keep hand-tuned constants by design. The loop detector needs
action-repetition evidence, which BaselineProfile deliberately does not store
(it keeps only aggregate counts, sums, and booleans per tool; project.md
section 1.4 privacy). CUSUM k/h define the alarm shape; calibration supplies
their mu0/sigma0 reference instead of replacing them.
"""

from __future__ import annotations

from dataclasses import dataclass

from snagline.baseline import BaselineProfile, load_baseline
from snagline.config import Config

# Tools with fewer observations than this are excluded from the per-tool
# percentile: their empirical rate is too noisy to plan thresholds from.
MIN_TOOL_SAMPLES = 20


@dataclass(frozen=True)
class CalibrationPlan:
    """Derived threshold overrides plus the profile they were derived from."""

    cascade_error_threshold: int
    cascade_consecutive_threshold: int
    baseline: BaselineProfile
    error_rate: float


def observed_error_rate(profile: BaselineProfile) -> float:
    """Healthy per-step failure probability used to plan thresholds.

    The max of the pooled error rate and the 99th-percentile well-sampled
    per-tool rate, so one unreliable but heavily exercised tool raises the
    planning rate while sparse tools cannot inflate it on their own.
    """
    total = sum(tb.count for tb in profile.tools.values())
    errors = sum(tb.error_count for tb in profile.tools.values())
    pooled = errors / total if total else 0.0
    rates = sorted(
        tb.error_rate for tb in profile.tools.values() if tb.count >= MIN_TOOL_SAMPLES
    )
    if not rates:
        return pooled
    # Nearest-rank 99th percentile over the well-sampled tools.
    rank = max(1, -(-len(rates) * 99 // 100))
    return max(pooled, rates[rank - 1])


def binomial_tail(n: int, k: int, p: float) -> float:
    """Return P(Bin(n, p) >= k) by walking the PMF upward from X = 0."""
    if k <= 0:
        return 1.0
    if k > n:
        return 0.0
    q = 1.0 - p
    if q <= 0.0:
        return 1.0
    pmf = q**n  # P(X = 0)
    cdf = pmf
    for i in range(1, k):
        pmf *= (n - i + 1) / i * p / q
        cdf += pmf
    return min(1.0, max(0.0, 1.0 - cdf))


def run_probability(n: int, k: int, p: float) -> float:
    """Return P(at least one run of >= k failures within n Bernoulli(p) steps).

    Dynamic program over the trailing failure-run length; construction-time
    cost only, never on the ingest path.
    """
    if k <= 0:
        return 1.0
    if k > n:
        return 0.0
    if p <= 0.0:
        return 0.0
    if p >= 1.0:
        return 1.0
    # dp[j]: no run of length >= k has occurred and the current trailing run
    # of failures is exactly j.
    dp = [0.0] * k
    dp[0] = 1.0
    q = 1.0 - p
    for _ in range(n):
        nxt = [0.0] * k
        for j in range(k):
            w = dp[j]
            if not w:
                continue
            nxt[0] += w * q
            if j + 1 < k:
                nxt[j + 1] += w * p
        dp = nxt
    return 1.0 - sum(dp)


def min_window_threshold(n: int, p: float, alpha: float) -> int:
    """Smallest t in [1, n] with P(Bin(n, p) >= t) <= alpha."""
    for t in range(1, n + 1):
        if binomial_tail(n, t, p) <= alpha:
            return t
    return n


def min_consecutive_threshold(n: int, p: float, alpha: float) -> int:
    """Smallest k in [1, n] with P(run of >= k failures in n steps) <= alpha."""
    for k in range(1, n + 1):
        if run_probability(n, k, p) <= alpha:
            return k
    return n


def build_plan(profile: BaselineProfile, cfg: Config) -> CalibrationPlan:
    """Derive calibrated thresholds from a healthy profile.

    Pure function of (profile, cfg); raises nothing for well-formed inputs.
    """
    rate = observed_error_rate(profile)
    window = cfg.cascade_window_size
    alpha = cfg.calibration_alpha
    raw_windowed = min_window_threshold(window, rate, alpha)
    raw_consecutive = min_consecutive_threshold(window, rate, alpha)
    # Tighten-only clamp: floor of 2 (a single error is never a cascade),
    # ceiling at the hand-tuned defaults so auto-calibration can never be
    # less sensitive than today's shipped behavior (issue #101).
    windowed = min(max(raw_windowed, 2), cfg.cascade_error_threshold)
    consecutive = min(max(raw_consecutive, 2), cfg.cascade_consecutive_threshold)
    return CalibrationPlan(
        cascade_error_threshold=windowed,
        cascade_consecutive_threshold=consecutive,
        baseline=profile,
        error_rate=rate,
    )


def resolve_baseline_profile(cfg: Config) -> BaselineProfile | None:
    """Resolve the calibration baseline: explicit object first, then path.

    Returns None when neither is configured. Raises when a configured path
    cannot be loaded; callers are expected to treat that as fail-open.
    """
    if cfg.calibration_baseline is not None:
        return cfg.calibration_baseline
    path = cfg.calibration_baseline_path
    if path:
        return load_baseline(path)
    return None
