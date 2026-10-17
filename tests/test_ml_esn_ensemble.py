"""Tests for the optional ESN + CUSUM ensemble detector (issue #80).

These tests need the ``ml`` extra (numpy). When numpy is absent the whole
module skips gracefully; CI runs a dedicated leg with the extra installed
so they are always exercised somewhere. The import-guard behavior for
numpy-less environments lives in test_ml_extra_guard.py.
"""

from __future__ import annotations

import logging
from typing import Any

import pytest

np = pytest.importorskip("numpy", reason="ml extra (numpy) not installed")

from snagline.baseline import BaselineProfile, ToolBaseline  # noqa: E402
from snagline.config import Config  # noqa: E402
from snagline.detectors.ml_ensemble import MLOrchestrator  # noqa: E402
from snagline.events import StepEvent  # noqa: E402
from snagline.ml.esn_ensemble import EsnCusumDetector  # noqa: E402
from snagline.monitor import Monitor  # noqa: E402
from snagline.risk import FailureRisk  # noqa: E402


def _fast(**overrides) -> EsnCusumDetector:
    """A detector with short warm-up and a low CUSUM threshold for tests."""
    params = dict(warmup_steps=5, cusum_h=1.0)
    params.update(overrides)
    return EsnCusumDetector(**params)  # type: ignore[arg-type]


def _ev(
    i: int,
    episode: str = "ep",
    *,
    tool: str = "t",
    latency: float = 100.0,
    error: bool = False,
    signature: str = "sig",
) -> StepEvent:
    return StepEvent(
        step_id=f"s{i}",
        episode_id=episode,
        timestamp=float(i),
        action_type="tool_call",
        action_signature=signature,
        tool_name=tool,
        latency_ms=latency,
        error=error,
    )


def _healthy(n: int, start: int = 0) -> list[StepEvent]:
    """Steady, predictable steps: stable latency, cycling signatures."""
    return [
        _ev(i, signature=f"a{i % 4}", latency=100.0 + (i % 3))
        for i in range(start, start + n)
    ]


def _unhealthy(n: int, start: int) -> list[StepEvent]:
    """Failing steps: novel signatures, huge latency, errors."""
    return [
        _ev(i, signature=f"x{i}", latency=9000.0 + i, error=True)
        for i in range(start, start + n)
    ]


class _Collector:
    """Sink that records every emitted risk."""

    def __init__(self) -> None:
        self.risks: list[FailureRisk] = []

    def emit(self, risk: FailureRisk) -> None:
        self.risks.append(risk)


class _Stub:
    """Detector stub with a fixed score, used to exercise noisy-OR fusion."""

    def __init__(self, score: float) -> None:
        self._score = score

    def observe(self, event: StepEvent) -> FailureRisk | None:
        return FailureRisk(
            event.episode_id,
            event.step_id,
            self._score,
            "loop",
            "stub",
            event.timestamp,
        )

    def reset(self, episode_id: str) -> None:
        pass


# --- Both sides: healthy stays silent ---------------------------------------


def test_healthy_stream_never_emits():
    det = _fast()
    for event in _healthy(40):
        assert det.observe(event) is None


def test_healthy_stream_with_baseline_stays_silent():
    tb = ToolBaseline(tool_name="t")
    for _ in range(50):
        tb.add(100.0, False)
    profile = BaselineProfile(tools={"t": tb}, total_steps=50)
    det = _fast(baseline=profile)
    for event in _healthy(40):
        assert det.observe(event) is None


def test_warmup_phase_is_silent_even_for_faulty_input():
    det = _fast(warmup_steps=30)
    for event in _unhealthy(25, start=0):
        assert det.observe(event) is None


def test_monitor_default_ml_flag_healthy_run_emits_nothing():
    cfg = Config()
    cfg.ml_ensemble_enabled = True
    sink = _Collector()
    mon = Monitor.default(config=cfg, sinks=[sink])
    # Unique signatures keep the deterministic loop detector quiet; stable
    # latency/error keep cascade, latency-CUSUM, and the ESN quiet too.
    for i in range(60):
        mon.ingest(_ev(i, signature=f"u{i}", latency=100.0 + (i % 3)))
    assert sink.risks == []


# --- Both sides: injected failure fires with the exact trigger --------------


def test_unhealthy_stream_fires_ml_ensemble_trigger():
    det = _fast()
    fired: list[FailureRisk] = []
    stream = _healthy(10) + _unhealthy(12, start=10)
    for event in stream:
        risk = det.observe(event)
        if risk is not None:
            fired.append(risk)
    assert len(fired) >= 1
    assert all(r.trigger == "ml_ensemble" for r in fired)
    assert all(0.5 <= r.score <= 1.0 for r in fired)


def test_cusum_rearms_after_firing():
    det = _fast()
    stream = _healthy(10) + _unhealthy(40, start=10)
    fired = [r for e in stream if (r := det.observe(e)) is not None]
    # A sustained fault must alarm more than once across 40 bad steps.
    assert len(fired) >= 2


def test_esn_score_feeds_noisy_or_inside_ml_orchestrator():
    params = dict(warmup_steps=5, cusum_h=1.0)
    cfg = Config()
    cfg.ml_ensemble_score_threshold = 0.5
    orch = MLOrchestrator([_Stub(0.3), _fast(**params)], config=cfg)
    direct = _fast(**params)
    stream = _healthy(6) + _unhealthy(12, start=6)
    saw_combined = False
    for event in stream:
        solo = direct.observe(event)
        combined = orch.observe(event)
        if solo is not None:
            # Independent hand computation of noisy-OR over 0.3 and solo.
            expected = 1.0 - (1.0 - 0.3) * (1.0 - solo.score)
            assert combined is not None
            assert abs(combined.score - expected) < 1e-9
            assert combined.trigger == "ml_ensemble"
            saw_combined = True
    assert saw_combined


def test_monitor_default_ml_flag_unhealthy_run_emits_ml_ensemble():
    cfg = Config()
    cfg.ml_ensemble_enabled = True
    sink = _Collector()
    mon = Monitor.default(config=cfg, sinks=[sink])
    for i in range(10):
        mon.ingest(_ev(i, signature=f"u{i}", latency=100.0 + (i % 3)))
    for j in range(15):
        mon.ingest(_ev(10 + j, signature=f"x{j}", latency=9000.0 + j, error=True))
    assert len(sink.risks) >= 1
    assert all(r.trigger == "ml_ensemble" for r in sink.risks)


# --- Mahalanobis baseline scoring: exact hand-computed values ---------------


def _profile(mean: float, std_ms: float, n: int = 50) -> BaselineProfile:
    tb = ToolBaseline(tool_name="t")
    if std_ms == 0.0:
        for _ in range(n):
            tb.add(mean, False)
    else:
        for i in range(n):
            tb.add(mean + (std_ms if i % 2 else -std_ms), False)
    return BaselineProfile(tools={"t": tb}, total_steps=n)


def test_mahalanobis_exact_value_two_sigma_latency_and_error():
    # mean=100, alternating +/-20 latency, every 5th step an error.
    tb = ToolBaseline(tool_name="t")
    for i in range(50):
        tb.add(120.0 if i % 2 else 80.0, error=(i % 5 == 0))
    profile = BaselineProfile(tools={"t": tb}, total_steps=50)
    det = _fast(baseline=profile)
    # Closed-form expectations from the documented diagonal formula, computed
    # here by hand rather than by calling the detector under test:
    # sample std uses the n-1 denominator: var = (sum_sq - sum^2/n)/(n-1).
    sum_sq = 25 * 120.0**2 + 25 * 80.0**2
    std = ((sum_sq - 50 * 100.0**2) / 49) ** 0.5
    p = 10 / 50
    sigma_e = max((p * (1 - p)) ** 0.5, 0.05)
    z_lat_wild = (140.0 - 100.0) / std
    z_err_wild = (1.0 - p) / sigma_e
    d2_wild = z_lat_wild**2 + z_err_wild**2
    expected_wild = d2_wild / (d2_wild + 2)
    wild = _ev(1, latency=140.0, error=True)
    assert det._mahalanobis_score(wild) == pytest.approx(expected_wild, abs=1e-9)
    # Calm event at the healthy mean: latency term is exactly 0; the error
    # term contributes ((0-p)/sigma_e)^2 = 0.25, dof=2 -> 0.25/2.25 = 1/9.
    calm = _ev(0, latency=100.0, error=False)
    assert det._mahalanobis_score(calm) == pytest.approx(1.0 / 9.0, abs=1e-9)


def test_mahalanobis_unknown_tool_scores_constant():
    profile = _profile(100.0, 0.0)
    det = _fast(baseline=profile)
    assert det._mahalanobis_score(_ev(0, tool="never_seen")) == pytest.approx(0.6)


def test_mahalanobis_disabled_without_baseline():
    det = _fast()
    assert det._mahalanobis_score(_ev(0, latency=9999.0, error=True)) == 0.0


def _untimed_profile(n: int = 50, error_every: int = 0) -> BaselineProfile:
    """A profile from a stream whose adapter reported no latency_ms.

    ``count`` grows on every call while ``latency_count`` stays 0, which is
    legitimate -- error rates are measured even on streams that report no
    timing (issue #101) -- but it means ``mean_latency``/``std_latency`` are
    0.0/0.0, not "zero-ms latency".
    """
    tb = ToolBaseline(tool_name="t")
    for i in range(n):
        tb.add(None, error=bool(error_every and i % error_every == 0))
    return BaselineProfile(tools={"t": tb}, total_steps=n)


def test_mahalanobis_untimed_baseline_does_not_score_latency():
    """The latency term must gate on latency_count, not count.

    Scoring a live latency against mean 0.0 with the 1 ms sigma floor makes
    any real latency a ~100-sigma deviation, saturating this term to 1.0.
    That feeds ``max(anomaly, mahalanobis)``, so it armed the CUSUM on every
    step of a healthy episode (issue #348, the same defect the zero-dep
    latency detector fixed).
    """
    det = _fast(baseline=_untimed_profile())
    # A perfectly ordinary latency, against a baseline that has no latency
    # samples at all: the latency term is absent, and the event matches the
    # baseline's zero error rate, so there is nothing to score.
    assert det._mahalanobis_score(_ev(0, latency=100.0)) == pytest.approx(0.0, abs=1e-9)
    # And a wild one: without timed samples the baseline has no latency
    # opinion, so 9000 ms is no more anomalous than 1 ms.
    assert det._mahalanobis_score(_ev(1, latency=9000.0)) == pytest.approx(
        0.0, abs=1e-9
    )


def test_untimed_baseline_healthy_stream_stays_silent():
    """End to end: an untimed baseline must not page on a healthy run.

    Before the gate this fired four times in 20 steps, because the saturated
    Mahalanobis term overrode the ESN's healthy residual on every step.
    """
    det = _fast(baseline=_untimed_profile())
    fires = [e.step_id for e in _healthy(40) if det.observe(e) is not None]
    assert fires == [], f"untimed baseline produced false alarms: {fires}"


def test_untimed_baseline_still_scores_genuine_errors():
    """Scope guard: the error term survives without timing.

    Removing the latency term must not remove the term that issue #101's
    split exists to preserve. A tool that never errored in the healthy
    baseline is genuinely anomalous when it errors live.
    """
    det = _fast(baseline=_untimed_profile())
    assert det._mahalanobis_score(_ev(0, latency=100.0, error=True)) > 0.9


def test_mahalanobis_timed_baseline_still_flags_latency():
    """Regression guard: the gate must not neuter the timed path."""
    det = _fast(baseline=_profile(100.0, 20.0))
    assert det._mahalanobis_score(_ev(0, latency=5000.0, error=False)) > 0.9


# --- Fail-open guarantees ----------------------------------------------------


def test_fail_open_internal_exception_is_swallowed_and_logged(caplog):
    det = _fast()

    def boom(self_event: StepEvent) -> object:
        raise RuntimeError("feature extraction exploded")

    det._features = boom  # type: ignore[assignment]
    with caplog.at_level(logging.ERROR, logger="snagline"):
        assert det.observe(_ev(0)) is None
    assert any("fail-open" in rec.getMessage() for rec in caplog.records)
    # Removing the shadowing attribute restores the class method: the
    # detector keeps working after the fault clears.
    del det._features  # type: ignore[attr-defined]
    assert det.observe(_ev(1)) is None


class _ObserveBoom(EsnCusumDetector):
    """Overrides observe() itself, bypassing the detector's own guard, so
    the MONITOR's fail-open layer is what must catch the fault."""

    def observe(self, event: StepEvent) -> FailureRisk | None:
        raise RuntimeError("ml path exploded")


def test_fail_open_monitor_survives_crashing_ml_detector():
    sink = _Collector()
    mon = Monitor([_ObserveBoom(warmup_steps=2)], [sink], fail_open=True)
    for i in range(18):
        mon.ingest(
            _ev(i, signature=f"u{i}")
            if i < 10
            else _ev(i, signature=f"x{i}", latency=9000.0, error=True)
        )  # must never raise into the host
    assert sink.risks == []
    assert mon.metrics()["detector_errors"] >= 1


# --- Determinism, reset, fit, memory bounds ---------------------------------


def test_same_seed_produces_identical_outputs():
    stream = _healthy(8) + _unhealthy(14, start=8)
    det_a, det_b = _fast(), _fast()
    scores_a = [r.score if (r := det_a.observe(e)) else None for e in stream]
    scores_b = [r.score if (r := det_b.observe(e)) else None for e in stream]
    assert scores_a == scores_b
    assert any(s is not None for s in scores_a)


def test_reset_restores_fresh_behavior():
    stream = _healthy(8) + _unhealthy(14, start=8)
    fresh = _fast()
    expected = [fresh.observe(e) for e in stream]
    reused = _fast()
    for event in stream[:9]:
        reused.observe(event)
    reused.reset("ep")
    again = [reused.observe(e) for e in stream]
    assert [(r.score if r else None) for r in expected] == [
        (r.score if r else None) for r in again
    ]


def test_fit_on_healthy_trajectory_enables_immediate_scoring():
    det = _fast(warmup_steps=5)
    det.fit(_healthy(30))
    # New episodes skip warm-up entirely: scoring starts at once.
    for event in _healthy(15):
        assert det.observe(event) is None
    fired = [det.observe(e) for e in _unhealthy(10, start=15)]
    assert any(r is not None and r.trigger == "ml_ensemble" for r in fired)


def test_fit_rejects_too_short_trajectories():
    det = _fast()
    with pytest.raises(ValueError):
        det.fit(_healthy(2))


def test_feature_vector_is_structural_and_bounded():
    """Features read only structure (error, latency, tokens), never content,
    and stay within [0, 1] even for extreme inputs."""
    det = _fast()
    extreme = StepEvent(
        step_id="s",
        episode_id="ep",
        timestamp=0.0,
        action_type="tool_call",
        action_signature="sig",
        tool_name="t",
        latency_ms=10_000_000.0,
        error=True,
        tokens_in=10_000_000,
        tokens_out=10_000_000,
        metadata={"prompt": "SECRET content that must not be read"},
    )
    x = det._features(extreme)
    assert x.shape == (4,)
    assert all(0.0 <= v <= 1.0 for v in x)


def test_noisy_but_healthy_stream_stays_silent():
    """A fixed-seed realistic stream: gaussian latency, sparse errors,
    varying token counts. Sustained-drift gating must keep it quiet."""
    import random

    rng = random.Random(7)
    det = _fast(warmup_steps=12, cusum_h=3.0)
    for i in range(80):
        event = _ev(
            i,
            signature=f"a{i % 6}",
            latency=max(1.0, rng.gauss(100, 25)),
            error=rng.random() < 0.03,
        )
        assert det.observe(event) is None


# --- Privacy ------------------------------------------------------------------


def test_features_ignore_metadata_content():
    secret = StepEvent(
        step_id="s0",
        episode_id="ep",
        timestamp=0.0,
        action_type="tool_call",
        action_signature="sig",
        tool_name="t",
        latency_ms=120.0,
        metadata={"prompt": "SECRET instructions do not leak"},
    )
    other = StepEvent(
        step_id="s0",
        episode_id="ep",
        timestamp=0.0,
        action_type="tool_call",
        action_signature="sig",
        tool_name="t",
        latency_ms=120.0,
        metadata={"prompt": "totally different content"},
    )
    stream_a = [secret] + _unhealthy(10, start=1)
    stream_b = [other] + _unhealthy(10, start=1)
    det_a, det_b = _fast(), _fast()
    scores_a = [r.score if (r := det_a.observe(e)) else None for e in stream_a]
    scores_b = [r.score if (r := det_b.observe(e)) else None for e in stream_b]
    assert scores_a == scores_b


# --- live scoring uses the fit-consistent pairing (issue #243) ----------------


def test_fitted_live_residuals_match_fit_pairing_on_healthy_traffic():
    """Issue #243: fit() solves beta for (state after x_{t-1}) -> x_t, but
    live scoring advanced the reservoir with x_t *first* and predicted x_t
    from the post-advance state. The step leaked into the context judging it,
    so even a perfectly healthy continuation carried a systematic residual
    bias over the trained mean, and the running residual-stat update absorbed
    it (res_mu drifting ~2 sigma on clean data). Drives the real observe()
    path: on healthy traffic the running res_mu must stay centered on the
    fitted mean.
    """
    det = EsnCusumDetector(seed=7, warmup_steps=5, cusum_h=1.0)
    healthy = [
        _ev(i, signature=f"a{i % 2}", latency=100.0 if i % 2 else 200.0)
        for i in range(40)
    ]
    det.fit(healthy)
    fitted_mu = det._fitted_res_mu
    fitted_sigma = det._fitted_res_sigma
    assert fitted_sigma > 0.0

    # Long, perfectly in-pattern continuation: nothing here is anomalous, so
    # every step updates the running residual stats with a "healthy" residual.
    for i in range(40, 240):
        det.observe(_ev(i, signature=f"a{i % 2}", latency=100.0 if i % 2 else 200.0))
    st = det._episodes["ep"]
    drift = st.res_mu - fitted_mu
    assert abs(drift) < 0.5 * fitted_sigma, (
        f"healthy continuation drifted res_mu by {drift / fitted_sigma:.2f} "
        f"sigma from the fitted mean: the live pairing does not match fit()"
    )


def test_fitted_first_live_step_is_silent_then_scores():
    """The first scored step has no pre-advance context (fit() skips the same
    step); it stays silent once, and the second step scores normally."""
    det = _fast()
    healthy = [_ev(i, signature=f"a{i % 4}", latency=100.0) for i in range(12)]
    det.fit(healthy)
    st = det._new_state()

    det._episodes["ep"] = st
    first = det._esn_anomaly(st, det._features(healthy[0]))
    assert first == 0.0, "no pre-advance context exists for the first step"
    assert st.context_prev is not None
    second = det._esn_anomaly(st, det._features(healthy[1]))
    assert second is not None  # scores from the previous context


# --- knob validation and fail-open log discipline (issue #386) ----------------


@pytest.mark.parametrize("bad", [0, -1, -32])
def test_reservoir_size_below_one_is_rejected(bad: int) -> None:
    """An empty reservoir cannot be built; fail at construction with a message
    that names the knob, not a numpy reduction error."""
    with pytest.raises(ValueError, match="reservoir_size must be >= 1"):
        EsnCusumDetector(reservoir_size=bad)  # type: ignore[arg-type]


@pytest.mark.parametrize("bad", [-0.001, -1.0, -100.0])
def test_negative_cusum_k_is_rejected(bad: float) -> None:
    """A negative slack inflates the accumulator every step (16 false positives
    on 50 healthy steps when measured); 0.0 is legitimate and stays allowed."""
    with pytest.raises(ValueError, match="cusum_k must be >= 0.0"):
        EsnCusumDetector(cusum_k=bad)


@pytest.mark.parametrize("bad", [0.0, -1e-9, -1.0])
def test_non_positive_cusum_h_is_rejected(bad: float) -> None:
    """cusum_h=0.0 divided by zero in the score formula on *every* step (50
    swallowed ZeroDivisionErrors, zero risks -- the detector was permanently
    dark and the log made it look merely noisy); a negative h can never be
    crossed by a non-negative accumulator, silently disabling detection."""
    with pytest.raises(ValueError, match="cusum_h must be > 0.0"):
        EsnCusumDetector(cusum_h=bad)


def test_validation_error_names_the_knob_and_its_effect() -> None:
    """An out-of-range knob is a configuration error: the message must say which
    knob and why, so it is distinguishable from an internal fault."""
    with pytest.raises(ValueError) as exc_info:
        EsnCusumDetector(cusum_h=0.0)
    message = str(exc_info.value)
    assert "cusum_h" in message
    assert "silently disabling" in message


def test_boundary_values_are_accepted() -> None:
    """Positive control: the rejected bounds are open, so the legitimate
    boundary still constructs and detects. cusum_k=0.0 makes the accumulator
    sum the raw signal, and a small cusum_h still fires on sustained drift."""
    det = EsnCusumDetector(warmup_steps=5, cusum_k=0.0, cusum_h=0.25)
    healthy = [det.observe(e) for e in _healthy(30)]
    assert all(r is None for r in healthy), "healthy traffic must stay silent"
    fired = [det.observe(e) for e in _unhealthy(20, start=30)]
    assert any(r is not None and r.trigger == "ml_ensemble" for r in fired)


def _faulty_det(exc_type: type[Exception]) -> EsnCusumDetector:
    """A detector whose feature extraction always raises the given class."""
    det = _fast()

    def boom(_event: StepEvent) -> object:
        raise exc_type("synthetic persistent fault")

    det._features = boom  # type: ignore[assignment]
    return det


def test_persistent_fault_is_logged_once_not_per_step(caplog) -> None:
    """The fail-open wrapper must dedupe: before the fix a fault raised on
    every step produced one ERROR record per step (50 for 50 steps), and
    MLOrchestrator._log_fault_once could not help because the exception never
    escaped observe() to reach its own handler."""
    det = _faulty_det(RuntimeError)
    with caplog.at_level(logging.ERROR, logger="snagline"):
        for i in range(50):
            assert det.observe(_ev(i)) is None  # never raises into the host
    failures = [r for r in caplog.records if r.exc_info is not None]
    assert len(failures) == 1, f"expected 1 logged fault, got {len(failures)}"
    assert "fail-open" in failures[0].getMessage()


def test_a_different_fault_after_the_first_is_still_reported(caplog) -> None:
    """The dedupe key carries the exception class: a *new* fault must still
    surface, otherwise the first fault would silence every later one."""
    det = _faulty_det(RuntimeError)
    with caplog.at_level(logging.ERROR, logger="snagline"):
        for _ in range(10):
            det.observe(_ev(0))

        def boom_value(_event: StepEvent) -> object:
            raise ValueError("a different fault")

        det._features = boom_value  # type: ignore[assignment]
        for _ in range(10):
            det.observe(_ev(0))

        def boom_key(_event: StepEvent) -> object:
            raise KeyError("yet another")

        det._features = boom_key  # type: ignore[assignment]
        for _ in range(10):
            det.observe(_ev(0))

    types = {
        r.exc_info[0].__name__  # type: ignore[index]
        for r in caplog.records
        if r.exc_info is not None
    }
    assert types == {"RuntimeError", "ValueError", "KeyError"}


def test_mutated_h_degrades_to_saturated_score_not_division_by_zero(caplog) -> None:
    """Defense in depth on the score denominator. __init__ rejects cusum_h <= 0,
    so this simulates a future path mutating _h after construction: the floor
    keeps the division non-zero, so the detector saturates -- every step scores
    in the valid range -- instead of raising (and being swallowed and logged)
    on every step."""
    det = _fast()
    det._h = 0.0  # type: ignore[attr-defined]
    with caplog.at_level(logging.ERROR, logger="snagline"):
        risks = [det.observe(e) for e in _healthy(20)]
    assert all(r is not None and 0.5 <= r.score <= 1.0 for r in risks)
    assert [r for r in caplog.records if r.exc_info is not None] == []


def test_valid_knobs_do_not_trip_the_validation() -> None:
    """The shipped construction path -- Monitor.default builds this detector
    with defaults -- must be unaffected."""
    det = EsnCusumDetector()
    assert det._h == 3.0
    assert det._k == 0.25
    for event in _healthy(40):
        assert det.observe(event) is None


def _esn_of(monitor: Monitor) -> EsnCusumDetector:
    """The ESN leg of the orchestrator Monitor.default() builds."""
    orch = monitor._detectors[0]
    assert isinstance(orch, MLOrchestrator)
    esn = orch._base[-1]
    assert isinstance(esn, EsnCusumDetector)
    return esn


# --- Restart survivability (issue #581) -------------------------------------
#
# Before #581 EsnCusumDetector had no dump_state/load_state, so
# MLOrchestrator.dump_state silently skipped it (it guards with getattr) and a
# snapshot/restore lost both the fitted readout and every live episode's
# reservoir + CUSUM. The restored detector re-warmed from the live stream --
# assuming it healthy -- and an anomaly shorter than warmup_steps was
# swallowed whole instead of alarming.


def test_dump_state_round_trips_the_fitted_readout():
    det = _fast()
    det.fit(_healthy(40))
    assert det._fitted_beta is not None
    dumped = det.dump_state()
    assert dumped is not None
    reborn = _fast()
    reborn.load_state(dumped)
    assert reborn._fitted_beta is not None
    assert np.allclose(det._fitted_beta, reborn._fitted_beta)
    assert reborn._fitted_res_n == det._fitted_res_n


def test_dump_state_round_trips_a_live_episodes_reservoir_and_cusum():
    det = _fast()
    det.fit(_healthy(40))
    for event in _healthy(30):
        assert det.observe(event) is None
    st = det._episodes["ep"]
    # Push the CUSUM partway so the accumulator is non-trivial to carry.
    for event in _unhealthy(3, 40):
        det.observe(event)
    dumped = det.dump_state()
    assert dumped is not None
    reborn = _fast()
    reborn.fit(_healthy(40))
    reborn.load_state(dumped)
    st2 = reborn._episodes["ep"]
    assert np.allclose(st.state, st2.state)
    assert np.allclose(st.gram, st2.gram)
    assert np.allclose(st.rhs, st2.rhs)
    assert st.cusum == st2.cusum
    assert st.res_n == st2.res_n
    assert st.warm_n == st2.warm_n


def test_dump_state_is_json_compatible():
    import json

    det = _fast()
    det.fit(_healthy(40))
    for event in _healthy(10):
        det.observe(event)
    # A snapshot is stdlib JSON, never pickle (StatefulDetector contract).
    json.dumps(det.dump_state())


def test_load_state_tolerates_a_reservoir_size_mismatch():
    """The live config owns the reservoir shape. A snapshot written by a
    differently-sized reservoir must not restore arrays that would raise on
    the next observe; the detector falls back to warm-up instead."""
    det = EsnCusumDetector(reservoir_size=16, warmup_steps=5, cusum_h=1.0)
    det.fit(_healthy(40))
    for event in _healthy(10):
        det.observe(event)
    dumped = det.dump_state()
    assert dumped is not None
    assert dumped["reservoir_size"] == 16

    other = EsnCusumDetector(reservoir_size=32, warmup_steps=5, cusum_h=1.0)
    other.load_state(dumped)
    # Nothing was adopted, but nothing raised either.
    assert other._episodes == {}
    assert other._fitted_beta is None
    # And the detector still works from cold.
    for event in _healthy(30):
        assert other.observe(event) is None


def test_load_state_defaults_an_absent_snapshot_cleanly():
    """An empty/foreign payload must leave a working cold-start detector."""
    det = _fast()
    det.load_state({})
    assert det._episodes == {}
    for event in _healthy(30):
        assert det.observe(event) is None


def test_ml_orchestrator_now_carries_the_esn_substate():
    """The delegation gap: MLOrchestrator.dump_state must reach this detector,
    not skip it for lacking the protocol."""
    det = _fast()
    det.fit(_healthy(40))
    for event in _healthy(10):
        det.observe(event)
    orch = MLOrchestrator([det])
    dumped = orch.dump_state()
    assert dumped is not None
    assert "esn_cusum" in dumped
    reborn = MLOrchestrator([_fast()])
    reborn.load_state(dumped)
    assert "ep" in reborn._base[0]._episodes


def test_monitor_snapshot_restore_preserves_the_fitted_readout():
    cfg = Config(ml_ensemble_enabled=True)
    m = Monitor.default(config=cfg, sinks=[_Collector()])
    esn = _esn_of(m)
    esn.fit(_healthy(40))
    for event in _healthy(20):
        m.ingest(event)
    snap = m.snapshot_dict()

    fresh = Monitor.default(config=cfg, sinks=[_Collector()])
    fresh.restore_dict(snap)
    esn2 = _esn_of(fresh)
    assert esn2._fitted_beta is not None
    assert np.allclose(esn._fitted_beta, esn2._fitted_beta)


def test_a_short_anomaly_is_not_swallowed_after_a_restart():
    """The failure this exists to prevent: an anomaly shorter than
    warmup_steps was missed entirely once the detector re-warmed from the live
    stream after a restore.

    The stream is built so the ESN is the only base that can produce a signal:
    unique signatures keep the loop detector quiet, no errors keep the cascade
    quiet, and the latency band barely moves so the latency CUSUM stays under
    threshold -- the anomaly is a token-side distribution shift, which only
    the one-class reservoir sees.
    """
    cfg = Config(ml_ensemble_enabled=True)
    tools = ["search", "lookup", "write", "calc", "fetch", "store", "parse", "render"]

    def step(i, latency, tokens_out):
        return StepEvent(
            step_id=f"s{i}",
            episode_id="ep",
            timestamp=float(i),
            action_type="tool_call",
            action_signature=f"sig-{i}",
            tool_name=tools[i % len(tools)],
            latency_ms=latency,
            tokens_in=100,
            tokens_out=tokens_out,
        )

    healthy = [step(i, 10.0, 40) for i in range(60)]
    # Fewer steps than warmup_steps (20): before #581 the re-warm swallowed it.
    anomaly = [step(i, 12.0, 9000) for i in range(60, 75)]

    def build():
        m = Monitor.default(config=cfg, sinks=[_Collector()])
        _esn_of(m).fit(healthy)
        return m

    continuous = build()
    for event in healthy:
        continuous.ingest(event)
    for event in anomaly:
        continuous.ingest(event)
    continuous.end_episode("ep")
    fired_continuous = sum(
        1 for r in continuous._sinks[0].risks if r.trigger == "ml_ensemble"
    )

    checkpointed = build()
    for event in healthy:
        checkpointed.ingest(event)
    snap = checkpointed.snapshot_dict()
    restored = build()
    restored.restore_dict(snap)
    for event in anomaly:
        restored.ingest(event)
    restored.end_episode("ep")
    fired_restored = sum(
        1 for r in restored._sinks[0].risks if r.trigger == "ml_ensemble"
    )

    assert fired_continuous > 0, "the probe must produce a signal to lose"
    assert fired_restored == fired_continuous, (
        "a restart mid-episode must not blind the ensemble: "
        f"{fired_restored} vs {fired_continuous} ml_ensemble risks"
    )


# --- the restore dropped what it had serialised (#581 follow-up) -------------
#
# load_state restored state/gram/rhs by validating the *snapshot's* shapes, but
# restored beta and context_prev by asking whether the *fresh* state already
# held them. _new_state() always leaves context_prev None, and leaves beta None
# on an unfitted detector -- which is the production path, since Monitor.default
# builds EsnCusumDetector without calling fit(). Both guards were therefore
# unreachable, and the round-trip silently discarded both fields.


def _warmed_unfitted() -> tuple[EsnCusumDetector, dict[str, Any]]:
    """An episode that solved its own readout during warm-up, no fit() called,
    paired with its own snapshot.

    This is the shape Monitor.default() produces in production: the detector is
    constructed unfitted and learns each episode's dynamics from its first
    warmup_steps steps. The snapshot is returned alongside so callers get the
    narrowed type; dump_state is typed ``dict | None`` and each test needs the
    payload.
    """
    det = _fast()
    for event in _healthy(8):
        assert det.observe(event) is None
    st = det._episodes["ep"]
    assert st.beta is not None, "the fixture must be past warm-up"
    dumped = det.dump_state()
    assert dumped is not None
    return det, dumped


def test_restore_keeps_a_warm_up_solved_readout_when_the_detector_is_unfitted():
    det, dumped = _warmed_unfitted()
    # An unfitted detector is the restore that broke: _new_state() copies beta
    # only from a fit(), so the fresh episode's beta was None and the guard
    # dropped the readout the episode had already solved.
    reborn = _fast()
    assert reborn._fitted_beta is None, "this must be the unfitted path"
    reborn.load_state(dumped)
    st = reborn._episodes["ep"]
    assert st.beta is not None, "the solved readout was dropped on restore"
    assert np.allclose(det._episodes["ep"].beta, st.beta)


def test_restore_keeps_the_scoring_context():
    det, dumped = _warmed_unfitted()
    reborn = _fast()
    reborn.load_state(dumped)
    st = reborn._episodes["ep"]
    # _esn_anomaly scores from the pre-advance context; without it the first
    # post-restore step returns 0.0 and the episode re-seeds context_prev,
    # costing one scored step per episode per restart.
    assert st.context_prev is not None, "context_prev was dropped on restore"
    assert np.allclose(det._episodes["ep"].context_prev, st.context_prev)


def test_restore_does_not_re_seed_residual_statistics_from_one_sample():
    # With the readout dropped, warm_n was restored past warmup_steps while
    # warm_ctx/warm_tgt were still empty (they are cleared when a warm-up
    # completes), so the next observe re-solved from a single new pair and
    # _seed_residual_stats re-seeded the band from that one residual --
    # collapsing res_n to 1 and discarding the accumulated statistics.
    # The live control is fed the same two steps the restored detector sees, so
    # the comparison is against a legitimately-extended running count.
    det, dumped = _warmed_unfitted()
    reborn = _fast()
    reborn.load_state(dumped)
    live_two = _healthy(2, start=8)
    for event in live_two:
        det.observe(event)
    for event in live_two:
        reborn.observe(event)
    before = det._episodes["ep"]
    after = reborn._episodes["ep"]
    assert after.res_n == before.res_n, (
        f"res_n diverged {before.res_n} -> {after.res_n}; the restored "
        "residual statistics were re-seeded from a single sample"
    )
    assert np.allclose(before.res_mu, after.res_mu)
    assert np.allclose(before.res_m2, after.res_m2)


def test_restored_episode_scores_immediately_without_re_warming():
    # The readout and context surviving means the episode keeps scoring from
    # step one after a restart, instead of going silent while it re-warms.
    # A restored detector must score the *same* anomaly a live one does on the
    # same step; pre-fix it returned 0.0 for the first post-restore step.
    det, dumped = _warmed_unfitted()
    reborn = _fast()
    reborn.load_state(dumped)
    live = det._episodes["ep"]
    restored = reborn._episodes["ep"]
    assert np.allclose(live.state, restored.state)
    probe = _ev(8, signature="a0", latency=9000.0)
    live_score = det._esn_anomaly(live, det._features(probe))
    restored_score = reborn._esn_anomaly(restored, reborn._features(probe))
    assert restored_score == live_score, (
        f"restored scored {restored_score} where live scored {live_score}"
    )
    assert live_score > 0.0, "the probe must be an anomaly for the control"
    assert restored_score > 0.0, (
        "the restored episode went silent on the first post-restore step"
    )


def test_a_corrupt_warmup_count_does_not_recalibrate_the_band():
    # warm_n outrunning the carried buffers is unreachable from dump_state, but
    # a foreign payload must re-warm honestly rather than solving from the
    # restored gram/rhs and re-seeding the band from one sample.
    _, dumped = _warmed_unfitted()
    ep = dumped["episodes"][0][1]
    # Claim a completed warm-up while the readout and buffers are absent.
    ep["beta"] = None
    ep["warm_n"] = 50
    ep["warm_ctx"] = []
    ep["warm_tgt"] = []
    reborn = _fast()
    reborn.load_state(dumped)
    st = reborn._episodes["ep"]
    assert st.beta is None
    assert st.warm_n == 0, "the count must clamp to the pairs actually carried"


def test_monitor_restore_keeps_a_warm_up_readout_on_the_default_path():
    # The end-to-end case: Monitor.default builds the ESN unfitted, so the
    # episode's warm-up readout is what a restart must survive. The default
    # warmup_steps is 20, so the stream must run past that before the snapshot.
    cfg = Config(ml_ensemble_enabled=True)
    m = Monitor.default(config=cfg, sinks=[_Collector()])
    for event in _healthy(30):
        m.ingest(event)
    esn = _esn_of(m)
    assert esn._fitted_beta is None, "the default path must be unfitted"
    assert esn._episodes["ep"].beta is not None
    snap = m.snapshot_dict()

    fresh = Monitor.default(config=cfg, sinks=[_Collector()])
    fresh.restore_dict(snap)
    esn2 = _esn_of(fresh)
    assert esn2._episodes["ep"].beta is not None, (
        "the warm-up readout was lost across a Monitor snapshot/restore"
    )
    assert np.allclose(esn._episodes["ep"].beta, esn2._episodes["ep"].beta)
