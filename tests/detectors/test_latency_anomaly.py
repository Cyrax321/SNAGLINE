"""Tests for the latency / CUSUM anomaly detector (project.md §5.3)."""

from __future__ import annotations

from snagline.baseline import BaselineProfile, ToolBaseline
from snagline.detectors.latency_anomaly import LatencyAnomalyDetector
from snagline.events import StepEvent, make_signature


def _sig(i: int) -> str:
    return make_signature("tool_call", "search", str(i))


def _event(step_id: int, latency: float, episode: str = "ep") -> StepEvent:
    return StepEvent(
        step_id=str(step_id),
        episode_id=episode,
        timestamp=float(step_id),
        action_type="tool_call",
        action_signature=_sig(step_id),
        tool_name="search",
        latency_ms=latency,
    )


def test_latency_spike_detected():
    # Healthy baseline is stable latency; a sustained shift to 400ms is a real
    # anomaly. With a long constant baseline, a single spike makes
    # (x - mean)/std large enough to cross h on the first spike step.
    d = LatencyAnomalyDetector(min_samples=15)
    risks = []
    for i in range(40):
        r = d.observe(_event(i, 100.0))
        if r is not None:
            risks.append(r)
    for i in range(40, 48):
        r = d.observe(_event(i, 400.0))
        if r is not None:
            risks.append(r)
    assert risks, "expected a latency anomaly risk"
    assert all(r.trigger == "latency_anomaly" for r in risks)
    assert risks[0].score >= 0.6


def test_no_false_positive_healthy():
    # A healthy run has stable latency (matching the project's healthy fixture),
    # so the std==0 guard prevents any alarm.
    d = LatencyAnomalyDetector(min_samples=20)
    for i in range(60):
        r = d.observe(_event(i, 100.0))
        assert r is None, f"false positive at step {i}: {r}"


def test_warmup_suppresses_early_noise():
    d = LatencyAnomalyDetector(min_samples=10)
    # a single early blip during warm-up must not alarm
    for i in range(9):
        assert d.observe(_event(i, 100.0)) is None
    assert d.observe(_event(9, 100.0)) is None


def test_reset_clears_state():
    d = LatencyAnomalyDetector(min_samples=5)
    for i in range(6):
        d.observe(_event(i, 100.0))
    d.reset("ep")
    # after reset, a single spike should not immediately alarm (no baseline yet)
    assert d.observe(_event(6, 400.0)) is None


def test_single_spike_detected_after_warmup():
    # Regression test for the CUSUM rewrite: a single large deviation from a
    # stable baseline must alarm immediately. The old implementation diluted
    # the deviation by including the anomaly in the running std and could only
    # fire after several sustained spikes (and never on a constant baseline).
    d = LatencyAnomalyDetector(min_samples=15)
    for i in range(15):
        assert d.observe(_event(i, 80.0)) is None
    r = d.observe(_event(15, 400.0))  # lone 5x spike
    assert r is not None, "single large spike should alarm"
    assert r.trigger == "latency_anomaly"
    assert r.score >= 0.6


def test_small_variation_does_not_false_positive():
    # Benign, symmetric jitter around a stable baseline must not alarm.
    d = LatencyAnomalyDetector(min_samples=15)
    risks = []
    for i in range(40):
        if i < 15:
            latency = 100.0
        else:
            # +/-5ms around the 100ms baseline (mean stays ~100ms, no real shift)
            latency = 100.0 + ((i % 3) - 1) * 5.0
        r = d.observe(_event(i, latency))
        if r is not None:
            risks.append(r)
    assert not risks, f"false positive on benign jitter: {risks}"


def test_sustained_shift_keeps_alarming():
    # A permanent regression must keep the CUSUM elevated (not be learned away
    # and forgotten). The old implementation reset to baseline and went quiet.
    d = LatencyAnomalyDetector(min_samples=15)
    for i in range(15):
        d.observe(_event(i, 100.0))
    alarmed = []
    for i in range(15, 25):
        r = d.observe(_event(i, 300.0))
        alarmed.append(r is not None)
    assert any(alarmed), "sustained shift should keep alerting"
    # it should not drop back to silent mid-shift
    assert alarmed[-1], "alert stopped during a still-elevated shift"


# --- calibrated start must not seed from a latency-less profile (#348) --------


def _latencyless_profile(tool: str = "search", n: int = 100) -> BaselineProfile:
    """A profile whose stream reported no timings: every count is an
    error-rate sample and none carries a ``latency_ms`` (the auto-calibration
    contract, issue #101, supports exactly this stream)."""
    profile = BaselineProfile()
    tb = ToolBaseline(tool_name=tool)
    for _ in range(n):
        tb.add(None, error=False)
    profile.tools[tool] = tb
    return profile


def test_latencyless_profile_does_not_alarm_on_first_step():
    # Seeding from ``count`` froze onto mean_latency 0.0. The reference spread
    # stays finite (sigma floors -- 1 ms absolute), but the first real call
    # then measures as a large multiple of that floor and crosses the CUSUM
    # threshold on the first step, paging critical on step 0 of every episode.
    d = LatencyAnomalyDetector(baseline=_latencyless_profile(), min_samples=5)
    assert d.observe(_event(0, 120.0)) is None


def test_latencyless_profile_falls_back_to_warmup():
    # Not seeding is not giving up: the detector learns a real baseline from
    # the live stream and still alarms on a genuine regression.
    d = LatencyAnomalyDetector(baseline=_latencyless_profile(), min_samples=5)
    risks = []
    for i in range(15):
        r = d.observe(_event(i, 100.0))
        if r is not None:
            risks.append(r)
    assert risks == [], "healthy warm-up must stay silent"
    for i in range(15, 23):
        r = d.observe(_event(i, 3000.0))
        if r is not None:
            risks.append(r)
    assert risks, "a real anomaly must still fire after a latency-less seed was refused"


def test_latency_bearing_profile_still_seeds_and_skips_warmup():
    # The gate moved to latency_count, not to "never seed": a tool whose
    # latency-bearing subset alone clears min_samples still starts frozen, so
    # a short episode stays monitorable from step 0.
    profile = BaselineProfile()
    tb = ToolBaseline(tool_name="search")
    for value in (400.0, 395.0, 405.0, 398.0, 402.0):
        tb.add(value, False)
    # Error-rate samples without timings push count past latency_count; the
    # latency gate must be the one that decides.
    for _ in range(20):
        tb.add(None, False)
    profile.tools["search"] = tb
    assert tb.count == 25 and tb.latency_count == 5

    d = LatencyAnomalyDetector(baseline=profile, min_samples=5)
    assert d.observe(_event(0, 395.0)) is None
    r = d.observe(_event(1, 2500.0))
    assert r is not None, "seeded baseline must alarm without warm-up"
    assert r.trigger == "latency_anomaly"


def test_seeding_gate_uses_latency_count_not_count():
    # Direct pin: latency_count below min_samples must refuse to seed even
    # when count clears it comfortably.
    d = LatencyAnomalyDetector(baseline=_latencyless_profile(), min_samples=5)
    d.observe(_event(0, 2500.0))
    state = d._states[("ep", "search")]
    # Refused the seed -> still in warm-up, learning from the live sample.
    assert state.frozen is False
    assert state.n == 1
