"""Tests for the latency / CUSUM anomaly detector (project.md §5.3)."""

from __future__ import annotations

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
