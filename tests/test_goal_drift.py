"""Tests for the goal-drift detector (next phase, step 2)."""

from __future__ import annotations

from snagline.baseline import BaselineProfile, ToolBaseline
from snagline.config import Config
from snagline.detectors.goal_drift import GoalDriftDetector
from snagline.events import StepEvent


def _ev(tool, latency, error=False, episode="ep"):
    return StepEvent(
        step_id="s",
        episode_id=episode,
        timestamp=1.0,
        action_type="tool_call",
        action_signature="sig",
        tool_name=tool,
        latency_ms=latency,
        error=error,
    )


def _healthy_baseline() -> BaselineProfile:
    # A healthy run: search is fast and error-free.
    prof = BaselineProfile()
    tb = ToolBaseline(tool_name="search")
    for _ in range(20):
        tb.add(100.0, error=False)
    prof.tools["search"] = tb
    return prof


def test_goal_drift_is_noop_without_baseline():
    det = GoalDriftDetector(config=Config())
    assert det.observe(_ev("search", 100.0)) is None


def test_goal_drift_flags_rising_error_rate():
    cfg = Config()
    cfg.goal_drift_min_samples = 5
    det = GoalDriftDetector(baseline=_healthy_baseline(), config=cfg)
    risks = [det.observe(_ev("search", 100.0, error=True)) for _ in range(6)]
    fired = [r for r in risks if r is not None]
    assert fired, "expected at least one goal_drift risk"
    assert fired[0].trigger == "goal_drift"
    assert fired[0].score >= cfg.goal_drift_score_threshold


def test_goal_drift_stays_silent_on_healthy_traffic():
    cfg = Config()
    cfg.goal_drift_min_samples = 5
    det = GoalDriftDetector(baseline=_healthy_baseline(), config=cfg)
    # Constant 100.0 latency matches the baseline exactly: no drift.
    risks = [det.observe(_ev("search", 100.0)) for _ in range(8)]
    assert all(r is None for r in risks)


def test_goal_drift_flags_unseen_tool():
    cfg = Config()
    cfg.goal_drift_min_samples = 5
    det = GoalDriftDetector(baseline=_healthy_baseline(), config=cfg)
    risks = [det.observe(_ev("mystery_tool", 50.0)) for _ in range(6)]
    assert any(r is not None for r in risks)


def test_goal_drift_dedupes_per_episode():
    cfg = Config()
    cfg.goal_drift_min_samples = 3
    det = GoalDriftDetector(baseline=_healthy_baseline(), config=cfg)
    fired = [det.observe(_ev("search", 100.0, error=True)) for _ in range(6)]
    assert sum(1 for r in fired if r is not None) == 1


def test_goal_drift_reset_clears_state():
    cfg = Config()
    cfg.goal_drift_min_samples = 3
    det = GoalDriftDetector(baseline=_healthy_baseline(), config=cfg)
    for _ in range(4):
        det.observe(_ev("search", 100.0, error=True))
    det.reset("ep")
    # After reset, accumulating fresh healthy traffic must not immediately alarm.
    risks = [det.observe(_ev("search", 100.0 + i)) for i in range(4)]
    assert all(r is None for r in risks)
