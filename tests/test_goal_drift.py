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


def test_goal_drift_rearms_after_recovery():
    """Issue #247: _fired latched once per episode and never re-armed, so a
    long-lived episode that recovered from one drift and later hit a second,
    independent (worse) one stayed silent forever. The latch now clears when
    the drift score drops back below the threshold, mirroring the re-arm
    semantics of every other shipped detector."""
    cfg = Config()
    cfg.goal_drift_min_samples = 3
    det = GoalDriftDetector(baseline=_healthy_baseline(), config=cfg)

    # Phase 1: a drift (errors) fires exactly once.
    phase1 = [det.observe(_ev("search", 100.0, error=True)) for _ in range(6)]
    assert sum(1 for r in phase1 if r is not None) == 1

    # Phase 2: healthy traffic brings the live profile back in line. Enough
    # steps to dilute the phase-1 errors below the error tolerance and push
    # the live mean back toward the baseline.
    for _ in range(40):
        det.observe(_ev("search", 100.0))
    assert det._fired.get("ep") is not True, "recovery must re-arm the latch"

    # Phase 3: a second, independent drift must alert again.
    phase3 = [det.observe(_ev("search", 20000.0, error=True)) for _ in range(6)]
    fired3 = [r for r in phase3 if r is not None]
    assert fired3, "a second independent drift in the same episode must fire"
    assert fired3[0].trigger == "goal_drift"


def test_goal_drift_still_dedupes_while_drift_persists():
    """Re-arm must not turn into alert spam: while the drift score stays above
    the threshold without an intervening recovery, exactly one risk fires."""
    cfg = Config()
    cfg.goal_drift_min_samples = 3
    det = GoalDriftDetector(baseline=_healthy_baseline(), config=cfg)
    risks = [det.observe(_ev("search", 5000.0, error=True)) for _ in range(10)]
    assert sum(1 for r in risks if r is not None) == 1
