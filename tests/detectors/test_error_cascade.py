"""Tests for the error-cascade detector (project.md §5.2)."""

from __future__ import annotations

from snagline.detectors.error_cascade import ErrorCascadeDetector
from snagline.events import StepEvent


def _event(step_id: int, error: bool, episode: str = "ep") -> StepEvent:
    return StepEvent(
        step_id=str(step_id),
        episode_id=episode,
        timestamp=float(step_id),
        action_type="tool_call",
        action_signature=f"s{step_id}",
        error=error,
    )


def test_consecutive_cascade_detected():
    d = ErrorCascadeDetector(consecutive_threshold=3)
    risks = []
    for i, err in enumerate([False, True, True, True]):
        r = d.observe(_event(i, err))
        if r is not None:
            risks.append(r)
    assert risks, "consecutive cascade not detected"
    assert risks[-1].trigger == "error_cascade"
    assert risks[-1].detail.startswith("3 consecutive")


def test_windowed_cascade_detected():
    d = ErrorCascadeDetector(
        window_size=10, error_threshold=3, consecutive_threshold=99
    )
    risks = []
    errors = [True, False, True, False, True] + [False] * 5
    for i, err in enumerate(errors):
        r = d.observe(_event(i, err))
        if r is not None:
            risks.append(r)
    assert risks, "windowed cascade not detected"
    assert risks[-1].trigger == "error_cascade"


def test_no_false_positive_healthy():
    d = ErrorCascadeDetector()
    # all clean
    for i in range(20):
        assert d.observe(_event(i, False)) is None
    # a single isolated error must not trip either rule
    d2 = ErrorCascadeDetector()
    for i in range(20):
        r = d2.observe(_event(i, i == 5))
        assert r is None, f"false positive at step {i}"


def test_reset_clears_state():
    d = ErrorCascadeDetector(consecutive_threshold=3)
    d.observe(_event(0, True))
    d.observe(_event(1, True))
    d.reset("ep")
    assert d.observe(_event(2, True)) is None
