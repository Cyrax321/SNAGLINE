"""Tests for the loop detector (project.md §5.1)."""

from __future__ import annotations

from snagline.detectors.loop import LoopDetector
from snagline.events import StepEvent, make_signature


def _sig(i: int) -> str:
    return make_signature("tool_call", "t", f"a{i}")


def _event(step_id: int, sig: str, episode: str = "ep") -> StepEvent:
    return StepEvent(
        step_id=str(step_id),
        episode_id=episode,
        timestamp=float(step_id),
        action_type="tool_call",
        action_signature=sig,
    )


def test_loop_detected():
    d = LoopDetector(window_size=5, repeat_threshold=3)
    risks = []
    for i, s in enumerate([_sig(1), _sig(2), _sig(1), _sig(1), _sig(1)]):
        r = d.observe(_event(i, s))
        if r is not None:
            risks.append(r)
    assert risks, "expected at least one loop risk"
    assert all(r.trigger == "loop" for r in risks)
    assert risks[0].score >= 0.5


def test_no_false_positive_healthy():
    d = LoopDetector()
    for i in range(20):
        r = d.observe(_event(i, _sig(i)))  # every signature unique
        assert r is None, f"false positive at step {i}"


def test_reset_clears_state():
    d = LoopDetector(window_size=4, repeat_threshold=3)
    d.observe(_event(0, _sig(1)))
    d.observe(_event(1, _sig(1)))
    d.reset("ep")
    assert d.observe(_event(2, _sig(1))) is None  # must rebuild from scratch
