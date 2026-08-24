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


def _fires(d: LoopDetector, signatures: list[str]) -> list[int]:
    """Feed ``signatures`` as consecutive steps; return the steps that alerted."""
    return [i for i, s in enumerate(signatures) if d.observe(_event(i, s)) is not None]


def test_sustained_loop_alerts_once_even_when_interleaved():
    """Regression: the dedupe flag was cleared by any step whose *own* signature
    had not repeated, so a loop interleaved with distinct steps -- an agent
    retrying one failing tool between reasoning turns, the common shape -- re-fired
    on every single repetition. That is the alert spam the flag exists to prevent
    (issue #4).
    """
    # A straight run of one signature is the case that already worked.
    d = LoopDetector(window_size=10, repeat_threshold=3)
    assert _fires(d, [_sig(1)] * 20) == [2]

    # The same loop with a unique step between each repeat must also alert once.
    d = LoopDetector(window_size=10, repeat_threshold=3)
    interleaved = [_sig(1)] * 3
    for i in range(20):
        interleaved += [_sig(100 + i), _sig(1)]
    fires = _fires(d, interleaved)
    assert fires == [2], f"expected one alert for one loop, got {len(fires)}: {fires}"


def test_loop_realerts_only_after_it_clears():
    """A repetition that stops and later returns is a second finding, so it must
    escalate again -- the re-arm this detector always intended. What must *not*
    re-arm it is an unrelated step arriving mid-loop.
    """
    d = LoopDetector(window_size=6, repeat_threshold=3)
    # Loop, then flush it out of the window entirely, then loop again.
    signatures = [_sig(1)] * 3 + [_sig(200 + i) for i in range(6)] + [_sig(1)] * 3
    assert _fires(d, signatures) == [2, 11]


def test_two_distinct_loops_each_alert():
    """Two different actions looping in one episode are two findings. The flag is
    per-signature, so neither suppresses nor re-arms the other.
    """
    d = LoopDetector(window_size=12, repeat_threshold=3)
    assert _fires(d, [_sig(1)] * 3 + [_sig(2)] * 3) == [2, 5]
