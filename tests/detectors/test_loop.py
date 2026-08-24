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


def test_fired_bookkeeping_is_released_when_the_loop_clears():
    """The per-signature fired set is pruned once it empties. An empty set left
    behind kept one dict entry for every episode that ever looped, alive until
    ``reset()`` -- the same unbounded per-episode growth this branch removes from
    the dedup path, reappearing in the detector.
    """
    d = LoopDetector(window_size=6, repeat_threshold=3)
    for i, s in enumerate([_sig(1)] * 3):
        d.observe(_event(i, s))
    assert d._fired["ep"] == {_sig(1)}, "the loop should have escalated"

    # Push the loop out of the window entirely: nothing repeats any more.
    for i in range(6):
        d.observe(_event(10 + i, _sig(200 + i)))
    assert "ep" not in d._fired, "empty bookkeeping outlived the loop"


def test_pruning_the_fired_set_does_not_break_realerting():
    """Guard on the fix above: the pruned entry has to be recreated through the
    dict, not mutated on the detached set.

    The prune and an escalation can land on the *same* ``observe()`` call -- the
    window slides, the old loop's signature drops below the threshold, and a new
    one reaches it -- and only then is the distinction observable. Recording
    that fire against the orphaned set loses it, so the next repeat escalates
    again: exactly the spam this bookkeeping exists to prevent.
    """
    # window 5 / threshold 3: the slide from [A,A,A,B,B] to [A,A,B,B,B] drops A
    # from 3 to 2 (prune) and lifts B from 2 to 3 (escalate) in one step.
    d = LoopDetector(window_size=5, repeat_threshold=3)
    fires = _fires(d, [_sig(1)] * 3 + [_sig(2)] * 4)
    assert fires == [2, 5], f"expected one alert per loop, got {fires}"
    assert d._fired["ep"] == {_sig(2)}, "the colliding fire was not recorded"
