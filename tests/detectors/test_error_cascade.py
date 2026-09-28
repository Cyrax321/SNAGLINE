"""Tests for the error-cascade detector (project.md §5.2)."""

from __future__ import annotations

from snagline.detectors.error_cascade import ErrorCascadeDetector
from snagline.events import StepEvent
from snagline.risk import severity_from_score


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


def test_second_cascade_after_recovery_escalates_again():
    # The dedupe flag must re-arm once the alarm clears, so a second,
    # independent cascade in the same episode still alerts. A long-lived
    # episode (a user session, or a sidecar episode that never calls
    # end_episode) would otherwise be muted after its first cascade.
    d = ErrorCascadeDetector(window_size=10, error_threshold=3, consecutive_threshold=3)
    step = 0

    def run(errors: list[bool]) -> list:
        nonlocal step
        out = []
        for err in errors:
            r = d.observe(_event(step, err))
            step += 1
            if r is not None:
                out.append(r)
        return out

    first = run([True] * 3)
    assert len(first) == 1, "first cascade must escalate exactly once"

    # Full recovery: enough clean steps to flush every error out of the window.
    assert run([False] * 15) == [], "healthy recovery must stay silent"

    second = run([True] * 3)
    assert len(second) == 1, "second cascade after recovery must escalate again"
    assert second[0].trigger == "error_cascade"


def test_sustained_cascade_still_escalates_only_once():
    # Issue #4 still holds, in its intended form: a sustained cascade must not
    # re-fire on *every* step. It alerts once per severity band instead, so a
    # cascade that deepens past the band it already reported escalates rather
    # than being silenced forever (issue #538). One alert per band is bounded,
    # which is what the issue was actually guarding against.
    d = ErrorCascadeDetector(window_size=10, error_threshold=3, consecutive_threshold=3)
    risks = [d.observe(_event(i, True)) for i in range(30)]
    scores = [r.score for r in risks if r is not None]
    assert scores == [0.5, 0.8, 1.0], scores


def _first_score(n_errors: int, episode: str) -> float:
    """Score of the first alarm after exactly ``n_errors`` consecutive errors.

    A fresh episode per call keeps the cascades independent -- the dedupe flag
    would otherwise suppress every fire after the first.
    """
    d = ErrorCascadeDetector(
        window_size=10, error_threshold=99, consecutive_threshold=3
    )
    last = None
    for i in range(n_errors):
        r = d.observe(_event(i, True, episode=episode))
        if r is not None:
            last = r
    assert last is not None, f"no alarm after {n_errors} consecutive errors"
    return last.score


def test_consecutive_first_crossing_is_graded_not_flat():
    # Issue #538: the alarm fires *at* the threshold, so dividing by the
    # threshold alone always yields >= 1 and every alert was emitted at 1.0 /
    # critical. The first crossing must land below the critical band (0.8)
    # instead, like every other count-based detector.
    score = _first_score(3, "cross")
    assert score < 0.8, f"first crossing must not be critical, got {score}"
    assert score == 0.5
    # Severity is derived from the score, so the band must follow it.
    assert severity_from_score(score) == "warning"
    # And it must not regress to the flat 1.0 the graded formula replaced.
    assert score != 1.0


def test_score_escalates_as_a_live_cascade_deepens():
    # The grading must respond to how far a *live* cascade has run, not just to
    # the crossing -- and the live path is the only one that matters. A plain
    # boolean dedupe flag pins the score at the first crossing for the rest of
    # the episode, so a genuine outage could never reach ``min_severity_for_halt``
    # and ``policy="halt_webhook"`` would never fire. Dedupe therefore tracks
    # the band, so a cascade that deepens past it alerts again (issue #538).
    d = ErrorCascadeDetector(window_size=100, error_threshold=99, consecutive_threshold=3)
    fires = [r for i in range(40) if (r := d.observe(_event(i, True))) is not None]
    scores = [r.score for r in fires]
    assert scores == [0.5, 0.8, 1.0], scores
    assert severity_from_score(scores[0]) == "warning"
    assert severity_from_score(scores[-1]) == "critical"
    # A real outage must be able to halt.
    assert any(s >= 0.8 for s in scores), "a deep cascade must reach the halt band"


def test_sustained_cascade_at_one_band_still_alerts_once():
    # Issue #4 must survive the band-aware dedupe: a cascade that stays inside
    # the band it already alerted on is a repeat, not new information, so it
    # must not re-fire on every step.
    d = ErrorCascadeDetector(window_size=100, error_threshold=99, consecutive_threshold=3)
    risks = [d.observe(_event(i, True)) for i in range(3)]
    assert [r.score for r in risks if r is not None] == [0.5]


def test_cleared_cascade_rearms_the_band():
    # When the cascade clears, the band must reset so an independent later
    # cascade alerts from the bottom again (mirrors ``LoopDetector``).
    d = ErrorCascadeDetector(window_size=10, error_threshold=99, consecutive_threshold=3)
    d.observe(_event(0, True))
    d.observe(_event(1, True))
    first = d.observe(_event(2, True))
    assert first is not None and first.score == 0.5
    d.observe(_event(3, False))  # clears the cascade
    d.observe(_event(4, True))
    d.observe(_event(5, True))
    second = d.observe(_event(6, True))
    assert second is not None, "a cleared cascade must re-arm"
    assert second.score == 0.5


def test_windowed_first_crossing_is_graded_not_flat():
    # Same property for the windowed/density rule.
    d = ErrorCascadeDetector(
        window_size=10, error_threshold=3, consecutive_threshold=99
    )
    errors = [True, False, True, False, True] + [False] * 5
    risks = []
    for i, err in enumerate(errors):
        r = d.observe(_event(i, err))
        if r is not None:
            risks.append(r)
    assert len(risks) == 1
    assert risks[0].score < 0.8, f"windowed first crossing must not be critical: {risks[0].score}"
    assert risks[0].score == 0.5
    assert risks[0].detail.startswith("3 errors in last")
