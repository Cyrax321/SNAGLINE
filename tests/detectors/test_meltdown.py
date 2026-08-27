"""Tests for the meltdown entropy detector (issue #85)."""

from __future__ import annotations

import pytest

from snagline.detectors.meltdown import MeltdownDetector
from snagline.events import StepEvent


def _event(step_id: int, tool: str, action_type: str = "tool_call") -> StepEvent:
    # Distinct signatures prove identity comes from tool_name, not the hash,
    # so args-varying repetition is still caught.
    return StepEvent(
        step_id=str(step_id),
        episode_id="ep",
        timestamp=float(step_id),
        action_type=action_type,
        action_signature=f"sig-{step_id}",
        tool_name=tool,
    )


def _run(d: MeltdownDetector, events: list[StepEvent]) -> list:
    return [r for e in events if (r := d.observe(e)) is not None]


def test_low_entropy_collapse_fires_once():
    d = MeltdownDetector(window_size=8)
    events = [_event(i, "search") for i in range(20)]  # same tool, varying args
    risks = _run(d, events)
    assert len(risks) == 1, "sustained collapse must escalate exactly once"
    assert risks[0].trigger == "meltdown"
    assert risks[0].score == 0.7
    assert "collapsed" in risks[0].detail


def test_rearm_after_recovery_allows_second_collapse():
    d = MeltdownDetector(window_size=8, rearm_steps=4)
    step = 0
    step += len(_run(d, [_event(step, "search") for _ in range(8)]))  # fires
    # In-band mixed work (two tools -> H = 1.0 bit) for >= rearm_steps full windows.
    for _ in range(6):
        step += len(_run(d, [_event(step, "read"), _event(step + 1, "write")]))
        step += 1
    second = _run(d, [_event(900 + i, "search") for i in range(8)])
    assert len(second) == 1, "a second independent collapse must re-alert"


def test_healthy_alternation_stays_silent():
    d = MeltdownDetector(window_size=8)
    tools = ("search", "read", "write", "list", "get")  # H ~ 2.32 bits, in band
    risks = []
    for i in range(60):
        risks.extend(_run(d, [_event(i * 5 + j, tools[j]) for j in range(5)]))
    assert risks == [], f"healthy purposeful alternation false-positive: {risks}"


def test_high_entropy_churn_fires():
    d = MeltdownDetector(window_size=8, high_entropy=2.2)
    tools = [f"tool{i}" for i in range(12)]  # any window of 8: H = 3.0 bits
    risks = []
    for i in range(16):
        risks.extend(_run(d, [_event(i, tools[i % 12])]))
    assert len(risks) == 1
    assert risks[0].score == 0.6
    assert "spiked" in risks[0].detail


def test_non_tool_steps_do_not_feed_window():
    d = MeltdownDetector(window_size=4)
    for i in range(10):
        assert d.observe(_event(i, "ignored", action_type="message")) is None
    assert all(len(w.window) == 0 for w in d._eps.values()) or not d._eps
    # Window never fills on non-tool traffic alone.
    assert _run(d, [_event(i, "x", action_type="plan_step") for i in range(10)]) == []


def test_reset_clears_state():
    d = MeltdownDetector(window_size=4)
    _run(d, [_event(i, "search") for i in range(4)])  # collapses and fires
    d.reset("ep")
    # The prior verdict must be forgotten: a fresh rote window may alert again.
    second = _run(d, [_event(i, "other") for i in range(4)])
    assert len(second) == 1, "reset must clear both window and fired flag"


def test_healthy_eight_tool_round_robin_stays_silent():
    """Regression for #180: uniform 8-tool rotation must not fire at default."""
    d = MeltdownDetector()  # default high 3.4, low 0.4, window 20
    tools = [f"tool_{i}" for i in range(8)]
    risks = []
    for i in range(60):
        risks.extend(_run(d, [_event(i, tools[i % 8])]))
    assert risks == [], (
        f"healthy 8-tool agent false-positive: {risks[0].detail if risks else ''}"
    )


def test_twelve_tool_churn_still_fires():
    """12+ distinct in one window must still fire after retuning."""
    d = MeltdownDetector()
    tools = [f"tool_{i}" for i in range(12)]
    risks = []
    for i in range(24):
        risks.extend(_run(d, [_event(i, tools[i % 12])]))
    assert len(risks) == 1
    assert "spiked" in risks[0].detail
    assert risks[0].trigger == "meltdown"


def test_inverted_thresholds_rejected():
    with pytest.raises(ValueError):
        MeltdownDetector(low_entropy=2.0, high_entropy=1.0)


def test_state_round_trip():
    d1 = MeltdownDetector(window_size=4)
    _run(d1, [_event(i, "search") for i in range(3)])  # partial window
    d2 = MeltdownDetector(window_size=4)
    d2.load_state(d1.dump_state())
    rest = [_event(i, "search") for i in range(3, 6)]
    r1 = [(r.trigger, r.score) for r in _run(d1, rest)]
    r2 = [(r.trigger, r.score) for r in _run(d2, rest)]
    assert r1 == r2 and r1, "restored detector must behave identically"
