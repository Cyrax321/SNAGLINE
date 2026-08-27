"""Tests for the token-runaway detector (issue #84)."""

from __future__ import annotations

from typing import Any

from snagline.detectors.token_runaway import TokenRunawayDetector
from snagline.events import StepEvent


def _event(
    step_id: int,
    tokens: int | None,
    episode: str = "ep",
    **kwargs: Any,
) -> StepEvent:
    return StepEvent(
        step_id=str(step_id),
        episode_id=episode,
        timestamp=float(step_id),
        action_type="tool_call",
        action_signature=f"s{step_id}",
        tokens_in=tokens,
        error=False,
        **kwargs,
    )


def _run(d: TokenRunawayDetector, events: list[StepEvent]) -> list:
    return [r for e in events if (r := d.observe(e)) is not None]


def test_sustained_burn_fires_after_warmup():
    d = TokenRunawayDetector(min_samples=10)
    warm = [_event(i, 100) for i in range(10)]
    assert _run(d, warm) == [], "warm-up must stay silent"
    hot = [_event(i, 400) for i in range(10, 15)]
    risks = _run(d, hot)
    assert risks, "sustained 4x burn must fire"
    assert risks[0].trigger == "token_runaway"


def test_stable_high_volume_no_false_positive():
    d = TokenRunawayDetector(min_samples=5)
    risks = _run(d, [_event(i, 5000) for i in range(40)])
    assert risks == [], f"stable volume false-positive: {risks}"


def test_envelope_warns_once_then_breaches_once():
    d = TokenRunawayDetector(budget_total_tokens=1000, warn_fraction=0.8)
    risks = []
    step = 0
    for expected_total in (300, 600, 900, 1200, 1500):
        risks.extend(_run(d, [_event(step, 300)]))
        step += 1
    triggers = [(r.trigger, r.score) for r in risks]
    # Step 3 (total 900 >= 80% of 1000): one warning. Step 4 (total 1200):
    # one breach. Step 5: silence -- envelope emits at most once per threshold.
    assert ("token_runaway", 0.8) in triggers
    assert ("budget_breach", 1.0) in triggers
    assert triggers.count(("budget_breach", 1.0)) == 1
    assert triggers.index(("budget_breach", 1.0)) > triggers.index(
        ("token_runaway", 0.8)
    )


def test_events_without_tokens_are_ignored():
    d = TokenRunawayDetector(budget_total_tokens=100)
    assert d.observe(_event(0, None)) is None
    assert d._totals == {}, "no-token events must not accumulate"


def test_reset_clears_envelope_and_cusum():
    d = TokenRunawayDetector(min_samples=2, budget_total_tokens=400)
    _run(d, [_event(0, 150), _event(1, 150), _event(2, 150)])  # crosses 80% (360)
    d.reset("ep")
    assert d._totals == {}
    risks = _run(d, [_event(3, 350)])
    assert [r.trigger for r in risks] == ["token_runaway"], (
        "after reset the warning must be able to fire again"
    )


def test_state_round_trip_preserves_behavior():
    d1 = TokenRunawayDetector(min_samples=3, budget_total_tokens=2000)
    _run(d1, [_event(i, 500) for i in range(4)])  # partial progress, warned at 2000*0.8
    d2 = TokenRunawayDetector(min_samples=3, budget_total_tokens=2000)
    d2.load_state(d1.dump_state())
    rest = [_event(i, 500) for i in range(4, 6)]  # 2000 -> 2500: breach
    assert [(r.trigger, r.step_id) for r in _run(d1, rest)] == [
        (r.trigger, r.step_id) for r in _run(d2, rest)
    ], "restored detector must behave identically"
