"""Tests for the token-runaway detector (issue #84)."""

from __future__ import annotations

from typing import Any

import pytest

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


def test_zero_budget_does_not_page_at_critical_on_the_first_step():
    """Issue #317 end-to-end: ``SNAGLINE_EPISODE_TOKEN_BUDGET=0`` used to reach
    the detector and emit a score-1.0 budget_breach on the *first* token-bearing
    step (``total >= budget`` was ``10 >= 0``). The bad value is now refused at
    configuration time, so a monitor built from it never exists and no critical
    risk can be dispatched."""
    from snagline import Monitor
    from snagline.config import Config
    from snagline.sinks.base import AlertSink

    class Collect(AlertSink):
        def __init__(self) -> None:
            self.risks: list = []

        def emit(self, risk) -> None:
            self.risks.append(risk)

    sink = Collect()
    with pytest.raises(ValueError, match="episode_token_budget"):
        Monitor.default(
            config=Config(token_runaway_enabled=True, episode_token_budget=0),
            sinks=[sink],
        )
    assert sink.risks == [], "no risk may be dispatched from a rejected config"


def test_nonpositive_budget_is_rejected():
    """Issue #317: direct construction with a non-positive budget is a
    configuration error, mirroring the StagnationDetector precedent
    (issue #132): direct kwargs skip the Config check, so the detector guards
    itself."""
    for budget in (0, -1, -1000):
        with pytest.raises(ValueError, match="budget_total_tokens"):
            TokenRunawayDetector(budget_total_tokens=budget)


def test_warn_fraction_out_of_range_is_rejected():
    """Issue #317: ``warn_fraction <= 0`` put the warning threshold at or below
    zero, so it fired on the first step; ``> 1.0`` put it above the budget, so
    the breach silenced it and the warning could never fire. Both are
    configuration errors."""
    for fraction in (0.0, -0.5, 1.5):
        with pytest.raises(ValueError, match="warn_fraction"):
            TokenRunawayDetector(budget_total_tokens=1000, warn_fraction=fraction)


def test_none_budget_still_disables_the_envelope():
    """Issue #317 regression guard: the new range check must not tighten the
    documented ``None disables envelope`` contract. The CUSUM path keeps its
    own state either way; only the envelope bookkeeping is budget-gated."""
    d = TokenRunawayDetector(budget_total_tokens=None, min_samples=5)
    _run(d, [_event(i, 100) for i in range(10)])
    assert d._totals == {}, "no budget means the envelope tracks nothing"


# --- load_state is atomic: a rejected snapshot leaves the detector untouched --
# restore_dict catches the exception and moves on (issue #384), so a snapshot
# applied attribute-by-attribute would silently pair restored episodes with the
# live totals/warned/breached that none of them describe, after the live state
# was already discarded (issue #417).


def test_load_state_is_atomic_when_an_entry_is_malformed():
    d = TokenRunawayDetector(min_samples=3, budget_total_tokens=2000)
    _run(d, [_event(i, 500) for i in range(4)])  # live "ep": 4*500 == budget
    assert "ep" in d._states and d._breached.get("ep"), "there must be state to lose"
    live_totals = dict(d._totals)
    live_breached = dict(d._breached)

    # A snapshot that mentions a *different*, malformed episode. Pre-fix
    # load_state cleared _states before repopulating, so the live episode is
    # discarded before the bad entry ever raises -- and _warned/_breached are
    # left describing episodes that no longer exist.
    bad = {
        "states": {"ep-bad": {"n": "not-an-int"}},
        "totals": {},
        "warned": {},
        "breached": {},
    }
    with pytest.raises(Exception):
        d.load_state(bad)

    assert "ep" in d._states, "a rejected snapshot must not discard live episodes"
    assert d._totals == live_totals, "totals must not be replaced by the snapshot"
    assert d._breached == live_breached


def test_load_state_applies_when_the_snapshot_is_good():
    d1 = TokenRunawayDetector(min_samples=3, budget_total_tokens=2000)
    _run(d1, [_event(i, 500) for i in range(4)])
    d2 = TokenRunawayDetector(min_samples=3, budget_total_tokens=2000)
    d2.load_state(d1.dump_state())
    assert set(d2._states) == set(d1._states)
    assert d2._warned == d1._warned


def test_load_state_rejects_a_non_numeric_counter():
    # ``dump_state`` copies the seven Welford/CUSUM counters out as-is, and a
    # snapshot is only JSON: a hand edit, a torn write, or a version skew can
    # hand back an entry where every key is present but a value is not a
    # number. That entry is structurally complete, so it cleared the guards
    # above and was published -- and the failure then moved out of restore and
    # into the next event, where ``learn_only``'s ``self.n += 1`` is
    # ``'not-a-number' + 1`` (issue #424).
    d = TokenRunawayDetector(min_samples=3, budget_total_tokens=2000)
    _run(d, [_event(i, 500) for i in range(4)])
    live = d.dump_state()

    bad = {
        "states": {
            "ep": {
                "n": "not-a-number",
                "mean": 5.0,
                "m2": 4.0,
                "cusum": 0.0,
                "mu0": 5.0,
                "sigma0": 2.0,
                "frozen": False,
            }
        },
        "totals": {},
        "warned": {},
        "breached": {},
    }
    with pytest.raises(Exception):
        d.load_state(bad)

    # Rejected at restore, so the live state survives (issue #417) rather than
    # being replaced by the poisoned one.
    assert d.dump_state() == live

    # ...and the episode is still scorable, not wedged. This is the observable
    # difference between "snapshot rejected" and "snapshot accepted and
    # broken": the old behaviour raised a TypeError here and on every event
    # after it, with the count left as a string.
    d.observe(_event(4, 500))
    assert isinstance(d._states["ep"].n, int)


def test_load_state_accepts_a_mid_warmup_snapshot():
    # ``mu0`` is the one field that may legitimately be absent: a state
    # captured before ``freeze`` has no baseline yet. Validating it with a bare
    # ``float()`` would reject exactly the snapshots ``dump_state`` writes, so
    # ``None`` has to round-trip (issue #424).
    d1 = TokenRunawayDetector(min_samples=5, budget_total_tokens=2_000_000)
    _run(d1, [_event(i, 500) for i in range(2)])  # under min_samples: unfrozen
    assert not d1._states["ep"].frozen, "this test needs a warm-up state"
    assert d1._states["ep"].mu0 is None

    d2 = TokenRunawayDetector(min_samples=5, budget_total_tokens=2_000_000)
    d2.load_state(d1.dump_state())
    assert d2._states["ep"].mu0 is None
    # The detector still warms up from the restored position rather than
    # starting over.
    _run(d2, [_event(i, 500) for i in range(2, 5)])
    assert d2._states["ep"].frozen
