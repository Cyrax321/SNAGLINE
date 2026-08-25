"""Tests for the loop hardening modes (issue #89): near-duplicate, cycle, stall.

Per the COMMON rules every mode gets both sides: an injected failure sequence
that fires (exact trigger name) and a healthy sequence that stays silent.
Modes-off regression proves the plain path is untouched.
"""

from __future__ import annotations

from snagline.config import Config
from snagline.detectors.loop import LoopDetector
from snagline.events import StepEvent


def _event(
    step_id: int,
    sig: str,
    episode: str = "ep",
    timestamp: float | None = None,
) -> StepEvent:
    return StepEvent(
        step_id=str(step_id),
        episode_id=episode,
        timestamp=float(step_id) if timestamp is None else float(timestamp),
        action_type="tool_call",
        action_signature=sig,
    )


def _feed(d: LoopDetector, sigs: list[str], timestamps: list[float] | None = None):
    """Feed consecutive steps; return [(step_index, risk)] for every alert."""
    out = []
    for i, s in enumerate(sigs):
        ts = None if timestamps is None else timestamps[i]
        r = d.observe(_event(i, s, timestamp=ts))
        if r is not None:
            out.append((i, r))
    return out


def _cycle_config(window: int = 6) -> Config:
    return Config(loop_cycle_enabled=True, loop_cycle_window_size=window)


# --- cycle mode --------------------------------------------------------------


def test_cycle_fires_on_exact_abc_period_three():
    """ABCABC with window 6: minimal period 3, two full periods in window."""
    d = LoopDetector(config=_cycle_config(6))
    a, b, c = "sig-a", "sig-b", "sig-c"
    fires = _feed(d, [a, b, c, a, b, c])
    assert len(fires) == 1, f"expected exactly one cycle alert, got {fires}"
    step, risk = fires[0]
    assert step == 5, f"window fills at step 5, fired at {step}"
    assert risk.trigger == "cycle"
    assert "period-3" in risk.detail
    assert risk.episode_id == "ep"


def test_cycle_abab_scan_finds_period_two():
    """ABAB with window 4: the candidate scan must report period 2, not 4."""
    d = LoopDetector(config=_cycle_config(4))
    fires = _feed(d, ["s1", "s2", "s1", "s2"])
    assert len(fires) == 1
    step, risk = fires[0]
    assert step == 3
    assert risk.trigger == "cycle"
    assert "period-2" in risk.detail


def test_cycle_healthy_varied_sequence_stays_silent():
    d = LoopDetector(config=_cycle_config(6))
    sigs = [f"unique-{i}" for i in range(24)]
    assert _feed(d, sigs) == [], "varied traffic must not trip the cycle scan"


def test_cycle_band_suppresses_faster_true_period():
    """A custom band filters on the window's TRUE minimal period: with
    min=3 a period-2 loop must stay silent, not re-fire as a multiple."""
    cfg = Config(
        loop_cycle_enabled=True,
        loop_cycle_window_size=6,
        loop_cycle_min_period=3,
        loop_cycle_max_period=4,
    )
    d = LoopDetector(config=cfg)
    out = [d.observe(_event(i, s)) for i, s in enumerate(["x", "y"] * 3)]
    cycles = [r.trigger for r in out if r is not None]
    assert "cycle" not in cycles, (
        f"true period 2 below configured min must not fire as cycle: {cycles}"
    )
    # Documented coexistence: the shared plain path still reports the raw
    # repetition; only the cycle trigger must stay silent.


def test_cycle_band_accepts_in_band_true_period():
    cfg = Config(
        loop_cycle_enabled=True,
        loop_cycle_window_size=6,
        loop_cycle_min_period=3,
        loop_cycle_max_period=4,
    )
    d = LoopDetector(config=cfg)
    fires = _feed(d, ["m", "n", "o", "m", "n", "o"])
    assert len(fires) == 1 and fires[0][1].trigger == "cycle"
    assert "period-3" in fires[0][1].detail


def test_uniform_repetition_is_not_a_cycle():
    """One signature repeated is loop/stall territory, never trigger 'cycle'."""
    d = LoopDetector(config=_cycle_config(6))
    fires = _feed(d, ["same"] * 8)
    triggers = {r.trigger for _, r in fires}
    assert "cycle" not in triggers, f"uniform repetition fired as cycle: {fires}"
    # The plain loop still catches it; that overlap is by design.
    assert "loop" in triggers


def test_cycle_rearms_after_pattern_breaks():
    """A second distinct cycle later in the episode escalates again."""
    d = LoopDetector(config=_cycle_config(6))
    first = ["p", "q", "r", "p", "q", "r"]
    filler = [f"noise-{i}" for i in range(6)]
    # Second pattern uses three distinct signatures so the plain loop (which
    # shares this detector) stays silent and cannot mask the cycle trigger.
    second = ["x", "y", "z", "x", "y", "z"]
    fires = _feed(d, first + filler + second)
    cycles = [(i, r) for i, r in fires if r.trigger == "cycle"]
    assert [i for i, _ in cycles] == [5, 17]


def test_reset_clears_cycle_state():
    d = LoopDetector(config=_cycle_config(6))
    _feed(d, ["p", "q", "r", "p"])
    d.reset("ep")
    # After reset the old pattern must be forgotten: three steps alone cannot
    # satisfy the two-full-periods requirement.
    assert _feed(d, ["q", "r", "p"]) == []


# --- near-duplicate mode ------------------------------------------------------

_UUID_A = "550e8400-e29b-41d4-a716-446655440000"
_UUID_B = "6ba7b810-9dad-11d1-80b4-00c04fd430c8"
_UUID_C = "f47ac10b-58cc-4372-a567-0e02b2c3d479"


def test_near_duplicate_uuid_variants_collapse_and_fire():
    """Same action, different uuid suffixes: raw signatures all differ (so the
    plain loop must stay silent) but normalization collapses them."""
    d = LoopDetector(config=Config(loop_near_duplicate_enabled=True))
    sigs = [
        f"get_user:id={_UUID_A}",
        f"get_user:id={_UUID_B}",
        f"get_user:id={_UUID_C}",
    ]
    fires = _feed(d, sigs)
    assert len(fires) == 1, f"expected one alert, got {fires}"
    step, risk = fires[0]
    assert step == 2
    assert risk.trigger == "near_duplicate_loop"
    # Proof the collapse did real work: the raw signatures are pairwise
    # distinct, so no plain-loop alert may exist for this stream.
    assert all(r.trigger == "near_duplicate_loop" for _, r in fires)


def test_near_duplicate_digit_suffixes_collapse_and_fire():
    d = LoopDetector(config=Config(loop_near_duplicate_enabled=True))
    fires = _feed(d, ["fetch_page:n=101", "fetch_page:n=202", "fetch_page:n=303"])
    assert [r.trigger for _, r in fires] == ["near_duplicate_loop"]


def test_near_duplicate_distinct_actions_stay_silent():
    d = LoopDetector(config=Config(loop_near_duplicate_enabled=True))
    sigs = [
        f"get_user:id={_UUID_A}",
        f"delete_pod:name={_UUID_B}",
        "search:index=q=hello",
        f"get_user:email={_UUID_C}",
    ] * 2
    assert _feed(d, sigs) == [], "genuinely different actions must stay silent"


def test_near_duplicate_hex_digests_do_not_collapse():
    """Honest limit of the heuristic: opaque hex digests whose normalized
    forms differ must NOT be merged (fixed strings, so deterministic)."""
    d = LoopDetector(config=Config(loop_near_duplicate_enabled=True))
    digests = ["a1b2c3d4e5f60718", "b2c3d4e5f6071829", "d4e5f6a70819283b"]
    assert _feed(d, digests * 2) == []


def test_normalizer_hook_is_replaceable():
    """The normalization hook accepts a custom strategy."""

    def drop_after_colon(sig: str) -> str:
        return sig.rsplit(":", 1)[0]

    d = LoopDetector(
        config=Config(loop_near_duplicate_enabled=True),
        normalizer=drop_after_colon,
    )
    fires = _feed(d, ["retry:1", "retry:22", "retry:333"])
    assert [r.trigger for _, r in fires] == ["near_duplicate_loop"]


# --- stall mode ---------------------------------------------------------------


def test_stall_fires_at_exactly_step_twenty_five_not_twenty_four():
    d = LoopDetector(config=Config(loop_stall_enabled=True))
    fires = _feed(d, ["stuck"] * 30)
    stalls = [(i, r) for i, r in fires if r.trigger == "stall"]
    assert [i for i, _ in stalls] == [24], (
        f"stall must fire once at the 25th identical step (index 24), got {fires}"
    )
    _, risk = stalls[0]
    assert risk.score > 0.0
    # A 25-step stall is also a plain repetition: the shared plain path alerts
    # once at its own threshold (documented coexistence, not double counting).
    others = {r.trigger for i, r in fires if i != 24}
    assert others == {"loop"}


def test_stall_accumulates_zero_delta_steps():
    """Zero wall-clock deltas are evidence of a stall, not a reason to reset:
    the streak must reach 25 even when the clock never advances."""
    d = LoopDetector(config=Config(loop_stall_enabled=True))
    fires = _feed(d, ["stuck"] * 25, timestamps=[0.0] * 25)
    stalls = [(i, r) for i, r in fires if r.trigger == "stall"]
    assert [i for i, _ in stalls] == [24]
    assert "0.000s elapsed" in stalls[0][1].detail


def test_stall_signature_change_resets_streak():
    d = LoopDetector(config=Config(loop_stall_enabled=True))
    sigs = ["stuck"] * 20 + ["moved"] + ["stuck"] * 25
    stalls = [(i, r) for i, r in _feed(d, sigs) if r.trigger == "stall"]
    assert [i for i, _ in stalls] == [45], (
        "one progress step must restart the count: next fire at index 45"
    )


def test_stall_healthy_alternation_never_fires():
    d = LoopDetector(config=Config(loop_stall_enabled=True))
    sigs = (["a1", "b2"] * 30) + ([f"c{i}" for i in range(25)])
    stalls = [(i, r) for i, r in _feed(d, sigs) if r.trigger == "stall"]
    assert stalls == [], "alternating and varied work is not a stall"


# --- modes off: regression -----------------------------------------------------


def test_default_modes_off_matches_plain_behavior():
    """With stock Config the detector emits only classic 'loop' triggers."""
    d = LoopDetector()
    # Alternating pair: the plain loop fires for each signature (documented
    # behavior), and no hardening trigger may ever appear.
    fires = _feed(d, ["x1", "y2"] * 12)
    assert {r.trigger for _, r in fires} == {"loop"}, fires
    assert fires[0][0] == 4  # third occurrence of x1 lands at index 4


def test_modes_off_uuid_variants_do_not_fire_anything():
    """Without near-duplicate mode, uuid-suffixed retries look unique and the
    detector correctly says nothing (that gap is what the mode exists for)."""
    d = LoopDetector()
    sigs = [
        f"get_user:id={_UUID_A}",
        f"get_user:id={_UUID_B}",
        f"get_user:id={_UUID_C}",
    ]
    assert _feed(d, sigs) == []


def test_modes_off_stall_shape_emits_only_plain_loop():
    d = LoopDetector()
    fires = _feed(d, ["stuck"] * 26)
    assert [r.trigger for _, r in fires] == ["loop"]
    assert [i for i, _ in fires] == [2], "plain loop fires once at threshold"


def test_config_defaults_match_spec():
    cfg = Config()
    assert cfg.loop_near_duplicate_enabled is False
    assert cfg.loop_cycle_enabled is False
    assert cfg.loop_cycle_window_size == 12
    assert cfg.loop_cycle_min_period == 2
    assert cfg.loop_cycle_max_period == 6
    assert cfg.loop_stall_enabled is False
    assert cfg.loop_stall_steps == 25


def test_hardening_coexists_with_plain_loop():
    """All modes on at once: each shape reports under its own trigger name."""
    cfg = Config(
        loop_near_duplicate_enabled=True,
        loop_cycle_enabled=True,
        loop_cycle_window_size=6,
        loop_stall_enabled=True,
    )
    d = LoopDetector(config=cfg)
    seen: set[str] = set()
    # Cycle phase.
    for i, s in enumerate(["ca", "cb", "cc"] * 2):
        r = d.observe(_event(i, s))
        if r is not None:
            seen.add(r.trigger)
    d.reset("ep")
    # Near-duplicate phase.
    for i, s in enumerate([f"act:{_UUID_A}", f"act:{_UUID_B}", f"act:{_UUID_C}"]):
        r = d.observe(_event(i, s))
        if r is not None:
            seen.add(r.trigger)
    d.reset("ep")
    # Stall phase.
    for i in range(25):
        r = d.observe(_event(i, "zzz"))
        if r is not None:
            seen.add(r.trigger)
    assert {"cycle", "near_duplicate_loop", "stall"} <= seen, seen
