"""Regression tests for loop-family arbitration (issue #604).

``loop``, ``cycle`` and ``stall`` are three thresholds on one statistic in
one detector, and ``observe()`` returned the coarse ``loop`` risk in
preference to the hardened one. Nothing retired the coarse alert once the
stricter condition held, so ``loop`` precision was degraded by coarse alerts
arriving 22-50 steps before the trigger that actually described the failure.

The fix arbitrates within the detector family: when ``stall`` or ``cycle``
fires, the risk declares that it *supersedes* earlier coarse ``loop`` findings
describing the same repeated actions (via ``supersedes`` and
``superseded_step_ids``). Detection behaviour is unchanged -- every condition
still fires, on the same step, at the same threshold -- but the reported
trigger identity is corrected by any consumer that reduces the episode's risks.
"""

from __future__ import annotations

from benchmarks.detection_accuracy import reduce_risks

from snagline.config import Config
from snagline.detectors.loop import LoopDetector
from snagline.events import StepEvent
from snagline.risk import FailureRisk


def _event(step_id: int | str, sig: str, episode: str = "ep") -> StepEvent:
    return StepEvent(
        step_id=str(step_id),
        episode_id=episode,
        timestamp=float(step_id)
        if isinstance(step_id, int) or step_id.isdigit()
        else 1.0,
        action_type="tool_call",
        action_signature=sig,
    )


def _feed(d: LoopDetector, sigs: list[str]) -> list[tuple[int, FailureRisk]]:
    """Feed consecutive steps; return [(step_index, risk)] for every alert."""
    out: list[tuple[int, FailureRisk]] = []
    for i, s in enumerate(sigs):
        r = d.observe(_event(i, s))
        if r is not None:
            out.append((i, r))
    return out


def _hardened_config() -> Config:
    """The harness configuration: both hardening modes on."""
    return Config(loop_cycle_enabled=True, loop_stall_enabled=True)


# --- the supersession declaration -------------------------------------------


def test_stall_declares_it_supersedes_the_plain_loop() -> None:
    """A stall is the specific shape of the repetition the loop alert named.

    ``stall`` requires 25 consecutive identical signatures; 25-in-a-row
    necessarily contains 3 within the 12-step loop window, so the plain loop
    condition is a strict subset. The stall risk says so and identifies
    which step_id it supersedes.
    """
    d = LoopDetector(config=_hardened_config())
    fires = _feed(d, ["stuck"] * 26)
    stalls = [r for _, r in fires if r.trigger == "stall"]
    assert len(stalls) == 1
    assert "loop" in stalls[0].supersedes
    assert stalls[0].superseded_step_ids == ("2",)


def test_cycle_declares_it_supersedes_the_plain_loop() -> None:
    """A short-period cycle also satisfies the plain loop rule (documented)."""
    d = LoopDetector(config=_hardened_config())
    fires = _feed(d, ["ca", "cb"] * 12)
    cycles = [r for _, r in fires if r.trigger == "cycle"]
    assert len(cycles) == 1
    assert "loop" in cycles[0].supersedes


def test_plain_loop_declares_no_supersession() -> None:
    """The coarse trigger never claims to replace anything."""
    d = LoopDetector()
    fires = _feed(d, ["x", "y", "x", "y", "x", "y"])
    loops = [r for _, r in fires if r.trigger == "loop"]
    assert loops
    assert all(r.supersedes == () for r in loops)


def test_near_duplicate_declares_no_supersession() -> None:
    """Near-duplicate mode names a different shape; it does not arbitrate."""
    cfg = Config(loop_near_duplicate_enabled=True)
    d = LoopDetector(config=cfg)
    import uuid

    sigs = [f"get_user:id={uuid.uuid4()}" for _ in range(4)]
    fires = _feed(d, sigs)
    near = [r for _, r in fires if r.trigger == "near_duplicate_loop"]
    assert near
    assert all(r.supersedes == () for r in near)


def test_supersedes_defaults_to_empty_everywhere_else() -> None:
    """The new field is additive: every other risk is unchanged."""
    d = LoopDetector()
    fires = _feed(d, ["a", "a", "a"])
    assert fires
    assert all(r.supersedes == () for _, r in fires)


# --- detection behaviour is preserved ----------------------------------------


def test_arbitration_does_not_silence_the_plain_loop() -> None:
    """The loop condition still fires, on the same step, at the same score.

    The fix is declarative metadata plus a consumer-side reduction; the
    detector still emits the plain loop risk so real-time sinks that do not
    understand supersession keep working.
    """
    d = LoopDetector(config=_hardened_config())
    fires = _feed(d, ["stuck"] * 26)
    by_trigger: dict[str, list[int]] = {}
    for i, r in fires:
        by_trigger.setdefault(r.trigger, []).append(i)
    assert by_trigger["loop"] == [2], "plain loop still fires once at step 2"
    assert by_trigger["stall"] == [24], "stall still fires at step 24"


def test_stall_still_fires_on_the_same_step() -> None:
    d = LoopDetector(config=_hardened_config())
    fires = _feed(d, ["stuck"] * 26)
    assert [i for i, r in fires if r.trigger == "stall"] == [24]


def test_cycle_still_fires_on_the_same_step() -> None:
    d = LoopDetector(config=_hardened_config())
    fires = _feed(d, ["ca", "cb"] * 12)
    assert [i for i, r in fires if r.trigger == "cycle"] == [3]


def test_supersession_does_not_re_arm_or_duplicate_the_loop() -> None:
    """The declaration appears exactly once, on the hardened risk."""
    d = LoopDetector(config=_hardened_config())
    fires = _feed(d, ["stuck"] * 40)
    assert sum(1 for _, r in fires if r.supersedes) == 1
    assert [r.trigger for _, r in fires if r.supersedes] == ["stall"]


# --- reduce_risks arbitration tests ------------------------------------------


def test_reduce_risks_retires_coarse_loop_on_stall() -> None:
    """In a stall episode, coarse loop finding on the stalled action is retired."""
    d = LoopDetector(config=_hardened_config())
    fires = [r for _, r in _feed(d, ["stuck"] * 26)]
    outcome = reduce_risks(fires, "ep", frozenset({"stall"}))
    assert outcome.predicted == {"stall"}


def test_reduce_risks_retires_coarse_loop_on_cycle() -> None:
    """In a cycle episode, coarse loop findings on cycling actions are retired."""
    d = LoopDetector(config=_hardened_config())
    fires = [r for _, r in _feed(d, ["ca", "cb"] * 12)]
    outcome = reduce_risks(fires, "ep", frozenset({"cycle"}))
    assert outcome.predicted == {"cycle"}


def test_reduce_risks_order_independence() -> None:
    """Outcome is order-independent: stall retiring loop works regardless of list order."""
    r_loop = FailureRisk("ep", "2", 0.5, "loop", "loop", 2.0)
    r_stall = FailureRisk(
        "ep",
        "24",
        0.5,
        "stall",
        "stall",
        24.0,
        supersedes=("loop",),
        superseded_step_ids=("2",),
    )
    # Natural order: loop then stall
    outcome1 = reduce_risks([r_loop, r_stall], "ep", None)
    assert outcome1.predicted == {"stall"}

    # Reversed order: stall then loop
    outcome2 = reduce_risks([r_stall, r_loop], "ep", None)
    assert outcome2.predicted == {"stall"}


def test_reduce_risks_loop_survives_after_stall_breaks() -> None:
    """26x z, then p, q, p, q, p: stall on z retires loop on z, but loop on p survives."""
    d = LoopDetector(config=_hardened_config())
    sigs = ["z"] * 26 + ["p", "q", "p", "q", "p"]
    fires = [r for _, r in _feed(d, sigs)]
    # Should have loop@2 (z), stall@24 (z), and loop@30 (p)
    triggers = [(r.step_id, r.trigger) for r in fires]
    assert ("2", "loop") in triggers
    assert ("24", "stall") in triggers
    assert ("30", "loop") in triggers

    outcome = reduce_risks(fires, "ep", None)
    assert outcome.predicted == {"stall", "loop"}


def test_reduce_risks_loop_survives_on_different_action_during_cycle() -> None:
    """a,b,a,b,c,c,c: cycle fires at step 3, separate loop on c fires at step 6."""
    d = LoopDetector(config=_hardened_config())
    sigs = ["a", "b", "a", "b", "c", "c", "c"]
    fires = [r for _, r in _feed(d, sigs)]
    triggers = [(r.step_id, r.trigger) for r in fires]
    assert ("3", "cycle") in triggers
    assert ("6", "loop") in triggers

    outcome = reduce_risks(fires, "ep", None)
    assert outcome.predicted == {"cycle", "loop"}
