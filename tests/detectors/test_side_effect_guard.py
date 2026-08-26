"""Tests for the side-effect guard detector (issue #88).

Every injected-failure scenario below has a mirrored healthy scenario: the
detector must fire (with the exact ``"side_effect_duplicate"`` trigger) on a
repeated host-declared non-idempotent action and must stay silent on
idempotent reads, distinct-argument retries, and other episodes' traffic.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from snagline.config import Config
from snagline.detectors.side_effect_guard import SideEffectGuardDetector
from snagline.events import StepEvent, make_signature
from snagline.monitor import Monitor


def _sig(i: int) -> str:
    return make_signature("tool_call", "payment_tool", f"a{i}")


def _event(
    step_id: int,
    sig: str,
    episode: str = "ep",
    *,
    tool_name: str | None = "payment_tool",
    side_effect: bool = True,
) -> StepEvent:
    return StepEvent(
        step_id=str(step_id),
        episode_id=episode,
        timestamp=float(step_id),
        action_type="tool_call",
        action_signature=sig,
        tool_name=tool_name,
        side_effect=side_effect,
    )


def _feed(d: SideEffectGuardDetector, events: list[StepEvent]) -> list:
    return [r for e in events if (r := d.observe(e)) is not None]


# --- Injected failure: duplicate non-idempotent action -----------------------


def test_duplicate_side_effect_fires_once_on_second_occurrence():
    """Two identical payment-shape steps fire exactly one risk, attached to the
    2nd step. The 3rd identical step stays silent: edge-triggered like the
    loop detector's dedupe, because alert spam during an ongoing duplicate is
    worse than one escalation."""
    d = SideEffectGuardDetector()
    sig = _sig(7)
    first = _feed(d, [_event(1, sig)])
    assert first == []
    second = _feed(d, [_event(2, sig)])
    assert len(second) == 1
    r = second[0]
    assert r.trigger == "side_effect_duplicate"
    assert r.step_id == "2"
    assert r.episode_id == "ep"
    assert r.score == 0.9
    assert r.severity == "critical"
    assert r.detail == "non-idempotent payment_tool fired 2x in episode"
    # Edge-triggered: nothing more from continued identical steps.
    assert _feed(d, [_event(3, sig), _event(4, sig)]) == []


def test_allowed_repeats_two_fires_on_third_occurrence():
    d = SideEffectGuardDetector(allowed_repeats=2)
    seq = [
        _event(1, _sig(1)),
        _event(2, _sig(1)),
        _event(3, _sig(1)),
        _event(4, _sig(1)),
    ]
    risks = _feed(d, seq)
    assert [r.step_id for r in risks] == ["3"]


def test_tool_name_is_part_of_the_key():
    """The same signature under two different tools is two distinct actions:
    interleaved, each tool escalates on its own 2nd occurrence."""
    d = SideEffectGuardDetector()
    sig = _sig(5)
    inter = [
        _event(1, sig, tool_name="charge_card"),
        _event(2, sig, tool_name="refund_card"),
        _event(3, sig, tool_name="charge_card"),
        _event(4, sig, tool_name="refund_card"),
    ]
    risks = _feed(d, inter)
    assert [r.step_id for r in risks] == ["3", "4"]
    assert "charge_card" in risks[0].detail
    assert "refund_card" in risks[1].detail


def test_none_tool_name_still_counts_and_fires():
    d = SideEffectGuardDetector()
    sig = _sig(9)
    assert _feed(d, [_event(1, sig, tool_name=None)]) == []
    risks = _feed(d, [_event(2, sig, tool_name=None)])
    assert [r.trigger for r in risks] == ["side_effect_duplicate"]


# --- Healthy traffic: the no-false-positive side -----------------------------


def test_idempotent_reads_never_fire_even_when_repeated():
    """Read-only steps repeated forever are exactly what must stay silent,
    regardless of how often they repeat."""
    d = SideEffectGuardDetector()
    sig = _sig(3)
    reads = [_event(i, sig, side_effect=False) for i in range(50)]
    assert _feed(d, reads) == []


def test_unmarked_repeats_of_a_marked_action_shape_stay_silent():
    """Only the boolean decides: the same action shape without the flag never
    fires, so legacy adapters that predate #88 see zero behavior change."""
    d = SideEffectGuardDetector()
    marked = _feed(d, [_event(1, _sig(1)), _event(2, _sig(1))])
    assert [r.step_id for r in marked] == ["2"]
    unmarked = [_event(i, _sig(1), side_effect=False) for i in range(10, 30)]
    assert _feed(d, unmarked) == []


def test_distinct_signature_retries_stay_silent():
    """Different-argument retries produce different signatures and never fire
    here (issue #88 acceptance); that failure shape belongs to LoopDetector."""
    d = SideEffectGuardDetector()
    retries = [_event(i, _sig(i), side_effect=True) for i in range(200)]
    assert _feed(d, retries) == []


def test_stays_silent_on_healthy_fixture_trajectory():
    fixture = (
        Path(__file__).resolve().parents[1]
        / "fixtures"
        / "trajectories"
        / "healthy_run.jsonl"
    )
    events = [
        StepEvent(**json.loads(line))
        for line in fixture.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    d = SideEffectGuardDetector()
    assert all(d.observe(e) is None for e in events)


# --- State: bounded per episode, isolated across episodes, resettable --------


def test_state_is_bounded_by_distinct_actions_not_by_repeats():
    """A thousand replays of one charge cost one counter entry: memory grows
    with distinct marked actions only."""
    d = SideEffectGuardDetector()
    sig = _sig(11)
    _feed(d, [_event(i, sig) for i in range(1000)])
    assert len(d._counts) == 1
    assert d._counts["ep"] == {("payment_tool", sig): 1000}


def test_episodes_are_isolated():
    """Episode b replaying the same charge is judged independently of episode
    a's history, and vice versa."""
    d = SideEffectGuardDetector()
    sig = _sig(13)
    events: list[StepEvent] = []
    events.append(_event(0, sig, "a"))  # a: 1st occurrence
    events.append(_event(1, sig, "b"))  # b: 1st occurrence
    events.append(_event(2, sig, "a"))  # a: fires
    events.append(_event(3, sig, "b"))  # b: fires
    events.append(_event(4, sig, "c"))  # c: still silent
    risks = _feed(d, events)
    assert [(r.episode_id, r.step_id) for r in risks] == [("a", "2"), ("b", "3")]


def test_reset_clears_episode_state_and_rearms_from_zero():
    d = SideEffectGuardDetector()
    sig = _sig(17)
    assert [r.step_id for r in _feed(d, [_event(1, sig), _event(2, sig)])] == ["2"]
    d.reset("ep")
    assert d._counts == {}
    # After reset even the same episode starts from zero: the next occurrence
    # is a 1st, not a 3rd.
    assert _feed(d, [_event(3, sig)]) == []
    assert [r.step_id for r in _feed(d, [_event(4, sig)])] == ["4"]


# --- End-to-end through Monitor.default ---------------------------------------


class _RecordingSink:
    def __init__(self) -> None:
        self.risks: list = []

    def emit(self, risk) -> None:
        self.risks.append(risk)


def _monitor_with_guard(**cfg_kwargs) -> tuple[Monitor, _RecordingSink]:
    sink = _RecordingSink()
    monitor = Monitor.default(
        config=Config(side_effect_guard_enabled=True, **cfg_kwargs), sinks=[sink]
    )
    return monitor, sink


def test_monitor_dispatches_exactly_one_duplicate_alert_end_to_end():
    from snagline.adapters.raw import watch

    monitor, sink = _monitor_with_guard()
    with watch(monitor, "ep") as step:
        step("tool_call", tool_name="search", args={"q": "x"})
        step(
            "tool_call", tool_name="charge_card", args={"amount": 42}, side_effect=True
        )
        step("tool_call", tool_name="search", args={"q": "x"})
        # The retry loop every production incident report contains: identical
        # charge re-fired after a timeout.
        step(
            "tool_call", tool_name="charge_card", args={"amount": 42}, side_effect=True
        )
        step(
            "tool_call", tool_name="charge_card", args={"amount": 42}, side_effect=True
        )
    dupes = [r for r in sink.risks if r.trigger == "side_effect_duplicate"]
    assert len(dupes) == 1
    assert dupes[0].severity == "critical"

    # end_episode teardown clears the guard: the same episode id replayed
    # fresh must start counting from zero again.
    with watch(monitor, "ep") as step:
        step(
            "tool_call", tool_name="charge_card", args={"amount": 42}, side_effect=True
        )
    dupes = [r for r in sink.risks if r.trigger == "side_effect_duplicate"]
    assert len(dupes) == 1


# --- Opt-in wiring and configuration -----------------------------------------


def test_default_off_in_zero_dependency_preset():
    m = Monitor.default(config=Config())
    assert all(getattr(det, "name", "") != "side_effect_guard" for det in m._detectors)


def test_config_flag_wires_detector_into_default_monitor():
    m = Monitor.default(config=Config(side_effect_guard_enabled=True))
    names = [getattr(det, "name", "") for det in m._detectors]
    assert names.count("side_effect_guard") == 1


def test_enabled_detector_joins_the_ml_ensemble_wrap():
    cfg = Config(side_effect_guard_enabled=True, ml_ensemble_enabled=True)
    m = Monitor.default(config=cfg)
    wrapped_names = [getattr(det, "name", "") for det in m._detectors[0]._base]
    assert "side_effect_guard" in wrapped_names


def test_settings_flow_through_env_overrides():
    cfg = Config.from_env(
        {
            "SNAGLINE_SIDE_EFFECT_GUARD_ENABLED": "true",
            "SNAGLINE_SIDE_EFFECT_ALLOWED_REPEATS": "3",
            "SNAGLINE_SIDE_EFFECT_SCORE": "0.95",
        }
    )
    assert cfg.side_effect_guard_enabled is True
    assert cfg.side_effect_allowed_repeats == 3
    assert cfg.side_effect_score == 0.95


def test_explicit_params_override_config_defaults():
    d = SideEffectGuardDetector(
        allowed_repeats=4,
        score=0.8,
        config=Config(side_effect_allowed_repeats=1),
    )
    assert d.allowed_repeats == 4
    assert d.score == 0.8


def test_invalid_constructor_arguments_raise():
    with pytest.raises(ValueError):
        SideEffectGuardDetector(allowed_repeats=0)
    with pytest.raises(ValueError):
        SideEffectGuardDetector(score=1.5)
