"""Tests for the stagnation detector (issue #87).

Every injected-failure scenario below has a mirrored healthy scenario: the
detector must fire (with the exact ``"stagnation"`` trigger) on collapsing
novelty and must stay silent on exploration and on repetitive-but-productive
traffic.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from snagline.config import Config
from snagline.detectors.loop import LoopDetector
from snagline.detectors.stagnation import StagnationDetector
from snagline.events import StepEvent, make_signature
from snagline.monitor import Monitor


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


def _feed(d: StagnationDetector, signatures: list[str], episode: str = "ep"):
    """Feed ``signatures`` as consecutive steps; return the risks emitted."""
    risks = []
    for i, s in enumerate(signatures):
        r = d.observe(_event(i, s, episode))
        if r is not None:
            risks.append(r)
    return risks


# --- Injected failure: sustained novelty collapse ----------------------------


def test_fires_once_after_two_consecutive_stale_windows():
    """Defaults (50/0.05/2): 50 unique steps fill the window fresh, then 50
    exact repeats starve it. The share of novel steps falls below 5% first at
    step 97 (2/50 novel, stale 1) and patience=2 is reached at step 98 (1/50,
    stale 2): one fire, and continuing staleness must not re-fire.
    """
    d = StagnationDetector()
    seq = [_sig(i) for i in range(50)] + [_sig(0)] * 50
    risks = _feed(d, seq)
    assert [r.step_id for r in risks] == ["98"]
    assert risks[0].trigger == "stagnation"
    assert risks[0].score == 0.6
    assert risks[0].severity == "warning"
    assert risks[0].episode_id == "ep"


def test_single_stale_window_short_of_patience_stays_silent():
    """One stale full-window observation is not enough: stop right after the
    first one (step 97 in the trace above) and nothing may fire.
    """
    d = StagnationDetector()
    seq = [_sig(i) for i in range(50)] + [_sig(0)] * 48
    assert _feed(d, seq) == []


def test_recovery_resets_the_stale_counter_and_rearms():
    """A fresh observation mid-stagnation must clear the stale counter (so the
    collapse has to rebuild patience from zero), and a recovered episode that
    collapses again fires a second time. Trace with window 10 / 0.2 / 2:
    fires at 19, goes fresh again at 31, fires again at 49.
    """
    d = StagnationDetector(window_size=10, min_novelty=0.2, patience=2)
    seq = (
        [_sig(i) for i in range(10)]  # explore: fresh
        + [_sig(0)] * 20  # starve: fire at 19, then quiet
        + [_sig(10 + i) for i in range(10)]  # recover: counter resets
        + [_sig(0)] * 10  # starve again: second fire at 49
    )
    risks = _feed(d, seq)
    assert [r.step_id for r in risks] == ["19", "49"]
    assert all(r.trigger == "stagnation" for r in risks)


def test_reset_clears_window_stale_counter_and_all_time_set():
    """reset() must drop everything: previously-seen signatures become novel
    again (no instant re-fire), and the detector can still fire afterwards.
    """
    d = StagnationDetector(window_size=10, min_novelty=0.2, patience=2)
    first = _feed(d, [_sig(i) for i in range(10)] + [_sig(0)] * 10)
    assert [r.step_id for r in first] == ["19"]

    d.reset("ep")
    assert d._windows == {}

    # The very signatures that starred in the first collapse are novel again.
    replay = [_event(100 + i, _sig(i), "ep2") for i in range(10)]
    assert all(d.observe(e) is None for e in replay)

    # And the detector is fully functional after the reset.
    more = [_event(110 + i, _sig(0), "ep2") for i in range(10)]
    risks = [r for r in (d.observe(e) for e in more) if r is not None]
    assert [r.step_id for r in risks] == ["119"]


def test_episodes_are_isolated():
    """State is keyed by episode: one episode starving cannot fire another
    that is exploring, and vice versa. Interleaved a (constant signature) and
    b (always-new): only a fires, at its own 20th step (index 38 of the
    interleaved stream maps to a-step 19).
    """
    d = StagnationDetector(window_size=10, min_novelty=0.2, patience=2)
    risks = []
    step = 0
    for k in range(25):
        for ep, sig in (("a", _sig(7)), ("b", _sig(100 + k))):
            r = d.observe(_event(step, sig, ep))
            if r is not None:
                risks.append(r)
            step += 1
    assert len(risks) == 1
    assert risks[0].episode_id == "a"
    assert risks[0].trigger == "stagnation"


# --- Injected failure that the loop detector structurally cannot see --------


def test_near_duplicate_varying_args_trip_stagnation_but_not_loop():
    """The distinctness requirement (issue #87): an agent that varies its
    arguments slightly produces recycled-but-spaced-out signatures from a tiny
    template space. Exact-match looping (window 12, threshold 3) never sees a
    repetition; the all-time novelty set still collapses. Same events, both
    detectors: loop silent, stagnation fires.
    """
    pool = [
        make_signature("tool_call", "search", f"query variant {i}") for i in range(40)
    ]
    seq = [pool[i % 40] for i in range(200)]

    loop = LoopDetector(window_size=12, repeat_threshold=3)
    loop_risks = [
        r for i, s in enumerate(seq) if (r := loop.observe(_event(i, s))) is not None
    ]
    assert loop_risks == [], "exact-match loop detector must not fire here"

    stag = StagnationDetector()
    risks = _feed(stag, seq)
    assert len(risks) == 1
    assert risks[0].trigger == "stagnation"
    assert risks[0].score == 0.6


# --- Healthy traffic: the no-false-positive side -----------------------------


def test_healthy_exploration_stays_silent():
    d = StagnationDetector()
    assert _feed(d, [_sig(i) for i in range(300)]) == []


def test_productive_maintenance_repetition_stays_silent():
    """Legitimate repetitive-but-productive phases (acceptance criteria): four
    identical maintenance calls per block interleaved with six genuinely new
    actions keeps every sliding window far above the novelty floor.
    """
    d = StagnationDetector()
    seq: list[str] = []
    for block in range(30):
        seq += [_sig(9000)] * 4 + [_sig(block * 6 + k) for k in range(6)]
    assert _feed(d, seq) == []


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
    d = StagnationDetector()
    assert all(d.observe(e) is None for e in events)


# --- Opt-in wiring and configuration -----------------------------------------


def test_default_off_in_zero_dependency_preset():
    m = Monitor.default(config=Config())
    assert all(getattr(det, "name", "") != "stagnation" for det in m._detectors)


def test_config_flag_wires_detector_into_default_monitor():
    m = Monitor.default(config=Config(stagnation_enabled=True))
    names = [getattr(det, "name", "") for det in m._detectors]
    assert names.count("stagnation") == 1


def test_enabled_detector_joins_the_ml_ensemble_wrap():
    cfg = Config(stagnation_enabled=True, ml_ensemble_enabled=True)
    m = Monitor.default(config=cfg)
    assert [getattr(det, "name", "") for det in m._detectors] == ["ml_ensemble"]
    wrapped_names = [getattr(det, "name", "") for det in m._detectors[0]._base]  # type: ignore[attr-defined]
    assert "stagnation" in wrapped_names


def test_stagnation_settings_flow_through_env_overrides():
    cfg = Config.from_env(
        {
            "SNAGLINE_STAGNATION_ENABLED": "true",
            "SNAGLINE_STAGNATION_WINDOW_SIZE": "20",
            "SNAGLINE_STAGNATION_MIN_NOVELTY": "0.1",
            "SNAGLINE_STAGNATION_PATIENCE": "5",
        }
    )
    assert cfg.stagnation_enabled is True
    assert cfg.stagnation_window_size == 20
    assert cfg.stagnation_min_novelty == 0.1
    assert cfg.stagnation_patience == 5


def test_explicit_params_override_config_defaults():
    d = StagnationDetector(
        window_size=8,
        min_novelty=0.5,
        patience=1,
        config=Config(stagnation_window_size=50),
    )
    assert (d.window_size, d.min_novelty, d.patience) == (8, 0.5, 1)


def test_invalid_constructor_arguments_raise():
    with pytest.raises(ValueError):
        StagnationDetector(window_size=0)
    with pytest.raises(ValueError):
        StagnationDetector(min_novelty=1.5)
    with pytest.raises(ValueError):
        StagnationDetector(min_novelty=0.0)  # unreachable fire condition
    with pytest.raises(ValueError):
        StagnationDetector(patience=0)


# --- Validation contract (issue #132) ----------------------------------------


def test_config_rejects_degenerate_stagnation_values():
    """Issue #132: degenerate stagnation settings are configuration errors.

    They fail loudly at Config construction instead of silently disabling the
    detector (min_novelty=0.0 made the fire condition unreachable because a
    novelty rate is always >= 0)."""
    with pytest.raises(ValueError, match="stagnation_window_size must be >= 1"):
        Config(stagnation_window_size=0)
    with pytest.raises(ValueError, match=r"stagnation_min_novelty must be within"):
        Config(stagnation_min_novelty=0.0)
    with pytest.raises(ValueError, match=r"stagnation_min_novelty must be within"):
        Config(stagnation_min_novelty=-0.1)
    with pytest.raises(ValueError, match=r"stagnation_min_novelty must be within"):
        Config(stagnation_min_novelty=1.5)
    with pytest.raises(ValueError, match="stagnation_patience must be >= 1"):
        Config(stagnation_patience=0)


def test_config_accepts_valid_stagnation_extremes():
    cfg = Config(
        stagnation_enabled=True,
        stagnation_window_size=1,
        stagnation_min_novelty=1.0,
        stagnation_patience=1,
    )
    d = StagnationDetector(config=cfg)
    risks = _feed(d, [_sig(0), _sig(0)])
    assert len(risks) == 1 and risks[0].trigger == "stagnation"


def test_valid_config_driven_detector_still_fires():
    """Accepted-and-functional side of the contract: tuned-through-Config
    knobs keep detecting (window 4 / 0.02 / 1 fires once novelty collapses)."""
    cfg = Config(
        stagnation_enabled=True,
        stagnation_window_size=4,
        stagnation_min_novelty=0.02,
        stagnation_patience=1,
    )
    d = StagnationDetector(config=cfg)
    seq = [_sig(i) for i in range(4)] + [_sig(0)] * 4
    risks = _feed(d, seq)
    assert risks and risks[0].trigger == "stagnation"
