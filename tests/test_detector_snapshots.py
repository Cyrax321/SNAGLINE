"""Round-trip snapshot/restore tests for StagnationDetector,
SideEffectGuardDetector, and CompactionTripwireDetector (issue #149).

These mirror the behavioral style of ``tests/test_monitor_snapshot.py``: state
goes through a true JSON round trip (``json.loads(json.dumps(...))``), then
the restored detector must behave identically to the never-restarted original.
The issue's acceptance criteria are covered verbatim: a key that already fired
pre-restart stays quiet after restore, and a key that had not yet fired still
fires afterwards. CompactionTripwireDetector is included per the maintainer's
scope-extension comment on #149 (it landed via #147 without the pair).
"""

from __future__ import annotations

import json

from snagline.detectors.compaction_tripwire import CompactionTripwireDetector
from snagline.detectors.side_effect_guard import SideEffectGuardDetector
from snagline.detectors.stagnation import StagnationDetector
from snagline.events import StepEvent
from snagline.monitor import Monitor


def _ev(
    step: int,
    *,
    signature: str = "s",
    tool_name: str | None = "api",
    side_effect: bool = False,
    action_type: str = "tool_call",
    metadata: dict | None = None,
    episode_id: str = "ep",
) -> StepEvent:
    return StepEvent(
        step_id=str(step),
        episode_id=episode_id,
        timestamp=float(step),
        action_type=action_type,
        action_signature=signature,
        tool_name=tool_name,
        side_effect=side_effect,
        metadata=metadata or {},
    )


def _risks(det, events):
    return [r for e in events if (r := det.observe(e)) is not None]


def _round_trip(det):
    """A true JSON round trip: floats/lists/dicts pass through repr()."""
    det.load_state(json.loads(json.dumps(det.dump_state())))


# --- SideEffectGuardDetector -------------------------------------------------


def _guard_stream(first: int, count: int):
    return [
        _ev(i, signature="charge", tool_name="payment", side_effect=True)
        for i in range(first, first + count)
    ]


def test_side_effect_guard_unfired_key_still_fires_after_restore():
    d1 = SideEffectGuardDetector(allowed_repeats=1)
    assert _risks(d1, _guard_stream(0, 1)) == [], "first occurrence is tolerated"

    d2 = SideEffectGuardDetector(allowed_repeats=1)
    d2.load_state(json.loads(json.dumps(d1.dump_state())))

    risks = _risks(d2, _guard_stream(1, 1))
    assert len(risks) == 1, "the repeat spanning the restart must still fire"
    assert risks[0].trigger == "side_effect_duplicate"


def test_side_effect_guard_already_fired_key_stays_quiet_after_restore():
    d1 = SideEffectGuardDetector(allowed_repeats=1)
    assert len(_risks(d1, _guard_stream(0, 2))) == 1, "fires on the 2nd occurrence"
    _round_trip(d1)

    # Third and fourth identical occurrences stay silent: edge-triggered latch
    # survives the restart, so a repeated payment cannot alert-spam across it.
    assert _risks(d1, _guard_stream(2, 2)) == []


def test_side_effect_guard_distinct_keys_independent_after_restore():
    d1 = SideEffectGuardDetector(allowed_repeats=1)
    _risks(d1, [_ev(0, signature="charge", tool_name="payment", side_effect=True)])
    _round_trip(d1)

    # A different (tool, signature) key was never seen pre-restart: it must get
    # its own full tolerance, not inherit the fired state of another key.
    fresh = [_ev(1, signature="deploy", tool_name="shipit", side_effect=True)]
    assert _risks(d1, fresh) == []
    again = [_ev(2, signature="deploy", tool_name="shipit", side_effect=True)]
    assert len(_risks(d1, again)) == 1


# --- StagnationDetector -------------------------------------------------------


def _stagnation_detector() -> StagnationDetector:
    # Small window so the scenario fits in a handful of steps: stale when fewer
    # than 1-in-4 actions are novel, firing after 2 consecutive stale windows.
    return StagnationDetector(window_size=4, min_novelty=0.25, patience=2)


def _warm_unique(det: StagnationDetector) -> None:
    det.observe(_ev(0, signature="u0"))
    det.observe(_ev(1, signature="u1"))
    det.observe(_ev(2, signature="u2"))
    det.observe(_ev(3, signature="u3"))


def _repeats(first: int, count: int):
    return [
        _ev(i, signature="stuck", tool_name="planner")
        for i in range(first, first + count)
    ]


def test_stagnation_unfired_collapse_still_fires_after_restore():
    d1 = _stagnation_detector()
    _warm_unique(d1)
    assert _risks(d1, _repeats(4, 4)) == [], "one stale window: below patience"

    d2 = _stagnation_detector()
    d2.load_state(json.loads(json.dumps(d1.dump_state())))

    risks = _risks(d2, _repeats(8, 4))
    assert len(risks) == 1, "collapse completing after restart must still fire"
    assert risks[0].trigger == "stagnation"


def test_stagnation_already_fired_collapse_stays_quiet_after_restore():
    d1 = _stagnation_detector()
    _warm_unique(d1)
    _risks(d1, _repeats(4, 4))  # stale_windows reaches 1
    fired = _risks(d1, _repeats(8, 4))  # stale_windows reaches patience: fires
    assert len(fired) == 1
    _round_trip(d1)

    # Continuing the same collapse must not re-fire (one finding per collapse),
    # and the restored stale counter is what keeps it quiet.
    assert _risks(d1, _repeats(12, 4)) == []


def test_stagnation_recovery_rearms_after_restore():
    d1 = _stagnation_detector()
    _warm_unique(d1)
    _risks(d1, _repeats(4, 8))  # fired mid-stream
    _round_trip(d1)

    # Novelty recovers (fresh signatures), then collapses again: the restored
    # re-arm state must allow exactly one new finding.
    recovery = [_ev(20 + i, signature=f"fresh{i}") for i in range(4)]
    assert _risks(d1, recovery) == [], "recovery resets the stale counter"
    refire = _risks(d1, _repeats(30, 4))
    assert refire == []  # only 1 stale window since recovery
    refire = _risks(d1, _repeats(40, 4))
    assert len(refire) == 1, "second collapse after recovered novelty fires again"


# --- CompactionTripwireDetector ----------------------------------------------


_PIN_A = "a" * 64
_PIN_B = "b" * 64


def _compaction(step: int, pins: list[str]) -> StepEvent:
    return _ev(
        step,
        # Stable signature: the tripwire keys on action_type/metadata only, and
        # stable strings keep the co-hosted stagnation scenarios predictable.
        signature="compact",
        tool_name=None,
        action_type="compaction",
        metadata={"pinned": pins},
    )


def _confirm(step: int, pin: str) -> StepEvent:
    return _ev(
        step,
        signature="confirm",
        tool_name=None,
        action_type="constraint_present",
        metadata={"pin": pin},
    )


def _filler(step: int) -> StepEvent:
    return _ev(step, signature="fill", tool_name="worker")


def test_compaction_tripwire_pending_window_survives_restore_and_still_fires():
    """The maintainer's acceptance sketch (#149 comment): snapshot after
    ``compaction`` but before any ``constraint_present``, restore, feed
    confirmations past the deadline; the unconfirmed pin still decays."""
    d1 = CompactionTripwireDetector(grace_steps=3)
    d1.observe(_compaction(0, [_PIN_A, _PIN_B]))
    d1.observe(_filler(1))  # ordinal 2 of 4; deadline not reached yet
    _round_trip(d1)

    # Restored: confirm A, then burn past the grace deadline. B was never
    # confirmed, so governance_decay must still fire -- exactly once.
    events = [_confirm(2, _PIN_A), _filler(3), _filler(4)]
    risks = _risks(d1, events)
    assert len(risks) == 1, "unconfirmed pin must decay after restore"
    assert risks[0].trigger == "governance_decay"
    # Detail names only unconfirmed pin prefixes: B decayed, A was confirmed.
    assert "b" * 16 in risks[0].detail
    assert "a" * 16 not in risks[0].detail


def test_compaction_tripwire_already_fired_stays_quiet_after_restore():
    d1 = CompactionTripwireDetector(grace_steps=3)
    d1.observe(_compaction(0, [_PIN_A, _PIN_B]))
    risks = _risks(
        d1,
        [_filler(1), _filler(2), _filler(3)],  # deadline passes unconfirmed
    )
    assert len(risks) == 1
    _round_trip(d1)

    # Confirming B late must not retract or re-fire anything, and pin A was
    # never confirmed: only the restored fired latch keeps this episode quiet.
    tail = [_confirm(4, _PIN_B), _filler(5), _filler(6)]
    assert _risks(d1, tail) == []


def test_compaction_tripwire_no_pending_window_round_trips_cleanly():
    d1 = CompactionTripwireDetector(grace_steps=3)
    d1.observe(_filler(0))  # ordinal tracked, nothing pending
    _round_trip(d1)
    # After restore a later compaction opens a normal window.
    d1.observe(_compaction(1, [_PIN_A]))
    risks = _risks(d1, [_filler(2), _filler(3), _filler(4)])
    assert len(risks) == 1


# --- Monitor-level participation ---------------------------------------------


class ListSink:
    def __init__(self) -> None:
        self.risks: list = []

    def emit(self, risk) -> None:
        self.risks.append(risk)


def _composition() -> tuple[list, list]:
    detectors: list = [
        SideEffectGuardDetector(allowed_repeats=1),
        StagnationDetector(window_size=4, min_novelty=0.25, patience=2),
        CompactionTripwireDetector(grace_steps=3),
    ]
    return detectors, [ListSink()]


def _mixed_stream() -> list[StepEvent]:
    """Signals from all three detectors land AFTER the snapshot boundary.

    Episodes are kept separate so scenarios cannot contaminate each other's
    per-episode detector state: "main" carries guard + tripwire, "stag" a
    self-contained stagnation collapse (same numbers as the unit tests above).
    """
    events: list[StepEvent] = [
        # Guard (main): first occurrence tolerated, fires only after boundary.
        _ev(0, signature="charge", tool_name="payment", side_effect=True),
        # Stagnation (stag): warm-up plus four repeats; patience not reached.
        _ev(100, signature="u0", episode_id="stag"),
        _ev(101, signature="u1", episode_id="stag"),
        _ev(102, signature="u2", episode_id="stag"),
        _ev(103, signature="u3", episode_id="stag"),
        _ev(104, signature="stuck", episode_id="stag"),
        _ev(105, signature="stuck", episode_id="stag"),
        _ev(106, signature="stuck", episode_id="stag"),
        _ev(107, signature="stuck", episode_id="stag"),
        # Tripwire (main): compaction opens a grace window of 3 events.
        _compaction(9, [_PIN_A, _PIN_B]),
        # Confirm A before the boundary; B stays pending across the restart.
        _confirm(10, _PIN_A),
    ]
    return events


def test_monitor_snapshot_restore_matches_never_restarted_twin(tmp_path):
    path = str(tmp_path / "state.json")

    m_source = Monitor(*_composition())
    m_twin = Monitor(*_composition())
    stream = _mixed_stream()
    for e in stream:
        m_source.ingest(e)
        m_twin.ingest(e)
    pre_tail_count = len(m_source._sinks[0].risks)
    m_source.snapshot(path)

    # Before this issue these three serialized as null and were skipped on
    # restore; participation itself is part of the acceptance criteria.
    dumped = json.loads(open(path, encoding="utf-8").read())
    states = [v for v in dumped["detectors"].values() if v is not None]
    assert len(states) == 3, "all three detectors must serialize non-null"

    m_restored = Monitor(*_composition())
    m_restored.restore(path)

    tail = [
        # Tripwire (main): ordinal 13 reaches the deadline; B decays once.
        _filler(12),
        _filler(13),
        # Stagnation (stag): the collapse completes after the boundary.
        _ev(108, signature="stuck", episode_id="stag"),
        _ev(109, signature="stuck", episode_id="stag"),
        _ev(110, signature="stuck", episode_id="stag"),
        _ev(111, signature="stuck", episode_id="stag"),
        # Guard (main): repeat fires.
        _ev(16, signature="charge", tool_name="payment", side_effect=True),
    ]
    for e in tail:
        m_source.ingest(e)
        m_restored.ingest(e)

    source_risks = [(r.step_id, r.trigger, r.score) for r in m_source._sinks[0].risks][
        pre_tail_count:
    ]
    restored_risks = [
        (r.step_id, r.trigger, r.score) for r in m_restored._sinks[0].risks
    ]
    assert source_risks == restored_risks
    triggers = {t for _, t, _ in source_risks}
    assert {"side_effect_duplicate", "stagnation", "governance_decay"} <= triggers, (
        "the tail must exercise all three restored detectors"
    )
