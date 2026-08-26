"""Tests for the compaction tripwire detector (issue #90).

Every injected-failure scenario below has a mirrored healthy scenario: the
detector must fire exactly one ``"governance_decay"`` risk when a pinned
constraint goes unconfirmed past the grace window, and must stay silent on
fully confirmed windows, empty pin sets, malformed metadata, and hosts that
never emit compaction events at all.

Privacy acceptance criterion from the issue: only 16-hex prefixes of pins may
ever appear in risk details; constraint text never reaches snagline.
"""

from __future__ import annotations

import hashlib
import os

import pytest

from snagline.cli import replay
from snagline.config import Config
from snagline.detectors.compaction_tripwire import CompactionTripwireDetector
from snagline.events import StepEvent, make_signature
from snagline.monitor import Monitor

FIX = os.path.join(os.path.dirname(__file__), "..", "fixtures", "trajectories")

# Real SHA-256 digests of canonical constraint strings. The plaintext exists
# only here; detector state and emitted risks never see it, which is exactly
# the privacy property under test.
PIN_A = "448a0c330ba83452956a4b294c1af26dc0def64cffbec53e8e6c99118655dbd1"
PIN_B = "a0ac1f1e6efc074d8a29b535770bada5a7a7bcf6f84319bba14348f499ce7d31"
PIN_C = hashlib.sha256(b"constraint:no-unsandboxed-shell").hexdigest()


class RecordingSink:
    def __init__(self) -> None:
        self.risks: list = []

    def emit(self, risk) -> None:
        self.risks.append(risk)


def _event(step_id: int, action_type: str, metadata: dict | None = None) -> StepEvent:
    return StepEvent(
        step_id=str(step_id),
        episode_id="ep",
        timestamp=1000.0 + float(step_id),
        action_type=action_type,
        action_signature=make_signature(action_type, None),
        metadata=metadata if metadata is not None else {},
    )


def _plain(step_id: int) -> StepEvent:
    """An ordinary tool step carrying no tripwire meaning."""
    return StepEvent(
        step_id=str(step_id),
        episode_id="ep",
        timestamp=1000.0 + float(step_id),
        action_type="tool_call",
        action_signature=make_signature("tool_call", "search"),
        tool_name="search",
    )


def _compaction(step_id: int, pins: list) -> StepEvent:
    return _event(step_id, "compaction", {"pinned": pins})


def _present(step_id: int, pin: str) -> StepEvent:
    return _event(step_id, "constraint_present", {"pin": pin})


def _ep_event(
    episode: str, step_id: int, action_type: str, metadata: dict | None = None
) -> StepEvent:
    e = _event(step_id, action_type, metadata)
    e = StepEvent(
        step_id=e.step_id,
        episode_id=episode,
        timestamp=e.timestamp,
        action_type=e.action_type,
        action_signature=e.action_signature,
        metadata=e.metadata,
    )
    return e


def _feed(d: CompactionTripwireDetector, seq: list[StepEvent]) -> list:
    risks = []
    for e in seq:
        r = d.observe(e)
        if r is not None:
            risks.append(r)
    return risks


# --- Injected failure: pinned constraint not re-confirmed ---------------------


def test_missing_pin_fires_at_exact_grace_boundary():
    # grace_steps=3: the compaction itself is observed first, then three
    # further events are allowed. Pin B confirms on the first follow-up;
    # pin A never does, so the single risk fires on the third follow-up.
    d = CompactionTripwireDetector(grace_steps=3)
    risks = _feed(
        d,
        [
            _compaction(0, [PIN_A, PIN_B]),
            _present(1, PIN_B),
            _plain(2),
            _plain(3),
        ],
    )
    assert len(risks) == 1
    r = risks[0]
    assert r.trigger == "governance_decay"
    assert r.score == 0.9
    assert r.step_id == "3"
    assert r.severity == "critical"
    assert PIN_A[:16] in r.detail
    assert PIN_A not in r.detail  # prefix only, never the full digest
    assert PIN_B[:16] not in r.detail


def test_fires_once_then_stays_quiet_until_next_compaction():
    d = CompactionTripwireDetector(grace_steps=3)
    seq: list[StepEvent] = [_compaction(0, [PIN_A])]
    seq += [_plain(i) for i in range(1, 10)]
    risks = _feed(d, seq)
    assert len(risks) == 1
    assert risks[0].step_id == "3"

    # A later compaction opens a fresh window, which can fire again once.
    seq2: list[StepEvent] = [_compaction(10, [PIN_C])]
    seq2 += [_plain(i) for i in range(11, 14)]
    risks += _feed(d, seq2)
    assert len(risks) == 2
    assert risks[1].step_id == "13"
    assert PIN_C[:16] in risks[1].detail


def test_new_compaction_resets_pending_set_and_deadline():
    # The first window's unconfirmed pin (A) is replaced wholesale by the new
    # compaction's set {B, C}; the deadline restarts from the newer event.
    d = CompactionTripwireDetector(grace_steps=3)
    risks = _feed(
        d,
        [
            _compaction(0, [PIN_A]),
            _plain(1),
            _compaction(2, [PIN_B, PIN_C]),
            _plain(3),
            _plain(4),
            _plain(5),  # third follow-up of the second compaction: fires here
        ],
    )
    assert len(risks) == 1
    assert risks[0].step_id == "5"
    assert PIN_A[:16] not in risks[0].detail
    assert PIN_B[:16] in risks[0].detail
    assert PIN_C[:16] in risks[0].detail


def test_multi_episode_isolation():
    # Two interleaved episodes share one detector: e1 never re-confirms and
    # fires on its own third follow-up; e2 confirms in time and stays silent.
    d = CompactionTripwireDetector(grace_steps=3)
    seq = [
        _ep_event("e1", 0, "compaction", {"pinned": [PIN_A]}),
        _ep_event("e2", 0, "compaction", {"pinned": [PIN_A]}),
        _ep_event("e2", 1, "constraint_present", {"pin": PIN_A}),
        _ep_event("e1", 1, "tool_call"),
        _ep_event("e1", 2, "tool_call"),
        _ep_event("e2", 2, "tool_call"),
        _ep_event("e1", 3, "tool_call"),  # e1's deadline lands here
        _ep_event("e2", 3, "tool_call"),  # e2 has nothing pending
    ]
    risks = _feed(d, seq)
    assert len(risks) == 1
    assert risks[0].episode_id == "e1"
    assert risks[0].step_id == "3"


# --- Healthy sequences: no false positives ------------------------------------


def test_all_pins_confirmed_within_grace_stays_silent():
    d = CompactionTripwireDetector(grace_steps=3)
    risks = _feed(
        d,
        [
            _compaction(0, [PIN_A, PIN_B]),
            _present(1, PIN_A),
            _present(2, PIN_B),
            _plain(3),
            _plain(4),
            _plain(5),
        ],
    )
    assert risks == []


def test_confirmation_on_final_grace_event_is_in_time():
    # Boundary semantics: a confirmation arriving exactly on the deadline
    # event still counts (effects apply before the deadline check).
    d = CompactionTripwireDetector(grace_steps=2)
    risks = _feed(
        d,
        [_compaction(0, [PIN_A]), _plain(1), _present(2, PIN_A), _plain(3)],
    )
    assert risks == []


def test_empty_pinned_list_stays_silent():
    d = CompactionTripwireDetector(grace_steps=2)
    seq: list[StepEvent] = [_event(0, "compaction", {"pinned": []})]
    seq += [_plain(i) for i in range(1, 5)]
    assert _feed(d, seq) == []


def test_constraint_present_without_pending_compaction_is_silent():
    d = CompactionTripwireDetector(grace_steps=3)
    assert _feed(d, [_present(0, PIN_A), _plain(1), _plain(2)]) == []


def test_malformed_metadata_stays_silent_and_never_raises():
    d = CompactionTripwireDetector(grace_steps=2)
    seq = [
        _event(0, "compaction", {}),  # pinned key missing entirely
        _event(1, "compaction", {"pinned": PIN_A}),  # string instead of list
        _event(2, "compaction", {"pinned": [1, None, ""]}),  # no usable pins
        _compaction(3, [PIN_A, 7, None]),  # one real pin among junk entries
        _event(4, "constraint_present", {}),  # pin key missing
        _plain(5),  # deadline event with the pin still pending: fires
        _present(6, PIN_A),  # too late to retract; must not raise or re-fire
        _plain(7),
    ]
    risks = _feed(d, seq)
    assert len(risks) == 1
    assert risks[0].trigger == "governance_decay"


def test_reset_clears_pending_window():
    d = CompactionTripwireDetector(grace_steps=2)
    _feed(d, [_compaction(0, [PIN_A]), _plain(1)])
    d.reset("ep")
    assert _feed(d, [_plain(i) for i in range(2, 6)]) == []


# --- Configuration and wiring --------------------------------------------------


def test_grace_steps_must_be_positive():
    with pytest.raises(ValueError):
        CompactionTripwireDetector(grace_steps=0)


def test_config_supplies_default_grace():
    cfg = Config(compaction_tripwire_enabled=True, compaction_tripwire_grace_steps=2)
    d = CompactionTripwireDetector(config=cfg)
    assert d.grace_steps == 2
    risks = _feed(d, [_compaction(0, [PIN_A]), _plain(1), _plain(2)])
    assert len(risks) == 1
    assert risks[0].step_id == "2"


def test_opt_in_wiring_in_monitor_default():
    off = Monitor.default(Config(), sinks=[RecordingSink()])
    assert "governance_decay" not in {d.name for d in off._detectors}
    on = Monitor.default(
        Config(compaction_tripwire_enabled=True), sinks=[RecordingSink()]
    )
    assert "governance_decay" in {d.name for d in on._detectors}


def test_zero_dep_preset_ignores_compaction_events_end_to_end():
    # Default-off means even an explicit decay sequence stays silent through
    # the full monitor: the zero-dependency preset keeps its bench numbers.
    # Tool names vary per step so the unrelated loop detector stays quiet too.
    mon = Monitor.default(Config(), sinks=[RecordingSink()])
    mon.ingest(_compaction(0, [PIN_A]))
    for i in range(1, 6):
        e = _plain(i)
        mon.ingest(
            StepEvent(
                step_id=e.step_id,
                episode_id=e.episode_id,
                timestamp=e.timestamp,
                action_type=e.action_type,
                action_signature=make_signature("tool_call", f"tool-{i}"),
                tool_name=f"tool-{i}",
            )
        )
    assert mon._sinks[0].risks == []


# --- Fixture replay parity ------------------------------------------------------


def test_replay_fixture_fires_exactly_once():
    sink = RecordingSink()
    mon = Monitor.default(Config(compaction_tripwire_enabled=True), sinks=[sink])
    n = replay(os.path.join(FIX, "injected_governance_decay.jsonl"), monitor=mon)
    assert n == 6
    gov = [r for r in sink.risks if r.trigger == "governance_decay"]
    assert len(gov) == 1
    assert gov[0].step_id == "4"
    assert PIN_A[:16] in gov[0].detail
    assert PIN_B[:16] not in gov[0].detail


def test_replay_healthy_fixture_has_no_governance_risks():
    sink = RecordingSink()
    mon = Monitor.default(Config(compaction_tripwire_enabled=True), sinks=[sink])
    replay(os.path.join(FIX, "healthy_run.jsonl"), monitor=mon)
    assert not any(r.trigger == "governance_decay" for r in sink.risks)
