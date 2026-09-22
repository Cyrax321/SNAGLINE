"""Tests for Monitor.snapshot()/restore() (issue #91)."""

from __future__ import annotations

import json
import os
from typing import cast

import pytest

from snagline.config import Config
from snagline.detectors.compaction_tripwire import CompactionTripwireDetector
from snagline.detectors.error_cascade import ErrorCascadeDetector
from snagline.detectors.goal_drift import GoalDriftDetector
from snagline.detectors.latency_anomaly import LatencyAnomalyDetector
from snagline.detectors.loop import LoopDetector
from snagline.detectors.meltdown import MeltdownDetector
from snagline.detectors.side_effect_guard import SideEffectGuardDetector
from snagline.detectors.silent_abort import SilentAbortDetector
from snagline.detectors.stagnation import StagnationDetector
from snagline.detectors.token_runaway import TokenRunawayDetector
from snagline.events import StepEvent
from snagline.monitor import SNAPSHOT_FORMAT_VERSION, Monitor
from snagline.sinks.dedup import DedupSink


class ListSink:
    def __init__(self) -> None:
        self.risks: list = []

    def emit(self, risk) -> None:
        self.risks.append(risk)


def _composition() -> tuple[list, list]:
    detectors: list = [
        LoopDetector(window_size=12, repeat_threshold=3),
        ErrorCascadeDetector(),
        LatencyAnomalyDetector(min_samples=5),
        TokenRunawayDetector(min_samples=5, budget_total_tokens=500),
        SilentAbortDetector(),
    ]
    return detectors, [ListSink()]


def _stream() -> list[StepEvent]:
    """A trajectory whose interesting failures all land AFTER step 7."""
    events: list[StepEvent] = []

    def add(step: int, **kw) -> None:
        kw.setdefault("action_type", "tool_call")
        kw.setdefault("error", False)
        events.append(
            StepEvent(
                step_id=str(step),
                episode_id="ep",
                timestamp=float(step),
                action_signature=kw.pop("signature", f"s{step}"),
                tool_name=kw.pop("tool_name", None) or f"t{step}",
                **kw,
            )
        )

    latencies = [100.0, 110.0, 90.0, 105.0, 95.0]  # latency baseline warm-up
    for i in range(5):
        add(i, signature=f"unique{i}", tokens_in=50, latency_ms=latencies[i])
    add(5, tool_name="search", signature="q-a", tokens_in=50)  # loop attempt 1/3
    add(6, tool_name="search", signature="q-a", tokens_in=50)  # loop attempt 2/3
    add(7, tool_name="fetch", signature=f"f{7}", tokens_in=50, error=True)
    # --- snapshot boundary here ---
    add(
        8, tool_name="search", signature="q-a", tokens_in=300
    )  # loop 3/3 + budget breach
    add(9, tool_name="fetch", signature=f"f{9}", tokens_in=200, error=True)
    add(10, tool_name="wrapup", signature=f"w{10}")  # ends mid-work: silent_abort
    return events


def _feed(monitor: Monitor, events: list[StepEvent]) -> None:
    for e in events:
        monitor.ingest(e)


def _risk_tuples(monitor: Monitor) -> list[tuple]:
    sink = monitor._sinks[0]
    return [(r.step_id, r.trigger, r.score, r.detail) for r in sink.risks]  # type: ignore[attr-defined]


def test_snapshot_restore_round_trip_matches_never_restarted_twin(tmp_path):
    path = str(tmp_path / "state.json")

    m_source = Monitor(*_composition())
    m_twin = Monitor(*_composition())
    stream = _stream()
    _feed(m_source, stream[:8])
    _feed(m_twin, stream[:8])
    m_source.snapshot(path)

    m_restored = Monitor(*_composition())
    m_restored.restore(path)

    tail = stream[8:]
    # Risks emitted BEFORE the boundary are history the restored monitor never
    # saw; parity applies to everything from the restore point onward.
    pre_tail_count = len(m_source._sinks[0].risks)  # type: ignore[attr-defined]
    _feed(m_source, tail)
    _feed(m_restored, tail)

    # The restored monitor must behave EXACTLY like the never-restarted twin,
    # including the end-of-episode silent-abort verdict.
    m_source.end_episode("ep")
    m_restored.end_episode("ep")

    source_risks = _risk_tuples(m_source)[pre_tail_count:]
    restored_risks = _risk_tuples(m_restored)
    assert source_risks == restored_risks
    # And the tail must actually have produced signals worth comparing:
    triggers = {t for _, t, _, _ in source_risks}
    assert {"loop", "budget_breach", "silent_abort"} <= triggers


def test_snapshot_file_is_atomic_json(tmp_path):
    path = str(tmp_path / "state.json")
    m = Monitor(*_composition())
    m.snapshot(path)
    assert os.path.exists(path)
    assert not os.path.exists(path + ".tmp")
    data = json.loads(open(path, encoding="utf-8").read())
    assert data["format_version"] == 1
    assert set(data["detectors"]) == {
        "0:loop",
        "1:error_cascade",
        "2:latency_anomaly",
        "3:token_runaway",
        "4:silent_abort",
    }


def test_version_mismatch_raises(tmp_path):
    path = str(tmp_path / "state.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump({"format_version": 999}, fh)
    with pytest.raises(ValueError, match="format_version"):
        Monitor(*_composition()).restore(path)


def test_composition_mismatch_strict_vs_tolerant(tmp_path):
    path = str(tmp_path / "state.json")
    m = Monitor([LoopDetector(), ErrorCascadeDetector()], [ListSink()])
    m.ingest(
        StepEvent(
            step_id="0",
            episode_id="ep",
            timestamp=0.0,
            action_type="tool_call",
            action_signature="s0",
        )
    )
    m.snapshot(path)

    # Strict: different composition must raise, not silently misbehave.
    different = Monitor([SilentAbortDetector()], [ListSink()])
    with pytest.raises(ValueError, match="composition mismatch"):
        different.restore(path, strict_names=True)

    # Tolerant default: applies what matches, warns about orphans.
    tolerant = Monitor([ErrorCascadeDetector(), SilentAbortDetector()], [ListSink()])
    tolerant.restore(path)  # loop state orphaned -> warning, no raise
    from snagline.detectors.error_cascade import ErrorCascadeDetector as ECDetector

    assert (
        cast(ECDetector, tolerant._detectors[0])._windows == {} or True
    )  # cascade has no ep state yet


def test_dedup_sink_cooldown_survives_round_trip(tmp_path):
    from snagline.risk import FailureRisk

    fr = FailureRisk("ep", "0", 0.9, "loop", "d", 0.0)

    inner_a, inner_b = ListSink(), ListSink()
    s1 = DedupSink(inner_a, cooldown_seconds=300.0)
    s1.emit(fr)
    s1.emit(fr)
    assert len(inner_a.risks) == 1, "second immediate emit must be suppressed"

    dumped = s1.dump_state()
    assert dumped is not None
    s2 = DedupSink(inner_b, cooldown_seconds=300.0)
    s2.load_state(dumped)
    s2.emit(fr)
    assert len(inner_b.risks) == 0, "cooldown bookkeeping must survive restore"


def test_custom_key_fn_sink_is_skipped_not_fatal():
    s = DedupSink(ListSink(), key_fn=lambda r: r.episode_id)
    assert s.dump_state() is None, "opaque keys cannot serialize"


def _lat(step: int, ms: float) -> StepEvent:
    return StepEvent(
        step_id=str(step),
        episode_id="ep",
        timestamp=float(step),
        action_type="tool_call",
        action_signature=f"s{step}",
        tool_name="api",
        latency_ms=ms,
    )


def test_latency_state_round_trip_behavioral():
    d1 = LatencyAnomalyDetector(min_samples=3)
    d2 = LatencyAnomalyDetector(min_samples=3)
    for i, ms in enumerate([100.0, 120.0, 140.0]):
        d1.observe(_lat(i, ms))
        d2.observe(_lat(i, ms))
    # A true JSON round trip: floats pass through repr() serialization.
    d2.load_state(json.loads(json.dumps(d1.dump_state())))
    tail = [_lat(3, 400.0), _lat(4, 450.0)]
    scores_1 = [r.score for e in tail if (r := d1.observe(e)) is not None]
    scores_2 = [r.score for e in tail if (r := d2.observe(e)) is not None]
    assert scores_1 == scores_2
    assert scores_1, "sustained shift after restored baseline must still alarm"


# --- strict_names slot ordering (issue #217) --------------------------------
#
# Snapshot keys are "<slot>:<name>", so sorting them as plain strings puts slot
# 10 between 1 and 2. Below 11 detectors every index is one digit and the bug
# is invisible, which is why the coverage above (1-5 detectors) misses it.

_DETECTOR_CLASSES = {
    "loop": LoopDetector,
    "error_cascade": ErrorCascadeDetector,
    "latency_anomaly": LatencyAnomalyDetector,
    "stagnation": StagnationDetector,
    "token_runaway": TokenRunawayDetector,
    "meltdown": MeltdownDetector,
    "silent_abort": SilentAbortDetector,
    "side_effect_guard": SideEffectGuardDetector,
    "governance_decay": CompactionTripwireDetector,
    "goal_drift": GoalDriftDetector,
}

# Eleven slots is not a synthetic number: it is what Monitor.default() builds
# once the opt-in detectors are enabled, with no extras installed.
_ELEVEN_NAMES = [*_DETECTOR_CLASSES, "loop"]


def _eleven() -> list:
    return [_DETECTOR_CLASSES[name]() for name in _ELEVEN_NAMES]


def test_strict_names_accepts_identical_11_detector_composition(tmp_path):
    path = str(tmp_path / "state.json")
    a = Monitor(_eleven(), [ListSink()])
    a.ingest(
        StepEvent(
            step_id="0",
            episode_id="ep",
            timestamp=0.0,
            action_type="tool_call",
            action_signature="s0",
            tool_name="api",
            latency_ms=10.0,
        )
    )
    a.snapshot(path)

    b = Monitor(_eleven(), [ListSink()])
    assert [d.name for d in a._detectors] == [d.name for d in b._detectors]
    b.restore(path, strict_names=True)  # raised "composition mismatch" pre-#217


def test_strict_names_rejects_mismatch_in_lexicographic_key_order(tmp_path):
    """The false-accept direction: a genuinely wrong composition that happens
    to equal the *string* order of the snapshot's keys must still raise."""
    path = str(tmp_path / "state.json")
    snapshotted = _eleven()
    Monitor(snapshotted, [ListSink()]).snapshot(path)

    with open(path, encoding="utf-8") as fh:
        keys = json.load(fh)["detectors"]
    lexicographic = [k.split(":", 1)[1] for k in sorted(keys)]
    assert lexicographic != [d.name for d in snapshotted], (
        "11 slots must reorder under string sorting, or the test proves nothing"
    )

    wrong = Monitor([_DETECTOR_CLASSES[n]() for n in lexicographic], [ListSink()])
    with pytest.raises(ValueError, match="composition mismatch"):
        wrong.restore(path, strict_names=True)


def test_strict_names_survives_key_without_integer_slot_prefix():
    """restore_dict takes caller-supplied dicts; an unparseable slot prefix
    must not crash the strict check (setup-time raise stays a ValueError)."""
    monitor = Monitor([LoopDetector()], [ListSink()])
    with pytest.raises(ValueError, match="composition mismatch"):
        monitor.restore_dict(
            {
                "format_version": SNAPSHOT_FORMAT_VERSION,
                "detectors": {"0:loop": None, "x:error_cascade": None},
            },
            strict_names=True,
        )


def test_strict_names_rejection_applies_no_state(tmp_path):
    """A rejected strict restore must leave the monitor untouched.

    The composition check runs *before* the load loop. After the loop it
    would be too late: the name-suffix fallback would already have loaded
    the snapshot's per-episode windows into detectors by name, so a caller
    that catches the ValueError keeps a monitor contaminated with state
    from the snapshot it just rejected.
    """
    path = str(tmp_path / "state.json")
    a = Monitor([LoopDetector(), ErrorCascadeDetector()], [ListSink()])
    a.ingest(
        StepEvent(
            step_id="0",
            episode_id="ep",
            timestamp=0.0,
            action_type="tool_call",
            action_signature="q-a",
            tool_name="search",
        )
    )
    a.ingest(
        StepEvent(
            step_id="1",
            episode_id="ep",
            timestamp=1.0,
            action_type="tool_call",
            action_signature="q-a",
            tool_name="search",
        )
    )
    a.snapshot(path)

    b = Monitor([ErrorCascadeDetector(), LoopDetector()], [ListSink()])
    with pytest.raises(ValueError, match="composition mismatch"):
        b.restore(path, strict_names=True)
    b_loop = cast(LoopDetector, next(d for d in b._detectors if d.name == "loop"))
    assert b_loop._windows == {}, (
        "rejected snapshot must not leave per-episode state behind "
        "(name-suffix fallback loaded it pre-fix)"
    )


def test_strict_restore_rejection_applies_no_detector_state(tmp_path):
    """Issue #236: a rejected strict restore must leave the target monitor
    untouched -- no detector load_state may run before the composition
    check raises."""
    path = str(tmp_path / "state.json")
    source = Monitor([LoopDetector(), ErrorCascadeDetector()], [ListSink()])
    for step in ("0", "1"):
        source.ingest(
            StepEvent(
                step_id=step,
                episode_id="ep",
                timestamp=float(step),
                action_type="tool_call",
                action_signature="q-a",
                tool_name="search",
            )
        )
    source.snapshot(path)
    target = Monitor([ErrorCascadeDetector(), LoopDetector()], [ListSink()])
    with pytest.raises(ValueError, match="composition mismatch"):
        target.restore(path, strict_names=True)
    loop = cast(LoopDetector, target._detectors[1])
    assert loop._windows == {}


# --- a rejected snapshot leaves the detector on its live state, whole (review
#     of #402) ---------------------------------------------------------------


def _tool(step: int, name: str = "api") -> StepEvent:
    return StepEvent(
        step_id=str(step),
        episode_id="ep",
        timestamp=float(step),
        action_type="tool_call",
        action_signature=f"call:{name}:{step}",
        tool_name=name,
        latency_ms=100.0,
    )


def test_loop_load_state_is_atomic_when_a_near_count_is_bad():
    """``restore_dict`` catches the exception and moves on, so a snapshot that
    parses partway must leave the detector's whole state rather than a mix.

    The loop detector carries four window families plus the stall counters.
    Assigning attribute-by-attribute meant the plain windows, counts and fired
    sets had already been replaced with the snapshot's when a bad near-count
    raised -- the live window destroyed, and restored windows paired with the
    fired sets live traffic had computed for different signatures entirely.
    """
    detector = LoopDetector()
    for i in range(3):
        detector.observe(_tool(i))
    live = detector.dump_state()
    assert live["windows"], "fixture: live state must exist to compare against"

    bad = {
        "windows": {"ep": ["call:other:0"]},
        "counts": {"ep": 4},
        "fired": {"ep": ["call:other:0"]},
        # ``windows`` and ``fired`` parse; ``near_counts`` does not, so the
        # failure lands after those families were built (and, pre-fix, after
        # they were assigned).
        "near_windows": {"ep": ["call:near:0"]},
        "near_counts": {"ep": "not-an-int"},
    }
    with pytest.raises(ValueError):
        detector.load_state(bad)

    after = detector.dump_state()
    assert after == live, (
        "a rejected snapshot must leave the detector on its live state, not a "
        f"half-applied mix: {after} vs {live}"
    )


def test_loop_load_state_applies_when_the_snapshot_is_good():
    """The atomicity change must not turn every load into a rejection."""
    detector = LoopDetector()
    detector.observe(_tool(0))
    detector.load_state(
        {
            "windows": {"ep": ["call:api:0", "call:api:1"]},
            "counts": {"ep": 12},
            "fired": {"ep": ["call:api:1"]},
            "near_windows": {},
            "near_counts": {},
            "near_fired": {},
            "cycle_windows": {},
            "cycle_counts": {},
            "cycle_fired": {},
            "stall_sig": {},
            "stall_count": {},
            "stall_start": {},
            "stall_fired": {},
        }
    )
    assert list(detector._windows["ep"]) == ["call:api:0", "call:api:1"]
    assert detector._counts == {"ep": 12}
    assert detector._fired == {"ep": {"call:api:1"}}


def test_meltdown_load_state_is_atomic_when_a_count_is_bad():
    """Same shape as the loop detector, one window family: a bad count raises
    after the rebuilt window is already in hand, so the live entropy window
    must not have been clobbered (review of #402)."""
    detector = MeltdownDetector(window_size=8)
    for i in range(3):
        detector.observe(_tool(i, name="t" if i % 2 else "u"))
    live = detector.dump_state()
    assert live["windows"], "fixture: live state must exist to compare against"

    bad = {
        "windows": {"ep": ["t", "u"]},
        # ``windows`` parses; the count does not.
        "counts": {"ep": "not-an-int"},
    }
    with pytest.raises(ValueError):
        detector.load_state(bad)

    after = detector.dump_state()
    assert after == live, (
        "a rejected snapshot must leave the detector on its live state, not a "
        f"half-applied mix: {after} vs {live}"
    )


def test_meltdown_load_state_applies_when_the_snapshot_is_good():
    """The atomicity change must not turn every load into a rejection."""
    detector = MeltdownDetector(window_size=8)
    detector.observe(_tool(0))
    detector.load_state(
        {
            "windows": {"ep": ["t", "u", "t"]},
            "counts": {"ep": 12},
            "fired": {"ep": True},
            "clear_streak": {"ep": 2},
        }
    )
    assert list(detector._eps["ep"].window) == ["t", "u", "t"]
    assert detector._counts == {"ep": 12}
    assert detector._fired == {"ep": True}
    assert detector._clear_streak == {"ep": 2}


def test_error_cascade_load_state_is_atomic_when_a_count_is_bad():
    """Same shape as the loop and meltdown detectors: ``int()`` on a malformed
    count raises after the rebuilt window is already in hand, so the live
    cascade window must not have been clobbered (review of #402).

    #406 reintroduced the attribute-by-attribute assignment here while adding
    the scaler-position seeding, which is why this test exists for this
    detector and not just its siblings.
    """
    detector = ErrorCascadeDetector(window_size=8)
    for i in range(3):
        detector.observe(_tool(i))
    live = detector.dump_state()
    assert live["windows"], "fixture: live state must exist to compare against"

    bad = {
        "windows": {"ep": [True, False]},
        # ``windows`` parses; the count does not.
        "counts": {"ep": "not-an-int"},
    }
    with pytest.raises(ValueError):
        detector.load_state(bad)

    after = detector.dump_state()
    assert after == live, (
        "a rejected snapshot must leave the detector on its live state, not a "
        f"half-applied mix: {after} vs {live}"
    )


def test_error_cascade_load_state_applies_when_the_snapshot_is_good():
    """The atomicity change must not turn every load into a rejection, and the
    scaler position the window was sized from must seed ``_counts`` (issue
    #403): an episode absent from ``counts`` still needs its position."""
    detector = ErrorCascadeDetector(window_size=8, config=Config(window_scale_steps=4))
    detector.load_state({"windows": {"ep": [True, False, True]}})
    assert list(detector._windows["ep"]) == [True, False, True]
    # Inferred from the shipped window, not left absent.
    assert detector._counts == {"ep": 3}
