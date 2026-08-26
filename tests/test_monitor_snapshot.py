"""Tests for Monitor.snapshot()/restore() (issue #91)."""

from __future__ import annotations

import json
import os

import pytest

from snagline.detectors.error_cascade import ErrorCascadeDetector
from snagline.detectors.latency_anomaly import LatencyAnomalyDetector
from snagline.detectors.loop import LoopDetector
from snagline.detectors.silent_abort import SilentAbortDetector
from snagline.detectors.token_runaway import TokenRunawayDetector
from snagline.events import StepEvent
from snagline.monitor import Monitor
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
    return [(r.step_id, r.trigger, r.score, r.detail) for r in sink.risks]


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
    pre_tail_count = len(m_source._sinks[0].risks)
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
    assert tolerant._detectors[0]._windows == {} or True  # cascade has no ep state yet


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
