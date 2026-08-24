"""Tests for FailureRisk severity and the DedupSink cooldown (P1, issue #4)."""

from __future__ import annotations

import time

from snagline.risk import (
    SEVERITY_CRITICAL,
    SEVERITY_INFO,
    SEVERITY_WARNING,
    FailureRisk,
    severity_from_score,
)
from snagline.sinks.dedup import DedupSink


def _risk(score: float, episode_id: str = "ep", **kw) -> FailureRisk:
    return FailureRisk(
        episode_id=episode_id,
        step_id="s1",
        score=score,
        trigger="loop",
        detail="repeating loop",
        timestamp=0.0,
        **kw,
    )


def test_severity_from_score_bands():
    assert severity_from_score(0.9) == SEVERITY_CRITICAL
    assert severity_from_score(0.6) == SEVERITY_WARNING
    assert severity_from_score(0.2) == SEVERITY_INFO


def test_risk_default_severity_derived_from_score():
    assert _risk(0.9).severity == SEVERITY_CRITICAL
    assert _risk(0.6).severity == SEVERITY_WARNING
    assert _risk(0.2).severity == SEVERITY_INFO


def test_risk_explicit_severity_kept():
    # When the caller sets severity explicitly, it is preserved exactly.
    assert _risk(0.9, severity=SEVERITY_INFO).severity == SEVERITY_INFO


class _RecordingSink:
    def __init__(self):
        self.emitted: list[FailureRisk] = []

    def emit(self, risk: FailureRisk) -> None:
        self.emitted.append(risk)


def test_dedup_suppresses_within_cooldown():
    inner = _RecordingSink()
    sink = DedupSink(inner, cooldown_seconds=10.0)
    sink.emit(_risk(0.9))
    sink.emit(_risk(0.9))  # same key, within cooldown -> suppressed
    assert len(inner.emitted) == 1


def test_dedup_emits_after_cooldown():
    inner = _RecordingSink()
    sink = DedupSink(inner, cooldown_seconds=0.05)
    sink.emit(_risk(0.9))
    time.sleep(0.08)
    sink.emit(_risk(0.9))  # cooldown elapsed -> re-emit
    assert len(inner.emitted) == 2


def test_dedup_distinct_keys_not_suppressed():
    inner = _RecordingSink()
    sink = DedupSink(inner, cooldown_seconds=100.0)
    sink.emit(_risk(0.9, episode_id="ep1"))
    sink.emit(_risk(0.9, episode_id="ep2"))
    assert len(inner.emitted) == 2


def test_dedup_custom_key_fn():
    inner = _RecordingSink()
    # Dedupe only by trigger, ignoring episode/severity.
    sink = DedupSink(inner, cooldown_seconds=100.0, key_fn=lambda r: r.trigger)
    sink.emit(_risk(0.9, episode_id="ep1"))
    sink.emit(_risk(0.2, episode_id="ep2"))  # different episode/severity, same trigger
    assert len(inner.emitted) == 1  # suppressed by custom key


def test_dedup_table_does_not_grow_without_bound():
    """Regression: the cooldown table was never swept, so with the default
    ``(episode_id, trigger, severity)`` key a long-lived monitor retained one
    entry per episode id for the life of the process -- unbounded growth in the
    sink whose whole job is production alerting hygiene.
    """
    sink = DedupSink(_RecordingSink(), cooldown_seconds=0.05)
    for i in range(2000):
        sink.emit(_risk(0.9, episode_id=f"old-{i}"))
    # Nothing has expired yet, so every key is still load-bearing.
    assert len(sink._last) == 2000
    time.sleep(0.08)  # the whole burst is now past its cooldown
    for i in range(10):
        sink.emit(_risk(0.9, episode_id=f"new-{i}"))
    assert len(sink._last) == 10, f"expired keys survived: {len(sink._last)}"
    assert all(k[0].startswith("new-") for k in sink._last)


def test_dedup_sweep_keeps_suppressing_live_keys():
    """Sweeping must only drop keys that can no longer suppress anything: every
    key still inside its cooldown has to keep working.
    """
    inner = _RecordingSink()
    sink = DedupSink(inner, cooldown_seconds=30.0)
    for i in range(500):
        sink.emit(_risk(0.9, episode_id=f"ep-{i}"))
    for i in range(500):
        sink.emit(_risk(0.9, episode_id=f"ep-{i}"))  # all repeats, all in window
    assert len(inner.emitted) == 500, "a repeat leaked through after sweeping"


def test_dedup_cooldown_survives_a_backward_wall_clock_step():
    """Regression: the cooldown was measured on ``time.time()``. A backward
    wall-clock step (NTP correction, an operator setting the date) made
    ``now - last`` negative, so the key stayed suppressed until real time caught
    up -- an alert-suppression wrapper silently going quiet. Elapsed time is now
    read from the monotonic clock, which no clock adjustment can move.
    """
    inner = _RecordingSink()
    sink = DedupSink(inner, cooldown_seconds=0.05)
    sink.emit(_risk(0.9))  # recorded against whichever clock the sink reads
    assert len(inner.emitted) == 1

    # The clock now steps a day backwards, mid-cooldown.
    real_time, real_monotonic = time.time, time.monotonic
    time.time = lambda: real_time() - 86_400.0
    try:
        time.sleep(0.08)  # the cooldown has genuinely elapsed in real time
        sink.emit(_risk(0.9))
    finally:
        time.time = real_time
    assert time.monotonic is real_monotonic  # the fix must not patch the clock
    assert len(inner.emitted) == 2, "alert stayed suppressed after a clock step"
