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


class _FakeClock:
    """A monotonic clock the test drives explicitly.

    Both sweep tests below turn on *when* the sweep runs relative to the
    cooldown window, which real elapsed time cannot express reliably. The bound
    test used to assume 2,000 emits complete inside a 0.05s window -- a slow
    machine crossed it, swept early, and failed a correct implementation -- and
    the live-key test never crossed the sweep threshold at all, so it passed
    whether or not the sweep dropped keys that were still suppressing.
    """

    def __init__(self, now: float = 1_000.0) -> None:
        self._now = now

    def __call__(self) -> float:
        return self._now

    def advance(self, seconds: float) -> None:
        self._now += seconds


def test_dedup_table_does_not_grow_without_bound(monkeypatch):
    """Regression: the cooldown table was never swept, so with the default
    ``(episode_id, trigger, severity)`` key a long-lived monitor retained one
    entry per episode id for the life of the process -- unbounded growth in the
    sink whose whole job is production alerting hygiene.
    """
    clock = _FakeClock()
    # Patched before construction: __init__ seeds ``_swept`` from this clock.
    monkeypatch.setattr(time, "monotonic", clock)
    sink = DedupSink(_RecordingSink(), cooldown_seconds=30.0)
    for i in range(2000):
        sink.emit(_risk(0.9, episode_id=f"old-{i}"))
    # No time has passed, so every key is still load-bearing.
    assert len(sink._last) == 2000
    clock.advance(31.0)  # the whole burst is now past its cooldown
    for i in range(10):
        sink.emit(_risk(0.9, episode_id=f"new-{i}"))
    assert len(sink._last) == 10, f"expired keys survived: {len(sink._last)}"
    assert all(k[0].startswith("new-") for k in sink._last)


def test_dedup_sweep_keeps_suppressing_live_keys(monkeypatch):
    """Sweeping must drop only keys that can no longer suppress anything: a key
    still inside its cooldown has to keep working *after* the sweep has rebuilt
    the table. The sweep therefore has to actually run for this to prove
    anything, which is what the previous version of this test never did.
    """
    clock = _FakeClock()
    monkeypatch.setattr(time, "monotonic", clock)
    inner = _RecordingSink()
    sink = DedupSink(inner, cooldown_seconds=30.0)

    sink.emit(_risk(0.9, episode_id="expired"))  # t+0, will age out
    clock.advance(29.0)
    sink.emit(_risk(0.9, episode_id="live"))  # t+29, 30s of cooldown still ahead
    clock.advance(2.0)
    # t+31 is one full cooldown past construction, so this emit sweeps.
    sink.emit(_risk(0.9, episode_id="sweeper"))
    assert {k[0] for k in sink._last} == {"live", "sweeper"}, "wrong keys swept"
    assert len(inner.emitted) == 3

    # "live" is 2s into a 30s cooldown and survived the sweep, so its repeat is
    # still suppressed. This is the assertion a sweep that evicts live keys
    # fails, and the reason the sweep above had to be triggered deliberately.
    sink.emit(_risk(0.9, episode_id="live"))
    assert len(inner.emitted) == 3, "the sweep dropped a key that was still live"
    # "expired" was swept because its window genuinely elapsed -- it alerts again.
    sink.emit(_risk(0.9, episode_id="expired"))
    assert len(inner.emitted) == 4


class _SweepCountingDedup(DedupSink):
    """``DedupSink`` that counts sweeps, so a test can assert the pass-through
    path never runs one. The end state alone cannot show this: without the
    guard, a non-positive cooldown makes every sweep cutoff ``>= now``, which
    empties the table on each emit and leaves it looking identical to never
    having recorded anything.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.sweeps = 0

    def _sweep(self, now: float) -> None:
        self.sweeps += 1
        super()._sweep(now)


def test_dedup_non_positive_cooldown_is_pass_through():
    """With no window nothing can be a repeat, so suppression is disabled. The
    sweep must not run either: its guard ``(now - _swept) >= cooldown`` is
    unconditionally true for a non-positive cooldown, which turned an O(1) path
    into an O(len) rebuild under the lock on *every* emit, stalling concurrent
    emitters on what is meant to be a pass-through.
    """
    for cooldown in (0.0, -5.0):
        inner = _RecordingSink()
        sink = _SweepCountingDedup(inner, cooldown_seconds=cooldown)
        for _ in range(100):
            sink.emit(_risk(0.9))  # the same key every time
        assert len(inner.emitted) == 100, f"suppressed with cooldown={cooldown}"
        assert sink.sweeps == 0, (
            f"{sink.sweeps} sweeps on a pass-through sink (cooldown={cooldown})"
        )
        assert sink._last == {}, f"table grew with cooldown={cooldown}"


def test_dedup_raising_key_fn_fails_open():
    """The class documents fail-open, but ``key_fn`` ran outside every handler,
    so a raising one propagated straight out of ``emit`` -- losing the alert
    and, used standalone rather than under ``Monitor``, raising into the host's
    ingest path.
    """
    inner = _RecordingSink()

    def boom(risk: FailureRisk) -> object:
        raise ValueError("key_fn is broken")

    sink = DedupSink(inner, cooldown_seconds=300.0, key_fn=boom)
    sink.emit(_risk(0.9))  # must not raise
    sink.emit(_risk(0.9))
    # Without a key there is no way to tell a repeat from a new finding, and
    # fail-open resolves that by delivering rather than dropping.
    assert len(inner.emitted) == 2


def test_dedup_unhashable_key_fails_open():
    """Same contract, different failure mode: an unhashable key raises inside
    the table lookup rather than in ``key_fn`` itself.
    """
    inner = _RecordingSink()
    sink = DedupSink(inner, cooldown_seconds=300.0, key_fn=lambda r: [r.episode_id])
    sink.emit(_risk(0.9))  # must not raise
    assert len(inner.emitted) == 1


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
