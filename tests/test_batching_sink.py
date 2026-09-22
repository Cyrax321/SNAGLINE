"""Tests for the BatchingSink (P1 item 5: batched/rate-limited dispatch)."""

from __future__ import annotations

import logging
import threading
import time

from snagline.risk import SEVERITY_INFO, FailureRisk
from snagline.sinks.batching import BatchingSink


def _risk(i: int = 0) -> FailureRisk:
    return FailureRisk(
        episode_id="ep",
        step_id=str(i),
        score=0.5,
        trigger="loop",
        detail="d",
        timestamp=1.0,
        severity=SEVERITY_INFO,
    )


class _RecordingSink:
    def __init__(self):
        self.emitted: list[FailureRisk] = []

    def emit(self, risk: FailureRisk) -> None:
        self.emitted.append(risk)


def test_batch_collects_and_flushes():
    inner = _RecordingSink()
    sink = BatchingSink(inner, max_batch=1000, flush_interval=5.0)
    try:
        for i in range(3):
            sink.emit(_risk(i))
        sink.flush_now()
        assert len(inner.emitted) == 3
    finally:
        sink.close()


def test_emit_is_non_blocking_and_background_flushes():
    inner = _RecordingSink()
    sink = BatchingSink(inner, max_batch=2, flush_interval=0.05)
    try:
        for i in range(5):
            sink.emit(_risk(i))
        # Wait for the background thread to flush at least once.
        for _ in range(50):
            if len(inner.emitted) >= 1:
                break
            time.sleep(0.02)
    finally:
        sink.close()
    assert len(inner.emitted) >= 1


def test_rate_limit_paces_delivery():
    inner = _RecordingSink()
    # 10 per second -> 3 risks take >= 0.2s to deliver.
    sink = BatchingSink(inner, max_batch=1000, flush_interval=5.0, max_per_second=10.0)
    try:
        import time

        start = time.monotonic()
        for i in range(3):
            sink.emit(_risk(i))
        sink.flush_now()
        elapsed = time.monotonic() - start
        assert elapsed >= 0.18
        assert len(inner.emitted) == 3
    finally:
        sink.close()


def test_rate_limit_persists_across_batches(monkeypatch):
    inner = _RecordingSink()
    now = [100.0]
    sleeps: list[float] = []

    def monotonic() -> float:
        return now[0]

    def sleep(seconds: float) -> None:
        sleeps.append(seconds)
        now[0] += seconds

    monkeypatch.setattr("snagline.sinks.batching.time.monotonic", monotonic)
    monkeypatch.setattr("snagline.sinks.batching.time.sleep", sleep)
    sink = BatchingSink(
        inner, max_batch=1000, flush_interval=3600.0, max_per_second=10.0
    )
    try:
        for i in range(3):
            sink.emit(_risk(i))
            sink.flush_now()
        assert len(inner.emitted) == 3
        assert sleeps == [0.1, 0.1]
    finally:
        sink.close()


def _wait_for(predicate, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.01)


def test_max_batch_flushes_without_an_interval_tick():
    # Issue #65: max_batch was stored and never read, so flush_interval was the
    # only trigger and the queue was unbounded between ticks. Deliberately no
    # flush_now() here -- an interval tick is an hour away, so the only thing
    # that can deliver is the max_batch threshold.
    inner = _RecordingSink()
    sink = BatchingSink(inner, max_batch=3, flush_interval=3600.0)
    try:
        for i in range(6):
            sink.emit(_risk(i))
        _wait_for(lambda: len(inner.emitted) == 6)
        assert len(inner.emitted) == 6
    finally:
        sink.close()


def test_close_drains_queued_risks():
    # Issue #65: close() set the stop event and joined, which made _run exit
    # without a final flush, so every risk enqueued since the last tick was
    # dropped. No flush_now() -- close() alone must deliver.
    inner = _RecordingSink()
    sink = BatchingSink(inner, max_batch=100, flush_interval=3600.0)
    for i in range(5):
        sink.emit(_risk(i))
    sink.close()
    assert len(inner.emitted) == 5


def test_emit_stays_non_blocking_when_the_batch_is_full():
    # The threshold flush must happen on the background thread: emit runs inside
    # ingest() on the host's thread and must not inherit the sink's latency.
    class _SlowSink:
        def __init__(self):
            self.emitted: list[FailureRisk] = []

        def emit(self, risk: FailureRisk) -> None:
            time.sleep(0.2)
            self.emitted.append(risk)

    inner = _SlowSink()
    sink = BatchingSink(inner, max_batch=1, flush_interval=3600.0)
    try:
        start = time.monotonic()
        for i in range(3):
            sink.emit(_risk(i))
        assert time.monotonic() - start < 0.1, "emit must not wait on the sink"
        _wait_for(lambda: len(inner.emitted) == 3)
        assert len(inner.emitted) == 3
    finally:
        sink.close()


# --- close() must not hang when the wrapped sink is stuck (issue #393) --------
# The flusher clears the queue before taking _delivery_lock, so the deadlock
# needs an enqueue *while* it is parked -- exactly what a live alert stream does.


class _ParkedSink:
    """A wrapped sink that parks inside ``emit`` until released, standing in
    for a network sink whose ``urlopen`` never returns: the sink's timeout
    bounds socket ops, not DNS, so an unresolvable host blocks indefinitely."""

    def __init__(self):
        self.parked = threading.Event()
        self.release = threading.Event()
        self.emitted: list[FailureRisk] = []

    def emit(self, risk: FailureRisk) -> None:
        self.parked.set()
        self.release.wait(timeout=10.0)
        self.emitted.append(risk)


def _parked_sink(flush_interval: float = 0.05) -> tuple[BatchingSink, _ParkedSink]:
    inner = _ParkedSink()
    sink = BatchingSink(inner, max_batch=100, flush_interval=flush_interval)
    sink.emit(_risk(0))
    _wait_for(lambda: inner.parked.is_set())
    return sink, inner


def test_close_does_not_hang_when_the_flusher_is_stuck():
    sink, inner = _parked_sink()
    # The flusher cleared the queue before parking; enqueue again so the
    # fallback flush has real work that would block on _delivery_lock.
    sink.emit(_risk(1))
    start = time.monotonic()
    sink.close()
    elapsed = time.monotonic() - start
    assert elapsed < 5.0, "close() must terminate, not hang on the delivery lock"
    # The parked delivery never finished, so the late alert is dropped rather
    # than blocking shutdown indefinitely.
    assert inner.emitted == []


def test_close_reports_undelivered_alerts_when_shutdown_times_out(caplog):
    sink, inner = _parked_sink(flush_interval=0.1)
    sink.emit(_risk(1))
    with caplog.at_level(logging.WARNING, logger="snagline"):
        sink.close()
    assert any("undelivered" in r.getMessage() for r in caplog.records), (
        "a stuck shutdown must be observable, not silent"
    )
    assert inner.emitted == []


def test_close_delivers_after_a_slow_delivery_recovers():
    # The acquire-with-timeout is not "always give up": if the wrapped sink is
    # merely slow and finishes mid-wait, the late alert still gets delivered.
    sink, inner = _parked_sink(flush_interval=0.1)
    sink.emit(_risk(1))
    release_after = threading.Timer(0.3, inner.release.set)
    release_after.start()
    try:
        sink.close()
    finally:
        release_after.cancel()
    # Both the parked batch and the late enqueue were delivered.
    assert len(inner.emitted) == 2


def test_close_does_not_hold_the_delivery_lock_across_its_own_flush(monkeypatch):
    # _flush() -> _deliver() re-acquires _delivery_lock, so close() must let go
    # of it before calling _flush(). Lock is not reentrant: holding it across
    # the call makes the re-acquire block forever, which is close() hanging
    # again -- on a path the join/acquire timeouts do not cover.
    #
    # To reach that path the flusher must still be alive once close()'s join
    # budget expires, so the wrapped sink delivers slower than that budget;
    # the lock is then free only after it finishes, which is exactly when
    # close()'s acquire succeeds and calls _flush(). The probe enqueues a late
    # alert on that path alone: the flusher reaches _flush without holding the
    # lock, close() reaches it holding it, so this fires once, on close().
    class _SlowSink:
        def __init__(self):
            self.started = threading.Event()
            self.emitted: list[FailureRisk] = []

        def emit(self, risk: FailureRisk) -> None:
            self.started.set()
            time.sleep(1.3)  # longer than the join budget of flush_interval + 1.0
            self.emitted.append(risk)

    inner = _SlowSink()
    sink = BatchingSink(inner, max_batch=100, flush_interval=0.05)
    real_flush = sink._flush

    def flush_with_a_late_alert() -> None:
        if sink._delivery_lock.locked():
            sink.emit(_risk(5))
        real_flush()

    monkeypatch.setattr(sink, "_flush", flush_with_a_late_alert)
    sink.emit(_risk(0))
    _wait_for(lambda: inner.started.is_set())  # mid-delivery, not after it

    finished = threading.Event()

    def do_close() -> None:
        sink.close()
        finished.set()

    threading.Thread(target=do_close, daemon=True).start()
    assert finished.wait(8.0), (
        "close() must terminate even when an alert arrives mid-flush; holding "
        "_delivery_lock across _flush() deadlocks shutdown"
    )
