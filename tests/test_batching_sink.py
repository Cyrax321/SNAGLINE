"""Tests for the BatchingSink (P1 item 5: batched/rate-limited dispatch)."""

from __future__ import annotations

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


def test_rate_limit_paces_across_separate_batches():
    # The pacing clock was a local in _deliver, reset to 0.0 on every flush, so
    # the first item of each batch always went out immediately and only pacing
    # *within* one batch worked. The test above hides that by putting all three
    # risks in a single batch (max_batch=1000 + one flush_now).
    #
    # A burst does not arrive as one tidy batch in production -- the flusher
    # ticks on flush_interval and delivers whatever accumulated, so N risks
    # normally arrive as N small batches. Here each risk is flushed on its own,
    # which delivered all 3 with zero delay before the fix.
    inner = _RecordingSink()
    sink = BatchingSink(
        inner, max_batch=1000, flush_interval=3600.0, max_per_second=10.0
    )
    try:
        start = time.monotonic()
        for i in range(3):
            sink.emit(_risk(i))
            sink.flush_now()
        elapsed = time.monotonic() - start
        # 3 deliveries at 10/s: the first is free, then two 0.1s gaps.
        assert elapsed >= 0.18, (
            f"max_per_second not enforced across batches: {elapsed:.3f}s"
        )
        assert len(inner.emitted) == 3
    finally:
        sink.close()


def test_rate_limit_unset_does_not_delay_delivery():
    # Guard the fix against over-correcting: with no max_per_second there is no
    # gap to honour, and the persisted clock must not introduce one.
    inner = _RecordingSink()
    sink = BatchingSink(inner, max_batch=1000, flush_interval=3600.0)
    try:
        start = time.monotonic()
        for i in range(20):
            sink.emit(_risk(i))
            sink.flush_now()
        elapsed = time.monotonic() - start
        assert elapsed < 0.1
        assert len(inner.emitted) == 20
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
