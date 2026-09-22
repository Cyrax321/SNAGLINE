"""Tests for the console sink (issue #19: fire-and-forget on a broken stream)."""

from __future__ import annotations

import concurrent.futures
import io
import logging
import threading

from snagline.risk import FailureRisk
from snagline.sinks import console
from snagline.sinks.console import ConsoleSink


def _risk() -> FailureRisk:
    return FailureRisk("ep-1", "step-1", 0.8, "loop", "test detail", 1.0)


def test_console_sink_writes_json_line_to_stream():
    import io

    buf = io.StringIO()
    sink = ConsoleSink(stream=buf)
    sink.emit(_risk())
    line = buf.getvalue().strip()
    assert line.startswith("{") and '"trigger": "loop"' in line


def test_console_sink_routes_through_logger():
    records: list[logging.LogRecord] = []

    class _Cap(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            records.append(record)

    logger = logging.getLogger("snagline.test.console")
    cap = _Cap()
    logger.addHandler(cap)
    try:
        sink = ConsoleSink(logger=logger, level=logging.WARNING)
        sink.emit(_risk())
    finally:
        logger.removeHandler(cap)
    assert len(records) == 1
    assert '"trigger": "loop"' in records[0].getMessage()


def test_console_sink_is_fire_and_forget_on_broken_stream() -> None:
    # Issue #19: a broken stream must not raise out of emit(); the sink is
    # part of the fail-open ingest path.
    class BrokenStream:
        def write(self, s: str) -> int:
            raise OSError("stream broken")

        def flush(self) -> None:
            raise OSError("stream broken")

    sink = ConsoleSink(stream=BrokenStream())  # type: ignore[arg-type]
    # Must not raise.
    sink.emit(_risk())


def test_console_sink_fault_latch_is_thread_safe(caplog):
    # PR #382 review, finding 4: sinks dispatch *outside* Monitor's per-episode
    # lock (monitor._dispatch), and DedupSink calls the wrapped sink outside
    # its own lock, so emits on a shared ConsoleSink race. Under that race the
    # latch must warn exactly once, not once per contending thread.
    #
    # Note on scope: this is an invariant guard, not a race reproduction. The
    # unsynchronized read-modify-write spans two bytecodes with no GIL release
    # between them, so on CPython the duplicate-warning window is not reliably
    # hittable from a test -- which is exactly why the fix is a lock rather
    # than a timing fix. What this pins is the post-fix contract: any number of
    # contending emits on a dead stream produce exactly one warning, and the
    # lock neither deadlocks nor lets an exception escape.
    n_threads = 32
    barrier = threading.Barrier(n_threads)

    class RacingBrokenStream:
        def write(self, s: str) -> int:
            # Park every thread at the latch boundary, then release them into
            # the check-and-set simultaneously. Bounded so a thread that never
            # arrives fails the test instead of hanging CI.
            barrier.wait(timeout=10)
            raise OSError("stream broken")

        def flush(self) -> None:
            raise OSError("stream broken")

    sink = ConsoleSink(stream=RacingBrokenStream())  # type: ignore[arg-type]
    with concurrent.futures.ThreadPoolExecutor(max_workers=n_threads) as pool:
        list(pool.map(lambda _: sink.emit(_risk()), range(n_threads)))

    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert "write to stream failed" in warnings[0].getMessage()
    assert sink._fault_logged is True


def test_console_sink_guards_the_latch_with_a_lock(monkeypatch):
    # PR #382 review, finding 4: the read-modify-write on the latch is not
    # atomic on its own, so it has to happen under a lock. This is the
    # deterministic half of the coverage -- the duplicate-warning window
    # itself is not reliably reproducible (see the thread-safety test above),
    # but whether the fix actually takes the lock, on both the failure and the
    # recovery path, is directly observable.
    class AuditedLock:
        def __init__(self) -> None:
            self._lock = threading.Lock()
            self.depth = 0
            self.enters = 0

        def __enter__(self) -> None:
            self._lock.acquire()
            self.depth += 1
            self.enters += 1

        def __exit__(self, *exc_info: object) -> None:
            self.depth -= 1
            self._lock.release()

        @property
        def held(self) -> bool:
            return self.depth > 0

    audited = AuditedLock()
    held_at_warning: list[bool] = []

    class _RecordingLogger:
        # The warning must go out *after* the lock is released: a logging
        # handler that re-enters this sink would deadlock against a lock we
        # still hold. Monitor's _log_fault_once logs outside _fault_lock for
        # the same reason (issue #14).
        def warning(self, *args: object, **kwargs: object) -> None:
            held_at_warning.append(audited.held)

    monkeypatch.setattr(console, "logger", _RecordingLogger())

    class BrokenStream:
        def write(self, s: str) -> int:
            raise OSError("stream broken")

        def flush(self) -> None:
            raise OSError("stream broken")

    sink = ConsoleSink(stream=BrokenStream())  # type: ignore[arg-type]
    sink._fault_lock = audited  # type: ignore[assignment]

    # Failure path: the latch is taken under the lock, and the (single)
    # warning is emitted with the lock released.
    sink.emit(_risk())
    sink.emit(_risk())
    assert audited.enters == 2
    assert held_at_warning == [False]
    assert sink._fault_logged is True

    # Recovery path: re-arming the latch also takes the lock, so a concurrent
    # failure cannot arm and be silently reset.
    sink._stream = io.StringIO()
    sink.emit(_risk())
    assert audited.enters == 3
    assert sink._fault_logged is False
    assert sink._stream.getvalue().startswith("{")
