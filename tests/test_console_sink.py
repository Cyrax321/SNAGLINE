"""Tests for the console sink (issue #19: fire-and-forget on a broken stream)."""

from __future__ import annotations

import logging

from snagline.risk import FailureRisk
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


def test_console_sink_is_fire_and_forget_on_closed_stream() -> None:
    # Issue #327: a closed stream raises ValueError ("I/O operation on closed
    # file"), not OSError -- the except OSError alone let it escape emit().
    import io

    stream = io.StringIO()
    stream.close()
    sink = ConsoleSink(stream=stream)
    # Must not raise, even across repeated emits on the same dead stream.
    sink.emit(_risk())
    sink.emit(_risk())


def test_console_sink_logs_stream_fault_only_once(caplog) -> None:
    # Issue #327: a permanently dead stream must warn once, not per alert
    # (the HeartbeatSink _fault_logged latch).
    class AlwaysClosed:
        def write(self, s: str) -> int:
            raise ValueError("I/O operation on closed file")

        def flush(self) -> None:
            raise ValueError("I/O operation on closed file")

    sink = ConsoleSink(stream=AlwaysClosed())  # type: ignore[arg-type]
    with caplog.at_level(logging.WARNING, logger="snagline"):
        for _ in range(5):
            sink.emit(_risk())
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert "fire-and-forget" in warnings[0].getMessage()


def test_console_sink_fault_latch_resets_on_recovery() -> None:
    # A stream that fails then recovers re-arms the latch, so a later failure
    # warns again instead of suffering in silence.
    class FlakyStream:
        def __init__(self) -> None:
            self.closed = True
            self.buf: list[str] = []

        def write(self, s: str) -> int:
            if self.closed:
                raise ValueError("I/O operation on closed file")
            self.buf.append(s)
            return len(s)

        def flush(self) -> None:
            if self.closed:
                raise ValueError("I/O operation on closed file")

    stream = FlakyStream()
    sink = ConsoleSink(stream=stream)  # type: ignore[arg-type]
    sink.emit(_risk())  # fails, arms the latch
    stream.closed = False
    sink.emit(_risk())  # succeeds, resets the latch
    assert stream.buf  # the recovered alert really landed
    stream.closed = True
    sink.emit(_risk())  # fails again
    assert sink._fault_logged
