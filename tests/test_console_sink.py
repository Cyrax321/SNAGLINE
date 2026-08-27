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
