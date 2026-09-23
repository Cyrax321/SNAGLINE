"""Sink contract bugs: LoggingSink dropping alerts a stream cannot encode
(#431), and --cooldown-seconds inf/nan defeating the dedup sink (#432)."""

from __future__ import annotations

import io
import json
import logging
import os
import tempfile

from snagline.cli import main
from snagline.risk import FailureRisk
from snagline.sinks.logging_sink import LoggingSink

UNICODE = "tool '翻訳ツール' repeated 5x in window"


def _risk(detail: str = UNICODE) -> FailureRisk:
    return FailureRisk("e", "s", 0.9, "loop", detail, 1.0)


def _cp1252_file_logger(path: str) -> logging.Logger:
    """A logger whose only handler writes through a cp1252 codepage, like a
    Windows host or a C-locale container with an ANSI log file."""
    lg = logging.getLogger("snagline.test.cp1252")
    lg.handlers.clear()
    lg.propagate = False
    lg.setLevel(logging.INFO)
    lg.addHandler(logging.FileHandler(path, encoding="cp1252"))
    return lg


# ---------------------------------------------------------------- #431 ----


def test_non_ascii_detail_is_not_dropped_by_a_narrow_codepage_stream():
    with tempfile.TemporaryDirectory() as d:
        log = os.path.join(d, "risk.log")
        lg = _cp1252_file_logger(log)
        try:
            sink = LoggingSink(logger=lg)
            for _ in range(3):
                sink.emit(_risk())
        finally:
            for h in list(lg.handlers):
                h.close()
                lg.removeHandler(h)
        with open(log, encoding="cp1252") as fh:
            lines = [ln for ln in fh.read().splitlines() if ln.strip()]

    assert len(lines) == 3, "every alert must reach the log file"
    for line in lines:
        # The line is pure ASCII: it survives any codepage.
        assert line.isascii()
        # ...and is still one parseable JSON object carrying the detail intact.
        assert json.loads(line)["detail"] == UNICODE


def test_utf8_stream_keeps_raw_non_ascii():
    """The common case is unchanged: a UTF-8-capable stream still receives the
    detail raw, not \\u-escaped."""
    lg = logging.getLogger("snagline.test.utf8")
    lg.handlers.clear()
    lg.propagate = False
    lg.setLevel(logging.INFO)
    # A UTF-8-capable stream: the raw line must survive untouched.
    stream = io.TextIOWrapper(io.BytesIO(), encoding="utf-8")
    lg.addHandler(logging.StreamHandler(stream))
    try:
        LoggingSink(logger=lg).emit(_risk())
    finally:
        for h in list(lg.handlers):
            lg.removeHandler(h)
    stream.seek(0)
    line = stream.read().strip()
    assert UNICODE in line
    assert json.loads(line)["detail"] == UNICODE


def test_narrow_stream_on_a_parent_logger_is_also_covered():
    """logging propagates to ancestors, so the root logger's handler counts."""
    with tempfile.TemporaryDirectory() as d:
        log = os.path.join(d, "root.log")
        root = logging.getLogger()
        saved = (root.handlers[:], root.level)
        root_fh = logging.FileHandler(log, encoding="cp1252")
        try:
            root.handlers[:] = [root_fh]
            root.setLevel(logging.INFO)
            lg = logging.getLogger("snagline.test.propagate")
            lg.handlers.clear()
            lg.propagate = True  # the default
            LoggingSink(logger=lg).emit(_risk())
        finally:
            root.handlers[:] = saved[0]
            root.setLevel(saved[1])
            root_fh.close()
        with open(log, encoding="cp1252") as fh:
            lines = [ln for ln in fh.read().splitlines() if ln.strip()]

    assert len(lines) == 1
    assert json.loads(lines[0])["detail"] == UNICODE


# ---------------------------------------------------------------- #432 ----


def _run(argv):
    old_out, old_err = io.StringIO(), io.StringIO()
    import sys

    saved_out, saved_err = sys.stdout, sys.stderr
    sys.stdout, sys.stderr = old_out, old_err
    try:
        code = main(argv)
    finally:
        sys.stdout, sys.stderr = saved_out, saved_err
    return code, old_out.getvalue() + old_err.getvalue()


def test_watch_rejects_infinite_cooldown():
    """inf used to make the first alert per key silence every repeat forever."""
    code, out = _run(["watch", "--cooldown-seconds", "inf"])
    assert code == 2, out
    assert "--cooldown-seconds must be a finite number" in out
    assert "inf" in out


def test_watch_rejects_nan_cooldown():
    """nan silently disabled the cooldown an operator asked for."""
    code, out = _run(["watch", "--cooldown-seconds", "nan"])
    assert code == 2, out
    assert "--cooldown-seconds must be a finite number" in out


def test_serve_rejects_infinite_cooldown():
    code, out = _run(["serve", "--cooldown-seconds", "inf", "--auth-token", "t"])
    assert code == 2, out
    assert "--cooldown-seconds must be a finite number" in out


def test_watch_zero_cooldown_is_still_a_documented_disable():
    """0 must keep meaning 'no dedup wrapper', not a usage error."""
    with tempfile.TemporaryDirectory() as d:
        empty = os.path.join(d, "empty.jsonl")
        with open(empty, "w", encoding="utf-8"):
            pass
        code, out = _run(["watch", "--file", empty, "--cooldown-seconds", "0"])
    assert code == 0, out
    assert "--cooldown-seconds must be a finite number" not in out


def test_watch_positive_cooldown_is_accepted():
    with tempfile.TemporaryDirectory() as d:
        empty = os.path.join(d, "empty.jsonl")
        with open(empty, "w", encoding="utf-8"):
            pass
        code, out = _run(["watch", "--file", empty, "--cooldown-seconds", "30"])
    assert code == 0, out
    assert "--cooldown-seconds must be a finite number" not in out
