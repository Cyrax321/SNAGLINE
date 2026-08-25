"""Tests for the structured JSON-lines logging sink (issue #99).

Every emitted line must parse as JSON with exactly seven structural keys,
escaping must survive hostile detail strings, serialization failures must
fail open, and a healthy agent stream must stay completely silent.
"""

from __future__ import annotations

import json
import logging

import pytest

from snagline.config import Config
from snagline.events import StepEvent, make_signature
from snagline.monitor import Monitor
from snagline.risk import FailureRisk
from snagline.sinks import LoggingSink

EXACT_KEYS = {"ts", "episode_id", "step_id", "trigger", "severity", "score", "detail"}
SORTED_KEYS = ["detail", "episode_id", "score", "severity", "step_id", "trigger", "ts"]


class _Cap(logging.Handler):
    """Minimal capturing handler: collects records, formats nothing."""

    def __init__(self) -> None:
        super().__init__()
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


def _capture_logger() -> tuple[logging.Logger, _Cap]:
    lg = logging.getLogger("snagline.test.logging_sink")
    cap = _Cap()
    lg.addHandler(cap)
    return lg, cap


def _risk(**overrides) -> FailureRisk:
    fields: dict = {
        "episode_id": "ep-99",
        "step_id": "step-7",
        "score": 0.85,  # >= 0.8 maps to "critical" per risk.py's documented rule
        "trigger": "loop",
        "detail": "signature repeated 3x in window",
        "timestamp": 1724600000.5,
    }
    fields.update(overrides)
    return FailureRisk(**fields)


def _event(step_id: int, sig: str) -> StepEvent:
    return StepEvent(
        step_id=str(step_id),
        episode_id="ep",
        timestamp=float(step_id),
        action_type="tool_call",
        action_signature=sig,
    )


def test_emit_produces_exactly_one_parseable_line_with_exact_keys():
    lg, cap = _capture_logger()
    try:
        sink = LoggingSink(logger=lg)
        sink.emit(_risk())
    finally:
        lg.removeHandler(cap)
    assert len(cap.records) == 1
    line = cap.records[0].getMessage()
    payload = json.loads(line)  # raises the test if not parseable
    assert set(payload.keys()) == EXACT_KEYS
    assert len(payload) == 7


def test_output_is_single_line_sorted_and_compact():
    lg, cap = _capture_logger()
    try:
        sink = LoggingSink(logger=lg)
        sink.emit(_risk(detail="plain"))
    finally:
        lg.removeHandler(cap)
    line = cap.records[0].getMessage()
    # One line only: no raw newline may leak into the stream.
    assert "\n" not in line
    # sort_keys=True: key order is alphabetical regardless of dict insertion.
    pairs = json.loads(line, object_pairs_hook=list)
    assert [key for key, _ in pairs] == SORTED_KEYS
    # Compact separators: no whitespace padding between items or keys.
    assert ", " not in line
    assert ": " not in line


def test_hostile_detail_round_trips_through_json():
    hostile = 'quote " backslash \\ newline\n tab\t unicode café 日本語'
    lg, cap = _capture_logger()
    try:
        sink = LoggingSink(logger=lg)
        sink.emit(_risk(detail=hostile))
    finally:
        lg.removeHandler(cap)
    line = cap.records[0].getMessage()
    assert "\n" not in line  # escaped as \\n, never a literal break
    assert json.loads(line)["detail"] == hostile
    # ensure_ascii=False: non-ASCII characters are emitted raw, not \\uXXXX.
    assert "café" in line


def test_severity_and_trigger_match_the_input_risk():
    lg, cap = _capture_logger()
    try:
        sink = LoggingSink(logger=lg)
        # Score < 0.5 derives severity "info"; 0.85 derives "critical".
        sink.emit(_risk(trigger="error_cascade", score=0.3))
        sink.emit(_risk())
    finally:
        lg.removeHandler(cap)
    assert len(cap.records) == 2
    first = json.loads(cap.records[0].getMessage())
    second = json.loads(cap.records[1].getMessage())
    assert first["trigger"] == "error_cascade"
    assert first["severity"] == "info"
    assert first["score"] == 0.3
    assert second["trigger"] == "loop"
    assert second["severity"] == "critical"
    assert second["ts"] == 1724600000.5


def test_formatter_sabotage_swallowed_with_plain_fallback(monkeypatch):
    lg, cap = _capture_logger()

    def _boom(*args, **kwargs):
        raise RuntimeError("formatter sabotaged")

    monkeypatch.setattr("snagline.sinks.logging_sink.json.dumps", _boom)
    try:
        sink = LoggingSink(logger=lg)
        sink.emit(_risk())  # must not raise, even with dumps broken
    finally:
        lg.removeHandler(cap)
    assert len(cap.records) == 1
    fallback = cap.records[0].getMessage()
    # Plain structural fallback, ids only, clearly marked as non-JSON.
    assert fallback.startswith("snagline risk (json encoding failed)")
    assert "ep-99" in fallback
    assert "step-7" in fallback
    with pytest.raises(ValueError):
        json.loads(fallback)


def test_custom_broken_formatter_never_propagates():
    class _Broken:
        def render(self, risk: FailureRisk) -> str:
            raise RuntimeError("broken custom formatter")

    lg, cap = _capture_logger()
    try:
        sink = LoggingSink(logger=lg)
        sink._formatter = _Broken()  # type: ignore[assignment]
        sink.emit(_risk())  # fail-open: swallowed, alert dropped
    finally:
        lg.removeHandler(cap)
    assert cap.records == []


def test_default_logger_is_named_snagline(caplog):
    sink = LoggingSink()
    with caplog.at_level(logging.WARNING, logger="snagline"):
        sink.emit(_risk())
    assert caplog.records[-1].name == "snagline"


def test_sink_exported_from_sinks_package():
    # Both import paths must resolve to the same class.
    from snagline.sinks.logging_sink import LoggingSink as Direct

    assert LoggingSink is Direct
    # The default formatter is wired and renders JSON for a real risk.
    lg, cap = _capture_logger()
    try:
        LoggingSink(logger=lg).emit(_risk())
    finally:
        lg.removeHandler(cap)
    assert json.loads(cap.records[0].getMessage())["episode_id"] == "ep-99"


# --- Healthy vs failing streams through a real Monitor -----------------------


def test_healthy_stream_produces_zero_records():
    lg, cap = _capture_logger()
    try:
        monitor = Monitor.default(sinks=[LoggingSink(logger=lg)])
        for i in range(20):
            monitor.ingest(_event(i, make_signature("tool_call", "t", f"a{i}")))
    finally:
        lg.removeHandler(cap)
    assert cap.records == []


def test_injected_loop_stream_fires_loop_records():
    lg, cap = _capture_logger()
    try:
        monitor = Monitor.default(sinks=[LoggingSink(logger=lg)])
        repeated = make_signature("tool_call", "t", "retry-me")
        for i in range(5):
            monitor.ingest(_event(100 + i, repeated))
    finally:
        lg.removeHandler(cap)
    assert cap.records, "expected the injected loop to reach the sink"
    triggers = [json.loads(r.getMessage())["trigger"] for r in cap.records]
    assert all(t == "loop" for t in triggers)


# --- Config wiring (12-factor key SNAGLINE_LOG_FORMAT) ----------------------


def test_config_log_format_defaults_to_text():
    cfg = Config()
    assert cfg.log_format == "text"


def test_config_log_format_from_env():
    cfg = Config.from_env({"SNAGLINE_LOG_FORMAT": "json"})
    assert cfg.log_format == "json"
    # Unset keeps the built-in default.
    assert Config.from_env({}).log_format == "text"


def test_config_log_format_from_file(tmp_path):
    path = tmp_path / "snagline.json"
    path.write_text('{"log_format": "json", "unknown_key": 1}', encoding="utf-8")
    cfg = Config.load_file(str(path))
    assert cfg.log_format == "json"
