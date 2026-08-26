"""End-to-end wiring of ``Config.log_format`` / ``SNAGLINE_LOG_FORMAT`` (#119).

Every contract here is tested from BOTH sides:

* fire side: ``log_format="json"`` makes risks surface as machine-readable
  JSON lines on the ``snagline`` logger through ``Monitor.default()``,
  ``watch``, and ``replay`` with zero code changes;
* silence side: the default ``"text"`` keeps that logging channel silent,
  and an explicitly passed ``sinks`` list is never augmented.

Validation is loud: values outside {"text", "json"} raise ValueError at
construction or resolve time instead of silently doing nothing.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from snagline import Config, Monitor
from snagline.cli import _build_parser, _build_sinks, main
from snagline.events import StepEvent

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _event(i: int, sig: str | None = None) -> dict:
    """A healthy step; latency-free so only injected loops can fire."""
    return {
        "step_id": str(i),
        "episode_id": "ep-log",
        "timestamp": 1_000.0 + i,
        "action_type": "tool_call",
        "action_signature": sig or f"unique-{i}",
        "tool_name": "search",
    }


def _trajectory(tmp_path: Path) -> Path:
    """Healthy steps plus one injected repetition loop (fires the detector)."""
    events = [_event(i) for i in range(5)]
    events += [_event(100, sig="retry-same-attempt") for _ in range(4)]
    path = tmp_path / "loop.jsonl"
    path.write_text("".join(json.dumps(e) + "\n" for e in events), encoding="utf-8")
    return path


class _FakeStdin:
    def __init__(self, lines: list[str]) -> None:
        self._lines = lines

    def __iter__(self):
        return iter(self._lines)


class _MarkerSink:
    """Explicit caller-provided sink; must never be augmented or replaced."""

    def __init__(self) -> None:
        self.risks: list = []

    def emit(self, risk) -> None:
        self.risks.append(risk)


def _snagline_records(caplog) -> list:
    return [r for r in caplog.records if r.name == "snagline"]


# ---------------------------------------------------------------------------
# config validation (loud failure, never silent no-op)
# ---------------------------------------------------------------------------


def test_log_format_defaults_to_text():
    assert Config().log_format == "text"


def test_constructor_rejects_invalid_log_format():
    with pytest.raises(ValueError, match="log_format must be one of"):
        Config(log_format="yaml")


def test_constructor_normalizes_case_and_whitespace():
    assert Config(log_format=" JSON ").log_format == "json"


@pytest.mark.parametrize("value", ["xml", "JSONL", "", "text,json"])
def test_invalid_values_are_rejected_everywhere(value, tmp_path):
    with pytest.raises(ValueError, match="log_format"):
        Config(log_format=value)
    with pytest.raises(ValueError, match="log_format"):
        Config.from_env({"SNAGLINE_LOG_FORMAT": value})
    with pytest.raises(ValueError, match="log_format"):
        Config.resolve(environ={"SNAGLINE_LOG_FORMAT": value})
    path = tmp_path / "cfg.json"
    path.write_text(json.dumps({"log_format": value}), encoding="utf-8")
    with pytest.raises(ValueError, match="log_format"):
        Config.load_file(str(path))


def test_resolve_env_layering_still_applies_valid_json():
    cfg = Config.resolve(environ={"SNAGLINE_LOG_FORMAT": " json "})
    assert cfg.log_format == "json"


# ---------------------------------------------------------------------------
# Monitor.default() composition (issues #99/#119)
# ---------------------------------------------------------------------------


def test_default_monitor_with_text_installs_console_only(monkeypatch):
    monkeypatch.delenv("SNAGLINE_LOG_FORMAT", raising=False)
    monitor = Monitor.default(Config())
    assert [type(s).__name__ for s in monitor._sinks] == ["ConsoleSink"]


def test_default_monitor_with_json_adds_logging_next_to_console():
    monitor = Monitor.default(Config(log_format="json"))
    # Order matters: LoggingSink sits NEXT TO ConsoleSink ("alongside
    # console"), it does not replace it.
    assert [type(s).__name__ for s in monitor._sinks] == [
        "ConsoleSink",
        "LoggingSink",
    ]


def test_default_monitor_never_augments_explicit_sinks_even_for_json():
    marker = _MarkerSink()
    monitor = Monitor.default(Config(log_format="json"), sinks=[marker])
    assert monitor._sinks == [marker]


def test_bare_default_monitor_resolves_env_for_json(monkeypatch):
    """Acceptance path: no config object, zero code changes, env only."""
    monkeypatch.setenv("SNAGLINE_LOG_FORMAT", "json")
    monitor = Monitor.default()
    assert [type(s).__name__ for s in monitor._sinks] == [
        "ConsoleSink",
        "LoggingSink",
    ]


def test_bare_default_monitor_stays_console_only_without_the_env(monkeypatch):
    monkeypatch.delenv("SNAGLINE_LOG_FORMAT", raising=False)
    monitor = Monitor.default()
    assert [type(s).__name__ for s in monitor._sinks] == ["ConsoleSink"]


# ---------------------------------------------------------------------------
# CLI sink selection honors the knob through Config.resolve()
# ---------------------------------------------------------------------------


def test_cli_watch_sink_selection_text_vs_json(monkeypatch):
    args = _build_parser().parse_args(["watch"])
    monkeypatch.delenv("SNAGLINE_LOG_FORMAT", raising=False)
    assert [type(s).__name__ for s in _build_sinks(args, Config())] == ["ConsoleSink"]
    assert [
        type(s).__name__ for s in _build_sinks(args, Config(log_format="json"))
    ] == ["ConsoleSink", "LoggingSink"]


def test_cli_serve_sink_selection_honors_json():
    args = _build_parser().parse_args(["serve"])
    assert [
        type(s).__name__ for s in _build_sinks(args, Config(log_format="json"))
    ] == ["ConsoleSink", "LoggingSink"]


# ---------------------------------------------------------------------------
# end-to-end: SNAGLINE_LOG_FORMAT=json with zero code changes
# ---------------------------------------------------------------------------

_JSON_KEYS = {"ts", "episode_id", "step_id", "trigger", "severity", "score", "detail"}


def test_replay_with_json_env_emits_parseable_json_lines(tmp_path, caplog, monkeypatch):
    monkeypatch.setenv("SNAGLINE_LOG_FORMAT", "json")
    with caplog.at_level(logging.WARNING, logger="snagline"):
        rc = main(["replay", str(_trajectory(tmp_path))])
    assert rc == 0
    records = _snagline_records(caplog)
    # Fire side: the injected loop surfaces as structured lines on the logger.
    assert records, "expected at least one risk record on the snagline logger"
    parsed = [json.loads(r.getMessage()) for r in records]
    for obj in parsed:
        assert set(obj) == _JSON_KEYS
    assert any(obj["trigger"] == "loop" for obj in parsed)


def test_replay_with_text_keeps_the_logging_channel_silent(
    tmp_path, caplog, capsys, monkeypatch
):
    monkeypatch.delenv("SNAGLINE_LOG_FORMAT", raising=False)
    with caplog.at_level(logging.WARNING, logger="snagline"):
        rc = main(["replay", str(_trajectory(tmp_path))])
    assert rc == 0
    # Silence side: no logging records at all in text mode...
    assert _snagline_records(caplog) == []
    # ...but humans still see the risk on stderr through the console sink.
    err = capsys.readouterr().err
    assert '"trigger": "loop"' in err


def test_watch_with_json_env_emits_json_lines_without_code_changes(monkeypatch, caplog):
    monkeypatch.setenv("SNAGLINE_LOG_FORMAT", "json")
    events = [_event(i) for i in range(2)]
    events += [_event(50 + i, sig="watch-loop-sig") for i in range(3)]
    monkeypatch.setattr("sys.stdin", _FakeStdin([json.dumps(e) for e in events]))
    with caplog.at_level(logging.WARNING, logger="snagline"):
        rc = main(["watch"])
    assert rc == 0
    records = _snagline_records(caplog)
    assert records, "expected the injected loop to reach the logging channel"
    parsed = [json.loads(r.getMessage()) for r in records]
    assert {set(obj) == _JSON_KEYS for obj in parsed} == {True}
    assert any(obj["trigger"] == "loop" for obj in parsed)


def test_watch_text_mode_logs_nothing(monkeypatch, caplog):
    monkeypatch.delenv("SNAGLINE_LOG_FORMAT", raising=False)
    events = [_event(i) for i in range(2)]
    events += [_event(50 + i, sig="watch-loop-sig") for i in range(3)]
    monkeypatch.setattr("sys.stdin", _FakeStdin([json.dumps(e) for e in events]))
    with caplog.at_level(logging.WARNING, logger="snagline"):
        rc = main(["watch"])
    assert rc == 0
    assert _snagline_records(caplog) == []


def test_healthy_unique_steps_fire_nothing_on_a_default_monitor():
    """No-false-positive side: 20 unique-signature steps stay silent."""
    marker = _MarkerSink()
    monitor = Monitor.default(Config(), sinks=[marker])
    for i in range(20):
        monitor.ingest(StepEvent(**_event(i)))
    assert marker.risks == []
