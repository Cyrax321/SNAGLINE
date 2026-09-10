"""Tests for ``snagline watch`` (live stdin mode) and ``snagline serve`` wiring."""

from __future__ import annotations

import json
from pathlib import Path

from snagline.cli import main

FIXTURES = Path(__file__).parent / "fixtures" / "trajectories"


class _FakeStdin:
    def __init__(self, lines):
        self._lines = lines

    def __iter__(self):
        return iter(self._lines)


def test_watch_ingests_stdin_and_reports(capsys, monkeypatch):
    lines = (FIXTURES / "injected_loop.jsonl").read_text().splitlines()
    monkeypatch.setattr("sys.stdin", _FakeStdin(lines))
    rc = main(["watch", "--episode-id", "ep-cli"])
    assert rc == 0
    err = capsys.readouterr().err
    assert "ingested" in err
    # The injected-loop fixture must fire the loop detector through the CLI.
    risks = [
        json.loads(line)
        for line in err.splitlines()
        if line.startswith("{") and '"trigger"' in line
    ]
    assert any(r["trigger"] == "loop" for r in risks)


def test_watch_skips_malformed_lines(capsys, monkeypatch):
    lines = ["{bad json", ""]
    monkeypatch.setattr("sys.stdin", _FakeStdin(lines))
    rc = main(["watch"])
    assert rc == 0
    assert "malformed" in capsys.readouterr().err


def test_watch_webhook_requires_url(capsys):
    assert main(["watch", "--sink", "webhook"]) == 2
    assert "--webhook-url" in capsys.readouterr().err


def test_watch_heartbeat_touches_per_ingest(tmp_path):
    """--heartbeat: the liveness file is created and stamped (issue #92)."""
    hb_path = tmp_path / "run" / "snagline" / "hb"
    rc = main(
        [
            "watch",
            "--file",
            str(FIXTURES / "healthy_run.jsonl"),
            "--heartbeat",
            str(hb_path),
        ]
    )
    assert rc == 0
    assert hb_path.exists()
    assert hb_path.read_bytes() == b""  # mtime only, never content


def test_watch_without_heartbeat_creates_nothing(tmp_path):
    rc = main(["watch", "--file", str(FIXTURES / "healthy_run.jsonl")])
    assert rc == 0
    assert list(tmp_path.iterdir()) == []


def test_iter_lines_calls_on_wait_while_following(tmp_path):
    """Idle follow-polls keep the heartbeat alive while no lines arrive."""
    from snagline.cli import _iter_lines

    watched = tmp_path / "events.jsonl"
    watched.write_text('{"step_id": "s1"}\n')
    waits: list[int] = []

    class _Done(Exception):
        pass

    def stop_after_three() -> None:
        waits.append(1)
        if len(waits) >= 3:
            raise _Done

    consumed: list[str] = []
    try:
        for line in _iter_lines(str(watched), True, on_wait=stop_after_three):
            consumed.append(line.strip())
    except _Done:
        pass
    assert consumed == ['{"step_id": "s1"}']
    assert len(waits) == 3


# --- issue #225: finalize-based detectors must see the real episode ids -------


def _event_line(step_id: str, episode_id: str, action_type: str) -> str:
    from snagline.events import make_signature

    return json.dumps(
        {
            "step_id": step_id,
            "episode_id": episode_id,
            "timestamp": 1718300000.0 + float(step_id),
            "action_type": action_type,
            "action_signature": make_signature(action_type, "t"),
            "tool_name": "t",
        }
    )


def test_watch_finalizes_ingested_episode_ids_not_the_filename(
    tmp_path, capsys, monkeypatch
):
    """The issue #225 repro, as a test: same events, same config, watch must
    report the silent_abort that replay reports. Pre-#225, end_episode was
    called with the *filename*, so the detector was asked about an episode
    that never existed and nothing was emitted.
    """
    path = tmp_path / "ep.jsonl"
    # An episode ending on a tool_call (not an output step) and not on an
    # error: exactly the silent-abort shape.
    path.write_text(
        _event_line("1", "real-ep", "message")
        + "\n"
        + _event_line("2", "real-ep", "tool_call")
        + "\n"
    )
    monkeypatch.setenv("SNAGLINE_SILENT_ABORT_ENABLED", "1")
    rc = main(["watch", "--file", str(path)])
    assert rc == 0
    err = capsys.readouterr().err
    risks = [
        json.loads(line)
        for line in err.splitlines()
        if line.startswith("{") and '"trigger"' in line
    ]
    assert any(
        r["trigger"] == "silent_abort" and r["episode_id"] == "real-ep" for r in risks
    ), err


def test_watch_finalizes_every_episode_in_a_multi_episode_file(
    tmp_path, capsys, monkeypatch
):
    """A multi-episode file finalizes each episode at teardown (issue #225's
    follow-session note): both ingested ids fire, exactly once each."""
    path = tmp_path / "multi.jsonl"
    lines = [
        _event_line("1", "ep-a", "message"),
        _event_line("2", "ep-a", "tool_call"),
        _event_line("3", "ep-b", "message"),
        _event_line("4", "ep-b", "tool_call"),
    ]
    path.write_text("\n".join(lines) + "\n")
    monkeypatch.setenv("SNAGLINE_SILENT_ABORT_ENABLED", "1")
    rc = main(["watch", "--file", str(path)])
    assert rc == 0
    err = capsys.readouterr().err
    risks = [
        json.loads(line)
        for line in err.splitlines()
        if line.startswith("{") and '"trigger"' in line
    ]
    aborts = [r for r in risks if r["trigger"] == "silent_abort"]
    assert {r["episode_id"] for r in aborts} == {"ep-a", "ep-b"}
    assert len(aborts) == 2, err


def test_watch_episode_id_override_still_fires_for_overridden_id(
    tmp_path, capsys, monkeypatch
):
    """--episode-id keeps working: events carry ids of their own here, but the
    override id is still finalized (zero-events fallback, issue #225)."""
    path = tmp_path / "ep.jsonl"
    path.write_text(
        _event_line("1", "real-ep", "message")
        + "\n"
        + _event_line("2", "real-ep", "tool_call")
        + "\n"
    )
    monkeypatch.setenv("SNAGLINE_SILENT_ABORT_ENABLED", "1")
    rc = main(["watch", "--file", str(path), "--episode-id", "real-ep"])
    assert rc == 0
    err = capsys.readouterr().err
    assert '"trigger": "silent_abort"' in err


def test_watch_zero_events_still_finalizes_the_fallback_id(capsys, monkeypatch):
    """No events ingested: the synthesized id is finalized as before (no
    regression in the empty-file/stdin case, issue #225)."""
    monkeypatch.setattr("sys.stdin", _FakeStdin([]))
    rc = main(["watch"])
    assert rc == 0
    assert "ingested 0 step(s)" in capsys.readouterr().err
