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
