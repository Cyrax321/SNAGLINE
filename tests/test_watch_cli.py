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
        json.loads(l)
        for l in err.splitlines()
        if l.startswith("{") and '"trigger"' in l
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
