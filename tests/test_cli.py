"""CLI smoke tests for the top-level ``main()`` dispatch (issue #11 coverage)."""

from __future__ import annotations

import io
import json

from snagline.cli import main
from snagline.sinks.dedup import DedupSink


def test_main_replay_quiet_and_summary(capsys, tmp_path):
    traj = tmp_path / "t.jsonl"
    traj.write_text(
        "\n".join(
            json.dumps(
                {
                    "step_id": str(i),
                    "episode_id": "ep-cli",
                    "timestamp": 1.0,
                    "action_type": "tool_call",
                    "action_signature": "SIG",
                }
            )
            for i in range(3)
        )
        + "\n"
    )
    assert main(["replay", str(traj), "--quiet", "--summary"]) == 0
    err = capsys.readouterr().err
    assert "replayed 3 steps" in err
    assert "risk(s) emitted" in err


def test_main_replay_skips_malformed_lines(capsys, tmp_path):
    traj = tmp_path / "bad.jsonl"
    traj.write_text(
        json.dumps(
            {
                "step_id": "0",
                "episode_id": "ep-cli",
                "timestamp": 1.0,
                "action_type": "tool_call",
                "action_signature": "SIG",
            }
        )
        + "\n"
        + "this is not json\n"
    )
    assert main(["replay", str(traj)]) == 0
    err = capsys.readouterr().err
    assert "skipping malformed line" in err


def test_main_bench_runs():
    assert main(["bench"]) == 0


def test_main_hook_forwards_to_out(tmp_path, monkeypatch):
    out = tmp_path / "out.jsonl"
    payload = json.dumps(
        {
            "hook_event_name": "PostToolUse",
            "session_id": "sess-cli",
            "tool_use_id": "tu-1",
            "tool_name": "search",
            "tool_input": {"q": "cat"},
        }
    )
    monkeypatch.setattr("sys.stdin", io.StringIO(payload))
    assert main(["hook", "--out", str(out)]) == 0
    assert out.exists()
    assert json.loads(out.read_text().splitlines()[0])["episode_id"] == "sess-cli"


def test_main_serve_starts_server(monkeypatch):
    started: dict = {}

    def _fake_serve(monitor, host="127.0.0.1", port=8787):
        started["ok"] = (host, port)

    monkeypatch.setattr("snagline.server.http_server.serve", _fake_serve)
    assert main(["serve"]) == 0
    assert started.get("ok") == ("127.0.0.1", 8787)


def test_maybe_dedup_wraps_sinks_only_when_cooldown_set():
    from snagline.cli import _maybe_dedup
    from snagline.sinks.console import ConsoleSink

    plain = [ConsoleSink()]
    # No cooldown -> sinks returned unchanged.
    assert _maybe_dedup(plain, 0.0) is plain
    assert _maybe_dedup(plain, 0) is plain
    # Cooldown > 0 -> each sink wrapped in a DedupSink.
    wrapped = _maybe_dedup(plain, 120.0)
    assert len(wrapped) == 1
    assert isinstance(wrapped[0], DedupSink)
    assert wrapped[0]._cooldown == 120.0
