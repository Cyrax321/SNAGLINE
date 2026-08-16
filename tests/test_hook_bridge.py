"""End-to-end tests for the external-process bridges:

* sidecar ``POST /hooks/claude-code`` (Claude Code native ``http`` hooks)
* ``snagline hook`` (command-hook bridge: stdin payload -> file/HTTP/local)
* ``snagline watch --file`` (file bridge)
"""

from __future__ import annotations

import json
import signal
import subprocess
import sys
import threading
import time
import urllib.request

from snagline.monitor import Monitor
from snagline.risk import FailureRisk
from snagline.server.http_server import make_server


class _RecordingSink:
    def __init__(self):
        self.risks: list[FailureRisk] = []

    def emit(self, risk: FailureRisk) -> None:
        self.risks.append(risk)


def _tool_payload(i: int, event: str = "PreToolUse") -> dict:
    # tool_input stays IDENTICAL across calls (only the volatile tool_use_id
    # differs) so the signature is stable and the loop detector can see the
    # repetition.
    return {
        "session_id": "sess-e2e",
        "hook_event_name": event,
        "tool_name": "Bash",
        "tool_input": {"command": "npm test"},
        "tool_use_id": f"toolu_{i}",
    }


def _post(url: str, obj: dict) -> dict:
    req = urllib.request.Request(
        url, data=json.dumps(obj).encode(), headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=5) as resp:
        return json.loads(resp.read())


def test_sidecar_accepts_native_claude_code_payloads_and_fires_loop():
    sink = _RecordingSink()
    server = make_server(Monitor.default(sinks=[sink]), host="127.0.0.1", port=0)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        # A pre/start + post/end pair, repeated with IDENTICAL tool_input
        # (tool_use_id differs; the signature ignores it on purpose).
        for i in range(4):
            assert (
                _post(base + "/hooks/claude-code", _tool_payload(i))["status"]
                == "ingested"
            )
            assert (
                _post(base + "/hooks/claude-code", _tool_payload(i, "PostToolUse"))[
                    "status"
                ]
                == "ingested"
            )
        # Unmapped lifecycle events are acknowledged and ignored.
        assert (
            _post(base + "/hooks/claude-code", {"hook_event_name": "SessionStart"})[
                "status"
            ]
            == "ignored"
        )
        assert any(r.trigger == "loop" for r in sink.risks)
        assert all(r.episode_id == "sess-e2e" for r in sink.risks)
    finally:
        server.shutdown()
        server.server_close()


def _run(args: list, stdin_text: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "snagline.cli", *args],
        input=stdin_text,
        capture_output=True,
        text=True,
        timeout=30,
    )


def test_hook_cli_appends_canonical_event_to_file(tmp_path):
    out = tmp_path / "events.jsonl"
    for i in range(4):
        r = _run(["hook", "--out", str(out)], json.dumps(_tool_payload(i)))
        assert r.returncode == 0  # fail-open: hook ALWAYS exits 0
    lines = [json.loads(line) for line in out.read_text().splitlines()]
    assert len(lines) == 4
    assert {line["episode_id"] for line in lines} == {"sess-e2e"}
    assert {line["action_type"] for line in lines} == {"tool_call"}
    # The appended file is valid snagline watch input and trips the detector.
    r = _run(["watch", "--file", str(out), "--episode-id", "sess-e2e"], "")
    assert r.returncode == 0
    assert '"trigger": "loop"' in r.stderr


def test_hook_cli_forwards_to_sidecar(tmp_path):
    sink = _RecordingSink()
    server = make_server(Monitor.default(sinks=[sink]), host="127.0.0.1", port=0)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{server.server_address[1]}/events"
    try:
        for i in range(4):
            r = _run(["hook", "--url", url], json.dumps(_tool_payload(i)))
            assert r.returncode == 0
        assert any(x.trigger == "loop" for x in sink.risks)
    finally:
        server.shutdown()
        server.server_close()


def test_hook_cli_never_fails_the_host():
    r = _run(["hook", "--out", "/nonexistent/dir/x.jsonl"], "not json at all")
    assert r.returncode == 0
    r2 = _run(
        ["hook", "--url", "http://127.0.0.1:1/nope"], json.dumps(_tool_payload(1))
    )
    assert r2.returncode == 0
    r3 = _run(["hook"], json.dumps({"hook_event_name": "SessionStart"}))
    assert r3.returncode == 0  # unmapped events are silently dropped


def test_watch_file_follow_sees_appended_lines(tmp_path):
    path = tmp_path / "live.jsonl"
    path.write_text(json.dumps(_base_event(0)) + "\n")
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "snagline.cli",
            "watch",
            "--file",
            str(path),
            "--follow",
            "--episode-id",
            "ep-follow",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        time.sleep(1.0)  # let it consume the first line
        with open(path, "a") as fh:
            for i in range(1, 4):
                fh.write(json.dumps(_base_event(i)) + "\n")
                fh.flush()
                time.sleep(0.3)
        time.sleep(0.8)
    finally:
        proc.send_signal(signal.SIGINT)  # graceful: same as Ctrl-C on the CLI
        out, err = proc.communicate(timeout=10)
    assert "ingested 4 step(s)" in err
    assert '"trigger": "loop"' in err


def _base_event(i: int) -> dict:
    from snagline.events import make_signature

    return {
        "step_id": str(i),
        "episode_id": "ep-follow",
        "timestamp": 1718300000.0 + i,
        "action_type": "tool_call",
        "action_signature": make_signature("tool_call", "Bash", "npm test"),
        "tool_name": "Bash",
    }
