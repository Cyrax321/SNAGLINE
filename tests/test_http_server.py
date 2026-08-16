"""End-to-end tests for the stdlib sidecar HTTP server (project.md §7)."""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request

from snagline.monitor import Monitor
from snagline.risk import FailureRisk
from snagline.server.http_server import make_server


class _RecordingSink:
    def __init__(self):
        self.risks: list[FailureRisk] = []

    def emit(self, risk: FailureRisk) -> None:
        self.risks.append(risk)


def _start_server(sink: _RecordingSink):
    server = make_server(Monitor.default(sinks=[sink]), host="127.0.0.1", port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, f"http://127.0.0.1:{server.server_address[1]}"


def test_health_endpoint():
    server, base = _start_server(_RecordingSink())
    try:
        with urllib.request.urlopen(base + "/health", timeout=5) as resp:
            assert resp.status == 200
            assert json.loads(resp.read()) == {"status": "ok"}
    finally:
        server.shutdown()
        server.server_close()


def test_events_endpoint_ingests_and_fires_loop_detector():
    sink = _RecordingSink()
    server, base = _start_server(sink)
    try:
        event = {
            "step_id": "0",
            "episode_id": "ep-http",
            "timestamp": 1718300000.0,
            "action_type": "tool_call",
            "action_signature": "aaaa1111bbbb2222",
            "tool_name": "search",
        }
        for _ in range(4):  # loop detector default: 3 repeats in window of 4
            req = urllib.request.Request(
                base + "/events",
                data=json.dumps(event).encode(),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=5) as resp:
                assert resp.status == 202
                assert json.loads(resp.read())["status"] == "ingested"
        assert any(r.trigger == "loop" for r in sink.risks)
    finally:
        server.shutdown()
        server.server_close()


def test_events_endpoint_rejects_malformed_body():
    server, base = _start_server(_RecordingSink())
    try:
        req = urllib.request.Request(base + "/events", data=b"not json", method="POST")
        try:
            urllib.request.urlopen(req, timeout=5)
            raise AssertionError("expected HTTP 400")
        except urllib.error.HTTPError as exc:
            assert exc.code == 400
    finally:
        server.shutdown()
        server.server_close()


def test_unknown_paths_are_404():
    server, base = _start_server(_RecordingSink())
    try:
        try:
            urllib.request.urlopen(base + "/nope", timeout=5)
            raise AssertionError("expected HTTP 404")
        except urllib.error.HTTPError as exc:
            assert exc.code == 404
    finally:
        server.shutdown()
        server.server_close()


def test_claude_code_hook_endpoint_ingests():
    # Issue #22 path: a native Claude Code hook payload is mapped and ingested.
    # Three identical PostToolUse events must map to repeated tool_call
    # signatures and trip the loop detector end to end.
    sink = _RecordingSink()
    server, base = _start_server(sink)
    try:
        payload = {
            "hook_event_name": "PostToolUse",
            "session_id": "sess-http",
            "tool_use_id": "tu-1",
            "tool_name": "search",
            "tool_input": {"q": "cat"},
        }
        for _ in range(3):
            req = urllib.request.Request(
                base + "/hooks/claude-code",
                data=json.dumps(payload).encode(),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=5) as resp:
                assert resp.status == 202
        assert any(r.trigger == "loop" for r in sink.risks)
    finally:
        server.shutdown()
        server.server_close()


def test_risks_endpoint_records_received_risk():
    sink = _RecordingSink()
    server, base = _start_server(sink)
    try:
        risk = {
            "episode_id": "ep-x",
            "step_id": "s1",
            "score": 0.9,
            "trigger": "loop",
            "detail": "repeated",
            "timestamp": 1.0,
        }
        req = urllib.request.Request(
            base + "/risks",
            data=json.dumps(risk).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            assert resp.status == 202
        with urllib.request.urlopen(base + "/risks", timeout=5) as resp:
            assert resp.status == 200
            assert json.loads(resp.read())["risks"]
    finally:
        server.shutdown()
        server.server_close()
