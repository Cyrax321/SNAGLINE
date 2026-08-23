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


def _start_server(
    sink: _RecordingSink,
    auth_token: str | None = None,
    max_body_bytes: int | None = None,
    max_risks: int | None = None,
):
    kwargs = {}
    if max_body_bytes is not None:
        kwargs["max_body_bytes"] = max_body_bytes
    if max_risks is not None:
        kwargs["max_risks"] = max_risks
    server = make_server(
        Monitor.default(sinks=[sink]),
        host="127.0.0.1",
        port=0,
        auth_token=auth_token,
        **kwargs,
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, f"http://127.0.0.1:{server.server_address[1]}"


def _get(base: str, path: str, headers: dict | None = None) -> int:
    """GET ``path`` and return the status code, HTTPError codes included."""
    req = urllib.request.Request(base + path, headers=headers or {}, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            return int(resp.status)
    except urllib.error.HTTPError as exc:
        return int(exc.code)


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


def test_post_requires_token_when_configured():
    sink = _RecordingSink()
    server, base = _start_server(sink, auth_token="secret")
    try:
        event = {
            "step_id": "0",
            "episode_id": "ep-auth",
            "timestamp": 1718300000.0,
            "action_type": "tool_call",
            "action_signature": "aaaa1111bbbb2222",
            "tool_name": "search",
        }
        # No token -> 401.
        req = urllib.request.Request(
            base + "/events", data=json.dumps(event).encode(), method="POST"
        )
        try:
            urllib.request.urlopen(req, timeout=5)
            raise AssertionError("expected HTTP 401")
        except urllib.error.HTTPError as exc:
            assert exc.code == 401

        # Wrong token -> 401.
        req = urllib.request.Request(
            base + "/events",
            data=json.dumps(event).encode(),
            headers={"Authorization": "Bearer wrong"},
            method="POST",
        )
        try:
            urllib.request.urlopen(req, timeout=5)
            raise AssertionError("expected HTTP 401")
        except urllib.error.HTTPError as exc:
            assert exc.code == 401

        # Correct bearer token -> 202.
        req = urllib.request.Request(
            base + "/events",
            data=json.dumps(event).encode(),
            headers={"Authorization": "Bearer secret"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            assert resp.status == 202

        # X-Snagline-Token header also accepted.
        req = urllib.request.Request(
            base + "/events",
            data=json.dumps(event).encode(),
            headers={"X-Snagline-Token": "secret"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            assert resp.status == 202
    finally:
        server.shutdown()
        server.server_close()


def test_events_endpoint_accepts_batch():
    sink = _RecordingSink()
    server, base = _start_server(sink)
    try:
        events = [
            {
                "step_id": str(i),
                "episode_id": "ep-batch",
                "timestamp": 1718300000.0 + i,
                "action_type": "tool_call",
                "action_signature": f"sig-{i}",
                "tool_name": "search",
            }
            for i in range(3)
        ]
        req = urllib.request.Request(
            base + "/events",
            data=json.dumps(events).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            assert resp.status == 202
            assert json.loads(resp.read())["count"] == 3
        assert len(sink.risks) == 0  # batch of unique signatures => no loop
    finally:
        server.shutdown()
        server.server_close()


def test_post_payload_too_large_returns_413():
    sink = _RecordingSink()
    server, base = _start_server(sink, max_body_bytes=10)
    try:
        event = {
            "step_id": "0",
            "episode_id": "ep-big",
            "timestamp": 1718300000.0,
            "action_type": "tool_call",
            "action_signature": "aaaa1111bbbb2222",
            "tool_name": "search",
        }
        req = urllib.request.Request(
            base + "/events",
            data=json.dumps(event).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            urllib.request.urlopen(req, timeout=5)
            raise AssertionError("expected HTTP 413")
        except urllib.error.HTTPError as exc:
            assert exc.code == 413
    finally:
        server.shutdown()
        server.server_close()


def test_health_open_without_token():
    server, base = _start_server(_RecordingSink(), auth_token="secret")
    try:
        with urllib.request.urlopen(base + "/health", timeout=5) as resp:
            assert resp.status == 200
    finally:
        server.shutdown()
        server.server_close()


def test_metrics_endpoint_reports_ingested_counts():
    server, base = _start_server(_RecordingSink())
    try:
        event = {
            "step_id": "0",
            "episode_id": "ep-metrics",
            "timestamp": 1718300000.0,
            "action_type": "tool_call",
            "action_signature": "aaaa1111bbbb2222",
            "tool_name": "search",
        }
        req = urllib.request.Request(
            base + "/events",
            data=json.dumps(event).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            assert resp.status == 202
        with urllib.request.urlopen(base + "/metrics", timeout=5) as resp:
            assert resp.status == 200
            body = json.loads(resp.read())
        assert body["events_ingested"] >= 1
        assert "risks_emitted" in body
        assert "detector_errors" in body
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


def test_get_metrics_requires_token_when_configured():
    """Regression: do_GET never consulted _authorized(), so /metrics leaked."""
    server, base = _start_server(_RecordingSink(), auth_token="secret")
    try:
        assert _get(base, "/metrics") == 401
        assert _get(base, "/metrics", {"Authorization": "Bearer wrong"}) == 401
        assert _get(base, "/metrics", {"Authorization": "Bearer secret"}) == 200
        assert _get(base, "/metrics", {"X-Snagline-Token": "secret"}) == 200
    finally:
        server.shutdown()
        server.server_close()


def test_get_risks_requires_token_when_configured():
    """Regression: /risks returned every risk the sidecar had ever received --
    episode ids, triggers, and detail strings -- to an unauthenticated caller."""
    sink = _RecordingSink()
    server, base = _start_server(sink, auth_token="secret")
    try:
        risk = {
            "episode_id": "ep-secret",
            "step_id": "s1",
            "score": 0.9,
            "trigger": "loop",
            "detail": "repeated",
            "timestamp": 1.0,
        }
        req = urllib.request.Request(
            base + "/risks",
            data=json.dumps(risk).encode(),
            headers={"Authorization": "Bearer secret"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            assert resp.status == 202

        assert _get(base, "/risks") == 401
        req = urllib.request.Request(
            base + "/risks", headers={"Authorization": "Bearer secret"}, method="GET"
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            assert resp.status == 200
            assert json.loads(resp.read())["risks"][0]["episode_id"] == "ep-secret"
    finally:
        server.shutdown()
        server.server_close()


def test_unknown_get_path_is_401_not_404_when_token_configured():
    """An unauthenticated caller must not be able to enumerate which paths
    exist, so the 404 fallthrough sits behind the token too."""
    server, base = _start_server(_RecordingSink(), auth_token="secret")
    try:
        assert _get(base, "/nope") == 401
        assert _get(base, "/nope", {"Authorization": "Bearer secret"}) == 404
    finally:
        server.shutdown()
        server.server_close()


def test_get_endpoints_stay_open_when_no_token_is_configured():
    """Without auth_token the sidecar is unchanged: GETs are open."""
    server, base = _start_server(_RecordingSink())
    try:
        assert _get(base, "/health") == 200
        assert _get(base, "/metrics") == 200
        assert _get(base, "/risks") == 200
    finally:
        server.shutdown()
        server.server_close()


def test_received_risks_are_bounded():
    """POST /risks is an open-ended ingest point; retention must be capped."""
    server, base = _start_server(_RecordingSink(), max_risks=3)
    try:
        for i in range(5):
            req = urllib.request.Request(
                base + "/risks",
                data=json.dumps(
                    {
                        "episode_id": "ep",
                        "step_id": str(i),
                        "score": 0.9,
                        "trigger": "loop",
                        "detail": "d",
                        "timestamp": float(i),
                    }
                ).encode(),
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=5) as resp:
                assert resp.status == 202
        with urllib.request.urlopen(base + "/risks", timeout=5) as resp:
            risks = json.loads(resp.read())["risks"]
        # Oldest dropped, newest kept, and still JSON-serializable.
        assert [r["step_id"] for r in risks] == ["2", "3", "4"]
    finally:
        server.shutdown()
        server.server_close()
