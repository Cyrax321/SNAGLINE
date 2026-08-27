"""``GET /directive`` exposes the halt-webhook answer over HTTP (#169).

The enforcement layer (#93) lands the halt endpoint's ``{"action": ...}``
response on the in-process ``Monitor.last_directive``, which non-Python hosts
-- the entire audience for server mode -- had no way to read. These tests pin
the endpoint's contract: it always answers 200 with a full directive, it
reflects a real webhook round trip, it sits behind the token exactly like
``GET /risks``, and it degrades to continue instead of 500 when the directive
cannot be read at all.
"""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from snagline.events import StepEvent
from snagline.monitor import Monitor
from snagline.risk import FailureRisk
from snagline.server.http_server import make_server


class _FixedRiskDetector:
    """Emits one high-score risk on every observe(), so a halt always fires."""

    name = "fixed_risk"

    def observe(self, event: StepEvent) -> FailureRisk | None:
        return FailureRisk(
            event.episode_id,
            event.step_id,
            0.95,
            "loop",
            "synthetic risk",
            event.timestamp,
        )

    def reset(self, episode_id: str) -> None:
        pass


class _PauseEndpoint:
    """A local halt endpoint that answers pause, as #93's contract allows."""

    def __init__(self) -> None:
        class _Handler(BaseHTTPRequestHandler):
            def log_message(self, format: str, *args) -> None:  # noqa: A002
                pass

            def do_POST(self) -> None:  # noqa: N802 - http.server naming
                self.rfile.read(int(self.headers.get("Content-Length") or 0))
                data = json.dumps(
                    {"action": "pause", "reason": "budget exceeded"}
                ).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        self._server.daemon_threads = True
        self.url = f"http://127.0.0.1:{self._server.server_port}/halt"
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=2)


def _start(monitor, **kwargs):
    server = make_server(monitor, host="127.0.0.1", port=0, **kwargs)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, f"http://127.0.0.1:{server.server_address[1]}"


def _get(base: str, path: str, headers: dict | None = None):
    """GET and return (status, parsed body); HTTPError codes included."""
    req = urllib.request.Request(base + path, headers=headers or {}, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            return int(resp.status), json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        return int(exc.code), json.loads(exc.read())


def _event(step_id: str = "s1") -> dict:
    return {
        "step_id": step_id,
        "episode_id": "ep-directive",
        "timestamp": 1718300000.0,
        "action_type": "tool_call",
        "action_signature": "aaaa1111bbbb2222",
    }


def test_directive_defaults_to_continue():
    """The resource exists as soon as the Monitor does: 200, never 404."""
    server, base = _start(Monitor([], []))
    try:
        status, body = _get(base, "/directive")
        assert status == 200
        assert body == {"action": "continue", "reason": "", "timestamp": 0.0}
    finally:
        server.shutdown()
        server.server_close()


def test_directive_reports_a_pause_after_a_halt_round_trip():
    """The regression this endpoint exists for: a pause is now readable."""
    endpoint = _PauseEndpoint()
    monitor = Monitor(
        [_FixedRiskDetector()],
        [],
        policy="halt_webhook",
        halt_url=endpoint.url,
    )
    server, base = _start(monitor)
    try:
        before_status, before = _get(base, "/directive")
        assert before_status == 200
        assert before["action"] == "continue"

        request = urllib.request.Request(
            base + "/events",
            data=json.dumps(_event()).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=5) as resp:
            assert resp.status == 202

        status, body = _get(base, "/directive")
        assert status == 200
        assert body["action"] == "pause"
        assert body["reason"] == "budget exceeded"
        # The timestamp is when the directive was received, so it is set now.
        assert body["timestamp"] > 0.0
    finally:
        server.shutdown()
        server.server_close()
        endpoint.stop()


def test_directive_is_behind_the_token_when_one_is_set():
    """A directive can pause the host, so it must never be world-readable."""
    server, base = _start(Monitor([], []), auth_token="secret")
    try:
        assert _get(base, "/directive")[0] == 401
        assert _get(base, "/directive", {"Authorization": "Bearer wrong"})[0] == 401
        ok_status, body = _get(base, "/directive", {"Authorization": "Bearer secret"})
        assert ok_status == 200
        assert body["action"] == "continue"
        # X-Snagline-Token is accepted here too, as on every other endpoint.
        assert _get(base, "/directive", {"X-Snagline-Token": "secret"})[0] == 200
        # /health stays open, so a probe cannot be broken by adding this route.
        assert _get(base, "/health")[0] == 200
    finally:
        server.shutdown()
        server.server_close()


class _UnreadableDirectiveMonitor:
    """A monitor whose directive cannot be read, e.g. one predating #93."""

    def __init__(self) -> None:
        self.metrics_calls = 0

    @property
    def last_directive(self):
        raise AttributeError("no directive on this monitor")

    def metrics(self) -> dict:
        self.metrics_calls += 1
        return {}


def test_directive_fails_open_to_continue_instead_of_500():
    """Fail-open, as everywhere else: a broken read reports continue."""
    server, base = _start(_UnreadableDirectiveMonitor())
    try:
        status, body = _get(base, "/directive")
        assert status == 200
        assert body == {"action": "continue", "reason": "", "timestamp": 0.0}
    finally:
        server.shutdown()
        server.server_close()


def test_unknown_get_paths_still_404():
    """Adding a route must not turn the fallthrough into a match."""
    server, base = _start(Monitor([], []))
    try:
        assert _get(base, "/directives")[0] == 404
        assert _get(base, "/directive/latest")[0] == 404
    finally:
        server.shutdown()
        server.server_close()
