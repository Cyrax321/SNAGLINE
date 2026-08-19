"""Sidecar server mode (project.md §7) -- for non-Python agents.

A minimal stdlib ``http.server`` endpoint any runtime (TypeScript, Node, a
shell script, a Claude Code hook) can POST ``StepEvent`` JSON to:

    POST /events             body: a StepEvent as a JSON object      -> monitor.ingest()
                               or a JSON array of StepEvent objects     (batched)
    POST /hooks/claude-code  body: a native Claude Code hook payload -> mapped + ingested
    GET  /health                                                     -> 200 OK
    GET  /metrics                                                    -> self-observability counters

POST bodies are capped at ``max_body_bytes`` (default 1 MB); larger payloads
get 413. This keeps the sidecar safe to expose and able to absorb buffered
telemetry from any host.

Optionally protect the POST endpoints with a shared secret by passing
``auth_token=`` to ``make_server``/``serve``. When set, POSTs must carry the
token via ``Authorization: Bearer <token>`` or ``X-Snagline-Token: <token>``;
missing or wrong tokens get 401. ``GET /health`` stays open for liveness
probes.

No framework, no third-party dependency -- this keeps the zero-dependency
principle intact even for server mode. For high-throughput production use,
front it with a real ASGI server / reverse proxy; that is the user's infra
choice, not this library's concern.

Risks produced during ingestion are dispatched to the Monitor's sinks as
usual (console by default), not returned in the HTTP response: ingestion is
one-way telemetry, and callers should not block on detection results.
"""

from __future__ import annotations

import json
import logging
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from snagline.adapters.claude_code import HookTracker, ingest_payload
from snagline.events import StepEvent
from snagline.monitor import Monitor

logger = logging.getLogger("snagline")


def make_handler(
    monitor: Monitor,
    auth_token: str | None = None,
    max_body_bytes: int = 1_000_000,
) -> type[BaseHTTPRequestHandler]:
    """Build a request-handler class bound to ``monitor``."""
    tracker = HookTracker()

    class _Handler(BaseHTTPRequestHandler):
        # Class attribute set below; typed as Any to satisfy the checker.
        snagline_monitor: Any = None
        snagline_tracker: Any = None
        snagline_risks: Any = None
        snagline_auth: str | None = None
        snagline_max_body: int = 1_000_000

        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
            # Route http.server's access log through logging, not stderr raw.
            logger.debug("snagline http: " + format, *args)

        def _authorized(self) -> bool:
            token = self.snagline_auth
            if not token:
                return True
            authz = self.headers.get("Authorization", "")
            if authz.startswith("Bearer "):
                return authz[len("Bearer ") :].strip() == token
            return self.headers.get("X-Snagline-Token") == token

        def do_GET(self) -> None:  # noqa: N802 - http.server naming
            if self.path == "/health":
                self._respond(200, {"status": "ok"})
            elif self.path == "/metrics":
                self._respond(200, self.snagline_monitor.metrics())
            elif self.path == "/risks":
                self._respond(200, {"risks": self.snagline_risks})
            else:
                self._respond(404, {"error": "not found"})

        def do_POST(self) -> None:  # noqa: N802 - http.server naming
            if not self._authorized():
                self._respond(401, {"error": "unauthorized"})
                return
            length = int(self.headers.get("Content-Length") or 0)
            if length > self.snagline_max_body:
                self._respond(413, {"error": "payload too large"})
                return
            if self.path == "/events":
                self._post_events()
            elif self.path == "/hooks/claude-code":
                self._post_claude_hook()
            elif self.path == "/risks":
                self._post_risks()
            else:
                self._respond(404, {"error": "not found"})

        def _post_risks(self) -> None:
            """Receive a ``FailureRisk`` JSON emitted by a WebhookSink elsewhere.

            This closes the loop with ``sinks.webhook.WebhookSink``: a remote
            or sibling agent posts its detected risks here, and the sidecar
            records/displays them. Validation is lenient and fail-open -- a
            malformed body is acknowledged (202) so the sender never retries.
            """
            body = self._read_body()
            try:
                risk = json.loads(body.decode("utf-8"))
                if not isinstance(risk, dict):
                    raise ValueError
            except (ValueError, json.JSONDecodeError):
                self._respond(400, {"error": "invalid risk JSON"})
                return
            self.snagline_risks.append(risk)
            logger.info("snagline sidecar received risk: %s", risk)
            print(
                f"[sidecar] RECEIVED risk -> trigger={risk.get('trigger')} "
                f"score={risk.get('score')} detail={risk.get('detail')}",
                file=sys.stderr,
                flush=True,
            )
            self._respond(202, {"status": "received", "trigger": risk.get("trigger")})

        def _post_events(self) -> None:
            body = self._read_body()
            try:
                obj = json.loads(body.decode("utf-8"))
            except (ValueError, json.JSONDecodeError):
                self._respond(400, {"error": "invalid StepEvent JSON"})
                return
            if isinstance(obj, list):
                # Batched ingestion: a JSON array of StepEvent objects. This
                # lets a host buffer telemetry and flush many steps in one
                # request (useful for high-throughput or offline replay).
                count = 0
                for item in obj:
                    if not isinstance(item, dict):
                        self._respond(400, {"error": "invalid StepEvent in batch"})
                        return
                    try:
                        event = StepEvent(**item)
                    except (ValueError, TypeError):
                        self._respond(400, {"error": "invalid StepEvent in batch"})
                        return
                    self.snagline_monitor.ingest(event)
                    count += 1
                self._respond(202, {"status": "ingested", "count": count})
                return
            if not isinstance(obj, dict):
                self._respond(
                    400, {"error": "event body must be a JSON object or array"}
                )
                return
            try:
                event = StepEvent(**obj)
            except (ValueError, TypeError):
                self._respond(400, {"error": "invalid StepEvent JSON"})
                return
            # Ingest itself is fail-open inside the Monitor; a bad event shape
            # above is the only client-visible error.
            self.snagline_monitor.ingest(event)
            self._respond(202, {"status": "ingested", "step_id": event.step_id})

        def _post_claude_hook(self) -> None:
            """Accept a native Claude Code hook payload (http hook type).

            Claude Code posts its own JSON (hook_event_name, session_id,
            tool_name, ...); the adapter maps it to a StepEvent. Used for
            unmapped lifecycle events too, responding 202 so the host never
            retries noise.
            """
            body = self._read_body()
            try:
                payload = json.loads(body.decode("utf-8"))
                if not isinstance(payload, dict):
                    raise ValueError
            except (ValueError, json.JSONDecodeError):
                self._respond(400, {"error": "invalid hook JSON"})
                return
            event = ingest_payload(
                self.snagline_monitor, payload, self.snagline_tracker
            )
            self._respond(
                202,
                {
                    "status": "ingested" if event is not None else "ignored",
                    "step_id": event.step_id if event else None,
                },
            )

        def _read_body(self) -> bytes:
            length = int(self.headers.get("Content-Length") or 0)
            return self.rfile.read(length)

        def _respond(self, code: int, payload: dict) -> None:
            data = (json.dumps(payload) + "\n").encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    _Handler.snagline_monitor = monitor
    _Handler.snagline_tracker = tracker
    _Handler.snagline_risks = []
    _Handler.snagline_auth = auth_token
    _Handler.snagline_max_body = max_body_bytes
    return _Handler


def make_server(
    monitor: Monitor,
    host: str = "127.0.0.1",
    port: int = 8787,
    auth_token: str | None = None,
    max_body_bytes: int = 1_000_000,
) -> ThreadingHTTPServer:
    """Construct a ready-to-``serve_forever()`` sidecar server."""
    return ThreadingHTTPServer(
        (host, port), make_handler(monitor, auth_token, max_body_bytes)
    )


def serve(
    monitor: Monitor,
    host: str = "127.0.0.1",
    port: int = 8787,
    auth_token: str | None = None,
    max_body_bytes: int = 1_000_000,
) -> None:
    """Run the sidecar server in the foreground until interrupted."""
    server = make_server(monitor, host, port, auth_token, max_body_bytes)
    logger.info(
        "snagline sidecar listening on http://%s:%d (POST /events, GET /health)",
        host,
        port,
    )
    try:
        server.serve_forever()
    finally:
        server.server_close()
