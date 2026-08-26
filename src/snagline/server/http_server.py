"""Sidecar server mode (project.md §7) -- for non-Python agents.

A minimal stdlib ``http.server`` endpoint any runtime (TypeScript, Node, a
shell script, a Claude Code hook) can POST ``StepEvent`` JSON to:

    POST /events             body: a StepEvent as a JSON object      -> monitor.ingest()
                               or a JSON array of StepEvent objects     (batched)
    POST /hooks/claude-code  body: a native Claude Code hook payload -> mapped + ingested
    GET  /health                                                     -> 200 OK
    GET  /metrics                                                    -> Prometheus text exposition (v0.0.4)

POST bodies are capped at ``max_body_bytes`` (default 1 MB); larger payloads
get 413. This keeps the sidecar safe to expose and able to absorb buffered
telemetry from any host.

Optionally protect the endpoints with a shared secret by passing
``auth_token=`` to ``make_server``/``serve``. When set, requests must carry the
token via ``Authorization: Bearer <token>`` or ``X-Snagline-Token: <token>``;
missing or wrong tokens get 401. ``GET /health`` stays open for liveness
probes -- everything else, GET and POST alike, is behind the token.

The listener can terminate TLS itself (issue #120): pass ``certfile=`` and
``keyfile=`` (or a ready-made ``ssl_context=``) and each accepted connection
is wrapped with stdlib ``ssl`` inside its own worker thread, so the sidecar
speaks ``https://`` on the same endpoints with identical auth semantics, and
one stalled handshake never blocks other senders. Without these arguments
the listener stays plain HTTP, byte-for-byte the behavior described above.

``POST /risks`` retains the most recent ``max_risks`` risks (default 1000) for
``GET /risks``; older ones are discarded so an unbounded sender cannot grow the
sidecar's memory without limit.

``GET /metrics`` speaks Prometheus text exposition version 0.0.4 by default
(issue #98), rendered with plain string formatting and no client library:
``snagline_events_total``, ``snagline_risks_total{trigger,severity}``,
``snagline_episodes_active``, the ``snagline_ingest_seconds`` count/sum pair,
and the raw ``Monitor.metrics()`` counters under ``snagline_monitor_*_total``.
The legacy JSON counters body is still served with ``?format=classic``, for
clients sending ``Accept: application/json``, or when the sidecar is started
with ``SNAGLINE_METRICS_FORMAT=classic`` (config key ``metrics_format``);
``?format=prometheus`` forces the new format back on per request.

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
import math
import ssl
import sys
import threading
import time
from collections import OrderedDict, deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import parse_qs, urlsplit

from snagline.adapters.claude_code import HookTracker, ingest_payload
from snagline.config import Config
from snagline.events import StepEvent
from snagline.monitor import Monitor

logger = logging.getLogger("snagline")

# Content type required for Prometheus text exposition format 0.0.4.
PROMETHEUS_CONTENT_TYPE = "text/plain; version=0.0.4; charset=utf-8"
# Supported values of the metrics_format config/env/request toggle.
METRICS_FORMATS = ("prometheus", "classic")
# Bound on distinct episode ids tracked for the episodes-active gauge. Episode
# ids are identifiers, never content; the cap keeps memory bounded even if a
# host streams unlimited unique ids through one long-lived sidecar process.
_MAX_TRACKED_EPISODES = 10_000

# Over-cap POSTs are drained and discarded before the 413 goes out, but only
# within this much above max_body_bytes (issue #121): reading the body lets a
# streaming client finish sending and still read the status line instead of
# dying on EPIPE/reset. Beyond that window the 413 is sent immediately so one
# hostile sender cannot tie up a handler thread indefinitely.
_MAX_OVERCAP_DRAIN_EXCESS = 65_536
# Drain reads happen in fixed chunks, so peak memory stays at one chunk no
# matter what Content-Length claims.
_DRAIN_CHUNK_BYTES = 16_384


def _escape_label(value: str) -> str:
    """Escape a Prometheus label value (backslash, quote, newline)."""
    return value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def _format_sample_value(value: float | int) -> str:
    """Render a sample value the way the exposition format expects."""
    number = float(value)
    if math.isnan(number):
        return "NaN"
    if math.isinf(number):
        return "+Inf" if number > 0 else "-Inf"
    return repr(float(value))


def _resolve_metrics_format(explicit: str | None) -> str:
    """Pick the startup-time default for GET /metrics.

    Precedence: explicit argument, then SNAGLINE_METRICS_FORMAT via Config,
    then the built-in Config default ("prometheus"). Unknown values are
    logged and replaced by "prometheus": configuration must never take the
    sidecar down (fail-open).
    """
    candidate = explicit
    if candidate is None:
        try:
            candidate = Config.from_env_overrides().get("metrics_format")
        except Exception:
            logger.debug(
                "snagline: env lookup for metrics_format failed", exc_info=True
            )
            candidate = None
    if candidate is None:
        candidate = Config().metrics_format
    name = str(candidate).strip().lower()
    if name in METRICS_FORMATS:
        return name
    logger.warning("snagline: unknown metrics format %r; serving prometheus", candidate)
    return "prometheus"


def _choose_metrics_format(
    query: dict[str, list[str]], accept_header: str | None, configured: str
) -> str:
    """Resolve the format for one GET /metrics request.

    An explicit ``?format=`` parameter wins. Without it, clients that send
    ``Accept: application/json`` keep getting the legacy JSON body; everyone
    else gets the configured default (prometheus since issue #98).
    """
    requested = (query.get("format") or [""])[-1].strip().lower()
    if requested:
        if requested in METRICS_FORMATS:
            return requested
        logger.warning(
            "snagline: unknown ?format=%r; falling back to negotiation", requested
        )
    accept = (accept_header or "").lower()
    if "application/json" in accept:
        return "classic"
    return configured


class SidecarMetricsCollector:
    """Process-lifetime sidecar counters, renderable as Prometheus text.

    Counts risks per ``(trigger, severity)`` by acting as one extra sink on
    the Monitor, plus HTTP-layer totals: events accepted, time spent inside
    ``Monitor.ingest``, and distinct episode ids seen (bounded table with
    least-recently-seen eviction, ids only, no content). Every entry point
    swallows its own exceptions: like any sink, this must never take down
    ingestion or serving.
    """

    def __init__(self, max_episodes: int = _MAX_TRACKED_EPISODES) -> None:
        self._lock = threading.Lock()
        self._risks: dict[tuple[str, str], int] = {}
        self._events_total = 0
        self._ingest_count = 0
        self._ingest_sum = 0.0
        # Episode id table: reseeing an id moves it to the end, so when the
        # cap bites we forget the episode quiet for the longest time, which
        # is the right approximation of "active" for a long-lived sidecar.
        self._episodes: OrderedDict[str, None] = OrderedDict()
        self._max_episodes = max(1, int(max_episodes))

    def emit(self, risk: Any) -> None:
        """AlertSink protocol entry point; counts one risk by label pair."""
        try:
            key = (str(risk.trigger), str(risk.severity))
            with self._lock:
                self._risks[key] = self._risks.get(key, 0) + 1
        except Exception:
            logger.debug("snagline: metrics collector emit failed", exc_info=True)

    def record_ingest(self, episode_id: str | None, elapsed_seconds: float) -> None:
        """Record one accepted event and its ingest wall time."""
        try:
            with self._lock:
                self._events_total += 1
                self._ingest_count += 1
                self._ingest_sum += max(0.0, float(elapsed_seconds))
                if episode_id is not None:
                    key = str(episode_id)
                    self._episodes.pop(key, None)
                    self._episodes[key] = None
                    while len(self._episodes) > self._max_episodes:
                        self._episodes.popitem(last=False)
        except Exception:
            logger.debug("snagline: metrics record failed", exc_info=True)

    def snapshot(self) -> dict[str, Any]:
        """Return a consistent copy of the counters for rendering."""
        with self._lock:
            return {
                "events_total": self._events_total,
                "risks": sorted(self._risks.items()),
                "episodes_active": len(self._episodes),
                "ingest_count": self._ingest_count,
                "ingest_sum": self._ingest_sum,
            }

    def render_prometheus(self, monitor_metrics: dict[str, int]) -> str:
        """Render the full exposition body (version 0.0.4 text format).

        ``monitor_metrics`` is the ``Monitor.metrics()`` snapshot; those four
        counters are exposed under ``snagline_monitor_*_total`` names so the
        sidecar view adds process-lifetime detail without renaming anything
        the JSON endpoint already published.
        """
        snap = self.snapshot()
        lines: list[str] = []
        lines.append(
            "# HELP snagline_events_total Steps ingested through this sidecar."
        )
        lines.append("# TYPE snagline_events_total counter")
        lines.append(f"snagline_events_total {snap['events_total']}")
        lines.append(
            "# HELP snagline_risks_total Risks emitted by trigger and severity."
        )
        lines.append("# TYPE snagline_risks_total counter")
        for (trigger, severity), count in snap["risks"]:
            lines.append(
                f'snagline_risks_total{{trigger="{_escape_label(trigger)}",'
                f'severity="{_escape_label(severity)}"}} {count}'
            )
        lines.append(
            "# HELP snagline_episodes_active Distinct episode ids seen since start."
        )
        lines.append("# TYPE snagline_episodes_active gauge")
        lines.append(f"snagline_episodes_active {snap['episodes_active']}")
        lines.append("# HELP snagline_ingest_seconds Wall time spent ingesting steps.")
        lines.append("# TYPE snagline_ingest_seconds summary")
        lines.append(f"snagline_ingest_seconds_count {snap['ingest_count']}")
        lines.append(
            f"snagline_ingest_seconds_sum {_format_sample_value(snap['ingest_sum'])}"
        )
        monitor_families = (
            (
                "events_ingested",
                "snagline_monitor_events_ingested_total",
                "Events ingested by the Monitor.",
            ),
            (
                "risks_emitted",
                "snagline_monitor_risks_emitted_total",
                "Risks emitted by the Monitor.",
            ),
            (
                "detector_errors",
                "snagline_monitor_detector_errors_total",
                "Detector exceptions swallowed (fail-open).",
            ),
            (
                "sink_errors",
                "snagline_monitor_sink_errors_total",
                "Sink exceptions swallowed (fail-open).",
            ),
        )
        for key, name, help_text in monitor_families:
            lines.append(f"# HELP {name} {help_text}")
            lines.append(f"# TYPE {name} counter")
            lines.append(f"{name} {monitor_metrics.get(key, 0)}")
        return "\n".join(lines) + "\n"


def _attach_metrics_collector(monitor: Any, collector: SidecarMetricsCollector) -> None:
    """Register ``collector`` as one extra sink so risks are counted per label.

    Uses the public ``Monitor.add_sink`` (issue #122), so any Monitor that
    honors the base-class contract works without reaching into privates.
    Guarded end to end: any failure leaves serving fully functional, only the
    per-label risk counters stay empty. Monitor dispatch already treats sinks
    as fail-open, matching project.md §1.2.
    """
    try:
        monitor.add_sink(collector)
    except Exception:
        logger.warning(
            "snagline: attaching metrics collector failed; risk labels stay empty",
            exc_info=True,
        )


def make_handler(
    monitor: Monitor,
    auth_token: str | None = None,
    max_body_bytes: int = 1_000_000,
    max_risks: int = 1000,
    metrics_format: str | None = None,
) -> type[BaseHTTPRequestHandler]:
    """Build a request-handler class bound to ``monitor``.

    ``metrics_format`` pins the default GET /metrics body; when omitted the
    SNAGLINE_METRICS_FORMAT environment variable decides, then the built-in
    "prometheus" default (see ``_resolve_metrics_format``).
    """
    tracker = HookTracker()

    class _Handler(BaseHTTPRequestHandler):
        # Class attribute set below; typed as Any to satisfy the checker.
        snagline_monitor: Any = None
        snagline_tracker: Any = None
        snagline_risks: Any = None
        snagline_auth: str | None = None
        snagline_max_body: int = 1_000_000
        snagline_collector: Any = None
        snagline_metrics_format: str = "prometheus"

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
            # /health is deliberately open: a liveness probe (k8s, ELB, docker
            # healthcheck) generally cannot be taught to carry a shared secret,
            # and it reveals nothing but reachability.
            if self.path == "/health":
                self._respond(200, {"status": "ok"})
                return
            # Everything else is behind the token, including the 404 fallthrough
            # so an unauthenticated caller cannot probe which paths exist.
            if not self._authorized():
                self._respond(401, {"error": "unauthorized"})
                return
            split = urlsplit(self.path)
            if split.path == "/metrics":
                self._serve_metrics(parse_qs(split.query))
            elif split.path == "/risks":
                self._respond(200, {"risks": list(self.snagline_risks)})
            else:
                self._respond(404, {"error": "not found"})

        def _serve_metrics(self, query: dict[str, list[str]]) -> None:
            """Serve either exposition format, failing open to an empty body.

            Rendering happens under try/except because a scrape endpoint that
            raises would break scrapers' confidence far more than an empty
            (valid) body does.
            """
            fmt = _choose_metrics_format(
                query, self.headers.get("Accept"), self.snagline_metrics_format
            )
            if fmt == "classic":
                self._respond(200, self.snagline_monitor.metrics())
                return
            try:
                body = self.snagline_collector.render_prometheus(
                    self.snagline_monitor.metrics()
                )
            except Exception:
                logger.exception("snagline: prometheus render failed")
                body = ""
            self._respond_text(200, body, PROMETHEUS_CONTENT_TYPE)

        def do_POST(self) -> None:  # noqa: N802 - http.server naming
            if not self._authorized():
                # Drain the declared body before replying: closing a
                # connection that still has unread inbound data makes the
                # kernel send RST and the sender can lose the 401 response
                # entirely (same rationale as the over-cap drain, #121;
                # observed over TLS where close timing shifts the race).
                self._discard_overcap_body(int(self.headers.get("Content-Length") or 0))
                self._respond(401, {"error": "unauthorized"})
                return
            length = int(self.headers.get("Content-Length") or 0)
            if length > self.snagline_max_body:
                # Consume the over-cap body first: closing with megabytes
                # unread makes the peer see EPIPE/reset instead of the 413
                # (issue #121).
                self._discard_overcap_body(length)
                self._respond(413, {"error": "payload too large"})
                return
            if self.path == "/events":
                self._post_events()
            elif self.path == "/hooks/claude-code":
                self._post_claude_hook()
            elif self.path == "/risks":
                self._post_risks()
            else:
                self._discard_overcap_body(int(self.headers.get("Content-Length") or 0))
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
                    self._ingest_recorded(event)
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
            self._ingest_recorded(event)
            self._respond(202, {"status": "ingested", "step_id": event.step_id})

        def _ingest_recorded(self, event: StepEvent) -> None:
            """Ingest one event while recording sidecar metrics around it.

            The collector swallows its own exceptions, so bookkeeping can
            never turn into a serving failure; the ingest call itself keeps
            exactly the Monitor's own fail-open contract.
            """
            collector = self.snagline_collector
            start = time.perf_counter()
            try:
                self.snagline_monitor.ingest(event)
            finally:
                if collector is not None:
                    elapsed = time.perf_counter() - start
                    collector.record_ingest(event.episode_id, elapsed)

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
            collector = self.snagline_collector
            start = time.perf_counter()
            event = ingest_payload(
                self.snagline_monitor, payload, self.snagline_tracker
            )
            if collector is not None and event is not None:
                elapsed = time.perf_counter() - start
                collector.record_ingest(event.episode_id, elapsed)
            self._respond(
                202,
                {
                    "status": "ingested" if event is not None else "ignored",
                    "step_id": event.step_id if event else None,
                },
            )

        def _discard_overcap_body(self, declared_length: int) -> None:
            """Read and throw away an over-cap body before replying 413.

            Bodies up to ``max_body_bytes + _MAX_OVERCAP_DRAIN_EXCESS`` are
            drained fully so the sender can finish writing and still read the
            response; anything larger is abandoned after that window (the 413
            still goes out at once). Reads happen in fixed-size chunks, so
            peak memory is bounded regardless of claimed Content-Length.
            Fail-open: a client that hangs up mid-drain is logged and
            forgotten, never turned into a serving error.
            """
            remaining = min(
                declared_length,
                self.snagline_max_body + _MAX_OVERCAP_DRAIN_EXCESS,
            )
            while remaining > 0:
                try:
                    chunk = self.rfile.read(min(remaining, _DRAIN_CHUNK_BYTES))
                except OSError:
                    logger.debug(
                        "snagline: over-cap body drain interrupted", exc_info=True
                    )
                    return
                if not chunk:
                    return  # client hung up early; nothing more to discard
                remaining -= len(chunk)

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

        def _respond_text(self, code: int, text: str, content_type: str) -> None:
            data = text.encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    _Handler.snagline_monitor = monitor
    _Handler.snagline_tracker = tracker
    # Bounded: POST /risks is an open-ended ingest point, so retain only the
    # most recent max_risks entries rather than growing without limit.
    _Handler.snagline_risks = deque(maxlen=max(1, max_risks))
    _Handler.snagline_auth = auth_token
    _Handler.snagline_max_body = max_body_bytes
    _Handler.snagline_collector = SidecarMetricsCollector()
    _attach_metrics_collector(monitor, _Handler.snagline_collector)
    _Handler.snagline_metrics_format = _resolve_metrics_format(metrics_format)
    return _Handler


def _resolve_ssl_context(
    ssl_context: ssl.SSLContext | None,
    certfile: str | None,
    keyfile: str | None,
) -> ssl.SSLContext | None:
    """Return the server-side TLS context for the listener, or None.

    An explicit ``ssl_context`` wins; otherwise ``certfile`` (with optional
    ``keyfile``) is loaded into a fresh ``ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)``.
    Contradictory or incomplete arguments raise ``ValueError`` at startup:
    silently ignoring half a TLS configuration would bind plaintext while the
    operator believes the listener is encrypted.
    """
    if ssl_context is not None:
        if certfile is not None or keyfile is not None:
            raise ValueError(
                "pass either ssl_context or certfile/keyfile to serve(), not both"
            )
        return ssl_context
    if certfile is None:
        if keyfile is not None:
            raise ValueError("keyfile requires certfile")
        return None
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(certfile, keyfile)
    return context


class _TLSThreadingHTTPServer(ThreadingHTTPServer):
    """ThreadingHTTPServer that terminates TLS inside the worker thread.

    The listening socket stays a plain socket; each accepted connection is
    wrapped (and its TLS handshake driven) in ``finish_request``, which
    ThreadingMixIn already runs on the per-connection worker thread. A client
    that opens a connection and stalls the handshake therefore ties up only
    its own thread, not the accept loop: one slow or hostile peer cannot
    freeze every other sender (the failure mode of wrapping the listener
    itself, where ``accept()`` performs the handshake inline).
    """

    def __init__(
        self,
        *args: Any,
        snagline_ssl_context: ssl.SSLContext,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.snagline_ssl_context = snagline_ssl_context

    def finish_request(self, request: Any, client_address: Any) -> None:
        try:
            tls_request = self.snagline_ssl_context.wrap_socket(
                request, server_side=True
            )
        except (ssl.SSLError, OSError):
            # Failed handshake (plaintext probe against the TLS port, port
            # scan, stale client): drop this one connection and keep serving.
            # Fail-open per project.md §1.2; never surface as a serving error.
            logger.debug(
                "snagline sidecar: TLS handshake failed from %s:%s",
                client_address[0],
                client_address[1],
                exc_info=True,
            )
            return
        # Annotated locally: typeshed binds this call's server parameter to
        # Self, which the subclass indirection here trips over.
        handler_class: Any = self.RequestHandlerClass
        handler_class(tls_request, client_address, self)


def make_server(
    monitor: Monitor,
    host: str = "127.0.0.1",
    port: int = 8787,
    auth_token: str | None = None,
    max_body_bytes: int = 1_000_000,
    max_risks: int = 1000,
    metrics_format: str | None = None,
    ssl_context: ssl.SSLContext | None = None,
    certfile: str | None = None,
    keyfile: str | None = None,
) -> ThreadingHTTPServer:
    """Construct a ready-to-``serve_forever()`` sidecar server.

    With ``ssl_context`` or ``certfile``/``keyfile`` the sidecar speaks
    HTTPS directly (issue #120): connections are wrapped server-side via
    stdlib ``ssl`` with the handshake running on each connection's own
    worker thread. Without them the server is plain HTTP, exactly as before.
    """
    tls_context = _resolve_ssl_context(ssl_context, certfile, keyfile)
    handler = make_handler(
        monitor, auth_token, max_body_bytes, max_risks, metrics_format
    )
    if tls_context is None:
        return ThreadingHTTPServer((host, port), handler)
    return _TLSThreadingHTTPServer(
        (host, port), handler, snagline_ssl_context=tls_context
    )


def serve(
    monitor: Monitor,
    host: str = "127.0.0.1",
    port: int = 8787,
    auth_token: str | None = None,
    max_body_bytes: int = 1_000_000,
    max_risks: int = 1000,
    metrics_format: str | None = None,
    ssl_context: ssl.SSLContext | None = None,
    certfile: str | None = None,
    keyfile: str | None = None,
) -> None:
    """Run the sidecar server in the foreground until interrupted."""
    tls_enabled = ssl_context is not None or certfile is not None
    server = make_server(
        monitor,
        host,
        port,
        auth_token,
        max_body_bytes,
        max_risks,
        metrics_format,
        ssl_context,
        certfile,
        keyfile,
    )
    logger.info(
        "snagline sidecar listening on %s://%s:%d (POST /events, GET /health)%s",
        "https" if tls_enabled else "http",
        host,
        port,
        "" if auth_token else " -- no auth token set, all endpoints are open",
    )
    try:
        server.serve_forever()
    finally:
        server.server_close()
