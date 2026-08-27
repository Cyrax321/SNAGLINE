"""Sidecar server mode (project.md §7) -- for non-Python agents.

A minimal stdlib ``http.server`` endpoint any runtime (TypeScript, Node, a
shell script, a Claude Code hook) can POST ``StepEvent`` JSON to:

    POST /events             body: a StepEvent as a JSON object      -> monitor.ingest()
                               or a JSON array of StepEvent objects     (batched)
    POST /hooks/claude-code  body: a native Claude Code hook payload -> mapped + ingested
    POST /episodes/end       body: {"episode_id": "<id>"}            -> monitor.end_episode()
    GET  /health                                                     -> 200 OK
    GET  /metrics                                                    -> Prometheus text exposition (v0.0.4)
    GET  /directive                                                  -> the latest halt-webhook directive

A host that knows an episode is over can signal it with ``POST
/episodes/end`` carrying ``{"episode_id": "..."}`` (issue #123): the sidecar
forwards to ``Monitor.end_episode()`` and drops the id from the
``snagline_episodes_active`` gauge immediately, so long-lived processes stop
counting finished episodes without waiting for cap eviction.

POST bodies are capped at ``max_body_bytes`` (default 1 MB); larger payloads
get 413. This keeps the sidecar safe to expose and able to absorb buffered
telemetry from any host. Each connection also has a read timeout
(``server_read_timeout``, default 30 s, ``SNAGLINE_SERVER_READ_TIMEOUT``)
so a stalled sender that declares a large ``Content-Length`` but dribbles
a few bytes cannot pin a handler thread indefinitely.

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

``GET /directive`` reports the newest halt-webhook directive as
``{"action": "continue"|"pause", "reason": ..., "timestamp": ...}`` (issue
#169). Without it, a sidecar started with ``--halt-forward URL`` had nowhere
to publish the answer: the directive landed on the in-process
``Monitor.last_directive`` and non-Python hosts, the whole point of server
mode, could not read it. The resource exists as soon as the Monitor does and
defaults to continue, so the endpoint always answers 200. It is auth-gated
like ``GET /risks``, because a directive can pause the host.

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
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import parse_qs, urlsplit

from snagline.adapters.claude_code import HookTracker, ingest_payload
from snagline.config import Config
from snagline.events import StepEvent
from snagline.monitor import HaltDirective, Monitor

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
# Default socket read timeout in seconds (issue #130). Applied to the
# handler class via ``StreamRequestHandler.timeout``; stdlib converts a
# stalled ``rfile.read(n)`` into a logged timeout and a closed connection.
_DEFAULT_READ_TIMEOUT = 30.0


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


def _resolve_read_timeout(explicit: float | None) -> float:
    """Pick the effective read timeout for sidecar connections.

    Precedence: explicit argument, then ``SNAGLINE_SERVER_READ_TIMEOUT`` via
    ``Config``, then the built-in ``Config`` default (30 s). Non-positive or
    non-numeric values are logged and replaced by the default so a bad
    knob never takes the sidecar down (fail-open, like
    ``_resolve_metrics_format``).
    """
    candidate: Any = explicit
    if candidate is None:
        try:
            candidate = Config.from_env_overrides().get("server_read_timeout")
        except Exception:
            logger.debug(
                "snagline: env lookup for server_read_timeout failed",
                exc_info=True,
            )
            candidate = None
    if candidate is None:
        candidate = Config().server_read_timeout
    try:
        value = float(candidate)
    except (TypeError, ValueError):
        logger.warning(
            "snagline: invalid server_read_timeout %r; using default %s",
            candidate,
            _DEFAULT_READ_TIMEOUT,
        )
        return _DEFAULT_READ_TIMEOUT
    if not math.isfinite(value) or value <= 0:
        logger.warning(
            "snagline: server_read_timeout %r must be positive; using default %s",
            candidate,
            _DEFAULT_READ_TIMEOUT,
        )
        return _DEFAULT_READ_TIMEOUT
    return value


def _resolve_episode_ttl(explicit: float | None) -> float | None:
    """Pick the effective episode TTL for the sidecar gauge (issue #173).

    Precedence: explicit argument, then ``SNAGLINE_EPISODE_TTL_SECONDS`` via
    ``Config``, then the built-in ``Config`` default (None = disabled).
    Malformed or non-positive values are logged and treated as disabled so a
    bad knob never takes the sidecar down (fail-open). ``0`` explicitly
    disables as well, keeping the default byte-identical behavior.
    Uses wall-clock seconds for the TTL value but monotonic time for expiry
    checks, so NTP jumps cannot mass-expire or freeze entries; docs note
    replay traffic will resurrect stale ids.
    """
    candidate: Any = explicit
    if candidate is None:
        try:
            candidate = Config.from_env_overrides().get("episode_ttl_seconds")
        except Exception:
            logger.debug(
                "snagline: env lookup for episode_ttl_seconds failed",
                exc_info=True,
            )
            candidate = None
    if candidate is None:
        candidate = Config().episode_ttl_seconds
    if candidate is None:
        return None
    try:
        value = float(candidate)
    except (TypeError, ValueError):
        logger.warning(
            "snagline: invalid episode_ttl_seconds %r; TTL disabled",
            candidate,
        )
        return None
    if not math.isfinite(value):
        logger.warning(
            "snagline: episode_ttl_seconds %r must be finite; TTL disabled",
            candidate,
        )
        return None
    if value == 0:
        return None
    if value < 0:
        logger.warning(
            "snagline: episode_ttl_seconds %r must be positive; TTL disabled",
            candidate,
        )
        return None
    return value


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


def _directive_payload(directive: Any) -> dict[str, Any]:
    """Render a :class:`HaltDirective` as the ``GET /directive`` body.

    Coerced rather than trusted: the handler holds whatever object the caller
    passed as ``monitor``, and a body that cannot be serialized would turn a
    read-only endpoint into a 500.
    """
    return {
        "action": str(directive.action),
        "reason": str(directive.reason),
        "timestamp": float(directive.timestamp),
    }


class SidecarMetricsCollector:
    """Process-lifetime sidecar counters, renderable as Prometheus text.

    Counts risks per ``(trigger, severity)`` by acting as one extra sink on
    the Monitor, plus HTTP-layer totals: events accepted, time spent inside
    ``Monitor.ingest``, and distinct episode ids seen (bounded table with
    least-recently-seen eviction, ids only, no content). Every entry point
    swallows its own exceptions: like any sink, this must never take down
    ingestion or serving.
    """

    def __init__(
        self,
        max_episodes: int = _MAX_TRACKED_EPISODES,
        episode_ttl_seconds: float | None = None,
        clock: Callable[[], float] | None = None,
    ) -> None:
        self._lock = threading.Lock()
        self._risks: dict[tuple[str, str], int] = {}
        self._events_total = 0
        self._ingest_count = 0
        self._ingest_sum = 0.0
        # Episode id table: reseeing an id moves it to the end, so when the
        # cap bites we forget the episode quiet for the longest time, which
        # is the right approximation of "active" for a long-lived sidecar.
        # With TTL enabled each id also carries a monotonic last-seen stamp;
        # the table stays ids-plus-a-float, never content, and the same cap
        # continues to bound memory (issue #173).
        self._episodes: OrderedDict[str, float] = OrderedDict()
        self._max_episodes = max(1, int(max_episodes))
        self._ttl = _resolve_episode_ttl(episode_ttl_seconds)
        self._clock = clock or time.monotonic

    def emit(self, risk: Any) -> None:
        """AlertSink protocol entry point; counts one risk by label pair."""
        try:
            key = (str(risk.trigger), str(risk.severity))
            with self._lock:
                self._risks[key] = self._risks.get(key, 0) + 1
        except Exception:
            logger.debug("snagline: metrics collector emit failed", exc_info=True)

    def _expire_stale_locked(self) -> None:
        """Remove ids not seen for longer than TTL (caller holds _lock)."""
        if self._ttl is None:
            return
        try:
            now = self._clock()
            # OrderedDict keeps insertion order = last-seen order because
            # record_ingest moves reseen ids to the end. Expired ids are
            # exactly those at the front whose age exceeds TTL; we can stop
            # at the first fresh entry. Bounded scan: at most one pass over
            # the table per sweep, and sweep happens only on ingest and
            # scrape, so amortized O(1) per step (project.md section 1.5).
            expired: list[str] = []
            for eid, last_seen in self._episodes.items():
                if now - last_seen > self._ttl:
                    expired.append(eid)
                else:
                    break
            for eid in expired:
                self._episodes.pop(eid, None)
        except Exception:
            logger.debug("snagline: TTL sweep failed", exc_info=True)

    def record_ingest(self, episode_id: str | None, elapsed_seconds: float) -> None:
        """Record one accepted event and its ingest wall time."""
        try:
            with self._lock:
                self._events_total += 1
                self._ingest_count += 1
                self._ingest_sum += max(0.0, float(elapsed_seconds))
                if episode_id is not None:
                    key = str(episode_id)
                    now = self._clock()
                    self._episodes.pop(key, None)
                    self._episodes[key] = now
                    self._expire_stale_locked()
                    while len(self._episodes) > self._max_episodes:
                        self._episodes.popitem(last=False)
                else:
                    self._expire_stale_locked()
        except Exception:
            logger.debug("snagline: metrics record failed", exc_info=True)

    def end_episode(self, episode_id: str) -> None:
        """Forget one episode id so the gauge reflects completion at once.

        Without this an ended episode keeps occupying a table slot until cap
        eviction forgets it (issue #123). Idempotent by design: ending an
        unknown or already-ended id is a no-op, never an error. Guarded like
        every other entry point: bookkeeping must never take serving down.
        """
        try:
            key = str(episode_id)
            with self._lock:
                self._episodes.pop(key, None)
        except Exception:
            logger.debug(
                "snagline: metrics collector end_episode failed", exc_info=True
            )

    def snapshot(self) -> dict[str, Any]:
        """Return a consistent copy of the counters for rendering."""
        with self._lock:
            self._expire_stale_locked()
            return {
                "events_total": self._events_total,
                "risks": sorted(self._risks.items()),
                "episodes_active": len(self._episodes),
                "ingest_count": self._ingest_count,
                "ingest_sum": self._ingest_sum,
            }

    def render_prometheus(self, monitor_metrics: dict[str, int]) -> str:
        """Render the full exposition body (version 0.0.4 text format).

        ``monitor_metrics`` is the ``Monitor.metrics()`` snapshot; those
        counters are exposed under ``snagline_monitor_*_total`` names so the
        sidecar view adds process-lifetime detail without renaming anything
        the JSON endpoint already published. A key the snapshot does not
        carry renders as 0, so an older Monitor still produces a valid body.
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
            "# HELP snagline_episodes_active Distinct episode ids seen since start;"
            " ids removed again on POST /episodes/end."
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
            (
                "policy_errors",
                "snagline_monitor_policy_errors_total",
                "Enforcement faults swallowed fail-open (#93): callback raises"
                " and halt-webhook timeout/error fallbacks.",
            ),
        )
        for key, name, help_text in monitor_families:
            lines.append(f"# HELP {name} {help_text}")
            lines.append(f"# TYPE {name} counter")
            lines.append(f"{name} {monitor_metrics.get(key, 0)}")
        # Issue #184: retained-episode count so the leak is observable rather
        # than inferred from RSS. Gauge, not counter; ids only.
        retained = monitor_metrics.get(
            "retained_episodes", monitor_metrics.get("live_episodes", 0)
        )
        lines.append(
            "# HELP snagline_monitor_retained_episodes Current live episode ids retained by Monitor (issue #184)."
        )
        lines.append("# TYPE snagline_monitor_retained_episodes gauge")
        lines.append(f"snagline_monitor_retained_episodes {retained}")
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
    read_timeout: float | None = None,
    episode_ttl_seconds: float | None = None,
) -> type[BaseHTTPRequestHandler]:
    """Build a request-handler class bound to ``monitor``.

    ``metrics_format`` pins the default GET /metrics body; when omitted the
    SNAGLINE_METRICS_FORMAT environment variable decides, then the built-in
    "prometheus" default (see ``_resolve_metrics_format``).
    ``read_timeout`` sets the socket read timeout in seconds; when omitted
    ``SNAGLINE_SERVER_READ_TIMEOUT`` / ``Config.server_read_timeout`` decides,
    then the built-in 30 s default (see ``_resolve_read_timeout``).
    ``episode_ttl_seconds`` sets the TTL for ``snagline_episodes_active`` ids;
    when omitted ``SNAGLINE_EPISODE_TTL_SECONDS`` / ``Config.episode_ttl_seconds``
    decides, then disabled (None) by default (see ``_resolve_episode_ttl``).
    """
    tracker = HookTracker()

    class _Handler(BaseHTTPRequestHandler):
        # BaseHTTPRequestHandler inherits ``timeout`` from
        # StreamRequestHandler (default None = block forever). Setting it
        # here bounds every ``rfile.read(n)``; stdlib turns a stall into a
        # logged "Request timed out" and a closed connection (issue #130).
        timeout = _DEFAULT_READ_TIMEOUT
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
            elif split.path == "/directive":
                self._serve_directive()
            else:
                self._respond(404, {"error": "not found"})

        def _serve_directive(self) -> None:
            """Serve the newest halt-webhook directive (issue #169).

            Read-only: the Monitor owns the directive and its lock, so this
            renders whatever the public property hands back. A monitor that
            predates enforcement, or any failure reading it, answers with the
            default continue rather than raising -- the same fail-open the
            webhook path itself takes on a timeout or a dead endpoint, and a
            500 here would tell a polling host nothing it could act on.
            """
            try:
                payload = _directive_payload(self.snagline_monitor.last_directive)
            except Exception:
                logger.debug(
                    "snagline: directive unreadable, reporting continue", exc_info=True
                )
                payload = _directive_payload(HaltDirective())
            self._respond(200, payload)

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
                declared = self._parse_content_length()
                if declared is None:
                    return
                self._discard_overcap_body(declared)
                self._respond(401, {"error": "unauthorized"})
                return
            length = self._parse_content_length()
            if length is None:
                return
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
            elif self.path == "/episodes/end":
                self._post_episodes_end()
            elif self.path == "/risks":
                self._post_risks()
            else:
                self._discard_overcap_body(length)
                self._respond(404, {"error": "not found"})

        def _post_risks(self) -> None:
            """Receive a ``FailureRisk`` JSON emitted by a WebhookSink elsewhere.

            This closes the loop with ``sinks.webhook.WebhookSink``: a remote
            or sibling agent posts its detected risks here, and the sidecar
            records/displays them. Validation is lenient and fail-open -- a
            malformed body is acknowledged (202) so the sender never retries.
            """
            body = self._read_body()
            if body is None:
                return
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
            if body is None:
                return
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

        def _post_episodes_end(self) -> None:
            """Accept an end-of-episode signal (issue #123).

            Body is ``{"episode_id": "<id>"}``. The id is forwarded to
            ``Monitor.end_episode`` (detector finalization plus state
            teardown) and removed from the metrics episodes table so the
            gauge stops counting it immediately. Ids only: the body is never
            logged or retained. Fail-open end to end -- malformed bodies get
            400, monitor failures are logged and swallowed, and the client
            always gets a clean response because this endpoint must never be
            the one that takes the sidecar down.
            """
            body = self._read_body()
            if body is None:
                return
            try:
                payload = json.loads(body.decode("utf-8"))
            except (ValueError, json.JSONDecodeError):
                self._respond(400, {"error": "invalid end-episode JSON"})
                return
            episode_id = (
                payload.get("episode_id") if isinstance(payload, dict) else None
            )
            if not isinstance(episode_id, str) or not episode_id:
                self._respond(400, {"error": "episode_id must be a non-empty string"})
                return
            try:
                self.snagline_monitor.end_episode(episode_id)
            except Exception:
                # Monitor dispatch is fail-open internally already; this guard
                # covers misconfigured monitors (fail_open=False) so a bad
                # signal can still never escape into the server plumbing.
                logger.exception("snagline: end_episode dispatch failed")
            finally:
                collector = self.snagline_collector
                if collector is not None:
                    collector.end_episode(episode_id)
            self._respond(200, {"status": "ended", "episode_id": episode_id})

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
            if body is None:
                return
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

        def _parse_content_length(self) -> int | None:
            """Parse Content-Length, or None if malformed (already answered 400).

            An absent Content-Length header is treated as length 0: the body
            is silently empty, which is acceptable for these telemetry-only
            endpoints. A present but non-numeric or negative value is answered
            with 400 {"error": "invalid Content-Length"} and the caller
            must return without reading any body bytes. Shared by ``do_POST``
            and ``_read_body`` so every entry point fails the same way.
            """
            raw = self.headers.get("Content-Length")
            if raw is None:
                return 0
            text = raw.strip() if isinstance(raw, str) else str(raw).strip()
            if text == "":
                self._respond(400, {"error": "invalid Content-Length"})
                return None
            try:
                length = int(text)
            except (ValueError, TypeError):
                self._respond(400, {"error": "invalid Content-Length"})
                return None
            if length < 0:
                self._respond(400, {"error": "invalid Content-Length"})
                return None
            return length

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

        def _read_body(self) -> bytes | None:
            """Read the request body, or None if Content-Length was malformed.

            On malformed Content-Length the helper has already sent the 400
            response; the caller must return immediately without sending
            another response, otherwise the connection would see two status
            lines.
            """
            length = self._parse_content_length()
            if length is None:
                return None
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
    _Handler.snagline_collector = SidecarMetricsCollector(
        episode_ttl_seconds=episode_ttl_seconds
    )
    _attach_metrics_collector(monitor, _Handler.snagline_collector)
    _Handler.snagline_metrics_format = _resolve_metrics_format(metrics_format)
    _Handler.timeout = _resolve_read_timeout(read_timeout)
    return _Handler


def _resolve_ssl_context(
    ssl_context: ssl.SSLContext | None,
    certfile: str | None,
    keyfile: str | None,
    client_ca: str | None = None,
) -> ssl.SSLContext | None:
    """Return the server-side TLS context for the listener, or None.

    An explicit ``ssl_context`` wins; otherwise ``certfile`` (with optional
    ``keyfile``) is loaded into a fresh ``ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)``.
    When ``client_ca`` is given the CA bundle is loaded via
    ``load_verify_locations`` and ``verify_mode`` is set to
    ``ssl.CERT_REQUIRED`` so every handshake must present a trusted client
    certificate (issue #145). Contradictory or incomplete arguments raise
    ``ValueError`` at startup: silently ignoring half a TLS configuration
    would bind plaintext while the operator believes the listener is encrypted.
    """
    if ssl_context is not None:
        if certfile is not None or keyfile is not None or client_ca is not None:
            raise ValueError(
                "pass either ssl_context or certfile/keyfile to serve(), not both"
            )
        return ssl_context
    if certfile is None:
        if keyfile is not None:
            raise ValueError("keyfile requires certfile")
        if client_ca is not None:
            raise ValueError("client-ca requires certfile")
        return None
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(certfile, keyfile)
    if client_ca is not None:
        context.load_verify_locations(client_ca)
        context.verify_mode = ssl.CERT_REQUIRED
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
    read_timeout: float | None = None,
    episode_ttl_seconds: float | None = None,
    ssl_context: ssl.SSLContext | None = None,
    certfile: str | None = None,
    keyfile: str | None = None,
    client_ca: str | None = None,
) -> ThreadingHTTPServer:
    """Construct a ready-to-``serve_forever()`` sidecar server.

    With ``ssl_context`` or ``certfile``/``keyfile`` the sidecar speaks
    HTTPS directly (issue #120): connections are wrapped server-side via
    stdlib ``ssl`` with the handshake running on each connection's own
    worker thread. Without them the server is plain HTTP, exactly as before.
    ``read_timeout`` bounds stalled body reads (issue #130); ``None`` resolves
    via ``SNAGLINE_SERVER_READ_TIMEOUT`` / ``Config.server_read_timeout``.
    ``episode_ttl_seconds`` sets the TTL for ``snagline_episodes_active`` ids;
    ``None`` resolves via ``SNAGLINE_EPISODE_TTL_SECONDS`` / ``Config.episode_ttl_seconds``
    and defaults to disabled (see ``_resolve_episode_ttl``). When
    ``client_ca`` is given the server requests a client certificate on
    every handshake and verifies it against that CA bundle (issue #145).
    """
    tls_context = _resolve_ssl_context(ssl_context, certfile, keyfile, client_ca)
    handler = make_handler(
        monitor,
        auth_token,
        max_body_bytes,
        max_risks,
        metrics_format,
        read_timeout,
        episode_ttl_seconds,
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
    read_timeout: float | None = None,
    episode_ttl_seconds: float | None = None,
    ssl_context: ssl.SSLContext | None = None,
    certfile: str | None = None,
    keyfile: str | None = None,
    client_ca: str | None = None,
) -> None:
    """Run the sidecar server in the foreground until interrupted."""
    tls_enabled = ssl_context is not None or certfile is not None
    server = make_server(
        monitor,
        host=host,
        port=port,
        auth_token=auth_token,
        max_body_bytes=max_body_bytes,
        max_risks=max_risks,
        metrics_format=metrics_format,
        read_timeout=read_timeout,
        episode_ttl_seconds=episode_ttl_seconds,
        ssl_context=ssl_context,
        certfile=certfile,
        keyfile=keyfile,
        client_ca=client_ca,
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
