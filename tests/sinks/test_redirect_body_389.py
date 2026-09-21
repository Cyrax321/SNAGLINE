"""Issue #389: the network sinks followed a 301/302/303 as a bodyless GET.

``urllib``'s redirect handler rebuilds the request with ``method="GET"`` and
no body for any of those codes (it does the same for 307/308, which the RFCs
say must keep the method). The redirect target then answered the GET with 200,
``emit`` never raised, and the alert was never delivered -- an endpoint that
moved, or one wanting the trailing slash the operator left off, silently
stopped paging with no error anywhere.

The three network sinks now go through ``bounded_post``'s opener, which
re-issues the redirect as a POST carrying the original body, provided the
redirect stays on the same scheme, host and port. An off-host redirect raises
instead: these sinks POST a credential, and following a redirect elsewhere
would hand it to whoever answers there.

Reproduced against a real local server, because the whole failure is in the
interaction between ``urllib``'s handler chain and a 302 -- mocking
``urlopen`` would mock away the bug along with the call.
"""

from __future__ import annotations

import json
import threading
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest import mock

import pytest

from snagline.risk import SEVERITY_CRITICAL, FailureRisk
from snagline.sinks.pagerduty import PagerDutySink
from snagline.sinks.slack import SlackSink
from snagline.sinks.webhook import WebhookSink

# urllib's own loop counter, which still applies through the custom handler.
_MAX_REDIRECTS = urllib.request.HTTPRedirectHandler.max_redirections


def _risk() -> FailureRisk:
    return FailureRisk(
        episode_id="ep-1",
        step_id="3",
        score=0.9,
        trigger="loop",
        detail="action repeated 3x in last 4 steps",
        timestamp=1718300000.0,
        severity=SEVERITY_CRITICAL,
    )


class _Server:
    """A local server that 302s ``/start`` to ``redirect_to`` and records
    every request it receives."""

    def __init__(self) -> None:
        # (path, method, body, content-type) per request, in arrival order.
        self.received: list[tuple[str, str, bytes, str | None]] = []
        # Set before the first request lands; a relative value stays on this
        # host, an absolute one can point at another server entirely.
        self.redirect_to = "/final"
        outer = self

        class _Handler(BaseHTTPRequestHandler):
            def _record(self) -> None:
                n = int(self.headers.get("Content-Length") or 0)
                body = self.rfile.read(n) if n else b""
                outer.received.append(
                    (self.path, self.command, body, self.headers.get("Content-Type"))
                )

            def do_POST(self) -> None:  # noqa: N802
                self._record()
                if self.path == "/start":
                    self.send_response(302)
                    self.send_header("Location", outer.redirect_to)
                    self.end_headers()
                    return
                self.send_response(200)
                self.end_headers()

            def do_GET(self) -> None:  # noqa: N802
                self._record()
                self.send_response(200)
                self.end_headers()

            def log_message(self, *args: object) -> None:
                pass  # a test server must not spam the suite's output

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        self._server.daemon_threads = True
        self.base = f"http://127.0.0.1:{self._server.server_address[1]}"
        self._thread = threading.Thread(target=self._server.serve_forever)
        self._thread.start()

    def stop(self) -> None:
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=2)


@pytest.fixture
def origin():
    server = _Server()
    yield server
    server.stop()


@pytest.fixture
def other_host():
    """A second server on a different port -- a different host:port to the
    redirect check, which is what makes the cross-origin case distinct from a
    same-host path redirect."""
    server = _Server()
    yield server
    server.stop()


# --- the body reaches the redirect target (issue #389) -----------------------


def test_webhook_body_survives_a_same_host_redirect(origin):
    sink = WebhookSink(origin.base + "/start")
    sink.emit(_risk())

    requests = origin.received
    assert requests[0][0] == "/start"
    delivered = next(r for r in requests if r[0] == "/final")
    assert delivered[1] == "POST", "the redirect must be re-issued as a POST"
    payload = json.loads(delivered[2])
    assert payload["episode_id"] == "ep-1"
    assert delivered[3] == "application/json", (
        "Content-Type must travel with the body, or it arrives as form data"
    )
    assert not any(r for r in requests if r[1] == "GET"), (
        "no GET may be issued: that is the dropped alert (issue #389)"
    )


def test_slack_body_survives_a_same_host_redirect(origin):
    sink = SlackSink(origin.base + "/start")
    sink.emit(_risk())

    delivered = next(r for r in origin.received if r[0] == "/final")
    assert delivered[1] == "POST"
    payload = json.loads(delivered[2])
    assert payload["text"] and "loop" in payload["text"]


def test_pagerduty_body_survives_a_same_host_redirect(origin):
    # The endpoint is a module constant, so point it at the local server.
    with mock.patch("snagline.sinks.pagerduty._EVENTS_API", origin.base + "/start"):
        sink = PagerDutySink("ROUTING-KEY")
        sink.emit(_risk())

    delivered = next(r for r in origin.received if r[0] == "/final")
    assert delivered[1] == "POST"
    payload = json.loads(delivered[2])
    assert payload["routing_key"] == "ROUTING-KEY"
    assert payload["event_action"] == "trigger"


def test_a_redirect_to_the_scheme_and_port_is_followed(origin, other_host):
    """A redirect to the same host but a different port is a different origin
    for these sinks: the credential leaves the endpoint it was configured
    for."""
    origin.redirect_to = other_host.base + "/final"
    sink = WebhookSink(origin.base + "/start")
    sink.emit(_risk())

    assert not other_host.received, (
        "an off-origin redirect must not deliver the body to the other host"
    )
    assert all(r[0] == "/start" for r in origin.received)


def test_off_origin_redirect_does_not_raise_out_of_emit(origin, other_host, caplog):
    """The sink stays fire-and-forget: the failure is logged, not thrown into
    the host's ingest path."""
    origin.redirect_to = other_host.base + "/final"
    sink = WebhookSink(origin.base + "/start")
    with caplog.at_level("ERROR", logger="snagline"):
        sink.emit(_risk())  # must not raise
    assert not other_host.received
    assert "refusing to follow a redirect" in caplog.text, (
        "the failure must say what happened and why: this is the line an "
        f"operator reads: {caplog.text!r}"
    )


def test_an_off_origin_redirect_logs_the_destination_not_the_body(
    origin, other_host, caplog
):
    """The routing key is a credential, so the failure line must name where
    the POST was headed without repeating what it carried."""
    origin.redirect_to = other_host.base + "/final"
    with mock.patch("snagline.sinks.pagerduty._EVENTS_API", origin.base + "/start"):
        sink = PagerDutySink("SECRET-ROUTING-KEY")
        with caplog.at_level("ERROR", logger="snagline"):
            sink.emit(_risk())
    assert not other_host.received
    assert "SECRET-ROUTING-KEY" not in caplog.text, "the credential reached the log"


def test_no_redirect_unchanged(origin):
    """Scope guard: the redirect handling must not change the happy path. A
    plain 200 is still a single POST that arrives once."""
    sink = WebhookSink(origin.base + "/final")
    sink.emit(_risk())
    assert [(r[0], r[1]) for r in origin.received] == [("/final", "POST")]


def test_redirect_loop_is_bounded(origin):
    """A redirect that points back at itself must terminate, not spin for the
    run -- urllib's loop counter still applies through the custom handler."""
    origin.redirect_to = "/start"
    sink = WebhookSink(origin.base + "/start", timeout=5)
    sink.emit(_risk())  # must return, not hang
    assert len(origin.received) <= _MAX_REDIRECTS + 1
