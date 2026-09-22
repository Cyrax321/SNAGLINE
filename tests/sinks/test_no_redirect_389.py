"""Issue #389: the network sinks followed a 3xx, which silently dropped the alert.

``urllib`` honours a ``301``/``302``/``303`` by re-issuing the request to the
``Location`` URL as a GET with no body -- see
``HTTPRedirectHandler.redirect_request``, which returns
``Request(newurl, method="GET", ...)`` with no ``data``. It then hands the final
2xx back to the caller. A sink that hit a redirect therefore reported a
successful delivery while its payload travelled with the POST the server
rejected, and went to a destination the server chose: a trailing-slash hop, a
proxy canonicalisation, or -- the worst pairing -- an auth redirect from an
expired credential that would otherwise have shown as a 401. A ``307``/``308``
instead raised ``HTTPError``, so the failure was silent for exactly the common
codes and loud for the rare ones.

The fix is a ``_NoRedirect`` handler on ``bounded_post``'s opener: returning
``None`` from ``redirect_request`` makes ``http_error_30x`` give up, which falls
through to ``HTTPDefaultErrorHandler`` and raises ``HTTPError`` for the 3xx,
which the sinks already log fail-open.

The behavioural tests go through a real opener against a loopback server,
because the claim is about urllib's redirect machinery end to end -- mocking the
handler would only restate the fix.
"""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from contextlib import contextmanager
from http.client import HTTPMessage
from http.server import BaseHTTPRequestHandler, HTTPServer
from io import BytesIO
from unittest import mock

import pytest

from snagline.risk import SEVERITY_CRITICAL, FailureRisk
from snagline.sinks.pagerduty import PagerDutySink
from snagline.sinks.slack import SlackSink
from snagline.sinks.webhook import WebhookSink


def _risk() -> FailureRisk:
    return FailureRisk(
        episode_id="ep-1",
        step_id="3",
        score=0.9,
        severity=SEVERITY_CRITICAL,
        trigger="loop",
        detail="action repeated 3x in last 4 steps",
        timestamp=1718300000.0,
    )


class _Recording(BaseHTTPRequestHandler):
    """Shared bookkeeping: what reached the server, and silent request logging.

    ``received`` holds ``(path, body)`` pairs so a test can tell the POST that
    carried the payload apart from the bodyless GET a redirect would have
    produced.
    """

    received: list[tuple[str, bytes]] = []

    def log_message(self, *args: object) -> None:
        return None


@contextmanager
def _server(handler_cls: type[_Recording]):
    handler_cls.received = []
    # Bind loopback explicitly: the host is known, so the URL does not depend on
    # how the platform reports its own address.
    httpd = HTTPServer(("127.0.0.1", 0), handler_cls)
    port = httpd.socket.getsockname()[1]
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{port}/webhook"
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=5)


# --- the bug: the payload must not be redirected away ------------------------


class _Redirecting(_Recording):
    """Answers a POST with a 302 elsewhere, so the opener must not follow it."""

    def do_POST(self) -> None:
        body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        _Redirecting.received.append((self.path, body))
        self.send_response(302)
        self.send_header("Location", "/elsewhere")
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_GET(self) -> None:
        _Redirecting.received.append((self.path, b"<empty>"))
        self.send_response(200)
        self.send_header("Content-Length", "2")
        self.end_headers()
        self.wfile.write(b"ok")


def test_a_redirect_does_not_become_a_silent_success(caplog) -> None:
    # The issue's reproduction: the server saw the POST that carried the alert,
    # then answered 302, and the sink counted the bodyless follow-up GET as the
    # delivery. Now the 3xx must surface as a logged failure instead.
    with caplog.at_level("ERROR", logger="snagline"), _server(_Redirecting) as url:
        WebhookSink(url, timeout=5.0).emit(_risk())
        paths = [p for p, _ in _Redirecting.received]
    assert "/webhook" in paths, "the POST must still be attempted"
    assert "/elsewhere" not in paths, (
        "the sink must not follow a redirect to a URL it never sent the payload to"
    )


def test_the_redirect_is_reported_as_a_failure(caplog) -> None:
    # The 302 is an HTTPError now, so the operator sees the endpoint is
    # misrouted rather than believing the alert landed.
    with caplog.at_level("ERROR", logger="snagline"), _server(_Redirecting) as url:
        WebhookSink(url, timeout=5.0).emit(_risk())
    assert "302" in caplog.text, "an unfollowed redirect must be logged, not swallowed"


def test_the_payload_does_not_leak_to_the_redirect_target() -> None:
    # The alert body travelled with the POST the server rejected; nothing may be
    # re-sent to the redirect target.
    with _server(_Redirecting) as url:
        WebhookSink(url, timeout=5.0).emit(_risk())
        bodies = [b for p, b in _Redirecting.received if p == "/elsewhere"]
    assert not bodies, "no payload may reach the redirect target"


@pytest.mark.parametrize(
    "module,sink",
    [
        ("snagline.sinks.webhook", WebhookSink("https://hooks.example/alerts")),
        (
            "snagline.sinks.slack",
            SlackSink("https://hooks.slack.example/services/T/B/secret"),
        ),
        ("snagline.sinks.pagerduty", PagerDutySink("routing-key")),
    ],
)
def test_each_network_sink_refuses_redirects(module: str, sink) -> None:
    # A structural guard: the policy rides on ``bounded_post``'s opener, so a
    # sink that regained a bare ``urlopen`` would silently regain the redirect
    # too. Pin the delegation, as with the deadline bound.
    with mock.patch(f"{module}.bounded_post", autospec=True) as post:
        sink.emit(_risk())
    post.assert_called_once()


# --- positive controls: the working path is unchanged ------------------------


class _OK(_Recording):
    """Answers 200 directly; the healthy path a redirect must not disturb."""

    def do_POST(self) -> None:
        body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        _OK.received.append((self.path, body))
        self.send_response(200)
        self.send_header("Content-Length", "2")
        self.end_headers()
        self.wfile.write(b"ok")


def test_a_direct_2xx_is_delivered_and_silent(caplog) -> None:
    # Passes both before and after: an endpoint that answers 200 directly still
    # receives the payload and still logs nothing, so refusing redirects adds
    # no failure to the healthy path.
    with caplog.at_level("ERROR", logger="snagline"), _server(_OK) as url:
        WebhookSink(url, timeout=5.0).emit(_risk())
    bodies = [b for _, b in _OK.received]
    assert bodies, "the POST must be delivered"
    assert json.loads(bodies[0])["episode_id"] == "ep-1"
    assert not caplog.records, "a delivered alert must log nothing"


class _TempRedirect(_Recording):
    """Answers 307, which urllib already refused to rewrite a POST into."""

    def do_POST(self) -> None:
        self.rfile.read(int(self.headers.get("Content-Length", 0)))
        self.send_response(307)
        self.send_header("Location", "/elsewhere")
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_GET(self) -> None:  # pragma: no cover - must never be reached
        self.send_response(200)
        self.send_header("Content-Length", "2")
        self.end_headers()
        self.wfile.write(b"ok")


def test_a_307_is_still_reported_and_never_followed(caplog) -> None:
    # urllib raised on a 307 for a POST already -- it preserves the method and
    # refuses to rewrite it -- so this is the path that worked before and must
    # keep working: reported, not silently misrouted.
    with caplog.at_level("ERROR", logger="snagline"), _server(_TempRedirect) as url:
        WebhookSink(url, timeout=5.0).emit(_risk())
    assert "307" in caplog.text, "a 307 must be reported, not followed"


# --- the handler itself ------------------------------------------------------


def test_the_handler_refuses_a_302() -> None:
    # The contract the opener depends on: ``redirect_request`` returns None --
    # "I can't handle this, let another handler try" -- and none can, so
    # ``OpenerDirector`` falls through to ``HTTPDefaultErrorHandler`` and raises.
    # ``http_error_302`` delegates to it, so this pins the refusal without
    # rebuilding the chain by hand. The import is local because the handler is
    # the fix itself -- it does not exist on an unfixed tree.
    from snagline.sinks.base import _NoRedirect

    handler = _NoRedirect()
    request = urllib.request.Request("http://127.0.0.1/webhook", data=b"{}")
    headers = HTTPMessage()
    headers["location"] = "/elsewhere"
    assert handler.http_error_302(request, BytesIO(), 302, "Found", headers) is None
