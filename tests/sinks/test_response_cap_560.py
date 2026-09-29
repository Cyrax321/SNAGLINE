"""Issue #560: the network sinks buffered an entire reply they then threw away.

``bounded_post`` (issue #395) documents its ``max_bytes`` parameter because "a
malicious or broken endpoint cannot make the exchange unbounded by streaming an
endless body." But the three network sinks -- and the ``snagline hook --url``
forward -- all called it as ``bounded_post(req, timeout)``, so the cap never
applied and the reply was read whole before being discarded. The wall-clock
deadline bounds *time*, not memory: a fast endless or chunked stream allocates
without bound well inside a 2 s budget, and with the pooled POSTs of issue #559
the address space multiplies. The halt webhook was the one caller passing its
cap; the sinks were the outliers.

The fix gives the fire-and-forget callers a shared ``_MAX_SINK_RESPONSE_BYTES``
-- 64 KiB, like the halt webhook's own ``_MAX_HALT_RESPONSE_BYTES`` -- and
passes it at every call site. A delivery is decided by the status code, not by
anything in the body, so a small ceiling is behaviour-preserving.

The behavioural tests go through a real opener against a loopback server that
streams a body far larger than the cap, because the claim is about how much
``read`` actually buffers -- mocking the exchange would only restate the fix.
The structural tests pin the delegation, so a sink that regained a bare
``bounded_post(req, timeout)`` call cannot silently lose the cap again.
"""

from __future__ import annotations

import io
import json
import sys
import threading
import tracemalloc
import urllib.request
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, HTTPServer
from unittest import mock

import pytest

from snagline.risk import SEVERITY_CRITICAL, FailureRisk
from snagline.sinks.base import bounded_post
from snagline.sinks.pagerduty import PagerDutySink
from snagline.sinks.slack import SlackSink
from snagline.sinks.webhook import WebhookSink

# Far larger than the 64 KiB cap, small enough to stay civil in the suite: the
# point is that the buffered peak is bounded by the cap, not by this number.
_BODY_BYTES = 16 * 1024 * 1024
# The shipped ceiling, restated locally so the memory test measures the bound
# against the number the fix ships rather than against the fix's own symbol.
_CAP_BYTES = 64 * 1024


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
    """Shared bookkeeping: what reached the server, and silent request logging."""

    received: list[bytes] = []

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


class _HugeReply(_Recording):
    """Streams ``_BODY_BYTES`` back on every POST, then the sink discards it.

    A healthy endpoint's ack is a line or two; this one answers with 16 MiB of
    padding, which is exactly the shape a broken or malicious endpoint can
    produce inside the deadline. ``tracemalloc`` therefore sees the whole body
    land in Python memory unless the read is capped.
    """

    _BODY = b"x" * _BODY_BYTES

    def do_POST(self) -> None:
        _HugeReply.received.append(
            self.rfile.read(int(self.headers.get("Content-Length", 0)))
        )
        self.send_response(200)
        self.send_header("Content-Length", str(_BODY_BYTES))
        self.end_headers()
        self.wfile.write(self._BODY)


# --- the mechanism the sinks now rely on -------------------------------------


def test_bounded_post_truncates_the_reply_at_the_cap() -> None:
    # The contract every call site now depends on: with a cap, ``read`` returns
    # at most that many bytes however large the endpoint's body is.
    with _server(_HugeReply) as url:
        req = urllib.request.Request(url, data=b"{}", method="POST")
        body = bounded_post(req, 5.0, 4096)
    assert len(body) == 4096, (
        "the reply must be truncated at max_bytes, not buffered in full"
    )


def test_bounded_post_without_a_cap_still_buffers_the_whole_body() -> None:
    # The negative control, and the reason the parameter matters at all: with
    # ``max_bytes=None`` the full 16 MiB comes back, which is what the sinks
    # used to do.
    with _server(_HugeReply) as url:
        req = urllib.request.Request(url, data=b"{}", method="POST")
        body = bounded_post(req, 5.0)
    assert len(body) == _BODY_BYTES


# --- the bug, end to end through a real sink ---------------------------------


def test_a_sink_buffers_only_the_cap_not_the_whole_reply() -> None:
    # The issue's reproduction, trimmed to suite-civil sizes: without the cap a
    # single emit peaks at the full body; with it the peak is bounded by the
    # shipped 64 KiB cap plus slack, never by what the endpoint chose to send.
    with _server(_HugeReply) as url:
        sink = WebhookSink(url, timeout=5.0)
        tracemalloc.start()
        try:
            before = tracemalloc.get_traced_memory()[0]
            sink.emit(_risk())
            peak = tracemalloc.get_traced_memory()[1]
        finally:
            tracemalloc.stop()
    assert peak - before < 1024 * 1024, (
        "one emit must not buffer the endpoint's whole reply: peaked at "
        f"{(peak - before) / 1024:.0f} KiB against a {_BODY_BYTES // (1024 * 1024)} MiB body"
    )
    assert peak - before >= _CAP_BYTES, (
        "the cap is what the sink reads -- a peak far below it would mean the "
        "test is no longer exercising the read at all"
    )


def test_the_delivery_still_reaches_the_endpoint_and_logs_no_failure(caplog) -> None:
    # Positive control: capping the read is behaviour-preserving -- the POST is
    # still sent in full, still counts as delivered on a 200, and logs nothing,
    # exactly as an uncapped emit against a terse ack already did.
    with caplog.at_level("ERROR", logger="snagline"), _server(_HugeReply) as url:
        WebhookSink(url, timeout=5.0).emit(_risk())
        bodies = list(_HugeReply.received)
    assert bodies, "the alert payload must still be posted in full"
    assert json.loads(bodies[0])["episode_id"] == "ep-1"
    assert not caplog.records, "a 200 with a large body is not a delivery failure"


# --- structural guard: no call site may drop the cap -------------------------


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
def test_every_network_sink_passes_a_response_cap(module: str, sink) -> None:
    # The issue was that the sinks called ``bounded_post(req, timeout)`` with no
    # third argument. Pin the cap at the delegation so it cannot regress to
    # ``None`` without a test noticing -- the deadline bound and the redirect
    # refusal are pinned the same way.
    with mock.patch(f"{module}.bounded_post", autospec=True) as post:
        sink.emit(_risk())
    post.assert_called_once()
    max_bytes = post.call_args.args[2] if len(post.call_args.args) > 2 else None
    assert max_bytes, (
        "a sink must cap the reply read, not buffer it unbounded (issue #560)"
    )


def test_the_hook_url_forward_also_caps_the_reply(monkeypatch) -> None:
    # ``snagline hook --url`` forwards a canonical StepEvent and discards the
    # reply too, so it shares the cap. Driven in-process through ``main`` so the
    # delegation can be pinned the same way as the sinks'.
    from snagline import cli

    payload = json.dumps(
        {
            "step_id": "s1",
            "episode_id": "e1",
            "timestamp": 1.0,
            "action_type": "tool_call",
            "action_signature": "Bash:npm test",
        }
    )
    monkeypatch.setattr(sys, "stdin", io.StringIO(payload))
    with mock.patch("snagline.cli.bounded_post", autospec=True) as post:
        assert cli.main(["hook", "--url", "https://forward.example/events"]) == 0
    post.assert_called_once()
    max_bytes = post.call_args.args[2] if len(post.call_args.args) > 2 else None
    assert max_bytes, "the hook forward must cap the reply read too (issue #560)"
