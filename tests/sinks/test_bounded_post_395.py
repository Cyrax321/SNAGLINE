"""Issue #395: the network sinks' ``timeout=`` was per-socket-operation only.

``urllib.request.urlopen`` applies its ``timeout=`` to individual socket
operations, and only after ``socket.create_connection`` has finished name
resolution. Two gaps follow: name resolution is not covered at all, and a
server that trickles its body one byte every interval just under the timeout
never trips a read timeout. Issue #395 measured a 2.0 s budget taking 36.2 s
on that trickle, on a request the sink reported as successful. The sinks are
dispatched from the ingest path, so the stall is the host agent's step.

The three network sinks now go through ``bounded_post``, which bounds the
whole exchange by a wall-clock deadline.

The behavioural tests below go through a sink rather than the helper, because
the claim is about what ``emit`` does to its caller -- and because a helper
that did not exist yet would break collection instead of demonstrating the
overrun.
"""

from __future__ import annotations

import socket
import time
from unittest import mock

import pytest

from snagline.risk import FailureRisk
from snagline.sinks.pagerduty import PagerDutySink
from snagline.sinks.slack import SlackSink
from snagline.sinks.webhook import WebhookSink

# A short deadline keeps the suite fast; the ratios are what carry the test.
_DEADLINE = 0.15
# Long enough that the per-operation contract alone cannot escape it.
_STALL = _DEADLINE * 4
# Separates "abandoned at the deadline" from "ran the stall to completion".
_MUST_BEAT = _DEADLINE * 2.5


def _risk() -> FailureRisk:
    return FailureRisk(
        episode_id="ep-1",
        step_id="3",
        score=0.5,
        trigger="loop",
        detail="action repeated 3x in last 4 steps",
        timestamp=1718300000.0,
    )


class _TrickleResponse:
    """A 200 whose body arrives one chunk per interval, past the deadline.

    Models the issue's measured server: every individual socket read finishes
    inside the timeout, so the per-operation contract never fires, but the
    exchange as a whole far outruns the budget.
    """

    def __init__(self, chunk_sleep: float, chunks: int) -> None:
        self._chunk_sleep = chunk_sleep
        self._chunks = chunks

    def __enter__(self) -> _TrickleResponse:
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def read(self, *args: object) -> bytes:
        # One ``read()`` call performs many socket reads internally; each of
        # them is under the timeout, and their sum is not.
        for _ in range(self._chunks):
            time.sleep(self._chunk_sleep)
        return b"x" * self._chunks


def _patch_opener(response_factory):
    # ``bounded_post`` goes through the module's own opener (the no-redirect
    # policy rides on it), so that is the seam a fake server plugs into -- not
    # ``urllib.request.urlopen``, which the helper no longer calls.
    class _Opener:
        def open(self, request, timeout=None):
            return response_factory(request, timeout)

    return mock.patch("snagline.sinks.base._opener", _Opener())


def _trickling_urlopen(chunk_sleep: float, chunks: int):
    return _patch_opener(lambda req, timeout: _TrickleResponse(chunk_sleep, chunks))


# --- the two gaps from the issue ---------------------------------------------


def test_a_trickling_body_does_not_outrun_the_deadline() -> None:
    # The issue's reproduction: a 2 s budget, a 36.2 s emit, reported success.
    # Here 0.15 s budgeted against a 0.4 s trickle -- the same 2.7x ratio.
    sink = WebhookSink("https://hooks.example/alerts", timeout=_DEADLINE)
    started = time.monotonic()
    with _trickling_urlopen(_DEADLINE / 3, 8):
        sink.emit(_risk())
    elapsed = time.monotonic() - started
    assert elapsed < _MUST_BEAT, (
        f"a trickling body must not hold emit past the {_DEADLINE}s deadline "
        f"(took {elapsed:.2f}s)"
    )


def test_a_stalled_resolver_does_not_outrun_the_deadline(monkeypatch) -> None:
    # This gap the issue established by reading ``socket.create_connection``
    # rather than by measurement, because it needs a controlled slow
    # resolver: getaddrinfo runs and blocks before the socket exists, so
    # urlopen's timeout is never in force while it does. Model the stall on
    # the resolver itself.
    real_getaddrinfo = socket.getaddrinfo

    def slow_getaddrinfo(host, port, *args, **kwargs):
        time.sleep(_STALL)
        # Answer with a loopback once resolution does return, so the request
        # never leaves the machine; the deadline abandons it first anyway.
        return real_getaddrinfo("127.0.0.1", port, *args, **kwargs)

    monkeypatch.setattr(socket, "getaddrinfo", slow_getaddrinfo)
    sink = WebhookSink("https://resolver.example.invalid/hook", timeout=_DEADLINE)
    started = time.monotonic()
    sink.emit(_risk())
    elapsed = time.monotonic() - started
    assert elapsed < _MUST_BEAT, (
        f"a stalled resolver must not hold emit past the {_DEADLINE}s deadline "
        f"(took {elapsed:.2f}s)"
    )


def test_an_overrun_is_reported_not_silently_succeeded(caplog) -> None:
    # The issue's other half: the trickle returned 200, so the sink logged
    # nothing and the alert was believed delivered. An overrun is now a
    # TimeoutError the sink logs, so an operator can see the escalation
    # endpoint is unreachable.
    sink = WebhookSink("https://hooks.example/alerts", timeout=_DEADLINE)
    with caplog.at_level("ERROR", logger="snagline"), _trickling_urlopen(_STALL, 4):
        sink.emit(_risk())
    # The failure is named by class only: an exception's own text can embed the
    # destination URL (issue #390), so the deadline message itself no longer
    # reaches the log.
    assert "TimeoutError" in caplog.text, (
        "an overrun must be logged as a failure, not reported as a delivery"
    )


# --- positive controls: the working path is unchanged ------------------------


def test_a_fast_post_is_delivered_and_silent(caplog) -> None:
    # Passes both before and after the fix: an ordinary request still
    # succeeds, which is the point -- the deadline bounds the broken path,
    # it does not add latency to the healthy one.
    captured = {}

    class _Resp:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def read(self):
            return b"ok"

    def fake_urlopen(req, timeout=None):
        captured["timeout"] = timeout
        captured["req"] = req
        return _Resp()

    with (
        caplog.at_level("ERROR", logger="snagline"),
        _patch_opener(fake_urlopen),
    ):
        WebhookSink("https://hooks.example/alerts", timeout=2.0).emit(_risk())
    assert captured["timeout"] == 2.0
    # The configured timeout still reaches the exchange as the per-operation
    # bound.
    assert not caplog.records, "a successful post must log nothing"


def test_an_endpoint_failure_is_still_relayed(caplog) -> None:
    # Passes both ways: the per-request failure handling the sinks rely on is
    # preserved -- the error is logged fail-open rather than raised. Named by
    # class only, since the exception's text can embed the destination (issue
    # #390).
    def refusing(req, timeout=None):
        raise OSError("connection refused")

    with (
        caplog.at_level("ERROR", logger="snagline"),
        _patch_opener(refusing),
    ):
        WebhookSink("https://hooks.example/alerts").emit(_risk())
    assert "OSError" in caplog.text


# --- the three network sinks actually route through it -----------------------


@pytest.mark.parametrize(
    "module,sink",
    [
        (
            "snagline.sinks.webhook",
            WebhookSink("https://hooks.example/alerts"),
        ),
        (
            "snagline.sinks.slack",
            SlackSink("https://hooks.slack.example/services/T/B/secret"),
        ),
        (
            "snagline.sinks.pagerduty",
            PagerDutySink("routing-key"),
        ),
    ],
)
def test_each_network_sink_bounds_its_post(module: str, sink) -> None:
    # A structural guard: a sink that talks directly to urlopen again would
    # silently regain the unbounded path, so pin the delegation. The sink
    # modules bind the helper into their own namespace, so patch it there --
    # that is the target a future regression would have to remove.
    from snagline.sinks.base import bounded_post  # noqa: F401  (guard import)

    with mock.patch(f"{module}.bounded_post", autospec=True) as post:
        sink.emit(_risk())
    post.assert_called_once()
    assert post.call_args.args[1] == sink._timeout
