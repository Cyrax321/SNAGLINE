"""Issues #415 / #416: the halt webhook's POST was neither bounded nor redirect-safe.

``Monitor._run_halt_webhook`` used bare ``urllib.request.urlopen`` for the
enforcement round trip. Two defects followed, both inherited from the sinks
(#395 / #389) and both worse in this position than they were there:

- ``urlopen``'s ``timeout=`` is applied per socket operation and only *after*
  name resolution, so a stalled resolver or a body trickled one byte per
  interval just under the timeout parks the exchange far past the configured
  ``halt_timeout_s``. The webhook is called synchronously from ``ingest()`` --
  ``ingest -> _dispatch -> _enforce -> _run_halt_webhook`` -- so the stall is
  this episode's step, and no ``policy_error`` is counted because nothing ever
  raised (#415).
- ``urllib`` rewrites a ``301``/``302``/``303`` POST into a bodyless GET to the
  ``Location`` URL and hands the final ``2xx`` back. The halt path parses that
  body into a ``HaltDirective``, so a redirect let the enforcement decision
  arrive from a server the operator never configured. A sink silently
  misroutes an alert; this silently misroutes the halt (#416).

Both call sites now go through the shared ``bounded_post`` from
``sinks/base.py``, which bounds the whole exchange by a wall-clock deadline on
a daemon thread and carries the ``_NoRedirect`` handler.

The behavioural tests use a real opener against loopback servers: the claims
are about urllib's machinery end to end, so mocking the helper would only
restate the fix.
"""

from __future__ import annotations

import json
import logging
import threading
import time
import urllib.request
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest import mock

from snagline.events import StepEvent
from snagline.monitor import Monitor
from snagline.risk import FailureRisk, TriggerType

# A short deadline keeps the suite fast; the ratios are what carry the test.
_DEADLINE = 0.5
# Each socket read must finish well inside the per-operation timeout, so no
# single read trips it while the exchange as a whole outruns the budget.
_TRICKLE_SLEEP = _DEADLINE / 10
# Separates "abandoned at the deadline" from "ran the trickle to completion".
_MUST_BEAT = _DEADLINE * 1.5


def _event(step_id: str = "s1", episode_id: str = "ep1") -> StepEvent:
    return StepEvent(
        step_id=step_id,
        episode_id=episode_id,
        timestamp=1.0,
        action_type="tool_call",
        action_signature=f"sig-{step_id}",
    )


class _FixedRiskDetector:
    """Emits one synthetic risk at a fixed score on every observe()."""

    name = "fixed_risk"

    def __init__(self, score: float = 0.9, trigger: TriggerType = "loop") -> None:
        self.score = score
        self.trigger: TriggerType = trigger

    def observe(self, event: StepEvent) -> FailureRisk | None:
        return FailureRisk(
            event.episode_id,
            event.step_id,
            self.score,
            self.trigger,
            "synthetic risk",
            event.timestamp,
        )

    def reset(self, episode_id: str) -> None:
        pass


def _monitor(halt_url: str, timeout: float = _DEADLINE) -> Monitor:
    return Monitor(
        [_FixedRiskDetector()],
        [],
        policy="halt_webhook",
        halt_url=halt_url,
        halt_timeout_s=timeout,
    )


class _Recording(BaseHTTPRequestHandler):
    """Shared bookkeeping: what reached the server, and silent request logging.

    ``received`` holds ``(path, body)`` pairs so a test can tell the POST that
    carried the payload apart from the bodyless GET a redirect would produce.
    """

    received: list[tuple[str, bytes]] = []

    def log_message(self, *args: object) -> None:
        return None


@contextmanager
def _server(handler_cls: type[_Recording]):
    handler_cls.received = []
    # Bind loopback explicitly: the host is known, so the URL does not depend
    # on how the platform reports its own address.
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler_cls)
    httpd.daemon_threads = True
    port = httpd.socket.getsockname()[1]
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{port}/halt"
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=5)


# --- #415: the budget is a wall-clock deadline, not a per-socket hint --------


class _Trickling(_Recording):
    """A 200 whose body arrives one byte per interval, past the deadline.

    Every individual socket read finishes inside the timeout, so the
    per-operation contract never fires, while the exchange as a whole outruns
    the budget -- the shape issue #395 measured at 2.0 s budgeted, 36.2 s
    taken.
    """

    sleep: float = _TRICKLE_SLEEP

    def do_POST(self) -> None:
        self.rfile.read(int(self.headers.get("Content-Length", 0)))
        body = json.dumps({"action": "pause", "reason": "late"}).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        for i in range(0, len(body)):
            time.sleep(_Trickling.sleep)
            self.wfile.write(body[i : i + 1])
            self.wfile.flush()
        _Trickling.received.append(("/halt", body))


def test_a_trickling_halt_endpoint_does_not_hold_ingest_past_its_budget() -> None:
    # Pre-fix the trickle completed inside the caller's view of the budget
    # several times over and the directive landed anyway; now the exchange is
    # abandoned at the deadline and counted as a policy error.
    with _server(_Trickling) as url:
        monitor = _monitor(url)
        started = time.perf_counter()
        monitor.ingest(_event())
        elapsed = time.perf_counter() - started
    assert elapsed < _MUST_BEAT, (
        f"halt webhook held ingest for {elapsed:.2f}s against a {_DEADLINE}s budget"
    )
    assert monitor.last_directive.action == "continue", (
        "a trickling endpoint must not still set the directive"
    )
    assert monitor.metrics()["policy_errors"] == 1


class _Stalled(_Recording):
    """Accepts the POST and never answers, so only a deadline can escape."""

    def do_POST(self) -> None:
        self.rfile.read(int(self.headers.get("Content-Length", 0)))
        _Stalled.received.append(("/halt", b"<read>"))
        time.sleep(30.0)


def test_a_stalled_halt_endpoint_is_abandoned_at_the_deadline() -> None:
    # Passes both before and after: a server that sends nothing does trip the
    # per-operation read timeout, so this pins the deadline bound without
    # changing the healthy failure path.
    with _server(_Stalled) as url:
        monitor = _monitor(url)
        started = time.perf_counter()
        monitor.ingest(_event())
        elapsed = time.perf_counter() - started
    assert elapsed < _MUST_BEAT
    assert monitor.last_directive.action == "continue"
    assert monitor.metrics()["policy_errors"] == 1


def test_the_halt_webhook_goes_through_bounded_post() -> None:
    # Structural guard: the deadline and the no-redirect policy both ride on
    # the helper, so a halt path that regained a bare urlopen would silently
    # regain both defects.
    monitor = _monitor("http://127.0.0.1:1/halt")
    with mock.patch("snagline.monitor.bounded_post", autospec=True) as post:
        monitor.ingest(_event())
    post.assert_called_once()
    # The configured deadline and the response byte cap are what the helper
    # receives -- the wiring the two fixes ride on.
    assert post.call_args.args[1] == _DEADLINE
    assert post.call_args.args[2] > 0, "the response byte cap must be passed"


# --- #416: a redirect must not choose the enforcement directive -------------


class _Redirecting(_Recording):
    """Answers the halt POST with a 302, so the opener must not follow it."""

    def do_POST(self) -> None:
        body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        _Redirecting.received.append((self.path, body))
        self.send_response(302)
        self.send_header("Location", "/elsewhere")
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_GET(self) -> None:
        _Redirecting.received.append((self.path, b"<empty>"))
        # The directive a redirect would have applied: nobody configured this.
        body = json.dumps({"action": "pause", "reason": "misrouted"}).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def test_a_redirect_cannot_set_the_directive(caplog) -> None:
    with (
        caplog.at_level(logging.ERROR, logger="snagline"),
        _server(_Redirecting) as url,
    ):
        monitor = _monitor(url, timeout=3.0)
        monitor.ingest(_event())
    assert monitor.last_directive.action == "continue", (
        "a redirect target must not be able to issue a directive at all"
    )
    assert monitor.last_directive.reason == "", (
        "a redirect target must not be able to supply the directive's reason"
    )
    assert monitor.metrics()["policy_errors"] == 1, (
        "an unfollowed redirect must count as a policy error, not a success"
    )
    # The monitor logs the failure one-line and fault-once (issue #14), so the
    # endpoint is named as unreachable rather than as a 3xx -- the counter and
    # the directive are what the redirect changed, and what this pins.
    assert "halt webhook" in caplog.text


def test_the_halt_payload_does_not_leak_to_the_redirect_target() -> None:
    with _server(_Redirecting) as url:
        monitor = _monitor(url, timeout=3.0)
        monitor.ingest(_event())
        bodies = [b for p, b in _Redirecting.received if p == "/elsewhere"]
    assert "/halt" in [p for p, _ in _Redirecting.received], "the POST must be tried"
    assert not bodies, "no payload may reach the redirect target"


# --- positive controls: the working path is unchanged -----------------------


class _Halting(_Recording):
    """Answers the directive the operator's endpoint would answer."""

    def do_POST(self) -> None:
        body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        _Halting.received.append((self.path, body))
        reply = json.dumps({"action": "pause", "reason": "too many retries"})
        data = reply.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def test_a_direct_2xx_still_sets_the_directive(caplog) -> None:
    # Passes both before and after: a healthy endpoint still decides, and
    # still logs nothing, so the deadline and redirect changes add no failure
    # to the working path.
    with caplog.at_level(logging.ERROR, logger="snagline"), _server(_Halting) as url:
        monitor = _monitor(url, timeout=3.0)
        monitor.ingest(_event())
    assert monitor.last_directive.action == "pause"
    assert monitor.last_directive.reason == "too many retries"
    assert monitor.metrics()["policy_errors"] == 0
    assert not [r for r in caplog.records if r.levelno >= logging.ERROR]


def test_bounded_post_returns_the_body_and_caps_it() -> None:
    # The helper now returns the reply, and honours the cap the halt path
    # already enforced -- an endpoint that streams an endless body must not
    # make the exchange unbounded by other means.
    from snagline.sinks.base import bounded_post

    class _Big(_Recording):
        body_bytes: int = 4096

        def do_POST(self) -> None:
            self.rfile.read(int(self.headers.get("Content-Length", 0)))
            data = b"x" * _Big.body_bytes
            self.send_response(200)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    with _server(_Big) as url:
        req = urllib.request.Request(url, data=b"{}", method="POST")
        capped = bounded_post(req, 3.0, max_bytes=16)
    assert len(capped) == 16

    with _server(_Big) as url:
        req = urllib.request.Request(url, data=b"{}", method="POST")
        whole = bounded_post(req, 3.0)
    assert len(whole) == 4096


# --- #390, same block: the halt URL is a credential on failure --------------


def test_the_halt_url_is_not_logged_with_its_credentials(caplog) -> None:
    # The halt URL can carry basic auth (``user:pass@host``), and the failure
    # line is what an operator reads when enforcement stops working.
    with caplog.at_level(logging.ERROR, logger="snagline"):
        # Nothing listens here, so the POST fails immediately.
        monitor = _monitor("http://halter:hunter@127.0.0.1:1/halt")
        monitor.ingest(_event())
    assert "halter:hunter" not in caplog.text, "the credential must not be logged"
    assert "127.0.0.1" in caplog.text, "the host must still be identifiable"
    assert monitor.metrics()["policy_errors"] == 1
