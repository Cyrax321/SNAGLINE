"""Issue #559: one dead endpoint starved every network sink in the process.

``bounded_post`` capped the number of in-flight sink POSTs with a single
module-global semaphore, and the cap itself is right (issue #423): a worker
holds a slot for its whole life and frees it in a ``finally``, so the ceiling
counts live threads rather than calls. The *scope* was wrong. An endpoint that
accepts the connection and never replies parks each POST's worker until its
deadline -- and a parked worker is what holds the slot. Every network sink in
the process (WebhookSink, SlackSink, PagerDutySink) plus the monitor's halt
webhook drew from that one pool, so a single dead escalation endpoint could
fill it and every other sink's deliveries would be dropped with
``SinkBusyError``, the enforcement directive included. The symptom looked in
the logs like a healthy sink failing.

The fix scopes the pool to each caller: a sink builds one in its constructor
and hands it to every POST, and the monitor keeps its own for the halt round
trip, so a dead endpoint can only ever exhaust the delivery budget of the sink
pointed at it. The module fallback pool remains for callers that pass none
(the CLI's ad-hoc forward, pre-#559 third-party sinks), so the #423 bound still
holds for them.

The end-to-end tests use real loopback servers, the way the issue was
reproduced; the isolation mechanics use a patched opener so the parked workers
are gated rather than slept.
"""

from __future__ import annotations

import json
import threading
import time
import urllib.request
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from unittest import mock

import pytest

import snagline.sinks.base as base
from snagline.events import StepEvent
from snagline.monitor import Monitor
from snagline.risk import FailureRisk, TriggerType
from snagline.sinks.base import SinkBusyError, bounded_post, new_inflight_pool
from snagline.sinks.pagerduty import PagerDutySink
from snagline.sinks.slack import SlackSink
from snagline.sinks.webhook import WebhookSink

# Short deadlines keep the suite fast; a worker abandoned at one stays parked.
_DEADLINE = 0.2
# How long a POST at the dead endpoint is left in flight. The caller's deadline
# and the worker's socket timeout are the same number, so a dead endpoint holds
# its slots only while its workers are still in flight -- see _park_a_burst.
_HOLD = 2.0
# The small per-sink cap the tests exercise; the shipped default is far larger.
_CAP = 3


@pytest.fixture
def small_cap(monkeypatch):
    """Shrink every sink's own pool to ``_CAP`` for one test.

    ``new_inflight_pool`` reads ``_MAX_INFLIGHT_POSTS`` at call time, so a sink
    built after this patch gets a small pool. Each pool is constructed fresh by
    the sinks the test builds, so parked workers from one test cannot leak
    slots into the next.
    """
    monkeypatch.setattr(base, "_MAX_INFLIGHT_POSTS", _CAP)


def _risk() -> FailureRisk:
    return FailureRisk(
        episode_id="ep-1",
        step_id="3",
        score=0.9,
        trigger="loop",
        detail="action repeated 3x in last 4 steps",
        timestamp=1718300000.0,
    )


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
        return None


# --- the dead endpoint, two ways --------------------------------------------


class _BlackHoleResponse:
    """A response whose body never arrives until ``gate`` is set.

    Models the dead endpoint: the worker parks inside ``read`` forever, exactly
    as it would on a socket that neither answers nor resets, so the caller's
    ``join(timeout)`` gives up and abandons a still-live thread -- still
    holding its slot.
    """

    def __init__(self, gate: threading.Event) -> None:
        self._gate = gate

    def __enter__(self) -> _BlackHoleResponse:
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def read(self, *args: object) -> bytes:
        # 5 s is a safety net so a failing test cannot hang the suite; every
        # test sets the gate in its own teardown well before then.
        self._gate.wait(5.0)
        return b""


class _OkResponse:
    """A completed exchange: the slot is freed at once."""

    def __enter__(self) -> _OkResponse:
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def read(self, *args: object) -> bytes:
        return b"{}"


def _url_routing_opener(dead_url: str, gate: threading.Event, calls: list[str]):
    """Patch the opener so only ``dead_url`` black-holes; everything else 200s.

    The isolation claim is about two destinations sharing one process, so the
    opener has to serve both under a single patch: the dead sink parks while
    the healthy one completes. Every exchange is recorded so a test can prove a
    POST was *started*, which a pool-exhausted caller never gets to do.
    """

    def open(req, timeout=None):
        calls.append(req.full_url)
        if req.full_url == dead_url:
            return _BlackHoleResponse(gate)
        return _OkResponse()

    return mock.patch.object(base._opener, "open", side_effect=open)


@contextmanager
def _server(handler_cls: type[_SilentHandler]):
    handler_cls.received = []
    # Bind loopback explicitly: the host is known, so the URL does not depend
    # on how the platform reports its own address.
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler_cls)
    httpd.daemon_threads = True
    port = httpd.socket.getsockname()[1]
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{port}/"
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=5)


class _SilentHandler(BaseHTTPRequestHandler):
    """Shared bookkeeping: what reached the server, and silent request logging."""

    received: list[bytes] = []

    def log_message(self, *args: object) -> None:
        return None


class _OkHandler(_SilentHandler):
    """A healthy endpoint: acks the POST and records the payload it received."""

    def do_POST(self) -> None:
        body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        reply = b"{}"
        self.send_response(200)
        self.send_header("Content-Length", str(len(reply)))
        self.end_headers()
        self.wfile.write(reply)
        _OkHandler.received.append(body)


class _BlackHoleHandler(_SilentHandler):
    """Accepts the connection and never replies until the test's gate opens.

    This is the failure mode the in-flight cap exists for: the socket is open,
    so no read timeout fires, and the caller's deadline is the only thing that
    ends the wait -- leaving the worker parked and holding its slot.
    """

    gate: threading.Event = threading.Event()

    def do_POST(self) -> None:
        self.rfile.read(int(self.headers.get("Content-Length", 0)))
        _BlackHoleHandler.gate.wait(10.0)
        self.send_response(200)
        self.send_header("Content-Length", "0")
        self.end_headers()
        _BlackHoleHandler.received.append(b"<parked>")


class _HaltHandler(_SilentHandler):
    """The enforcement endpoint: answers with a directive the monitor applies."""

    def do_POST(self) -> None:
        body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        reply = json.dumps({"action": "pause", "reason": "halted"}).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(reply)))
        self.end_headers()
        self.wfile.write(reply)
        _HaltHandler.received.append(body)


# --- each network sink gets a pool of its own --------------------------------


def _make_sink(cls: type, url: str) -> Any:
    """Build a network sink for the URL, or a routing key for PagerDuty."""
    if cls is PagerDutySink:
        return cls("routing-key-secret")
    return cls(url)


@pytest.mark.parametrize("cls", [WebhookSink, SlackSink, PagerDutySink])
def test_each_network_sink_builds_its_own_pool(small_cap, cls) -> None:
    # Two sinks of the same kind, built in one process, must not share an
    # in-flight cap: a pool per sink is what stops one dead destination from
    # spending another's delivery budget.
    one = _make_sink(cls, "https://a.example.invalid/hook")
    two = _make_sink(cls, "https://b.example.invalid/hook")
    assert one._inflight is not two._inflight
    assert one._inflight is not base._inflight_posts
    # ... and it is sized by the module default at construction time.
    assert one._inflight._value == _CAP


def test_new_inflight_pool_reads_the_cap_at_call_time(small_cap) -> None:
    assert new_inflight_pool()._value == _CAP
    assert new_inflight_pool(7)._value == 7
    # A cap below one is not a cap at all; clamp rather than build an unusable
    # pool that would reject every POST.
    assert new_inflight_pool(0)._value == 1


# --- the isolation claim, against a patched opener --------------------------


def test_a_dead_endpoint_exhausts_only_its_own_sink(small_cap) -> None:
    gate = threading.Event()
    calls: list[str] = []
    dead = WebhookSink("https://dead.example.invalid/hook", timeout=_DEADLINE)
    healthy = WebhookSink("https://ok.example.invalid/hook", timeout=_DEADLINE)
    try:
        with _url_routing_opener(dead._url, gate, calls):
            # Hammer the dead endpoint: every POST is abandoned at its deadline
            # with its worker still parked in read(), so it keeps its slot.
            for _ in range(_CAP * 3):
                dead.emit(_risk())  # fail-open: never raises
            # The dead sink has spent every slot it has...
            assert dead._inflight._value == 0
            # ...and the healthy sink's pool is untouched. Before #559 both
            # drew from one semaphore and this read 0 too.
            assert healthy._inflight._value == _CAP
            # So the healthy POST is still started -- a pool-exhausted caller
            # raises SinkBusyError before it ever reaches the opener.
            delivered = calls.count(healthy._url)
            healthy.emit(_risk())
            assert calls.count(healthy._url) == delivered + 1, (
                "a healthy sink must still deliver while another sink's endpoint "
                "holds all of its own slots open (issue #559)"
            )
    finally:
        gate.set()


# --- the issue's reproduction, against real loopback servers -----------------


def _park_a_burst(sink: Any, count: int) -> None:
    """Fire ``count`` emits at ``sink`` at once and let every one park.

    A dead endpoint only keeps its slots held while its workers are still in
    flight: the caller's ``join(deadline)`` and the worker's socket timeout are
    the same number, so a worker released by its own timeout frees its slot
    shortly after its caller gives up on it. A sequential loop therefore never
    fills the pool -- emit N's worker has already released by the time emit N+1
    runs. Issue #559's pile-up was many POSTs started at once, and that
    concurrency is what walks the pool to its ceiling, so the burst is fired
    on threads rather than in a loop. ``_HOLD`` is comfortably longer than the
    settle window below, so the slots stay held through the assertions.
    """
    threads = [
        threading.Thread(target=sink.emit, args=(_risk(),), daemon=True)
        for _ in range(count)
    ]
    for t in threads:
        t.start()
    # Loopback connect is fast; this only has to beat the socket timeout.
    time.sleep(_HOLD / 8)


def test_two_real_sinks_one_dead_endpoint_the_healthy_one_still_delivers(
    small_cap, monkeypatch
) -> None:
    gate = threading.Event()
    _BlackHoleHandler.gate = gate
    # Shrink the module fallback pool too: before #559 both sinks drew from it,
    # and it is a 64-slot pool at import time, so a test that only parks a few
    # POSTs would never reach the ceiling and the bug would look fixed. With
    # the fallback this small, the shared-pool behaviour drops the healthy
    # delivery outright.
    monkeypatch.setattr(base, "_inflight_posts", new_inflight_pool(_CAP))
    with _server(_OkHandler) as ok_url, _server(_BlackHoleHandler) as dead_url:
        dead = WebhookSink(dead_url, timeout=_HOLD)
        healthy = WebhookSink(ok_url, timeout=_DEADLINE)
        try:
            # Park more POSTs at the dead endpoint than it has slots, all at
            # once -- the issue's "72 POSTs against a black hole".
            _park_a_burst(dead, _CAP * 3)
            # The healthy sink shares only the process. Its delivery must land.
            healthy.emit(_risk())
            assert _OkHandler.received, (
                "a healthy sink must deliver while another sink's endpoint is "
                "dead; a shared in-flight pool drops this POST with "
                "SinkBusyError (issue #559)"
            )
        finally:
            gate.set()


# --- the sharpest consequence: the enforcement directive ---------------------


def test_the_halt_webhook_has_a_pool_of_its_own(small_cap) -> None:
    sink = WebhookSink("https://sink.example.invalid/hook")
    monitor = Monitor(
        [_FixedRiskDetector()],
        [sink],
        policy="halt_webhook",
        halt_url="https://halt.example.invalid/halt",
        halt_timeout_s=_DEADLINE,
    )
    # A stalled user sink must not be able to spend the enforcement round
    # trip's budget, and neither may the module fallback's other users.
    assert monitor._halt_inflight is not sink._inflight
    assert monitor._halt_inflight is not base._inflight_posts


def test_the_halt_webhook_passes_its_own_pool(small_cap) -> None:
    # Structural guard: the deadline and the no-redirect policy ride on
    # bounded_post, and now so does the pool the round trip draws from. A halt
    # path that regained the module default would silently regain #559.
    monitor = Monitor(
        [_FixedRiskDetector()],
        [],
        policy="halt_webhook",
        halt_url="https://halt.example.invalid/halt",
        halt_timeout_s=_DEADLINE,
    )
    with mock.patch("snagline.monitor.bounded_post", autospec=True) as post:
        monitor.ingest(_event())
    post.assert_called_once()
    assert post.call_args.kwargs["pool"] is monitor._halt_inflight, (
        "the halt round trip must draw from the monitor's own pool, not the "
        "sinks' shared one (issue #559)"
    )


def test_a_starved_sink_cannot_suppress_the_halt_directive(
    small_cap, monkeypatch
) -> None:
    # The issue's worst case: a user-configured escalation endpoint stalls, and
    # the halt directive -- the enforcement the operator asked for -- is the
    # delivery that gets dropped. The halt webhook now has its own budget.
    gate = threading.Event()
    _BlackHoleHandler.gate = gate
    # Shrink the module fallback too, as above: before #559 the halt round trip
    # drew from it alongside the sinks, and a 64-slot pool would hide the
    # starvation from a test that parks only a few POSTs.
    monkeypatch.setattr(base, "_inflight_posts", new_inflight_pool(_CAP))
    with _server(_HaltHandler) as halt_url, _server(_BlackHoleHandler) as dead_url:
        dead = WebhookSink(dead_url, timeout=_HOLD)
        monitor = Monitor(
            [_FixedRiskDetector()],
            [dead],
            policy="halt_webhook",
            halt_url=halt_url,
            halt_timeout_s=_DEADLINE * 2,
        )
        # Exhaust the user sink's pool on its dead endpoint first, so the halt
        # round trip runs while no sink slot is free.
        _park_a_burst(dead, _CAP * 3)
        try:
            monitor.ingest(_event())
            assert monitor.last_directive.action == "pause", (
                "the enforcement directive must survive a stalled user sink; "
                "when the halt webhook shared the sinks' pool it was dropped "
                "as a SinkBusyError and the policy silently went on logging "
                "failures (issue #559)"
            )
            assert _HaltHandler.received, "the halt POST itself must have landed"
        finally:
            gate.set()


# --- callers that pass no pool keep the #423 bound ---------------------------


def _request() -> urllib.request.Request:
    return urllib.request.Request(
        "https://example.invalid/hook", data=b"{}", method="POST"
    )


def test_a_caller_without_a_pool_falls_back_to_the_module_pool(monkeypatch) -> None:
    # The CLI's ad-hoc forward and any pre-#559 third-party sink call
    # bounded_post with no pool; they still draw from the module fallback and
    # are still bounded against a dead endpoint (issue #423).
    fallback = new_inflight_pool(1)
    monkeypatch.setattr(base, "_inflight_posts", fallback)
    assert fallback.acquire(blocking=False)
    try:
        with pytest.raises(SinkBusyError):
            bounded_post(_request(), _DEADLINE)
    finally:
        fallback.release()


def test_an_explicit_pool_shadows_the_module_fallback(monkeypatch) -> None:
    # A caller's own pool is authoritative for it: a full module fallback must
    # not drop a caller that passes a pool of its own.
    own = new_inflight_pool(2)
    monkeypatch.setattr(base, "_inflight_posts", new_inflight_pool(1))
    assert base._inflight_posts.acquire(blocking=False)  # fallback is full
    try:
        with mock.patch.object(
            base._opener, "open", side_effect=lambda req, timeout=None: _OkResponse()
        ):
            assert bounded_post(_request(), _DEADLINE, pool=own) == b"{}"
    finally:
        base._inflight_posts.release()
