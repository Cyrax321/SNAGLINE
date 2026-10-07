"""Issue #559: the in-flight POST pool was shared process-global.

``bounded_post`` gated *every* sink POST in the process through one
module-level semaphore, and the Monitor's halt webhook drew from the same one.
An endpoint that accepts the connection and never replies parks each worker
(and its slot) until its deadline; once enough are parked, the next POST is
refused with ``SinkBusyError`` without ever being attempted. Because the pool
was shared, one dead escalation endpoint silently stopped deliveries for every
*other* network sink -- a cross-sink failure that looks in the logs exactly
like a healthy sink being broken -- and, worst, a stalled user-configured sink
could suppress the halt directive itself.

The fix scopes the cap to one sink: each network sink and the halt webhook hold
their own pool, so a dead destination can only exhaust its own delivery
budget. The #423 ceiling (parked threads and file descriptors) is preserved per
destination.
"""

from __future__ import annotations

import threading
import urllib.request
from unittest import mock

import pytest

import snagline.sinks.base as base
from snagline.risk import FailureRisk
from snagline.sinks.base import SinkBusyError, bounded_post
from snagline.sinks.slack import SlackSink
from snagline.sinks.webhook import WebhookSink

_DEADLINE = 0.1
# A small per-sink cap so a test fills it in a handful of emits.
_CAP = 3


def _risk() -> FailureRisk:
    return FailureRisk(
        episode_id="ep-1",
        step_id="3",
        score=0.9,
        trigger="meltdown",
        detail="tool-choice entropy collapsed",
        timestamp=1718300000.0,
    )


def _request(url: str) -> urllib.request.Request:
    return urllib.request.Request(url, data=b"{}", method="POST")


class _Ok:
    """A response whose body arrives at once -- the healthy endpoint."""

    def __enter__(self) -> _Ok:
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def read(self, *args: object) -> bytes:
        return b"{}"


class _BlackHole:
    """A response whose body never arrives until ``gate`` is set.

    The worker parks inside ``read`` exactly as it would on a socket that
    neither answers nor resets, so the caller abandons a still-live thread that
    keeps holding its slot.
    """

    def __init__(self, gate: threading.Event) -> None:
        self._gate = gate

    def __enter__(self) -> _BlackHole:
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def read(self, *args: object) -> bytes:
        self._gate.wait(5.0)
        return b""


@pytest.fixture
def small_sink_cap(monkeypatch):
    """Shrink the *per-sink* pool to ``_CAP`` for one test.

    ``make_post_pool`` resolves its default at call time, so patching the
    module global resizes every pool built afterwards without touching the
    shipped default.
    """
    monkeypatch.setattr(base, "_MAX_SINK_INFLIGHT_POSTS", _CAP)


def test_each_network_sink_gets_its_own_pool() -> None:
    # The isolation is structural, not accidental: two sinks of the same class
    # must not share a pool object, or a dead one would still starve the other.
    webhook = WebhookSink("https://hooks.example/one")
    other = WebhookSink("https://hooks.example/two")
    slack = SlackSink("https://hooks.example/slack")
    assert webhook._post_pool is not other._post_pool
    assert webhook._post_pool is not slack._post_pool


def test_a_dead_sink_does_not_starve_a_healthy_one(small_sink_cap) -> None:
    # The issue's reproduction: park enough workers on the dead endpoint to
    # fill a pool, then emit through a *different* sink pointed at a healthy
    # one. Before #559 both drew from the same pool and the healthy delivery
    # was dropped with SinkBusyError.
    gate = threading.Event()
    delivered: list[urllib.request.Request] = []

    def routing_urlopen(req, timeout=None):
        if req.full_url.endswith("/dead"):
            return _BlackHole(gate)
        delivered.append(req)
        return _Ok()

    dead = WebhookSink("https://hooks.example/dead")
    healthy = SlackSink("https://hooks.example/healthy")
    try:
        with mock.patch.object(base._opener, "open", side_effect=routing_urlopen):
            # Park workers on the dead endpoint until its own pool is full.
            # Sinks are fire-and-forget, so drive the pool directly here; the
            # point is what it does to the *other* sink's delivery below.
            for _ in range(_CAP * 2):
                with pytest.raises((TimeoutError, SinkBusyError)):
                    bounded_post(
                        _request("https://hooks.example/dead"),
                        _DEADLINE,
                        pool=dead._post_pool,
                    )
            # The healthy sink must still deliver, untouched by the dead one.
            healthy.emit(_risk())
        assert len(delivered) == 1, (
            "a dead endpoint must not starve a healthy sink's pool; the "
            f"healthy delivery happened {len(delivered)} time(s), expected 1"
        )
    finally:
        gate.set()


def test_the_halt_webhook_pool_is_isolated_from_the_sinks(small_sink_cap) -> None:
    # The sharpest consequence: the enforcement directive shared the sinks'
    # pool, so a stalled user-configured sink could suppress the halt. The
    # Monitor's pool is its own, so the directive fires while the sink starves
    # only itself.
    from snagline.monitor import Monitor

    gate = threading.Event()
    posted: list[urllib.request.Request] = []

    def routing_urlopen(req, timeout=None):
        if req.full_url.endswith("/dead"):
            return _BlackHole(gate)
        posted.append(req)
        return _Ok()

    monitor = Monitor(
        [],
        [],
        policy="halt_webhook",
        halt_url="https://halt.example/enforce",
        halt_timeout_s=_DEADLINE,
    )
    dead = WebhookSink("https://hooks.example/dead")
    try:
        with mock.patch.object(base._opener, "open", side_effect=routing_urlopen):
            for _ in range(_CAP * 2):
                with pytest.raises((TimeoutError, SinkBusyError)):
                    bounded_post(
                        _request("https://hooks.example/dead"),
                        _DEADLINE,
                        pool=dead._post_pool,
                    )
            monitor._run_halt_webhook(_risk())
        assert len(posted) == 1, (
            "the halt directive must still be delivered while a user sink is "
            f"starving its own pool; it fired {len(posted)} time(s), expected 1"
        )
    finally:
        gate.set()


def test_a_full_pool_still_refuses_only_its_own_sink(small_sink_cap, caplog) -> None:
    # The #423 guarantee is preserved per sink: once *this* sink's pool is full
    # the next POST on that sink is refused before a worker starts, and a
    # second sink is unaffected.
    dead = WebhookSink("https://hooks.example/dead")
    healthy = WebhookSink("https://hooks.example/healthy")
    for _ in range(_CAP):
        assert dead._post_pool.acquire(blocking=False)
    try:
        with mock.patch.object(
            base._opener, "open", side_effect=lambda req, timeout=None: _Ok()
        ) as spy:
            with caplog.at_level("ERROR", logger="snagline"):
                dead.emit(_risk())  # swallowed, but must not start a worker
                healthy.emit(_risk())  # must deliver
        assert spy.call_count == 1, (
            "only the healthy sink's POST may start a worker; the dead sink's "
            f"pool-full POST started {spy.call_count - 1}"
        )
        assert "SinkBusyError" in caplog.text, (
            "the refused delivery on the full pool must be logged, not silent"
        )
    finally:
        for _ in range(_CAP):
            dead._post_pool.release()


def test_a_caller_without_a_pool_still_gets_the_global_one() -> None:
    # ``bounded_post`` keeps its default: a caller that passes no pool (the
    # one-shot ``snagline hook --url`` forward) draws the process-global pool,
    # which the sink-scoped pools deliberately do not touch.
    assert base._inflight_posts is not None
    with mock.patch.object(
        base._opener, "open", side_effect=lambda req, timeout=None: _Ok()
    ):
        body = bounded_post(_request("https://hooks.example/ok"), _DEADLINE)
    assert body == b"{}"
