"""Issue #423: nothing bounded how many abandoned sink POSTs could pile up.

``bounded_post`` (issue #395) gives up on a POST at its deadline but cannot
cancel it -- the worker thread, and the socket it holds, lives on until the
exchange resolves on its own. Against a merely slow endpoint the worker drains
and exits shortly after; against a *dead* one -- a black hole that neither
answers nor resets -- every alert spawns a worker that never returns, so a
monitor under load would accumulate one parked thread (and file descriptor)
per emit without bound.

The fix caps the number of POSTs in flight at once with a process-global
non-blocking semaphore: a worker holds a slot for its whole life and frees it
in a ``finally``, so the ceiling counts live threads rather than calls. When
every slot is taken the next POST is dropped with ``SinkBusyError`` -- fast,
and fail-open at the sink -- rather than adding another parked thread.

These tests drive a small cap so the ceiling is reached in a handful of calls;
the real default (``_MAX_INFLIGHT_POSTS``) is generous by design.
"""

from __future__ import annotations

import threading
import urllib.request
from unittest import mock

import pytest

import snagline.sinks.base as base
from snagline.risk import FailureRisk
from snagline.sinks.base import SinkBusyError, bounded_post
from snagline.sinks.webhook import WebhookSink

# A short deadline keeps the suite fast; a worker abandoned at it stays parked.
_DEADLINE = 0.1
# The small cap the tests exercise; the shipped default is far larger.
_CAP = 3


@pytest.fixture
def small_cap(monkeypatch):
    """Shrink the process-global pool to ``_CAP`` for one test.

    ``bounded_post`` reads both module globals at call time, so replacing them
    resizes the pool without touching the shipped default -- and each test gets
    a fresh semaphore, so parked workers from one cannot leak slots into the
    next.
    """
    monkeypatch.setattr(base, "_MAX_INFLIGHT_POSTS", _CAP)
    monkeypatch.setattr(base, "_inflight_posts", threading.BoundedSemaphore(_CAP))


def _risk() -> FailureRisk:
    return FailureRisk(
        episode_id="ep-1",
        step_id="3",
        score=0.5,
        trigger="loop",
        detail="action repeated 3x in last 4 steps",
        timestamp=1718300000.0,
    )


class _BlackHole:
    """A response whose body never arrives until ``gate`` is set.

    Models the dead endpoint: the worker parks inside ``read`` forever, exactly
    as it would on a socket that neither answers nor resets, so the caller's
    ``join(timeout)`` gives up and abandons a still-live thread.
    """

    def __init__(self, gate: threading.Event) -> None:
        self._gate = gate

    def __enter__(self) -> _BlackHole:
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def read(self, *args: object) -> bytes:
        # 5 s is a safety net so a failed test cannot hang the suite forever;
        # every test sets the gate in its own teardown well before then.
        self._gate.wait(5.0)
        return b""


def _black_holing_urlopen(gate: threading.Event):
    return mock.patch.object(
        urllib.request,
        "urlopen",
        side_effect=lambda req, timeout=None: _BlackHole(gate),
    )


def _request() -> urllib.request.Request:
    return urllib.request.Request(
        "https://blackhole.example.invalid/hook", data=b"{}", method="POST"
    )


def test_a_full_pool_rejects_the_next_post_without_starting_it(small_cap) -> None:
    gate = threading.Event()
    calls: list[object] = []

    def counting_urlopen(req, timeout=None):
        calls.append(req)
        return _BlackHole(gate)

    try:
        with mock.patch.object(urllib.request, "urlopen", side_effect=counting_urlopen):
            # Fill every slot: each POST is abandoned at its deadline with its
            # worker still parked in read(), so it keeps holding its slot.
            for _ in range(_CAP):
                with pytest.raises(TimeoutError):
                    bounded_post(_request(), _DEADLINE)
            assert len(calls) == _CAP

            # The pool is now full. The next POST must be refused before a
            # worker is ever started -- so urlopen is not reached again, and
            # the caller neither waits a deadline nor adds a parked thread.
            with pytest.raises(SinkBusyError):
                bounded_post(_request(), _DEADLINE)
        assert len(calls) == _CAP, (
            "a rejected POST must not start a worker: urlopen was called "
            f"{len(calls)} times, expected {_CAP}"
        )
    finally:
        gate.set()


def test_live_worker_threads_stay_bounded_by_the_cap(small_cap) -> None:
    # The issue's reproduction: many emits at a black hole. Without the cap the
    # parked-worker count climbs with every call; with it, it plateaus at _CAP.
    gate = threading.Event()

    def live_workers() -> int:
        return sum(
            1
            for t in threading.enumerate()
            if t.name == "snagline-sink-post" and t.is_alive()
        )

    before = live_workers()
    try:
        with _black_holing_urlopen(gate):
            for _ in range(_CAP * 5):
                with pytest.raises((TimeoutError, SinkBusyError)):
                    bounded_post(_request(), _DEADLINE)
            parked = live_workers() - before
        assert parked <= _CAP, (
            f"parked worker threads ({parked}) must not exceed the cap ({_CAP}); "
            "an unbounded pile-up is exactly issue #423"
        )
    finally:
        gate.set()


def test_a_completed_post_frees_its_slot(small_cap) -> None:
    # A slot must be reusable once its worker finishes -- the cap bounds
    # concurrency, it does not permanently consume capacity. Run more healthy
    # POSTs than the cap through a fast endpoint; each frees its slot at once.
    class _Ok:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def read(self, *a):
            return b"ok"

    with mock.patch.object(
        urllib.request, "urlopen", side_effect=lambda req, timeout=None: _Ok()
    ):
        for _ in range(_CAP * 3):
            bounded_post(_request(), _DEADLINE)  # must not raise SinkBusyError


def test_a_sink_logs_a_full_pool_fail_open_without_leaking_the_url(
    small_cap, caplog
) -> None:
    # Drain the pool by hand so the sink's own emit hits a full one, then check
    # it is logged fail-open (the sink never raises) and named by class only --
    # the URL carries basic-auth credentials and must not reach the log (#390).
    for _ in range(_CAP):
        assert base._inflight_posts.acquire(blocking=False)
    try:
        sink = WebhookSink("https://user:s3cret@hooks.example/alerts")
        with caplog.at_level("ERROR", logger="snagline"):
            sink.emit(_risk())  # must not raise
        assert "SinkBusyError" in caplog.text, (
            "a dropped delivery must be logged as a failure, not silently lost"
        )
        assert "s3cret" not in caplog.text and "user:" not in caplog.text, (
            "the destination credential must never reach the log (issue #390)"
        )
    finally:
        for _ in range(_CAP):
            base._inflight_posts.release()
