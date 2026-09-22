"""Batched, optionally rate-limited sink dispatch (ATTACH_ANY_SYSTEM P1 item 5).

A network sink (webhook/Slack/PagerDuty) can be slow or rate-limited. Emitting
every risk synchronously inside ``ingest()`` couples detector throughput to
the sink's latency. ``BatchingSink`` decouples them: ``emit`` just enqueues
(cheap, non-blocking), and a background thread flushes the queue on an interval
or once it reaches ``max_batch``. An optional ``max_per_second`` rate limit
paces the actual deliveries so a burst of risk does not trip the destination's
rate limiter.

``max_batch`` is what bounds the queue between interval ticks, so reaching it
wakes the flusher immediately rather than delivering on the caller's thread --
``emit`` stays non-blocking even when the wrapped sink is slow.

A non-positive ``flush_interval`` is rejected at construction rather than
clamped: ``_wake.wait`` returns immediately for one, so the flusher spins
through an empty queue hundreds of thousands of times per second, pinning a
full core for the life of the process while alerts still deliver and nothing
else looks wrong. Unlike ``max_batch`` (clamped, since any size still paces)
a non-positive interval has no meaningful reading -- the sink's whole point
is pacing (issue #358).

Fail-open: a delivery error is swallowed (never blocks ingest or the queue).
``close()`` drains the queue before returning, so a clean shutdown inside one
``flush_interval`` of a detection still delivers the alert.
"""

from __future__ import annotations

import collections
import contextlib
import threading
import time

from snagline.risk import FailureRisk
from snagline.sinks.base import AlertSink


class BatchingSink:
    """Buffer ``FailureRisk`` emits and flush them on a background thread."""

    def __init__(
        self,
        sink: AlertSink,
        max_batch: int = 100,
        flush_interval: float = 5.0,
        max_per_second: float | None = None,
    ) -> None:
        self._sink = sink
        self._max_batch = max(1, max_batch)
        if flush_interval <= 0.0:
            # A non-positive interval makes _wake.wait return immediately, so
            # the flusher busy-spins on an empty queue (~780k passes/sec, a
            # full core, for the life of the process) while still delivering
            # alerts -- nothing else looks wrong (issue #358). Reject it here
            # rather than clamping: unlike max_batch there is no size that
            # still paces, and the join in close() uses this same value as its
            # timeout, so a negative one also starves the shutdown drain.
            raise ValueError(f"flush_interval must be positive; got {flush_interval!r}")
        self._flush_interval = flush_interval
        self._min_gap = 1.0 / max_per_second if max_per_second else 0.0
        self._queue: collections.deque[FailureRisk] = collections.deque()
        self._lock = threading.Lock()
        # Delivery pacing is shared across every batch and flush path. Keep it
        # separate from the queue lock so sleeping or calling the wrapped sink
        # never blocks ``emit``.
        self._delivery_lock = threading.Lock()
        self._last_delivery: float | None = None
        self._stop = threading.Event()
        # Set when the queue reaches ``max_batch`` (or on close) so the flusher
        # can wake early instead of sitting out the rest of the interval.
        self._wake = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def emit(self, risk: FailureRisk) -> None:
        # Non-blocking enqueue; the background thread does the actual delivery.
        with self._lock:
            self._queue.append(risk)
            full = len(self._queue) >= self._max_batch
        if full:
            # Threshold reached. Wake the flusher rather than delivering here:
            # ``emit`` runs on the caller's thread (inside ``ingest``) and must
            # stay non-blocking even when the wrapped sink is slow.
            self._wake.set()

    def _run(self) -> None:
        try:
            while not self._stop.is_set():
                # Whichever comes first: the interval elapsing or a full batch.
                self._wake.wait(self._flush_interval)
                self._wake.clear()
                if self._stop.is_set():
                    break
                self._flush()
        finally:
            # Drain anything enqueued before the stop so a clean shutdown does
            # not silently lose alerts.
            self._flush()

    def _flush(self) -> None:
        with self._lock:
            if not self._queue:
                return
            batch = list(self._queue)
            self._queue.clear()
        self._deliver(batch)

    def _deliver(self, batch: list[FailureRisk]) -> None:
        with self._delivery_lock:
            for risk in batch:
                if self._min_gap:
                    now = time.monotonic()
                    if self._last_delivery is not None:
                        wait = self._min_gap - (now - self._last_delivery)
                        if wait > 0:
                            time.sleep(wait)
                    # Record the actual delivery start, including failed
                    # attempts, so failures cannot create an unpaced burst.
                    self._last_delivery = time.monotonic()
                with contextlib.suppress(Exception):
                    # Fail-open: a delivery failure must not poison the queue.
                    self._sink.emit(risk)

    def flush_now(self) -> None:
        """Force an immediate flush (used at shutdown and in tests)."""
        self._flush()

    def close(self) -> None:
        """Stop the flusher and deliver whatever is still queued."""
        self._stop.set()
        self._wake.set()  # unblock the interval wait so the drain happens now
        self._thread.join(timeout=self._flush_interval + 1.0)
        # If the thread did not finish its drain within the join timeout, flush
        # on the caller's thread. ``_flush`` is a no-op on an empty queue, so
        # this is safe either way.
        self._flush()
