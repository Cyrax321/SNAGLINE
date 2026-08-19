"""Batched, optionally rate-limited sink dispatch (ATTACH_ANY_SYSTEM P1 item 5).

A network sink (webhook/Slack/PagerDuty) can be slow or rate-limited. Emitting
every risk synchronously inside ``ingest()`` couples detector throughput to
the sink's latency. ``BatchingSink`` decouples them: ``emit`` just enqueues
(cheap, non-blocking), and a background thread flushes the queue on an interval
or once it reaches ``max_batch``. An optional ``max_per_second`` rate limit
paces the actual deliveries so a burst of risk does not trip the destination's
rate limiter.

Fail-open: a delivery error is swallowed (never blocks ingest or the queue).
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
        self._max_batch = max_batch
        self._flush_interval = flush_interval
        self._min_gap = 1.0 / max_per_second if max_per_second else 0.0
        self._queue: collections.deque[FailureRisk] = collections.deque()
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def emit(self, risk: FailureRisk) -> None:
        # Non-blocking enqueue; the background thread does the actual delivery.
        with self._lock:
            self._queue.append(risk)

    def _run(self) -> None:
        while not self._stop.wait(self._flush_interval):
            self._flush()

    def _flush(self) -> None:
        with self._lock:
            if not self._queue:
                return
            batch = list(self._queue)
            self._queue.clear()
        self._deliver(batch)

    def _deliver(self, batch: list[FailureRisk]) -> None:
        last = 0.0
        for risk in batch:
            if self._min_gap:
                now = time.monotonic()
                wait = self._min_gap - (now - last)
                if wait > 0:
                    time.sleep(wait)
                last = time.monotonic()
            with contextlib.suppress(Exception):
                # Fail-open: a delivery failure must not poison the queue.
                self._sink.emit(risk)

    def flush_now(self) -> None:
        """Force an immediate flush (used at shutdown and in tests)."""
        self._flush()

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=self._flush_interval + 1.0)
