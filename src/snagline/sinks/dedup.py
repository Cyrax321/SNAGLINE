"""Deduplicating / cooldown sink wrapper (ATTACH_ANY_SYSTEM P1, issue #4).

Production alerting must not storm on-call: the same failure repeating every
second across thousands of steps should page once, not thousands of times.
``DedupSink`` wraps any ``AlertSink`` and suppresses repeats of the same key
within a cooldown window. The key defaults to ``(episode_id, trigger,
severity)`` but can be customized via ``key_fn`` (e.g. to dedupe purely by
detector type, or by the full detail string).

The cooldown table is bounded: once a key's cooldown has elapsed it can never
suppress anything again, so expired keys are pruned instead of being retained
for the life of the process. With the default key that would otherwise be one
permanent entry per episode id, and a long-lived monitor watching many short
runs is exactly this sink's intended deployment.

Elapsed time is measured on the monotonic clock. Wall-clock time can step
(NTP correction, an operator setting the date), and a backward step would
stretch a cooldown into an indefinite silence -- the one failure mode an
alert-suppression wrapper must not have.
"""

from __future__ import annotations

import contextlib
import threading
import time
from collections.abc import Callable
from typing import Any

from snagline.risk import FailureRisk
from snagline.sinks.base import AlertSink


def _default_key(risk: FailureRisk) -> tuple[Any, ...]:
    return (risk.episode_id, risk.trigger, risk.severity)


class DedupSink:
    """Wrap ``sink`` so identical alerts are emitted at most once per
    ``cooldown_seconds``. Fail-open: a bookkeeping error never blocks the
    wrapped sink.
    """

    def __init__(
        self,
        sink: AlertSink,
        cooldown_seconds: float = 300.0,
        key_fn: Callable[[FailureRisk], Any] | None = None,
    ) -> None:
        self._sink = sink
        self._cooldown = cooldown_seconds
        self._key_fn = key_fn or _default_key
        self._last: dict[Any, float] = {}
        self._lock = threading.Lock()
        self._swept = time.monotonic()

    def emit(self, risk: FailureRisk) -> None:
        key = self._key_fn(risk)
        now = time.monotonic()
        with self._lock:
            last = self._last.get(key)
            if last is not None and (now - last) < self._cooldown:
                return
            self._last[key] = now
            # At most one sweep per cooldown window. Anything that expired since
            # the previous sweep is expired now, so this is frequent enough to
            # keep the table from growing without bound, and rare enough that the
            # O(len) pass is amortized away across the window's emits.
            if (now - self._swept) >= self._cooldown:
                self._sweep(now)
        with contextlib.suppress(Exception):
            # Never let a wrapped-sink failure corrupt our cooldown bookkeeping
            # (fail-open), and do not re-raise into the monitor.
            self._sink.emit(risk)

    def _sweep(self, now: float) -> None:
        """Drop keys whose cooldown has elapsed. Caller must hold ``_lock``.

        An entry older than the cooldown is dead weight: the suppression test in
        ``emit`` is already false for it, so dropping it changes no decision this
        sink would make. Keys still inside their window are kept -- they are what
        makes suppression work, so retaining them is correctness, not a leak.

        The table therefore holds the keys seen in roughly the last two cooldown
        windows (a key can expire just after one sweep and wait until the next),
        rather than every key seen for the life of the process.
        """
        cutoff = now - self._cooldown
        self._last = {k: t for k, t in self._last.items() if t > cutoff}
        self._swept = now
