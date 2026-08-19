"""Deduplicating / cooldown sink wrapper (ATTACH_ANY_SYSTEM P1, issue #4).

Production alerting must not storm on-call: the same failure repeating every
second across thousands of steps should page once, not thousands of times.
``DedupSink`` wraps any ``AlertSink`` and suppresses repeats of the same key
within a cooldown window. The key defaults to ``(episode_id, trigger,
severity)`` but can be customized via ``key_fn`` (e.g. to dedupe purely by
detector type, or by the full detail string).
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

    def emit(self, risk: FailureRisk) -> None:
        key = self._key_fn(risk)
        now = time.time()
        with self._lock:
            last = self._last.get(key)
            if last is not None and (now - last) < self._cooldown:
                return
            self._last[key] = now
        with contextlib.suppress(Exception):
            # Never let a wrapped-sink failure corrupt our cooldown bookkeeping
            # (fail-open), and do not re-raise into the monitor.
            self._sink.emit(risk)
