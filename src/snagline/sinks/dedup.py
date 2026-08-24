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

A non-positive ``cooldown_seconds`` disables suppression: ``emit`` becomes a
pass-through that never touches the table. Wrapping is normally avoided
upstream in that case (``cli._maybe_dedup``), but the sink is public API and
must behave sanely when constructed directly.
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

    "Fail-open" covers ``key_fn`` too. A ``key_fn`` that raises -- or returns
    something unhashable -- leaves this sink unable to decide whether the alert
    is a repeat, so the alert is delivered rather than dropped, and the error is
    not re-raised at the caller. That holds standalone, not just under
    ``Monitor``: an alerting wrapper must never turn a bad key into a lost
    alert or an exception in the host's ingest path.

    A non-positive ``cooldown_seconds`` disables suppression entirely and makes
    ``emit`` a pass-through: with no window, every alert is already outside it.
    """

    def __init__(
        self,
        sink: AlertSink,
        cooldown_seconds: float = 300.0,
        key_fn: Callable[[FailureRisk], Any] | None = None,
    ) -> None:
        self._sink = sink
        self._cooldown = cooldown_seconds
        # Suppression needs a positive window to mean anything. Precomputed so
        # the disabled case costs one attribute read on the hot path rather
        # than a comparison plus an always-true sweep (see ``emit``).
        self._enabled = cooldown_seconds > 0
        self._key_fn = key_fn or _default_key
        self._last: dict[Any, float] = {}
        self._lock = threading.Lock()
        self._swept = time.monotonic()

    def emit(self, risk: FailureRisk) -> None:
        # The whole suppression decision is inlined here rather than delegated to
        # a helper: this is the per-alert hot path, and an extra Python call
        # measured ~15% of it.
        if self._enabled:
            try:
                # ``key_fn`` is caller-supplied and runs inside this handler.
                # Outside it, a raising key_fn -- or one returning an unhashable
                # key, which raises in the lookup below -- propagated straight
                # out of ``emit``, losing the alert and, standalone rather than
                # under ``Monitor``, raising into the host's ingest path.
                key = self._key_fn(risk)
                now = time.monotonic()
                with self._lock:
                    last = self._last.get(key)
                    if last is not None and (now - last) < self._cooldown:
                        return
                    self._last[key] = now
                    # At most one sweep per cooldown window. Anything that
                    # expired since the previous sweep is expired now, so this
                    # is frequent enough to keep the table from growing without
                    # bound, and rare enough that the O(len) pass is amortized
                    # away across the window's emits.
                    if (now - self._swept) >= self._cooldown:
                        self._sweep(now)
            except Exception:
                # Fail-open: bookkeeping we cannot complete must not silence the
                # alert. Falling through delivers it, which is the right way to
                # fail for a wrapper whose only job is *suppressing* duplicates.
                pass
        # Disabled (non-positive cooldown) reaches here directly: with no window
        # nothing can be a repeat, so the table is never touched and the sweep
        # guard below -- unconditionally true for a non-positive cooldown -- can
        # never turn this pass-through into an O(len) rebuild under the lock.
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
