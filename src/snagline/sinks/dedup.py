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

Snapshots (``dump_state``/``load_state``, issue #91) record a wall-clock
companion alongside every monotonic timestamp so restore can tell a same-boot
process restart (monotonic continuous, suppression continues seamlessly) from
a host reboot (monotonic reset; restored values would sit in the fresh clock's
future and silently suppress alerts for roughly the previous uptime -- issue
#136). Entries that are provably stale or of unverifiable provenance are
dropped at load with one log line and never restored: an alert-suppression
wrapper fails by delivering too much, never by staying silent.

A non-positive ``cooldown_seconds`` disables suppression: ``emit`` becomes a
pass-through that never touches the table. Wrapping is normally avoided
upstream in that case (``cli._maybe_dedup``), but the sink is public API and
must behave sanely when constructed directly.
"""

from __future__ import annotations

import contextlib
import logging
import math
import threading
import time
from collections.abc import Callable
from typing import Any

from snagline.risk import FailureRisk
from snagline.sinks.base import AlertSink

logger = logging.getLogger("snagline")


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

    def dump_state(self) -> dict[str, Any] | None:
        """Cooldown map for ``Monitor.snapshot`` (issue #91).

        Only serializable when the default ``(episode_id, trigger, severity)``
        key function is in use; a custom ``key_fn`` is an opaque callable, so
        this returns ``None`` and snapshot/restore skip it (a restored dedup
        sink then starts with empty cooldowns rather than failing setup).

        Each entry carries its monotonic timestamp plus a wall-clock companion
        derived at dump time (``now_wall - (now_mono - t)``), so ``load_state``
        can judge entry age across a host reboot without any extra per-emit
        bookkeeping on this sink's hot path. The derivation assumes both clocks
        advanced together since the emit, which holds within one boot; across a
        reboot the companion is exactly what exposes the discontinuity.
        """
        if self._key_fn is not _default_key:
            return None
        with self._lock:
            now_mono = time.monotonic()
            now_wall = time.time()
            entries = sorted(self._last.items(), key=lambda kv: repr(kv[0]))
            return {
                "cooldown_seconds": self._cooldown,
                # Tuples become lists on JSON round-trip; load_state converts back.
                "last": [[list(k), t] for k, t in entries],
                # Wall-clock moment of each entry's last emit (issue #136).
                "wall": [[list(k), now_wall - (now_mono - t)] for k, t in entries],
            }

    def load_state(self, state: dict[str, Any]) -> None:
        """Restore cooldowns from :meth:`dump_state` (setup-time only, #91).

        A snapshot may have crossed a host reboot since it was written: the
        fresh clock then starts near zero while restored monotonic timestamps
        still hold previous-uptime values, so trusting them suppresses every
        restored key until ``now`` catches up with roughly the old uptime
        (#136). Restore therefore drops, with one warning and no exception:

        - entries whose wall-clock age already exceeds ``cooldown_seconds``;
        - entries sitting in the future of the current monotonic clock (proof
          of a clock reset, whatever the wall clock claims);
        - entries without a finite wall companion (snapshots from older
          versions carry no provenance, so they cannot be trusted either way);
        - anything malformed (fail open to delivery, never raise into setup).

        Surviving entries keep their stored monotonic value: they only survive
        when the monotonic clock is demonstrably continuous, which is exactly
        the same-boot restart scenario #91 shipped for.
        """
        if self._key_fn is not _default_key:
            return
        if not isinstance(state, dict):
            logger.warning(
                "snagline: dedup snapshot malformed (%r); starting with "
                "empty cooldowns",
                type(state).__name__,
            )
            return
        try:
            last_entries: list[Any] = state.get("last", []) or []
            wall_map = {
                tuple(k): w
                for k, w in (state.get("wall") or [])
                if isinstance(w, (int, float))
            }
            now_mono = time.monotonic()
            now_wall = time.time()
            restored: dict[Any, float] = {}
            dropped = 0
            for k, ts in last_entries:
                key = tuple(k)
                wall = wall_map.get(key)
                if (
                    not isinstance(ts, (int, float))
                    or not math.isfinite(float(ts))
                    or wall is None
                    or not math.isfinite(wall)
                    or (now_wall - wall) >= self._cooldown
                    or float(ts) > now_mono
                ):
                    dropped += 1
                    continue
                restored[key] = float(ts)
        except Exception:
            # Fail open: an unreadable snapshot must not break setup, let alone
            # surface later inside ingest (#91 contract). Delivering duplicates
            # beats silently suppressing fresh alerts for hours.
            logger.warning(
                "snagline: dedup snapshot unreadable; starting with empty cooldowns",
                exc_info=True,
            )
            return
        if dropped:
            logger.warning(
                "snagline: dedup snapshot carried %d stale or clock-reset "
                "cooldown entrie(s); dropped them so affected alerts "
                "re-deliver",
                dropped,
            )
        with self._lock:
            self._last = restored
