"""Pluggable state backend (ATTACH_ANY_SYSTEM P1, item 4).

The Monitor's per-episode detector state (loop windows, cascade counters,
CUSUM baselines) is naturally keyed by ``episode_id``. This module provides a
``StateBackend`` that owns the concurrency primitive, so the ingest path can
shard its lock by episode instead of serializing all episodes behind one
global lock (the verified bottleneck in monitor.py).

The default ``MemoryStateBackend`` is process-local. ``RedisStateBackend``
(optional, behind the ``redis`` extra) provides a shared lock across workers
so a horizontally-scaled deployment does not double-count; it is imported
lazily so the core stays zero-dependency.
"""

from __future__ import annotations

import logging
import os
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Protocol

logger = logging.getLogger("snagline")


class StateBackend(Protocol):
    """Owns the lock used to serialize a single episode's ingest path."""

    @contextmanager
    def episode_lock(self, episode_id: str) -> Iterator[None]:
        """Yield while holding the lock for ``episode_id`` (re-entrant safe)."""
        ...


class ReleasableStateBackend(StateBackend, Protocol):
    """A ``StateBackend`` that can drop what it holds for a finished episode.

    Optional capability: ``Monitor.end_episode`` probes for ``release`` and
    skips backends that do not expose it, so a backend written against the
    narrower ``StateBackend`` above keeps working unchanged.
    """

    def release(self, episode_id: str) -> None:
        """Discard any per-episode state held for ``episode_id``."""
        ...


class MemoryStateBackend:
    """Process-local backend: one re-entrant lock per episode id."""

    def __init__(self) -> None:
        self._meta = threading.Lock()
        self._locks: dict[str, threading.RLock] = {}

    @contextmanager
    def episode_lock(self, episode_id: str) -> Iterator[None]:
        with self._meta:
            lock = self._locks.get(episode_id)
            if lock is None:
                lock = threading.RLock()
                self._locks[episode_id] = lock
        with lock:
            yield

    def release(self, episode_id: str) -> None:
        """Drop the lock allocated for a finished episode.

        Without this the dict grows by one entry per episode id and never
        shrinks, so a long-lived Monitor watching many short runs retains a
        lock for every episode it has ever seen.

        ``Monitor.end_episode`` calls this while holding the episode lock, so
        no other thread can be inside the critical section when the entry is
        dropped. A thread that ingests for the same id *after* the release
        simply allocates a fresh lock -- correct, because an episode that has
        ended has no detector state left to serialize.
        """
        with self._meta:
            self._locks.pop(episode_id, None)


class RedisStateBackend:  # pragma: no cover - optional, requires redis
    """Shared backend: a Redis lock so workers coordinate across processes."""

    def __init__(self, url: str, prefix: str = "snagline:") -> None:
        import redis

        self._r = redis.Redis.from_url(url)
        self._prefix = prefix

    @contextmanager
    def episode_lock(self, episode_id: str) -> Iterator[None]:
        lock = self._r.lock(self._prefix + "lock:" + episode_id, timeout=30)
        acquired = lock.acquire(blocking=True, blocking_timeout=30)
        try:
            yield
        finally:
            if acquired:
                lock.release()

    def release(self, episode_id: str) -> None:
        """No-op: Redis locks are per-acquisition and expire on their own.

        Nothing is retained between ``episode_lock`` calls, so a finished
        episode leaves nothing behind to discard.
        """


def default_state_backend() -> StateBackend:
    """Pick a backend from ``SNAGLINE_STATE_BACKEND`` env (memory|redis)."""
    kind = os.environ.get("SNAGLINE_STATE_BACKEND", "memory").lower()
    if kind == "redis":
        url = os.environ.get("SNAGLINE_STATE_REDIS_URL")
        if url:
            try:
                return RedisStateBackend(url)
            except ImportError:
                logger.warning(
                    "snagline: redis backend requested but redis not installed; "
                    "falling back to in-memory state"
                )
    return MemoryStateBackend()
