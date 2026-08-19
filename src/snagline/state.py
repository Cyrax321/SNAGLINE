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
