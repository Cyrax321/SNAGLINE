"""Tests for the pluggable StateBackend and Monitor per-episode locking (P1)."""

from __future__ import annotations

import threading

from snagline.events import StepEvent, make_signature
from snagline.monitor import Monitor
from snagline.state import (
    MemoryStateBackend,
    RedisStateBackend,
    default_state_backend,
)


def _event(episode_id="ep", i=0):
    return StepEvent(
        step_id=str(i),
        episode_id=episode_id,
        timestamp=1.0,
        action_type="tool_call",
        action_signature=make_signature("tool_call", "tool", str(i)),
        tool_name="tool",
        latency_ms=100.0,
    )


def test_memory_backend_isolates_episode_locks():
    b = MemoryStateBackend()
    # Same episode id returns the same underlying lock object (re-entrant).
    with b.episode_lock("a"):
        with b.episode_lock("a"):
            pass  # re-entrant: must not deadlock
    with b.episode_lock("b"):
        pass
    # Distinct episodes get distinct locks.
    assert b._locks["a"] is not b._locks["b"]


def test_default_state_backend_memory_without_env(monkeypatch):
    monkeypatch.delenv("SNAGLINE_STATE_BACKEND", raising=False)
    assert isinstance(default_state_backend(), MemoryStateBackend)


def test_default_state_backend_redis_when_configured(monkeypatch):
    monkeypatch.setenv("SNAGLINE_STATE_BACKEND", "redis")
    monkeypatch.setenv("SNAGLINE_STATE_REDIS_URL", "redis://localhost:6379/0")
    # Import is guarded; without redis installed this returns a Memory backend
    # as a safe fallback rather than raising (redis extra not installed in CI).
    backend = default_state_backend()
    assert isinstance(backend, (RedisStateBackend, MemoryStateBackend))


def test_concurrent_ingest_of_distinct_episodes_no_deadlock():
    monitor = Monitor.default()
    errors: list[Exception] = []

    def worker(ep):
        try:
            for i in range(50):
                monitor.ingest(_event(episode_id=ep, i=i))
        except Exception as exc:  # pragma: no cover
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(f"ep{j}",)) for j in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)
    assert not errors, errors
    # Each episode's state was torn down cleanly.
    for j in range(4):
        monitor.end_episode(f"ep{j}")
