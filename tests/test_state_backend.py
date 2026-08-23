"""Tests for the pluggable StateBackend and Monitor per-episode locking (P1)."""

from __future__ import annotations

import threading
from contextlib import contextmanager

import pytest

from snagline.config import Config
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


def test_end_episode_releases_the_episode_lock():
    """Regression: the lock dict must not grow once per episode forever.

    A long-lived Monitor (sidecar, hook bridge) sees one fresh episode id per
    agent run, so a lock retained past ``end_episode`` is an unbounded leak.
    """
    backend = MemoryStateBackend()
    monitor = Monitor.default(state_backend=backend)
    for n in range(500):
        ep = f"episode-{n}"
        monitor.ingest(_event(episode_id=ep, i=n))
        monitor.end_episode(ep)
    assert backend._locks == {}


def test_release_keeps_other_episodes_untouched():
    backend = MemoryStateBackend()
    with backend.episode_lock("a"):
        pass
    with backend.episode_lock("b"):
        pass
    kept = backend._locks["b"]
    backend.release("a")
    assert "a" not in backend._locks
    assert backend._locks["b"] is kept


def test_release_of_an_unknown_episode_is_a_no_op():
    backend = MemoryStateBackend()
    backend.release("never-seen")  # must not raise
    assert backend._locks == {}


def test_end_episode_tolerates_a_backend_without_release():
    """``release`` is optional: a backend implementing only the narrower
    ``StateBackend`` protocol must keep working unchanged."""

    class MinimalBackend:
        def __init__(self):
            self.locked: list[str] = []

        @contextmanager
        def episode_lock(self, episode_id):
            self.locked.append(episode_id)
            yield

    backend = MinimalBackend()
    monitor = Monitor.default(state_backend=backend)
    monitor.ingest(_event(episode_id="ep", i=0))
    monitor.end_episode("ep")  # must not raise AttributeError
    assert backend.locked == ["ep", "ep"]


def test_end_episode_is_fail_open_when_release_raises():
    class BrokenBackend(MemoryStateBackend):
        def release(self, episode_id):
            raise RuntimeError("backend down")

    monitor = Monitor.default(state_backend=BrokenBackend())
    monitor.end_episode("ep")  # fail-open: logged, not raised


def test_end_episode_propagates_release_error_when_not_fail_open():
    class BrokenBackend(MemoryStateBackend):
        def release(self, episode_id):
            raise RuntimeError("backend down")

    monitor = Monitor.default(
        config=Config(fail_open=False), state_backend=BrokenBackend()
    )
    with pytest.raises(RuntimeError):
        monitor.end_episode("ep")
