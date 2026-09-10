"""Tests for the pluggable StateBackend and Monitor per-episode locking (P1)."""

from __future__ import annotations

import threading
import time
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


# --- release() vs parked waiters (issue #238) -------------------------------


def test_parked_waiter_keeps_exclusive_section_after_release():
    """The exact race from issue #238, made deterministic.

    Thread A (end_episode) holds the episode lock inside finalize(); thread
    B (ingest) fetches the same entry under _meta and parks on the lock; A
    runs release() and returns. Pre-fix, release() popped the entry
    immediately, so when B woke it entered on the orphaned lock while a
    later thread C found no entry and allocated a fresh one: B and C ran
    observe() for the same episode concurrently. Post-fix, the entry is
    marked released and stays until every waiter drains, so B and C
    serialize on the same lock; C only enters after B leaves.
    """
    in_finalize = threading.Event()
    finalize_go = threading.Event()
    in_observe = threading.Event()
    observe_go = threading.Event()

    class _GatedDetector:
        name = "gated"

        def observe(self, event):
            in_observe.set()
            assert observe_go.wait(timeout=10)
            return None

        def finalize(self, episode_id):
            in_finalize.set()
            assert finalize_go.wait(timeout=10)
            return None

        def reset(self, episode_id):
            pass

    backend = MemoryStateBackend()
    monitor = Monitor.default(sinks=[], state_backend=backend)
    monitor._detectors = [_GatedDetector()]
    monitor.ingest(_event(episode_id="ep", i=0))  # create the entry

    # A holds the lock inside finalize().
    a = threading.Thread(target=lambda: monitor.end_episode("ep"))
    a.start()
    assert in_finalize.wait(timeout=5), "A must reach finalize"

    # B fetches the entry and parks on the lock (A holds it).
    b = threading.Thread(target=lambda: monitor.ingest(_event("ep", 1)))
    b.start()
    time.sleep(0.1)  # let B park

    # A finishes: release() runs inside end_episode, then A exits the lock.
    finalize_go.set()
    a.join(timeout=10)

    # B wakes and enters observe() on the (now released-marked) entry.
    assert in_observe.wait(timeout=5), "B must reach observe after A leaves"

    # C arrives after the release. Pre-fix it got a FRESH lock and entered
    # observe() concurrently with B; post-fix it shares B's entry and parks.
    c = threading.Thread(target=lambda: monitor.ingest(_event("ep", 2)))
    c.start()
    time.sleep(0.2)
    # The decisive check: B and C must be counted on ONE shared entry; a
    # fresh entry for C would mean two live locks guard the same episode.
    with backend._meta:
        entries = dict(backend._locks)
    assert "ep" in entries, "B's entry must still exist while B holds it"
    assert entries["ep"].waiters >= 2, (
        "B and C must be counted on one shared entry; a fresh entry for C "
        "would mean two live locks guard the same episode (issue #238)"
    )

    # Let B leave; C then enters on the same entry.
    observe_go.set()
    b.join(timeout=10)
    assert in_observe.wait(timeout=5)
    observe_go.set()
    c.join(timeout=10)
    a.join(timeout=10)

    # Once every waiter drains, the released entry is removed. C's own
    # observe runs under the entry it fetched; after it drains the table is
    # empty again (the leak guard from #67 keeps holding).
    deadline = time.monotonic() + 5
    while backend._locks and time.monotonic() < deadline:
        time.sleep(0.01)
    assert "ep" not in backend._locks, "entry must drain after the last waiter"


def test_release_during_parked_waiter_blocks_new_entry_until_drain():
    """Directly at the primitive: release() while a waiter is parked must not
    hand a *fresh* lock to a later fetcher -- the parked waiter and the later
    fetcher must serialize on the same entry."""
    backend = MemoryStateBackend()

    def _run_ctx(ctx):
        with ctx:
            pass

    # Hold the lock in this thread, and park a second waiter on it.
    with backend.episode_lock("ep"):
        with_park = backend.episode_lock("ep")
        parked = threading.Thread(target=lambda: _run_ctx(with_park))
        parked.start()
        time.sleep(0.1)  # parked.waiters counted, blocked on the lock
        backend.release("ep")  # entry must be marked, not dropped
        assert "ep" in backend._locks, "released entry with a parked waiter stays"
    # Holder exits; the parked waiter runs and drains the entry on exit.
    parked.join(timeout=10)
    deadline = time.monotonic() + 5
    while "ep" in backend._locks and time.monotonic() < deadline:
        time.sleep(0.01)
    assert "ep" not in backend._locks
