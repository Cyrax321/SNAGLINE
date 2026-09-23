"""Tests for the pluggable StateBackend and Monitor per-episode locking (P1)."""

from __future__ import annotations

import sys
import threading
import time
import types
from contextlib import contextmanager

import pytest

from snagline.config import Config
from snagline.events import StepEvent, make_signature
from snagline.monitor import Monitor
from snagline.state import (
    MemoryStateBackend,
    RedisStateBackend,
    _RedisLockRenewer,
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


def test_default_state_backend_redis_when_configured(track_backends, monkeypatch):
    monkeypatch.setenv("SNAGLINE_STATE_BACKEND", "redis")
    monkeypatch.setenv("SNAGLINE_STATE_REDIS_URL", "redis://localhost:6379/0")
    # Import is guarded; without redis installed this returns a Memory backend
    # as a safe fallback rather than raising (redis extra not installed in CI).
    backend = track_backends(default_state_backend())
    assert isinstance(backend, (RedisStateBackend, MemoryStateBackend))


def test_default_state_backend_warns_when_redis_url_missing(monkeypatch, caplog):
    """Regression (#308): a missing redis URL must not degrade silently.

    The redis backend coordinates episodes across processes; in-memory state is
    per-process, so a misconfigured deployment would otherwise get N workers
    each holding their own lock -- no coordination, and nothing in the logs.
    """
    monkeypatch.setenv("SNAGLINE_STATE_BACKEND", "redis")
    monkeypatch.delenv("SNAGLINE_STATE_REDIS_URL", raising=False)
    with caplog.at_level("WARNING", logger="snagline"):
        backend = default_state_backend()
    assert isinstance(backend, MemoryStateBackend)
    assert any(
        "SNAGLINE_STATE_REDIS_URL is unset" in record.message
        for record in caplog.records
    ), [record.message for record in caplog.records]


def test_default_state_backend_quiet_when_redis_not_requested(monkeypatch, caplog):
    """The new warning must stay scoped to an explicit redis request (#308).

    Default and in-memory selections are the documented happy path, so a
    warning there would be noise on every plain ``Monitor.default()``.
    """
    monkeypatch.setenv("SNAGLINE_STATE_BACKEND", "memory")
    with caplog.at_level("WARNING", logger="snagline"):
        default_state_backend()
    assert not [record for record in caplog.records if "state" in record.message], [
        record.message for record in caplog.records
    ]


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


# --- Redis backend: lock TTL and renewal (issue #326) -----------------------
#
# CI does not install redis, so the tests below inject a time-aware fake
# ``redis`` module into ``sys.modules``. The fake genuinely models TTL expiry,
# ``extend`` pushing the deadline out, and ownership failures raising a
# ``LockError`` subclass -- the three behaviours the fix depends on -- so the
# tests are not mocked into vacuousness.


class _FakeLockError(Exception):
    """Mirrors ``redis.lock.LockError``: an ownership-level lock failure."""


class _FakeLockNotOwnedError(_FakeLockError):
    """Mirrors redis-py's LockNotOwnedError: the lock is no longer ours."""


class _FakeRedisLock:
    """A time-aware stand-in for redis-py's ``Lock``.

    Models the three behaviours the real one has and the fix depends on: the
    TTL actually expires (so a section that outlives it loses the lock),
    ``extend`` pushes the deadline out, and an ownership failure raises a
    ``LockError`` subclass rather than a generic error. ``extend_errors`` lets
    a test make a pass fail with a *transient* error, which the renewer must
    not mistake for lock loss, and ``extend_hook`` lets a test hold a pass
    mid-``extend`` to reproduce a race against the table. ``acquire_returns``
    models redis-py returning ``False`` once ``blocking_timeout`` lapses,
    which -- unlike the ``with lock:`` form -- is not an exception (#325).
    """

    def __init__(
        self,
        name,
        timeout=None,
        thread_local=True,
        acquire_returns=True,
        extend_hook=None,
    ):
        self.name = name
        self.timeout = timeout
        self.thread_local = thread_local
        self.acquire_returns = acquire_returns
        self.extend_hook = extend_hook
        self.acquired = False
        self.expires_at = None
        self.extend_calls = []
        self.extend_errors = []
        self.release_calls = 0

    def acquire(self, blocking=None, blocking_timeout=None, token=None):
        if not self.acquire_returns:
            # redis-py's deadline lapse: a silent False, not a LockError.
            return False
        self.acquired = True
        self.expires_at = time.monotonic() + (self.timeout or 0)
        return True

    def extend(self, additional_time, replace_ttl=False):
        # A failure queued by a test: real redis-py raises ConnectionError
        # (not a LockError) when the connection drops, which the renewer has
        # to distinguish from losing the lock.
        if self.extend_errors:
            raise self.extend_errors.pop(0)
        if self.extend_hook is not None:
            # Held open by a test that needs the pass in flight to reproduce
            # a race against the renewal table.
            self.extend_hook()
        if not self.acquired:
            raise _FakeLockNotOwnedError("Cannot extend a lock that is no longer owned")
        if self.timeout is not None and time.monotonic() > self.expires_at:
            # The key expired in Redis: redis-py's Lua token check fails and
            # extend raises, even though this process still thinks it holds
            # the lock. That is exactly the loss the renewer must survive.
            raise _FakeLockNotOwnedError("Cannot extend a lock that is no longer owned")
        now = time.monotonic()
        base = now if replace_ttl else self.expires_at
        self.expires_at = base + additional_time
        self.extend_calls.append((additional_time, replace_ttl))

    def release(self):
        self.release_calls += 1
        if not self.acquired:
            raise _FakeLockNotOwnedError(
                "Cannot release a lock that is no longer owned"
            )
        self.acquired = False
        if self.timeout is not None and time.monotonic() > self.expires_at:
            # redis-py fails the token check here once the key has expired.
            raise _FakeLockNotOwnedError(
                "Cannot release a lock that is no longer owned"
            )


class _FakeRedis:
    """Records the locks the backend asks for, keyed by name."""

    def __init__(self):
        self.locks: dict[str, _FakeRedisLock] = {}
        self.acquire_returns = True

    def lock(self, name, timeout=None, thread_local=True):
        self.locks[name] = _FakeRedisLock(
            name,
            timeout=timeout,
            thread_local=thread_local,
            acquire_returns=self.acquire_returns,
        )
        return self.locks[name]


def _install_fake_redis(monkeypatch) -> tuple[_FakeRedis, types.ModuleType]:
    """Put a stub ``redis`` package in ``sys.modules`` and return its client.

    ``RedisStateBackend`` imports ``redis`` lazily inside ``__init__`` and
    ``redis.lock`` inside the renewer, so patching ``sys.modules`` is enough
    -- no real redis install needed.
    """
    client = _FakeRedis()

    lock_mod = types.ModuleType("redis.lock")
    lock_mod.Lock = _FakeRedisLock
    lock_mod.LockError = _FakeLockError
    lock_mod.LockNotOwnedError = _FakeLockNotOwnedError

    redis_mod = types.ModuleType("redis")

    class _Redis:
        @staticmethod
        def from_url(url):
            return client

    redis_mod.Redis = _Redis
    redis_mod.lock = lock_mod

    monkeypatch.setitem(sys.modules, "redis", redis_mod)
    monkeypatch.setitem(sys.modules, "redis.lock", lock_mod)
    return client, redis_mod


@pytest.fixture
def track_backends():
    """Register Redis backends so their renewer threads stop on teardown.

    A backend with renewal on owns a daemon thread; leaking one per test
    leaves threads waking on a cadence for the rest of the session. The
    fallback memory backend has no thread, so only Redis backends are closed.
    """
    made: list[RedisStateBackend] = []

    def _track(backend):
        made.append(backend)
        return backend

    yield _track
    for backend in made:
        if isinstance(backend, RedisStateBackend):
            backend.close()


def test_redis_backend_defaults_to_a_longer_lock_timeout(track_backends, monkeypatch):
    """The fixed 30s TTL expired under realistic ingest load (#326).

    A whole episode's detector work runs inside the lock, so the timeout has
    to size to episodes, not to a single step.
    """
    client, _ = _install_fake_redis(monkeypatch)
    backend = track_backends(RedisStateBackend("redis://example/0"))
    with backend.episode_lock("ep"):
        pass
    lock = client.locks["snagline:lock:ep"]
    assert lock.timeout == 300.0, "default TTL must be 300s, not the old 30s"


def test_redis_backend_lock_timeout_is_configurable(track_backends, monkeypatch):
    """Operators can size the TTL to their own episodes (#326)."""
    client, _ = _install_fake_redis(monkeypatch)
    backend = track_backends(RedisStateBackend("redis://example/0", lock_timeout=900))
    with backend.episode_lock("ep"):
        pass
    assert client.locks["snagline:lock:ep"].timeout == 900


def test_redis_lock_renewal_defaults_to_a_third_of_the_ttl(track_backends, monkeypatch):
    """The default cadence keeps a full TTL of margin even past a delay."""
    _install_fake_redis(monkeypatch)
    backend = track_backends(RedisStateBackend("redis://example/0", lock_timeout=9))
    assert backend._renewer._interval == 3.0


@pytest.mark.parametrize("bad", [0, -1, -30])
def test_redis_backend_rejects_non_positive_lock_timeout(monkeypatch, bad):
    """A zero or negative TTL is a misconfiguration, not a mode: fail loudly."""
    _install_fake_redis(monkeypatch)
    with pytest.raises(ValueError, match="lock_timeout"):
        RedisStateBackend("redis://example/0", lock_timeout=bad)


def test_redis_backend_rejects_negative_renew_interval(monkeypatch):
    _install_fake_redis(monkeypatch)
    with pytest.raises(ValueError, match="lock_renew_interval"):
        RedisStateBackend("redis://example/0", lock_timeout=10, lock_renew_interval=-1)


def test_redis_lock_raises_when_acquisition_times_out(track_backends, monkeypatch):
    """A failed acquire must not yield into the critical section (#325).

    redis-py returns ``False`` when ``blocking_timeout`` lapses instead of
    raising; yielding anyway would run the whole detector section with no
    lock held, silently disabling the guarantee the backend exists for. The
    section has to be refused outright.
    """
    client, _ = _install_fake_redis(monkeypatch)
    backend = track_backends(RedisStateBackend("redis://example/0"))
    client.acquire_returns = False  # the deadline lapses inside acquire()
    with pytest.raises(RuntimeError, match="could not acquire Redis episode lock"):
        with backend.episode_lock("ep"):
            pass
    assert "ep" not in backend._renewer._renewals, "nothing was held to renew"


def test_redis_lock_outliving_ttl_is_tolerated_not_raised(
    track_backends, monkeypatch, caplog
):
    """The exact repro from #326: the section outlasts the TTL.

    Pre-fix, ``release()`` raised ``LockNotOwnedError`` out of the finally and
    surfaced from the caller's ``ingest``/``end_episode`` looking like the
    episode's own work raised. The lock is already gone -- there is nothing to
    release -- so the exit must log, not raise.
    """
    _install_fake_redis(monkeypatch)
    # Renewal off, so the short TTL genuinely lapses mid-section.
    backend = track_backends(
        RedisStateBackend("redis://example/0", lock_timeout=0.05, lock_renew_interval=0)
    )
    with caplog.at_level("WARNING", logger="snagline"):
        with backend.episode_lock("ep"):
            time.sleep(0.15)  # ~3x the TTL, inside the section
    assert any(
        "was lost before release" in record.message for record in caplog.records
    ), [record.message for record in caplog.records]


def test_redis_lock_renewed_while_section_stays_alive(
    track_backends, monkeypatch, caplog
):
    """A section that outlives the nominal TTL keeps its lock (#326).

    The renewer extends on a cadence, so the TTL bounds a *stuck* section
    rather than merely a slow one. Without renewal the same hold would have
    expired and the release would have warned.
    """
    client, _ = _install_fake_redis(monkeypatch)
    backend = track_backends(
        RedisStateBackend(
            "redis://example/0", lock_timeout=0.6, lock_renew_interval=0.1
        )
    )
    with caplog.at_level("WARNING", logger="snagline"):
        with backend.episode_lock("ep"):
            time.sleep(0.8)  # well past the 0.6s TTL
            lock = client.locks["snagline:lock:ep"]
            # The renewer has pushed the deadline out at least once.
            assert lock.extend_calls, "renewer must extend a live section"
            assert lock.acquired, "a live section must keep its lock"
    assert not any(
        "was lost before release" in record.message for record in caplog.records
    ), [record.message for record in caplog.records]


def test_redis_lock_renewal_can_be_disabled(track_backends, monkeypatch):
    """``lock_renew_interval=0`` opts out of the renewer thread entirely."""
    client, _ = _install_fake_redis(monkeypatch)
    backend = track_backends(
        RedisStateBackend("redis://example/0", lock_timeout=10, lock_renew_interval=0)
    )
    assert backend._renewer is None
    with backend.episode_lock("ep"):
        pass
    # No renewer exists to have extended anything...
    assert client.locks["snagline:lock:ep"].extend_calls == []
    # ...and redis-py's thread-local default token is kept, since nothing
    # outside this thread needs to see it.
    assert client.locks["snagline:lock:ep"].thread_local is True


def test_redis_lock_unregisters_on_exit(track_backends, monkeypatch):
    """A finished section must not keep a lock in the renewer's table.

    Otherwise the table grows once per episode id, the same unbounded leak the
    memory backend's ``release`` guards against (#67), and the renewer keeps
    extending locks nobody holds.
    """
    _install_fake_redis(monkeypatch)
    backend = track_backends(
        RedisStateBackend("redis://example/0", lock_timeout=10, lock_renew_interval=0.1)
    )
    with backend.episode_lock("ep"):
        assert "ep" in backend._renewer._renewals
    assert backend._renewer._renewals == {}


def test_redis_lock_uses_shared_token_for_renewal(track_backends, monkeypatch):
    """The renewer runs on its own thread, so the lock cannot be thread-local.

    redis-py stores the token in thread-local storage by default; the renewer
    would then see no token and refuse to extend. Renewal requires
    ``thread_local=False``.
    """
    client, _ = _install_fake_redis(monkeypatch)
    backend = track_backends(
        RedisStateBackend("redis://example/0", lock_timeout=10, lock_renew_interval=0.1)
    )
    with backend.episode_lock("ep"):
        pass
    assert client.locks["snagline:lock:ep"].thread_local is False


def test_redis_lock_renewal_stops_when_lock_is_lost(track_backends, monkeypatch):
    """A lock that expired or was released elsewhere is dropped, not retried.

    Extending a dead lock raises every pass; the renewer must unregister it so
    the table drains instead of churning against a gone key.
    """
    client, _ = _install_fake_redis(monkeypatch)
    backend = track_backends(
        RedisStateBackend("redis://example/0", lock_timeout=10, lock_renew_interval=0.1)
    )
    renewer = backend._renewer
    with backend.episode_lock("ep"):
        lock = client.locks["snagline:lock:ep"]
        # Simulate the key expiring in Redis while the section is still live:
        # the holder still believes it holds the lock, but every extend fails.
        lock.expires_at = time.monotonic() - 1
        deadline = time.monotonic() + 5
        while "ep" in renewer._renewals and time.monotonic() < deadline:
            time.sleep(0.02)
        assert "ep" not in renewer._renewals, "a lost lock must be dropped"
        # The renewer stopped touching it.
        assert lock.extend_calls == []


def test_redis_lock_loss_is_reported_while_the_section_is_still_running(
    track_backends, monkeypatch, caplog
):
    """A loss the renewer detects is reported then, not at the section's exit.

    The renewer drops a lock whose ``extend`` fails the ownership check: the
    key expired or was released elsewhere, and a second worker may already
    hold it. The holder's own exit reports the same loss, but a section still
    running -- or hung -- does not reach that exit for a while, and the
    violation has to be visible while it is happening, not afterwards.
    """
    client, _ = _install_fake_redis(monkeypatch)
    backend = track_backends(
        RedisStateBackend("redis://example/0", lock_timeout=10, lock_renew_interval=0.1)
    )
    renewer = backend._renewer
    with caplog.at_level("WARNING", logger="snagline"):
        with backend.episode_lock("ep"):
            client.locks["snagline:lock:ep"].expires_at = time.monotonic() - 1
            deadline = time.monotonic() + 5
            while "ep" in renewer._renewals and time.monotonic() < deadline:
                time.sleep(0.02)
            assert "ep" not in renewer._renewals
            # Named by the renewer, mid-section, while the holder is still
            # inside the ``with`` -- not deferred to its exit.
            assert any(
                "was lost mid-section" in record.message for record in caplog.records
            ), "the renewer must name the loss when it sees it"


def test_default_state_backend_lock_timeout_from_env(track_backends, monkeypatch):
    """``SNAGLINE_STATE_REDIS_LOCK_TIMEOUT`` sizes the TTL without code (#326).

    The documented config path is env vars (12-factor), so the constructor knob
    has to be reachable from the environment an operator actually deploys.
    """
    _install_fake_redis(monkeypatch)
    monkeypatch.setenv("SNAGLINE_STATE_BACKEND", "redis")
    monkeypatch.setenv("SNAGLINE_STATE_REDIS_URL", "redis://localhost:6379/0")
    monkeypatch.setenv("SNAGLINE_STATE_REDIS_LOCK_TIMEOUT", "900")
    backend = track_backends(default_state_backend())
    assert isinstance(backend, RedisStateBackend)
    assert backend._lock_timeout == 900.0


def test_default_state_backend_rejects_unparsable_lock_timeout(monkeypatch):
    """An unparsable TTL must not silently fall back to the default.

    Otherwise an operator's typo looks like a working configuration while
    running at 300s -- exactly the silent misconfiguration #308 taught us to
    announce.
    """
    _install_fake_redis(monkeypatch)
    monkeypatch.setenv("SNAGLINE_STATE_BACKEND", "redis")
    monkeypatch.setenv("SNAGLINE_STATE_REDIS_URL", "redis://localhost:6379/0")
    monkeypatch.setenv("SNAGLINE_STATE_REDIS_LOCK_TIMEOUT", "not-a-number")
    with pytest.raises(ValueError, match="SNAGLINE_STATE_REDIS_LOCK_TIMEOUT"):
        default_state_backend()


@pytest.mark.parametrize("bad", ["inf", "nan", "Infinity"])
def test_default_state_backend_rejects_a_non_finite_lock_timeout(monkeypatch, bad):
    """The env path parses with ``float()``, which accepts inf and nan.

    A typo-free but meaningless value must not look like a working
    configuration while the backend runs at the default TTL -- the same
    silent-misconfiguration trap the unparsable-string check closes (#308).
    """
    _install_fake_redis(monkeypatch)
    monkeypatch.setenv("SNAGLINE_STATE_BACKEND", "redis")
    monkeypatch.setenv("SNAGLINE_STATE_REDIS_URL", "redis://localhost:6379/0")
    monkeypatch.setenv("SNAGLINE_STATE_REDIS_LOCK_TIMEOUT", bad)
    with pytest.raises(ValueError, match="SNAGLINE_STATE_REDIS_LOCK_TIMEOUT"):
        default_state_backend()


def test_end_episode_tolerates_a_lost_redis_lock(track_backends, monkeypatch, caplog):
    """The swallowed loss must also be safe through the real ingest path.

    A slow detector holds the episode lock past its TTL. Pre-fix, the
    ``LockNotOwnedError`` from the finally surfaced from ``ingest`` and
    ``end_episode`` as if the episode's own work had raised; the caller has no
    way to attribute it to locking. Post-fix both log and complete.
    """
    _install_fake_redis(monkeypatch)
    backend = track_backends(
        RedisStateBackend("redis://example/0", lock_timeout=0.05, lock_renew_interval=0)
    )
    monitor = Monitor.default(sinks=[], state_backend=backend)

    class _SlowDetector:
        name = "slow"

        def observe(self, event):
            time.sleep(0.15)  # ~3x the TTL, inside the locked section
            return None

        def finalize(self, episode_id):
            time.sleep(0.15)
            return None

        def reset(self, episode_id):
            pass

    monitor._detectors = [_SlowDetector()]
    with caplog.at_level("WARNING", logger="snagline"):
        monitor.ingest(_event(episode_id="ep", i=0))  # must not raise
        monitor.end_episode("ep")  # must not raise either
    assert any(
        "was lost before release" in record.message for record in caplog.records
    ), [record.message for record in caplog.records]


# --- review follow-ups on the renewer (#326) ---------------------------------


def test_redis_lock_renewal_survives_a_transient_error(
    track_backends, monkeypatch, caplog
):
    """A transient redis failure is *not* lock loss, and must not end renewal.

    redis-py raises ``ConnectionError``/``TimeoutError`` -- not ``LockError``
    -- when the connection drops, while ``extend`` can raise either. Pre-fix
    ``except Exception`` treated them alike: the entry was dropped, renewal
    stopped, and a lock the worker still held then expired at the next TTL
    boundary. A second worker acquired it, and both mutated the same episode's
    detector state -- the exact violation this renewer exists to prevent,
    with only a misleading "TTL expired" warning to hint at it.
    """
    client, _ = _install_fake_redis(monkeypatch)
    backend = track_backends(
        RedisStateBackend("redis://example/0", lock_timeout=10, lock_renew_interval=0.1)
    )
    renewer = backend._renewer
    with caplog.at_level("WARNING", logger="snagline"):
        with backend.episode_lock("ep"):
            lock = client.locks["snagline:lock:ep"]
            # The connection drops for two renew passes in a row.
            lock.extend_calls.clear()
            lock.extend_errors.extend(
                [
                    ConnectionError("redis went away"),
                    ConnectionError("redis went away again"),
                ]
            )
            deadline = time.monotonic() + 10
            while len(lock.extend_calls) < 2 and time.monotonic() < deadline:
                # The entry survives each failed pass, so the next pass can
                # still reach this lock.
                assert "ep" in renewer._renewals, (
                    "a transient error must keep the entry"
                )
                time.sleep(0.02)
            assert len(lock.extend_calls) >= 2, (
                "renewal must retry after a transient error"
            )
        transient = [r for r in caplog.records if "could not renew" in r.message]
        assert transient, "a transient failure must be logged"
        assert len(transient) == 1, (
            "the warning fires once per section, not once per failed pass"
        )
        assert not any("was lost" in record.message for record in caplog.records), (
            "a transient failure is not a lost lock"
        )


def test_renewer_keeps_a_later_holders_registration(monkeypatch):
    """A renew pass must not drop a *later* holder's entry for the same episode.

    The pass snapshots the table under ``_lock`` while exit proceeds: holder A
    unregisters and releases, worker B acquires the same episode and registers,
    then the pass's ``extend`` on A's lock fails. Pre-fix it called
    ``unregister(episode_id)`` -- keyed by id alone -- and removed B's entry;
    B's lock was then never extended and expired mid-section. Passing the
    entry back makes the removal identity-checked.
    """
    _install_fake_redis(monkeypatch)
    renewer = _RedisLockRenewer(interval=0.1, ttl=10)
    try:
        # A's lock: the key expired in Redis while its section was still
        # live, so every extend fails the ownership check.
        a_lock = _FakeRedisLock("ep", timeout=1)
        a_lock.acquired = True
        a_lock.expires_at = time.monotonic() - 1
        a_entry = renewer.register("ep", a_lock)
        # Hold A's extend open *after* the pass has snapshotted the table, so
        # A's exit and B's registration land while the pass is still in flight
        # against A's stale entry.
        in_extend = threading.Event()
        let_extend_finish = threading.Event()

        def _block_inside_extend():
            in_extend.set()
            let_extend_finish.wait(5)

        a_lock.extend_hook = _block_inside_extend
        pass_thread = threading.Thread(target=renewer.renew_all)
        pass_thread.start()
        assert in_extend.wait(5), "the pass must reach the stale extend"
        # A exits its section; B acquires the same episode id right after.
        renewer.unregister("ep", a_entry)
        b_lock = _FakeRedisLock("ep", timeout=10)
        b_lock.acquired = True
        b_lock.expires_at = time.monotonic() + 10
        b_entry = renewer.register("ep", b_lock)
        let_extend_finish.set()
        pass_thread.join(5)
        assert renewer._renewals.get("ep") is b_entry, (
            "the pass must drop its own stale entry, not the live holder's"
        )
        renewer.renew_all()  # now reaches B
        assert b_lock.extend_calls, "B's lock must still be renewed"
    finally:
        renewer.stop()


def test_renewer_warns_when_a_section_outlives_the_ttl(
    track_backends, monkeypatch, caplog
):
    """A hung-but-alive section is renewed forever, so it has to be visible.

    With renewal the TTL no longer bounds a *live* holder: a detector that
    hangs inside ``observe`` holds the distributed lock for the process
    lifetime with no warning, where the old behaviour at least broke the
    deadlock after one TTL. Renewal keeps mutual exclusion intact, which is
    the right call, so the bound that survives is visibility -- once a
    section has held the lock past a full ``lock_timeout``, it is named once.
    """
    _install_fake_redis(monkeypatch)
    backend = track_backends(
        RedisStateBackend(
            "redis://example/0", lock_timeout=1.0, lock_renew_interval=0.25
        )
    )
    with caplog.at_level("WARNING", logger="snagline"):
        with backend.episode_lock("ep"):
            deadline = time.monotonic() + 15
            while (
                not any(
                    "has been held for" in record.message for record in caplog.records
                )
                and time.monotonic() < deadline
            ):
                time.sleep(0.05)
        assert any(
            "has been held for" in record.message for record in caplog.records
        ), "a section past one TTL must be named once, not silently renewed"


def test_stuck_warning_fires_at_or_after_one_full_ttl(monkeypatch, caplog):
    """The slow-section warning lands after a full TTL, not before it.

    Floor division rounded the renewal count down, so a 1s TTL renewed every
    0.3s warned at 0.9s -- ninety percent of the TTL -- and float rounding
    could pull a ``ttl / 3`` cadence below one TTL a renewal early. Ceiling
    division puts the warning on the first renewal at or past the TTL.
    """
    _install_fake_redis(monkeypatch)
    renewer = _RedisLockRenewer(interval=0.3, ttl=1.0)
    try:
        # 1.0 / 0.3 = 3.33: floor is 3 renewals (0.9s, short of the TTL),
        # ceiling is 4 (1.2s, the first renewal past it).
        assert renewer._stuck_after == 4, "ceiling, not floor, of ttl / interval"
        lock = _FakeRedisLock("ep", timeout=10)
        lock.acquired = True
        lock.expires_at = time.monotonic() + 10
        renewer.register("ep", lock)
        with caplog.at_level("WARNING", logger="snagline"):
            for _ in range(renewer._stuck_after - 1):
                renewer.renew_all()
            assert not any("has been held for" in r.message for r in caplog.records), (
                "no warning before one full TTL has elapsed"
            )
            renewer.renew_all()
            assert any("has been held for" in r.message for r in caplog.records), (
                "the warning lands on the first renewal at or past the TTL"
            )
    finally:
        renewer.stop()


@pytest.mark.parametrize("bad", [0.09, 0.001, 1e-9])
def test_redis_backend_rejects_a_sub_floor_renew_interval(monkeypatch, bad):
    """A tiny positive interval is a busy-loop, not a tuning knob.

    ``time.sleep(1e-9)`` is effectively a tight loop, so the renewer would
    spin and fire one extend script per live lock per pass.
    """
    _install_fake_redis(monkeypatch)
    with pytest.raises(ValueError, match="lock_renew_interval"):
        RedisStateBackend("redis://example/0", lock_renew_interval=bad)


@pytest.mark.parametrize("interval", [10, 20])
def test_redis_backend_rejects_an_interval_at_or_above_ttl(monkeypatch, interval):
    """An interval that meets or beats the TTL cannot renew anything.

    The first attempt lands after the lock has already expired, so each value
    looks fine alone while the pair is self-defeating -- renewal silently
    buys nothing and the TTL is the only bound left.
    """
    _install_fake_redis(monkeypatch)
    with pytest.raises(ValueError, match="less than lock_timeout"):
        RedisStateBackend(
            "redis://example/0", lock_timeout=10, lock_renew_interval=interval
        )


@pytest.mark.parametrize("bad", [float("inf"), float("-inf"), float("nan")])
def test_redis_backend_rejects_a_non_finite_lock_timeout(monkeypatch, bad):
    """inf/nan pass a plain positivity check and only fail later, in acquire.

    ``int(inf * 1000)`` overflows inside redis-py, and a nan deadline makes
    the acquisition loop never stop -- both escape ``episode_lock`` into
    ``ingest``.
    """
    _install_fake_redis(monkeypatch)
    with pytest.raises(ValueError, match="lock_timeout"):
        RedisStateBackend("redis://example/0", lock_timeout=bad, lock_renew_interval=0)


@pytest.mark.parametrize("bad", [float("inf"), float("nan")])
def test_redis_backend_rejects_a_non_finite_renew_interval(monkeypatch, bad):
    """A nan interval silently disables renewal; an infinite one never fires."""
    _install_fake_redis(monkeypatch)
    with pytest.raises(ValueError, match="lock_renew_interval"):
        RedisStateBackend("redis://example/0", lock_renew_interval=bad)


def test_renewer_thread_stops_on_close(track_backends, monkeypatch):
    """``close()`` puts the daemon thread down deterministically.

    ``Monitor.default`` can replace the backend that ``__init__`` built, and
    every construction would otherwise leave a renewer running for the
    process lifetime. The table drains itself as sections exit, so an
    unstopped renewer does no work -- but a thread that cannot be stopped is
    still a leak.
    """
    _install_fake_redis(monkeypatch)
    backend = track_backends(
        RedisStateBackend("redis://example/0", lock_renew_interval=0.1)
    )
    thread = backend._renewer._thread
    assert thread.is_alive()
    backend.close()
    thread.join(timeout=5)
    assert not thread.is_alive(), "close() must stop the renewer"
    assert backend._renewer is None
    backend.close()  # idempotent, not an AttributeError
