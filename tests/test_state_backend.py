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


# --- a bad redis URL degrades to in-memory instead of crashing startup (#392)


@pytest.fixture
def fake_redis(monkeypatch):
    """Stand in for redis-py: ``from_url`` validates the scheme at
     construction, before any socket is opened, and raises ``ValueError`` for
     anything that is not redis/rediss/unix.

    redis-py 8.x's ``ConnectionPool.from_url`` behaves exactly this way, so the
     stub reproduces the real failure shape without needing a server or the
     extra installed.
    """
    import sys
    import types
    from urllib.parse import urlsplit

    module = types.ModuleType("redis")

    class FakeRedis:
        @staticmethod
        def from_url(url):
            scheme = urlsplit(url).scheme.lower()
            if scheme not in ("redis", "rediss", "unix"):
                raise ValueError(
                    "Redis URL must specify one of the following schemes "
                    "(redis://, rediss://, unix://)"
                )
            return FakeRedis()

    module.Redis = FakeRedis  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "redis", module)
    return module


@pytest.mark.parametrize(
    "url",
    [
        "postgres://host:5432/db",  # copied from a sibling service
        "host:6379",  # a bare host, not a URL
        "${REDIS_URL}",  # an unsubstituted secrets-manager placeholder
        "http://localhost:6379",  # the wrong scheme entirely
    ],
)
def test_bad_redis_url_falls_back_not_raises(monkeypatch, caplog, fake_redis, url):
    """A URL redis-py rejects must warn and degrade, not escape into startup.

    The knob's whole purpose is to be optional: both neighbouring arms warn
    and fall back, so an unparseable URL being the one that crashes
    ``Monitor.default()`` was an inconsistency rather than a policy.
    """
    monkeypatch.setenv("SNAGLINE_STATE_BACKEND", "redis")
    monkeypatch.setenv("SNAGLINE_STATE_REDIS_URL", url)
    with caplog.at_level("WARNING", logger="snagline"):
        backend = default_state_backend()
    assert isinstance(backend, MemoryStateBackend)
    assert any("not a usable redis URL" in r.message for r in caplog.records), [
        r.message for r in caplog.records
    ]


def test_good_redis_url_still_builds_the_backend(monkeypatch, fake_redis):
    """Regression guard: the new arm must not swallow a URL the parser
    accepts. A widening that fell back on every URL would be worse than the
    crash it fixes."""
    monkeypatch.setenv("SNAGLINE_STATE_BACKEND", "redis")
    monkeypatch.setenv("SNAGLINE_STATE_REDIS_URL", "redis://localhost:6379/0")
    backend = default_state_backend()
    assert isinstance(backend, RedisStateBackend)


def test_bad_redis_url_warning_names_the_target(monkeypatch, caplog, fake_redis):
    """The warning still names the offending host and scheme: the fallback is
    silent at the protocol level (in-memory state looks fine), so the log line
    is the operator's only signal that their coordination knob did not take --
    and identifying *which* endpoint was misconfigured is what makes the line
    actionable."""
    monkeypatch.setenv("SNAGLINE_STATE_BACKEND", "redis")
    monkeypatch.setenv("SNAGLINE_STATE_REDIS_URL", "postgres://host:5432/db")
    with caplog.at_level("WARNING", logger="snagline"):
        default_state_backend()
    record = next(r for r in caplog.records if "not a usable redis URL" in r.message)
    assert "postgres://host:5432/" in record.message


@pytest.mark.parametrize(
    "url, secret",
    [
        # redis-py takes the password from the URL, so this is the documented
        # way to configure the backend -- and a typo in the scheme is exactly
        # when the operator reads this line.
        ("redis://:hunter2-secret@redis.internal:6379/0", "hunter2-secret"),
        # The user:pass spelling, and a query-string password (accepted by
        # redis-py as an alternative to userinfo).
        ("redis://alice:hunter2@redis.internal:6379/0", "hunter2"),
        ("rediss://redis.internal?password=[REDACTED]", "hunter2"),
    ],
)
def test_bad_redis_url_warning_redacts_the_credential(
    monkeypatch, caplog, echo_redis, url, secret
):
    """The URL is the credential: redis-py reads the password out of it, so
    logging it verbatim leaks the secret into the operator's log collector at
    the exact moment they go looking (review of #409). The host stays, since
    identifying the misconfigured endpoint is the point of the line."""
    monkeypatch.setenv("SNAGLINE_STATE_BACKEND", "redis")
    monkeypatch.setenv("SNAGLINE_STATE_REDIS_URL", url)
    with caplog.at_level("WARNING", logger="snagline"):
        backend = default_state_backend()
    assert isinstance(backend, MemoryStateBackend)
    record = next(r for r in caplog.records if "not a usable redis URL" in r.message)
    assert secret not in record.message, "credential reached the log line"
    assert "redis.internal" in record.message, "the host must still be named"


@pytest.fixture
def echo_redis(monkeypatch):
    """A redis-py stub whose parse failure quotes the offending URL back in
    the exception text, as redis-py's own ``parse_url`` does for some inputs.

    The scheme validator in ``fake_redis`` does not, so this separate stub
    exists to cover the second leak path: the exception, not just the URL.
    """
    import sys
    import types

    module = types.ModuleType("redis")

    class FakeRedis:
        @staticmethod
        def from_url(url):
            raise ValueError(
                f"invalid connection parameters in {url!r}; see redis-py docs"
            )

    module.Redis = FakeRedis  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "redis", module)
    return module


def test_bad_redis_url_withholds_exception_text_that_echoes_it(
    monkeypatch, caplog, echo_redis
):
    """The exception can carry the URL verbatim, so once the URL holds a
    credential only the exception's *type* is logged. Without the credential
    the message is diagnosis and stays (review of #409)."""
    monkeypatch.setenv("SNAGLINE_STATE_BACKEND", "redis")
    monkeypatch.setenv("SNAGLINE_STATE_REDIS_URL", "redis://:hunter2@host:6379/0")
    with caplog.at_level("WARNING", logger="snagline"):
        default_state_backend()
    record = next(r for r in caplog.records if "not a usable redis URL" in r.message)
    assert "hunter2" not in record.message
    assert "ValueError" in record.message, "the exception type must still be named"


def test_bad_redis_url_keeps_exception_text_when_url_has_no_secret(
    monkeypatch, caplog, echo_redis
):
    """The gate is not a blanket suppression: a credential-free URL keeps the
    full message, which is what tells the operator what is actually wrong."""
    monkeypatch.setenv("SNAGLINE_STATE_BACKEND", "redis")
    monkeypatch.setenv("SNAGLINE_STATE_REDIS_URL", "redis-s://host:6379/0")
    with caplog.at_level("WARNING", logger="snagline"):
        default_state_backend()
    record = next(r for r in caplog.records if "not a usable redis URL" in r.message)
    assert "redis-s://host:6379/0" in record.message
    assert "invalid connection parameters" in record.message


@pytest.mark.parametrize(
    "url, expected",
    [
        ("redis://:pw@host:6379/0", "redis://host:6379/"),
        ("redis://host:6379/0", "redis://host:6379/"),
        ("unix:///var/run/redis.sock", "unix:///var/run/redis.sock"),
        ("not a url at all", "<invalid redis url>"),
        ("", "<invalid redis url>"),
    ],
)
def test_redact_url(url, expected):
    from snagline.state import _redact_url

    assert _redact_url(url) == expected
