"""Pluggable state backend (ATTACH_ANY_SYSTEM P1, item 4).

The Monitor's per-episode detector state (loop windows, cascade counters,
CUSUM baselines) is naturally keyed by ``episode_id``. This module provides a
``StateBackend`` that owns the concurrency primitive, so the ingest path can
shard its lock by episode instead of serializing all episodes behind one
global lock (the verified bottleneck in monitor.py).

The default ``MemoryStateBackend`` is process-local. ``RedisStateBackend``
(optional; ``pip install redis`` -- there is no ``snagline[redis]`` extra)
provides a shared lock across workers so a horizontally-scaled deployment
does not double-count; it is imported lazily so the core stays
zero-dependency.
"""

from __future__ import annotations

import logging
import math
import os
import threading
from collections.abc import Iterator
from contextlib import AbstractContextManager, contextmanager
from typing import Any, Literal, Protocol

logger = logging.getLogger("snagline")


class StateBackend(Protocol):
    """Owns the lock used to serialize a single episode's ingest path.

    ``episode_lock`` returns any context manager, not specifically a
    generator: the memory backend measures ~0.34 us/step cheaper as a plain
    object (issue #299), and a caller only ever enters and exits. A backend
    that prefers to ``yield`` still type-checks, because
    ``contextlib._GeneratorContextManager`` satisfies this shape.
    """

    def episode_lock(self, episode_id: str) -> AbstractContextManager[None]:
        """Hold the lock for ``episode_id`` for the duration of a ``with``."""
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


class _LockEntry:
    """One episode's lock plus the bookkeeping that keeps it safe to drop.

    ``waiters`` counts threads that have fetched this entry -- holding its
    lock or parked waiting for it. ``released`` marks an episode whose lock
    the backend has been asked to drop. The entry leaves the table only
    once every waiter has drained, so a thread parked on the lock when
    ``release()`` ran can never end up inside the critical section beside
    a fetcher that arrived afterwards: both serialize on this same entry
    until the last one exits and the entry is finally removed (issue #238).
    """

    __slots__ = ("lock", "waiters", "released")

    def __init__(self) -> None:
        self.lock = threading.RLock()
        self.waiters: int = 0
        self.released: bool = False


class _HeldLock(AbstractContextManager[None]):
    """The live half of ``MemoryStateBackend.episode_lock`` (issue #299).

    ``episode_lock`` used to be a ``@contextmanager``, so every ingest paid
    for a generator frame -- create, resume into the ``with``, throw on exit
    -- on top of the locks themselves. Measured, that machinery alone was
    ~0.34 us of the ~1.9 us default per-step budget, roughly a fifth of it,
    and it bought nothing: the caller only ever enters and exits.

    Splitting the protocol's context manager into a cheap object avoids the
    generator entirely while keeping ``with backend.episode_lock(eid):`` and
    the issue #238 waiter semantics byte-for-byte. The entry is resolved and
    the waiter counted eagerly, in ``episode_lock`` itself, so a waiter that
    blocks on the RLock is already accounted for before it parks -- that is
    what makes ``release()`` safe against a thread parked mid-acquire.
    """

    __slots__ = ("_backend", "_entry", "_episode_id")

    def __init__(
        self, backend: MemoryStateBackend, entry: _LockEntry, episode_id: str
    ) -> None:
        self._backend = backend
        self._entry = entry
        self._episode_id = episode_id

    def __enter__(self) -> None:
        self._entry.lock.acquire()

    def __exit__(self, *exc_info: object) -> Literal[False]:
        self._entry.lock.release()
        # Mirror the generator's original finally block: this entry is only
        # safe to forget once nothing is left parked on it. Fetched under
        # _meta so a concurrent release() cannot observe a half-decremented
        # waiter count. The lookup can come back None if release() already
        # drained this entry -- the waiter count then has nothing left to
        # adjust, and there is nothing to pop.
        backend = self._backend
        with backend._meta:
            entry = backend._locks.get(self._episode_id)
            if entry is None:
                return False
            entry.waiters -= 1
            if entry.released and entry.waiters == 0:
                backend._locks.pop(self._episode_id, None)
        return False


class MemoryStateBackend:
    """Process-local backend: one re-entrant lock per episode id."""

    def __init__(self) -> None:
        self._meta = threading.Lock()
        self._locks: dict[str, _LockEntry] = {}

    def episode_lock(self, episode_id: str) -> _HeldLock:
        """A context manager holding the lock for ``episode_id``.

        Returns a cheap object rather than yielding from a generator: the
        generator frame was ~0.34 us of every ingest, about a fifth of the
        default per-step budget, and the caller only ever enters and exits
        (issue #299). Resolving the entry and counting this caller as a
        waiter under ``_meta`` before the RLock is acquired is what makes a
        concurrent ``release()`` safe against a waiter parked mid-acquire.
        """
        with self._meta:
            entry = self._locks.get(episode_id)
            if entry is None:
                entry = _LockEntry()
                self._locks[episode_id] = entry
            entry.waiters += 1
        return _HeldLock(self, entry, episode_id)

    def release(self, episode_id: str) -> None:
        """Drop the lock allocated for a finished episode.

        Without this the dict grows by one entry per episode id and never
        shrinks, so a long-lived Monitor watching many short runs retains a
        lock for every episode it has ever seen.

        ``Monitor.end_episode`` calls this while holding the episode lock.
        Other threads may still be *parked* on that lock -- they fetched it
        before the release and are waiting for the holder to finish -- so
        the entry is only removed once every such waiter has drained
        (issue #238). Until then a parked waiter and any fetcher that
        arrives later share this same entry, keeping the episode's critical
        section mutually exclusive; a thread that fetches after the entry
        is finally removed allocates a fresh one, which is correct because
        an ended episode has no detector state left to serialize.
        """
        with self._meta:
            entry = self._locks.get(episode_id)
            if entry is None:
                return
            entry.released = True
            if entry.waiters == 0:
                self._locks.pop(episode_id, None)


_RENEW_INTERVAL_FLOOR = 0.1
"""The smallest renewal cadence the renewer will run at.

A positive interval below this is a misconfiguration, not a tuning knob: the
renewer would spin in a near-tight loop and fire one extend script per live
lock per pass, hammering Redis to buy a fraction of a second of extra margin
that nobody asked for.
"""


class _RenewalEntry:
    """One live section's lock plus the bookkeeping renewal needs.

    ``renewals`` counts successful extensions, and the two ``warned_*`` flags
    keep each diagnostic to a single line per section instead of one per pass.
    """

    __slots__ = ("lock", "renewals", "warned_stuck", "warned_transient")

    def __init__(self, lock: Any) -> None:
        self.lock = lock
        self.renewals = 0
        self.warned_stuck = False
        self.warned_transient = False


def _is_lock_loss(exc: BaseException) -> bool:
    """True if ``exc`` means the lock is no longer ours to extend.

    redis-py raises ``LockError`` -- and its ``LockNotOwnedError`` subclass --
    for ownership failures: the key expired, was released elsewhere, or failed
    the token check. A connection error or a timeout is *not* ownership loss:
    the lock may still be held and the failure is transient. Both come out of
    ``extend``, so the renewer has to tell them apart. Treating a transient
    failure as loss ends renewal for a lock we still hold, and it then expires
    at the next TTL boundary -- the exact mutual-exclusion break this renewer
    exists to prevent.
    """
    try:
        from redis.lock import LockError
    except ImportError:  # pragma: no cover - renewer only runs with redis
        return False
    return isinstance(exc, LockError)


class _RedisLockRenewer:
    """Keeps a held Redis lock alive while its critical section is still running.

    redis-py's ``Lock`` expires at its TTL no matter how long the holder is
    still inside the section, and renewing from inside is impossible -- the
    holder's thread is the one running the detector work. A daemon thread
    outside it walks the live sections and calls ``extend`` on a fixed cadence.

    One renewer per ``RedisStateBackend``, not one thread per ``episode_lock``
    call: ``episode_lock`` runs on every ingest, and thread startup alone
    (~50 us) would dwarf the ~1.9 us per-step budget the memory backend is
    tuned for (issue #299). The renewer only ever *extends* a lock, never
    acquires or releases one, so it cannot steal a lock from its holder.

    Renewal does not, and cannot, *guarantee* mutual exclusion on its own: it
    only keeps a lock alive for a section that is still running. If the lock
    is nonetheless lost, the section keeps going and another worker may enter
    the same episode, so the loss is reported when the renewer sees it rather
    than waiting for the section's own exit, which a hung section never
    reaches.
    """

    __slots__ = (
        "_interval",
        "_lock",
        "_renewals",
        "_stop",
        "_stuck_after",
        "_thread",
        "_ttl",
    )

    def __init__(self, interval: float, ttl: float) -> None:
        self._interval = interval
        self._ttl = ttl
        self._lock = threading.Lock()
        # episode_id -> entry, so one iteration reaches every live section. A
        # section unregisters itself on exit, so the table never holds a lock
        # whose section has finished.
        self._renewals: dict[str, _RenewalEntry] = {}
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        # Warn once a section has been extended past one full TTL of holding:
        # at that point it would have expired without the renewer, so the TTL
        # no longer bounds it and an operator should see the episode named (an
        # alive-but-hung section is otherwise renewed forever, silently).
        # Ceiling, not floor: rounding down warns before a full TTL has
        # actually elapsed.
        self._stuck_after = max(1, math.ceil(ttl / interval))

    def register(self, episode_id: str, lock: Any) -> _RenewalEntry:
        entry = _RenewalEntry(lock)
        with self._lock:
            self._renewals[episode_id] = entry
        return entry

    def unregister(
        self, episode_id: str, expected: _RenewalEntry | None = None
    ) -> None:
        with self._lock:
            # Identity-check before popping: a renew pass can be carrying an
            # entry whose holder has already exited while another worker
            # acquired the same episode and registered under a fresh lock.
            # Keyed by episode id alone, the table cannot tell whose
            # registration it is removing, and dropping the *new* entry would
            # let that new holder's lock expire mid-section.
            if expected is None or self._renewals.get(episode_id) is expected:
                self._renewals.pop(episode_id, None)

    def renew_all(self) -> None:
        """One pass: extend every lock still held, dropping those that left."""
        with self._lock:
            live = list(self._renewals.items())
        for _episode_id, entry in live:
            try:
                # Reset to a full TTL, not the interval: the timeout is the
                # bound on how long a dead holder keeps the lock, and it
                # should not silently shrink to a third once renewal starts.
                # Renewing at timeout/3 still leaves a full TTL of margin
                # even if a pass is delayed.
                entry.lock.extend(self._ttl, replace_ttl=True)
            except Exception as exc:
                if _is_lock_loss(exc):
                    # The lock is gone, and unlike the holder's own release
                    # this thread cannot know how far the section still has to
                    # run: a slow or hung holder keeps going for a while before
                    # its exit reports the loss, and by then a second worker
                    # may already be mutating the same episode. Report the
                    # violation when it happens, then stop renewing the dead
                    # key so the table drains instead of churning against it.
                    logger.warning(
                        "snagline: Redis episode lock for %r was lost "
                        "mid-section (TTL expired or released elsewhere); "
                        "mutual exclusion may be violated while the section "
                        "keeps running -- consider a larger lock_timeout",
                        _episode_id,
                    )
                    self.unregister(_episode_id, entry)
                    continue
                # Not ownership loss: a connection error, a timeout. The lock
                # may still be held, so renewal must not stop -- the next pass
                # retries, and the holder's release still reports a real loss
                # if one eventually happens.
                if not entry.warned_transient:
                    entry.warned_transient = True
                    logger.warning(
                        "snagline: could not renew Redis episode lock for %r "
                        "(%s); renewal will retry on the next pass",
                        _episode_id,
                        exc,
                    )
                continue
            entry.renewals += 1
            if not entry.warned_stuck and entry.renewals >= self._stuck_after:
                # Renewal keeps mutual exclusion intact, which is the point,
                # but a section this slow has outlived the bound the TTL is
                # supposed to put on it. Name it once, so a hung-but-alive
                # detector is visible instead of holding the distributed lock
                # for the process lifetime in silence.
                entry.warned_stuck = True
                logger.warning(
                    "snagline: Redis episode lock for %r has been held for "
                    "longer than lock_timeout (%ss); the section may be "
                    "stuck, and renewal keeps holding the lock",
                    _episode_id,
                    self._ttl,
                )

    def run(self) -> None:
        """Renew on a fixed cadence until stopped or the process exits.

        ``_stop.wait`` stands in for ``time.sleep`` so a stop lands within one
        interval instead of waiting out a full sleep. The thread is still a
        daemon: if the process dies it dies with it and holds nothing that
        needs joining.
        """
        while not self._stop.wait(self._interval):
            self.renew_all()

    def stop(self, timeout: float | None = None) -> None:
        """Signal the renewer's loop to exit and wait for it.

        A discarded backend -- a ``Monitor.default`` that replaces the one
        ``__init__`` built, or a test tearing down -- otherwise leaves a
        daemon thread running for the process lifetime. The renewal table
        drains itself as sections exit, so an unstopped renewer does no work,
        but it should still be possible to put the thread down.
        """
        self._stop.set()
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout)


class RedisStateBackend:  # pragma: no cover - optional, requires redis
    """Shared backend: a Redis lock so workers coordinate across processes."""

    def __init__(
        self,
        url: str,
        prefix: str = "snagline:",
        lock_timeout: float = 300.0,
        lock_blocking_timeout: float = 30.0,
        lock_renew_interval: float | None = None,
    ) -> None:
        import redis

        self._r = redis.Redis.from_url(url)
        self._prefix = prefix
        # TTL on the held lock. The critical section covers a whole episode's
        # ingest work, so the old fixed 30s expired under realistic load --
        # a slow detector, a large replay, GC pressure -- and made release()
        # raise LockNotOwnedError out of the finally into ingest/end_episode
        # (issue #326). 300s is a safer default and is operator-tunable to
        # the shape of the site's own episodes. NaN and infinity have to be
        # rejected explicitly: both pass a simple positivity check and only
        # blow up later, deep inside acquire().
        if not math.isfinite(lock_timeout) or lock_timeout <= 0:
            raise ValueError(
                "lock_timeout must be a positive, finite number of seconds, "
                f"got {lock_timeout}"
            )
        self._lock_timeout = lock_timeout
        self._lock_blocking_timeout = lock_blocking_timeout
        # Renew while the section is alive, so the TTL bounds a *stuck*
        # section rather than merely a slow one. The interval defaults to a
        # third of the TTL, which keeps the deadline comfortably ahead of the
        # renewer even when a pass is delayed. Pass 0 to disable renewal and
        # keep the old expire-and-lose behaviour.
        if lock_renew_interval is None:
            interval = lock_timeout / 3
        else:
            interval = lock_renew_interval
        if interval == 0:
            self._renewer: _RedisLockRenewer | None = None
        else:
            if not math.isfinite(interval) or interval < _RENEW_INTERVAL_FLOOR:
                raise ValueError(
                    "lock_renew_interval must be 0 (renewal disabled) or at "
                    f"least {_RENEW_INTERVAL_FLOOR}s so the renewer cannot "
                    f"spin in a tight loop, got a cadence of {interval}s"
                )
            if interval >= lock_timeout:
                # Each value looks fine alone; together they are
                # self-defeating. The first renewal attempt lands after the
                # lock has already expired, so renewal could never preserve
                # it and the TTL is silently back to being the only bound.
                raise ValueError(
                    "lock_renew_interval must be less than lock_timeout "
                    f"({lock_timeout}s) -- at this cadence the lock expires "
                    f"before its first renewal, got {interval}s"
                )
            self._renewer = _RedisLockRenewer(interval, lock_timeout)
        if self._renewer is not None:
            thread = threading.Thread(
                target=self._renewer.run,
                name="snagline-redis-lock-renewer",
                daemon=True,
            )
            self._renewer._thread = thread
            thread.start()

    @contextmanager
    def episode_lock(self, episode_id: str) -> Iterator[None]:
        timeout = self._lock_timeout
        renewer = self._renewer
        # thread_local=False is required for renewal: redis-py keeps the lock
        # token in thread-local storage by default, so the renewer thread
        # would see no token and refuse to extend. The renewer only ever
        # extends, so sharing the token across threads is safe here.
        lock = self._r.lock(
            self._prefix + "lock:" + episode_id,
            timeout=timeout,
            thread_local=renewer is None,
        )
        acquired = lock.acquire(
            blocking=True, blocking_timeout=self._lock_blocking_timeout
        )
        if not acquired:
            # redis-py's acquire() returns False on blocking_timeout, it does
            # not raise (only the `with lock:` form raises LockError). Yielding
            # anyway would run the whole detector critical section with no lock
            # held, silently disabling the one guarantee this backend exists to
            # provide (issue #325). Fail loudly instead.
            raise RuntimeError(
                f"could not acquire Redis episode lock for {episode_id!r} "
                f"within {self._lock_blocking_timeout}s"
            )
        # Registered only while a renewer exists, so the exit path can rely on
        # ``renewal`` alone to tell it whether to unregister.
        renewal: _RenewalEntry | None = None
        if renewer is not None:
            renewal = renewer.register(episode_id, lock)
        try:
            yield
        finally:
            if renewer is not None and renewal is not None:
                # Unregister before releasing so the renewer cannot extend a
                # lock this thread is in the middle of dropping. Handing the
                # entry back is the identity check: a renew pass already
                # carrying this section's entry must not remove a later
                # holder's registration for the same episode id.
                renewer.unregister(episode_id, renewal)
            try:
                lock.release()
            except Exception:
                # The TTL expired mid-section: redis-py fails the token check
                # and raises LockNotOwnedError out of this finally, which would
                # surface from the caller's ingest/end_episode looking like the
                # episode's own work raised. The lock is already gone -- there
                # is nothing to release -- so log and move on (issue #326).
                logger.warning(
                    "snagline: Redis episode lock for %r was lost before "
                    "release (TTL expired); mutual exclusion may have been "
                    "violated -- consider a larger lock_timeout",
                    episode_id,
                )

    def release(self, episode_id: str) -> None:
        """No-op: Redis locks are per-acquisition and expire on their own.

        Nothing is retained between ``episode_lock`` calls, so a finished
        episode leaves nothing behind to discard.
        """

    def close(self) -> None:
        """Stop the background renewer thread.

        The backend owns one daemon thread for the process lifetime by
        default; the renewal table drains itself as sections exit, so leaving
        it running costs nothing. Call this when a backend is discarded ahead
        of process exit -- a test tearing down, or a ``Monitor.default`` that
        replaces the backend built at import -- to put the thread down
        deterministically instead of leaving it for the interpreter.
        """
        renewer = self._renewer
        if renewer is not None:
            renewer.stop()
            self._renewer = None


def default_state_backend() -> StateBackend:
    """Pick a backend from ``SNAGLINE_STATE_BACKEND`` env (memory|redis)."""
    kind = os.environ.get("SNAGLINE_STATE_BACKEND", "memory").lower()
    if kind == "redis":
        url = os.environ.get("SNAGLINE_STATE_REDIS_URL")
        if not url:
            # The redis backend exists to coordinate episodes across
            # *processes*; in-memory state is per-process, so a missing URL
            # silently gives a scaled deployment N workers each holding their
            # own lock. Announce the fallback for the same reason the
            # ImportError arm below does (issue #308).
            logger.warning(
                "snagline: redis backend requested but SNAGLINE_STATE_REDIS_URL "
                "is unset; falling back to in-memory state"
            )
        else:
            kwargs: dict[str, Any] = {}
            raw_timeout = os.environ.get("SNAGLINE_STATE_REDIS_LOCK_TIMEOUT")
            if raw_timeout is not None:
                # Operators size the TTL to their own episodes (issue #326);
                # a value this layer cannot parse must not silently fall back
                # to the default and leave them thinking it took effect. The
                # same holds for inf/nan: float() accepts both, and each only
                # fails later, inside acquire().
                try:
                    parsed = float(raw_timeout)
                except ValueError:
                    raise ValueError(
                        "SNAGLINE_STATE_REDIS_LOCK_TIMEOUT must be a number of "
                        f"seconds, got {raw_timeout!r}"
                    ) from None
                if not math.isfinite(parsed):
                    raise ValueError(
                        "SNAGLINE_STATE_REDIS_LOCK_TIMEOUT must be a finite "
                        f"number of seconds, got {raw_timeout!r}"
                    ) from None
                kwargs["lock_timeout"] = parsed
            try:
                return RedisStateBackend(url, **kwargs)
            except ImportError:
                logger.warning(
                    "snagline: redis backend requested but redis not installed; "
                    "falling back to in-memory state"
                )
    return MemoryStateBackend()
