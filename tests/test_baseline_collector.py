"""Tests for the live BaselineCollector (P1 item 6).

Covers the capture contract and issue #357: ``snapshot()`` used to hand back
the live accumulator, so a caller that mutated the returned profile corrupted
what a later ``commit()`` persisted, and a reader raced ``observe()`` with no
lock on either side.
"""

from __future__ import annotations

import threading

from snagline.baseline_store import BaselineCollector, BaselineStore
from snagline.events import StepEvent, make_signature


def _event(i: int, latency_ms: float = 10.0, error: bool = False) -> StepEvent:
    return StepEvent(
        step_id=str(i),
        episode_id="ep",
        timestamp=1000.0 + i,
        action_type="tool_call",
        action_signature=make_signature("tool_call", "t", "a"),
        tool_name="t",
        latency_ms=latency_ms,
        error=error,
    )


def test_observe_accumulates_into_the_profile():
    col = BaselineCollector()
    for i in range(3):
        col.observe(_event(i))
    snap = col.snapshot()
    assert snap.total_steps == 3
    assert sorted(snap.tools) == ["t"]


def test_snapshot_is_a_copy_not_the_live_accumulator():
    # The bug (issue #357): snapshot() returned self._profile itself, so
    # mutating the result mutated the accumulator a later commit() persisted.
    col = BaselineCollector()
    col.observe(_event(0))
    snap = col.snapshot()
    assert snap is not col._profile
    snap.tools.clear()
    snap.total_steps = 999
    # The live profile is untouched by edits to the snapshot.
    assert col._profile.total_steps == 1
    assert sorted(col._profile.tools) == ["t"]


def test_snapshot_after_mutation_still_sees_subsequent_events():
    col = BaselineCollector()
    col.observe(_event(0))
    first = col.snapshot()
    col.observe(_event(1))
    second = col.snapshot()
    assert first.total_steps == 1
    assert second.total_steps == 2
    assert col._profile.total_steps == 2


def test_commit_persists_the_accumulated_profile_independently(tmp_path):
    store = BaselineStore(root_dir=str(tmp_path / "store"))
    col = BaselineCollector(store=store)
    col.observe(_event(0))
    col.observe(_event(1, latency_ms=20.0))
    captured = col.snapshot()
    version = col.commit()
    assert version is not None
    # What landed on disk is the profile as of commit()...
    loaded = store.load()
    assert loaded is not None
    assert loaded.total_steps == 2
    assert sorted(loaded.tools) == ["t"]
    assert captured.total_steps == 2
    # ...and mutating the snapshot the caller kept cannot reach it.
    captured.tools.clear()
    reloaded = store.load()
    assert reloaded is not None
    assert reloaded.total_steps == 2


def test_commit_copies_so_a_snapshot_taken_later_cannot_reach_the_save(tmp_path):
    # commit() used to hand the live profile to save(); a caller snapshot()
    # taken afterwards returned that same object, so mutating it rewrote what
    # had just been persisted (issue #357).
    store = BaselineStore(root_dir=str(tmp_path / "store"))
    col = BaselineCollector(store=store)
    col.observe(_event(0))
    col.commit()
    col.snapshot().tools.clear()
    persisted = store.load()
    assert persisted is not None
    assert sorted(persisted.tools) == ["t"]


def test_commit_without_a_store_is_a_safe_noop():
    col = BaselineCollector()
    col.observe(_event(0))
    assert col.commit() is None


def test_concurrent_observe_and_snapshot_stay_consistent():
    # snapshot() iterating the profile while observe() adds to it used to be
    # unguarded on both sides (issue #357); the shared lock keeps the copy
    # coherent without holding it across the save.
    col = BaselineCollector()
    stop = threading.Event()

    def ingest() -> None:
        i = 0
        while not stop.is_set():
            col.observe(_event(i))
            i += 1

    threads = [threading.Thread(target=ingest) for _ in range(2)]
    for t in threads:
        t.start()
    try:
        snapshots = []
        for _ in range(50):
            snapshots.append(col.snapshot().total_steps)
    finally:
        stop.set()
        for t in threads:
            t.join(timeout=5.0)
    # Every snapshot is a coherent point-in-time count, monotonically
    # non-decreasing against the live accumulation.
    assert snapshots == sorted(snapshots)
    assert col.snapshot().total_steps >= snapshots[-1]
