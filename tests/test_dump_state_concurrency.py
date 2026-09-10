"""``dump_state`` must not walk a live per-episode dict (#229 follow-up, #231).

Every stateful detector keys its state by ``episode_id``, and the Monitor holds
only a *per-episode* lock while ingesting. ``Monitor.snapshot()`` calls each
detector's ``dump_state`` with no lock at all, so a Python-level comprehension
over ``self._some_dict.items()`` yields between bytecodes and an ingest meeting
a new episode inserts a key mid-walk::

    File "src/snagline/detectors/error_cascade.py", line 141, in dump_state
      "windows": {ep: list(w) for ep, w in self._windows.items()},
    RuntimeError: dictionary keys changed during iteration

That escapes ``Monitor.snapshot`` -- a public API with no fail-open guard,
because snapshot/restore faults are deliberately loud. The fix routes each walk
through ``detectors.base.snapshot_items``, which copies the pairs in a single C
call that cannot be interrupted.

The race tests drive the collision for a bounded wall time with several
threads; they reproduce in well under a second on the unfixed code and are a
no-op once the copies are in place. The rest pin the payloads so the change
stays behavior-preserving.
"""

from __future__ import annotations

import itertools
import threading
from collections.abc import Callable
from typing import Any

import pytest

from snagline.baseline import BaselineProfile
from snagline.detectors.base import snapshot_items
from snagline.detectors.compaction_tripwire import CompactionTripwireDetector
from snagline.detectors.error_cascade import ErrorCascadeDetector
from snagline.detectors.goal_drift import GoalDriftDetector
from snagline.detectors.loop import LoopDetector
from snagline.detectors.meltdown import MeltdownDetector
from snagline.detectors.side_effect_guard import SideEffectGuardDetector
from snagline.detectors.silent_abort import SilentAbortDetector
from snagline.detectors.stagnation import StagnationDetector
from snagline.detectors.token_runaway import TokenRunawayDetector
from snagline.events import StepEvent, make_signature
from snagline.monitor import Monitor

_RACE_SECONDS = 1.5

# Every detector whose dump_state walks a per-episode dict. Built so that the
# shared _ev() below populates that dict on the first observe.
_FACTORIES: dict[str, Callable[[], Any]] = {
    "loop": LoopDetector,
    "error_cascade": ErrorCascadeDetector,
    "stagnation": StagnationDetector,
    "meltdown": MeltdownDetector,
    "token_runaway": TokenRunawayDetector,
    "compaction_tripwire": CompactionTripwireDetector,
    "silent_abort": SilentAbortDetector,
    "side_effect_guard": SideEffectGuardDetector,
    # Inert without a healthy reference, so _live never fills.
    "goal_drift": lambda: GoalDriftDetector(baseline=BaselineProfile()),
}


def _ev(episode: str, i: int) -> StepEvent:
    """One event rich enough that every detector above records the episode."""
    return StepEvent(
        step_id=str(i),
        episode_id=episode,
        timestamp=float(i),
        action_type="tool_call",
        action_signature=make_signature("tool_call", "write_file", str(i)),
        tool_name="write_file",
        latency_ms=10.0,
        tokens_in=5,
        tokens_out=5,
        side_effect=True,
        error=i % 3 == 0,
    )


class _Worker:
    """A stoppable loop that records the first exception it sees."""

    def __init__(self, stop: threading.Event, errors: list[BaseException]) -> None:
        self.stop = stop
        self.errors = errors

    def __call__(self) -> None:
        for n in itertools.count():
            if self.stop.is_set():
                return
            try:
                self.step(n)
            except BaseException as exc:
                self.errors.append(exc)
                self.stop.set()
                return

    def step(self, n: int) -> None:  # pragma: no cover - overridden
        raise NotImplementedError


def _race(workers: list[Any], seconds: float = _RACE_SECONDS) -> None:
    """Start every worker, let them collide, then stop and join."""
    threads = [threading.Thread(target=w, daemon=True) for w in workers]
    for thread in threads:
        thread.start()
    threading.Event().wait(timeout=seconds)
    for worker in workers:
        worker.stop.set()
    for thread in threads:
        thread.join(timeout=10.0)


# --- the reported failure: Monitor.default() under live ingest --------------


def test_monitor_default_snapshot_dict_survives_concurrent_ingest() -> None:
    """The exact #231 repro: the full detector stack, snapshotted while busy."""
    monitor = Monitor.default()
    stop = threading.Event()
    errors: list[BaseException] = []

    class Ingest(_Worker):
        def __init__(self, tag: str) -> None:
            super().__init__(stop, errors)
            self.tag = tag

        def step(self, n: int) -> None:
            monitor.ingest(_ev(f"{self.tag}-{n}", n))

    class Snapshot(_Worker):
        def step(self, n: int) -> None:
            monitor.snapshot_dict()

    _race([Ingest("w0"), Ingest("w1"), Ingest("w2"), Snapshot(stop, errors)])

    assert not errors, f"snapshot_dict raced ingest: {errors[0]!r}"


def test_monitor_default_snapshot_file_survives_concurrent_ingest(tmp_path) -> None:
    """Same race through the public on-disk path, which also JSON-encodes."""
    monitor = Monitor.default()
    stop = threading.Event()
    errors: list[BaseException] = []
    path = str(tmp_path / "snap.json")

    class Ingest(_Worker):
        def step(self, n: int) -> None:
            monitor.ingest(_ev(f"ep-{n}", n))

    class Snapshot(_Worker):
        def step(self, n: int) -> None:
            monitor.snapshot(path)

    _race([Ingest(stop, errors), Ingest(stop, errors), Snapshot(stop, errors)])

    assert not errors, f"snapshot raced ingest: {errors[0]!r}"


# --- per-detector coverage -------------------------------------------------


@pytest.mark.parametrize("factory", list(_FACTORIES.values()), ids=list(_FACTORIES))
def test_detector_dump_state_survives_concurrent_observe(factory) -> None:
    detector = factory()
    stop = threading.Event()
    errors: list[BaseException] = []

    class Observe(_Worker):
        def __init__(self, tag: str) -> None:
            super().__init__(stop, errors)
            self.tag = tag

        def step(self, n: int) -> None:
            # A brand-new episode every step: maximum key-set churn.
            detector.observe(_ev(f"{self.tag}-{n}", n))

    class Dump(_Worker):
        def step(self, n: int) -> None:
            detector.dump_state()

    _race([Observe("o0"), Observe("o1"), Observe("o2"), Dump(stop, errors)])

    assert not errors, f"dump_state raced observe: {errors[0]!r}"


@pytest.mark.parametrize("factory", list(_FACTORIES.values()), ids=list(_FACTORIES))
def test_detector_dump_state_still_records_every_live_episode(factory) -> None:
    """The copy must not drop entries: two episodes in, two episodes out."""
    detector = factory()
    for i in range(6):
        detector.observe(_ev("a", i))
        detector.observe(_ev("b", i))

    dumped = detector.dump_state()

    walked = [v for v in dumped.values() if isinstance(v, dict) and v]
    assert walked, f"no per-episode dict populated in {dumped!r}"
    for section in walked:
        assert set(section) == {"a", "b"}


@pytest.mark.parametrize("factory", list(_FACTORIES.values()), ids=list(_FACTORIES))
def test_detector_state_round_trip_is_unchanged(factory) -> None:
    detector = factory()
    for i in range(6):
        detector.observe(_ev("a", i))
        detector.observe(_ev("b", i))
    dumped = detector.dump_state()

    target = factory()
    target.load_state(dumped)

    assert target.dump_state() == dumped


# --- snapshot_items itself -------------------------------------------------


def test_snapshot_items_preserves_every_pair() -> None:
    source = {"a": 1, "b": 2, "c": 3}

    assert snapshot_items(source) == list(source.items())


def test_snapshot_items_returns_an_independent_copy() -> None:
    """A later insert must not be visible to a walk already under way."""
    source = {"a": 1}
    copied = snapshot_items(source)
    source["b"] = 2

    assert copied == [("a", 1)]


def test_snapshot_items_tolerates_an_empty_dict() -> None:
    assert snapshot_items({}) == []
