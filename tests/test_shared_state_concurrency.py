"""Concurrency regressions for cross-episode shared state (#228, #229).

Two dicts in the hot path are keyed across episodes rather than partitioned by
one, so the Monitor's *per-episode* lock does not protect their key sets:

* ``Monitor._clocks`` -- ``snapshot_dict`` walked it with no lock while
  ``_advance_clock`` inserted under only the episode lock, so a snapshot taken
  while the host kept ingesting raised ``RuntimeError`` out of a public API
  that has no fail-open guard (#228).
* ``LatencyAnomalyDetector._states`` -- keyed by ``(episode_id, tool_name)``,
  so ``reset`` has to scan instead of popping. Teardown for one episode raced
  an ingest meeting a new tool in another; the Monitor swallowed the
  ``RuntimeError`` fail-open and the episode's state leaked past the #184
  retention cap (#229).

Both are timing-dependent, so each test drives the race for a bounded wall
time with several threads. They reproduce reliably on the unfixed code
(sub-second) and are a no-op once the locks are in place.
"""

from __future__ import annotations

import itertools
import threading

from snagline.config import Config
from snagline.detectors.latency_anomaly import LatencyAnomalyDetector
from snagline.events import StepEvent, make_signature
from snagline.monitor import Monitor

# Any horizon knob activates the time axis; with both unset _advance_clock
# returns early and _clocks stays empty, which is why the rest of the suite
# never exercises this path.
_HORIZON = Config(idle_warn_seconds=3600.0)

_RACE_SECONDS = 1.5


def _ev(episode: str, i: int, tool: str = "t") -> StepEvent:
    return StepEvent(
        step_id=str(i),
        episode_id=episode,
        timestamp=float(i),
        action_type="tool_call",
        action_signature=make_signature("tool_call", tool, str(i)),
        tool_name=tool,
        latency_ms=10.0,
    )


def _race(workers: list, seconds: float = _RACE_SECONDS) -> None:
    """Start every callable, let them run, then join."""
    threads = [threading.Thread(target=w, daemon=True) for w in workers]
    for thread in threads:
        thread.start()
    threading.Event().wait(timeout=seconds)
    for worker in workers:
        worker.stop.set()  # type: ignore[attr-defined]
    for thread in threads:
        thread.join(timeout=10.0)


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


# --- #228: Monitor._clocks -------------------------------------------------


def test_snapshot_dict_survives_concurrent_ingest_of_new_episodes() -> None:
    monitor = Monitor(detectors=[], sinks=[], config=_HORIZON)
    stop, errors = threading.Event(), []

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

    assert not errors, f"snapshot raced ingest: {errors[0]!r}"


def test_snapshot_dict_survives_concurrent_end_episode() -> None:
    """Teardown pops from _clocks; the snapshot walk must tolerate that too."""
    monitor = Monitor(detectors=[], sinks=[], config=_HORIZON)
    stop, errors = threading.Event(), []

    class Churn(_Worker):
        def __init__(self, tag: str) -> None:
            super().__init__(stop, errors)
            self.tag = tag

        def step(self, n: int) -> None:
            episode = f"{self.tag}-{n}"
            monitor.ingest(_ev(episode, n))
            monitor.ingest(_ev(episode, n + 1))
            monitor.end_episode(episode)

    class Snapshot(_Worker):
        def step(self, n: int) -> None:
            monitor.snapshot_dict()

    _race([Churn("c0"), Churn("c1"), Churn("c2"), Snapshot(stop, errors)])

    assert not errors, f"snapshot raced end_episode: {errors[0]!r}"


def test_time_axis_snapshot_content_is_unchanged() -> None:
    """The lock must not alter what a single-threaded snapshot records."""
    monitor = Monitor(detectors=[], sinks=[], config=_HORIZON)
    monitor.ingest(_ev("ep", 0))
    monitor.ingest(_ev("ep", 1))

    time_axis = monitor.snapshot_dict()["time_axis"]

    assert set(time_axis) == {"ep"}
    assert time_axis["ep"]["last_ts"] == 1.0
    assert time_axis["ep"]["idle_fired"] is False


def test_first_event_of_an_episode_still_establishes_the_clock_quietly() -> None:
    """The reworked _advance_clock head must keep its no-delta early return."""
    received: list[object] = []

    class Sink:
        def emit(self, risk: object) -> None:
            received.append(risk)

    monitor = Monitor(detectors=[], sinks=[Sink()], config=_HORIZON)
    monitor.ingest(_ev("ep", 0))

    assert received == []
    assert set(monitor.snapshot_dict()["time_axis"]) == {"ep"}


def test_time_axis_round_trip_still_restores(tmp_path) -> None:
    monitor = Monitor(detectors=[], sinks=[], config=_HORIZON)
    monitor.ingest(_ev("ep", 0))
    monitor.ingest(_ev("ep", 5))
    path = str(tmp_path / "snap.json")
    monitor.snapshot(path)

    restored = Monitor(detectors=[], sinks=[], config=_HORIZON)
    restored.restore(path)

    assert restored.snapshot_dict()["time_axis"]["ep"]["last_ts"] == 5.0


# --- #229: LatencyAnomalyDetector._states ---------------------------------


def test_reset_survives_concurrent_observe_of_other_episodes() -> None:
    detector = LatencyAnomalyDetector()
    stop, errors = threading.Event(), []

    class Observe(_Worker):
        def __init__(self, tag: str) -> None:
            super().__init__(stop, errors)
            self.tag = tag

        def step(self, n: int) -> None:
            # A new (episode, tool) pair every step: maximum key-set churn.
            detector.observe(_ev(f"{self.tag}-{n}", n, tool=f"tool-{n}"))

    class Teardown(_Worker):
        def step(self, n: int) -> None:
            detector.observe(_ev("mine", n, tool=f"t{n}"))
            detector.reset("mine")

    _race([Observe("o0"), Observe("o1"), Observe("o2"), Teardown(stop, errors)])

    assert not errors, f"reset raced observe: {errors[0]!r}"
    leaked = [k for k in detector.dump_state()["states"] if k[0][0] == "mine"]
    assert not leaked, f"reset left {len(leaked)} entries behind"


def test_dump_state_survives_concurrent_observe() -> None:
    detector = LatencyAnomalyDetector()
    stop, errors = threading.Event(), []

    class Observe(_Worker):
        def __init__(self, tag: str) -> None:
            super().__init__(stop, errors)
            self.tag = tag

        def step(self, n: int) -> None:
            detector.observe(_ev(f"{self.tag}-{n}", n, tool=f"tool-{n}"))

    class Dump(_Worker):
        def step(self, n: int) -> None:
            detector.dump_state()

    _race([Observe("o0"), Observe("o1"), Observe("o2"), Dump(stop, errors)])

    assert not errors, f"dump_state raced observe: {errors[0]!r}"


def test_reset_still_only_drops_the_named_episode() -> None:
    detector = LatencyAnomalyDetector()
    detector.observe(_ev("a", 0, tool="x"))
    detector.observe(_ev("a", 1, tool="y"))
    detector.observe(_ev("b", 2, tool="x"))

    detector.reset("a")

    remaining = {tuple(k) for k, _ in detector.dump_state()["states"]}
    assert remaining == {("b", "x")}


def test_state_round_trip_is_unchanged_by_the_load_state_rework() -> None:
    detector = LatencyAnomalyDetector()
    for i in range(10):
        detector.observe(_ev("ep", i, tool="x"))
    dumped = detector.dump_state()

    target = LatencyAnomalyDetector()
    target.load_state(dumped)

    assert target.dump_state() == dumped


def test_load_state_replaces_rather_than_merges() -> None:
    detector = LatencyAnomalyDetector()
    detector.observe(_ev("stale", 0, tool="x"))

    detector.load_state({"states": []})

    assert detector.dump_state() == {"states": []}
