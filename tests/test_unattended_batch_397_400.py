"""Four unattended contract violations, verified against pristine master.

#397  ``semantic_drift_tolerance`` was unvalidated, so a value at/above 2.0
      silently deadened the goal-drift detector (the comparison ``dev <= tol``
      is always true, so the signal is pinned at 0.0) and a NaN one stormed a
      full-strength alarm on every step.
#398  The live-episode LRU cap evicted episodes that were still running and
      destroyed their in-flight detection state with no log line at all.
#399  ``_dispatch`` iterated the live sink list, so a concurrent
      ``remove_sink`` shifted the tail underneath the iterator and lost or
      duplicated alerts.
#400  A colon-less detector key under ``strict_names`` raised ``IndexError``
      out of a method whose contract says to catch ``ValueError``.
"""

from __future__ import annotations

import threading

import pytest

from snagline.config import Config
from snagline.events import StepEvent, make_signature
from snagline.monitor import Monitor
from snagline.risk import FailureRisk


def _event(episode_id: str, step: int) -> StepEvent:
    return StepEvent(
        step_id=f"s{step}",
        episode_id=episode_id,
        timestamp=float(step),
        action_type="tool_call",
        action_signature=make_signature("tool_call", "tool"),
        tool_name="tool",
    )


class _CountingSink:
    """Records how many risks it saw; optionally mutates the sink list."""

    name = "counting"

    def __init__(self, monitor: Monitor | None = None, other: object | None = None):
        self.count = 0
        self._monitor = monitor
        self._other = other

    def emit(self, risk: FailureRisk) -> None:
        self.count += 1
        if self._monitor is not None and self._other is not None:
            # Re-entrancy exercised in both directions: removing the *next*
            # sink (a shift that used to skip it) and registering a new one
            # (an append that a live iterator used to visit in-flight).
            if self._other in self._monitor._sinks:
                self._monitor.remove_sink(self._other)
            else:
                self._monitor.add_sink(self._other)

    def dump_state(self) -> dict:
        return {}

    def load_state(self, state: dict) -> None:
        pass


# --- #397: the goal-drift noise threshold has a bounded domain --------------


@pytest.mark.parametrize("bad", [2.0, 2.5, 10.0, float("nan"), float("inf"), -0.1])
def test_semantic_drift_tolerance_out_of_range_is_rejected(bad):
    """A tolerance at/above 2.0 deadens the detector and NaN storms alarms."""
    with pytest.raises(ValueError, match="semantic_drift_tolerance"):
        Config(semantic_drift_tolerance=bad)


@pytest.mark.parametrize("good", [0.0, 0.3, 1.9])
def test_semantic_drift_tolerance_accepts_the_valid_range(good):
    assert Config(semantic_drift_tolerance=good).semantic_drift_tolerance == good


def test_semantic_drift_tolerance_is_rejected_after_env_layering():
    """SNAGLINE_SEMANTIC_DRIFT_TOLERANCE=2 must abort startup, not run inert."""
    with pytest.raises(ValueError, match="semantic_drift_tolerance"):
        Config.resolve(environ={"SNAGLINE_SEMANTIC_DRIFT_TOLERANCE": "2"})
    cfg = Config.resolve(environ={"SNAGLINE_SEMANTIC_DRIFT_TOLERANCE": "0.15"})
    assert cfg.semantic_drift_tolerance == 0.15


# --- #398: eviction over cap must be observable ----------------------------


def _cap_monitor(cap: int) -> Monitor:
    from snagline.detectors.loop import LoopDetector

    return Monitor(
        [LoopDetector(repeat_threshold=3)],
        [],
        config=Config(max_live_episodes=cap, loop_repeat_threshold=3),
    )


def test_eviction_over_cap_warns_instead_of_silently_losing_detection(caplog):
    """The cap can still evict a running episode (the monitor cannot tell
    'between events' from 'gone forever'), but the loss must be visible."""
    mon = _cap_monitor(1)
    with caplog.at_level("ERROR", logger="snagline"):
        mon.ingest(_event("A", 0))
        mon.ingest(_event("B", 0))  # over cap -> evict A
    assert any("max_live_episodes" in r.message for r in caplog.records), (
        "eviction over cap must log the fault so a too-small cap is visible"
    )


def test_eviction_warning_is_deduplicated_not_per_event(caplog):
    """A too-small cap evicts on nearly every ingest; one warning is enough."""
    mon = _cap_monitor(1)
    with caplog.at_level("ERROR", logger="snagline"):
        for i in range(50):
            mon.ingest(_event(f"ep{i}", 0))
    warnings = [r for r in caplog.records if "max_live_episodes" in r.message]
    assert len(warnings) == 1


def test_eviction_still_resets_state_and_never_raises():
    """Observability must not change the fail-open teardown contract."""
    mon = _cap_monitor(1)
    mon.ingest(_event("A", 0))
    mon.ingest(_event("B", 0))  # evict A
    assert "A" not in mon._clocks
    assert "B" in mon._live_episodes


# --- #399: dispatch must iterate a snapshot of the sink list ---------------


def test_dispatch_survives_a_removal_during_iteration():
    """A sink removed by an earlier sink's emit must still see this risk:
    list.remove shifts the tail left under a live iterator and skips it."""
    mon = Monitor([], [])
    a, b, c = _CountingSink(), _CountingSink(), _CountingSink()
    for s in (a, b, c):
        mon.add_sink(s)
    a._monitor = mon
    a._other = b  # a's emit removes b, who is next in dispatch order
    risk = FailureRisk("ep", "s", 0.9, "loop", "d", 1.0)
    mon._dispatch(risk)
    assert (a.count, b.count, c.count) == (1, 1, 1), (
        "a concurrent remove_sink must not drop a sink from the in-flight risk"
    )


def test_dispatch_does_not_skip_a_sink_after_a_concurrent_removal():
    """The real defect: ``list.remove`` shifts the tail left under a live
    iterator, so a sink *after* the removed one is skipped even though it never
    stopped being registered. The two stable sinks sit either side of the
    victim, so both must see every dispatched risk."""
    n_rounds = 200
    mon = Monitor([], [])
    before, victim, after = _CountingSink(), _CountingSink(), _CountingSink()
    for s in (before, victim, after):
        mon.add_sink(s)
    risk = FailureRisk("ep", "s", 0.9, "loop", "d", 1.0)
    stop = False

    def hammer() -> None:
        while not stop:
            mon.remove_sink(victim)
            mon.add_sink(victim)

    thread = threading.Thread(target=hammer, daemon=True)
    thread.start()
    try:
        for _ in range(n_rounds):
            mon._dispatch(risk)
    finally:
        stop = True
        thread.join()
    # ``before`` and ``after`` are registered for the whole run; ``after`` sits
    # past the victim in dispatch order, so it is the one the shift used to
    # skip. A snapshot makes both see all n_rounds risks.
    assert before.count == n_rounds, f"before={before.count}"
    assert after.count == n_rounds, (
        f"after={after.count} of {n_rounds}: a registered sink was skipped by a "
        "concurrent remove_sink"
    )


def test_a_sink_added_mid_dispatch_waits_for_the_next_risk():
    """Snapshot semantics: a sink registered during dispatch joins from the
    following risk, rather than receiving the one already in flight (which a
    live-list iterator can hand it, since append grows the list under it)."""
    mon = Monitor([], [])
    a, late = _CountingSink(), _CountingSink()
    mon.add_sink(a)
    a._monitor = mon
    a._other = late  # not yet registered, so a's emit adds it
    risk = FailureRisk("ep", "s", 0.9, "loop", "d", 1.0)
    mon._dispatch(risk)
    assert late.count == 0, "the in-flight risk must not reach a late sink"
    mon._dispatch(risk)
    assert late.count == 1


# --- #400: strict restore must raise ValueError, never IndexError ----------


def test_strict_restore_rejects_a_colonless_key_with_value_error():
    mon = Monitor([], [])
    snap = {
        "format_version": 1,
        "detectors": {"loop": {"windows": {}, "counts": {}}},
    }
    with pytest.raises(ValueError, match="must be '<slot>:<name>'"):
        mon.restore_dict(snap, strict_names=True)


def test_strict_restore_still_reports_composition_mismatch():
    """The existing contract is unchanged for well-formed keys."""
    mon = Monitor([], [])
    snap = {
        "format_version": 1,
        "detectors": {"0:loop": {"windows": {}, "counts": {}}},
    }
    with pytest.raises(ValueError, match="composition mismatch"):
        mon.restore_dict(snap, strict_names=True)
