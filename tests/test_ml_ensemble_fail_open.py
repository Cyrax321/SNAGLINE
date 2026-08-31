"""Regression tests for issue #227: MLOrchestrator must isolate base detectors.

``Monitor.ingest`` guards every detector slot individually, but enabling
``ml_ensemble`` collapses the base detectors into a single slot, so the
Monitor's guard has nothing left to isolate. The orchestrator itself must
therefore honor the README's "one bad detector doesn't block others"
guarantee on every path it delegates: observe, reset, finalize, dump_state
and load_state.
"""

from __future__ import annotations

from typing import Any

from snagline.config import Config
from snagline.detectors.ml_ensemble import MLOrchestrator
from snagline.events import StepEvent, make_signature
from snagline.monitor import Monitor
from snagline.risk import FailureRisk


class _Boom:
    """A base detector that raises on every delegated call."""

    name = "boom"

    def observe(self, event: StepEvent) -> FailureRisk | None:
        raise RuntimeError("observe exploded")

    def reset(self, episode_id: str) -> None:
        raise RuntimeError("reset exploded")

    def finalize(self, episode_id: str) -> FailureRisk | None:
        raise RuntimeError("finalize exploded")

    def dump_state(self) -> dict[str, Any]:
        raise RuntimeError("dump_state exploded")

    def load_state(self, state: dict[str, Any]) -> None:
        raise RuntimeError("load_state exploded")


class _Healthy:
    """A base detector that records every delegated call it receives."""

    name = "healthy"

    def __init__(self, score: float = 0.9) -> None:
        self._score = score
        self.seen = 0
        self.resets = 0
        self.finalized = 0
        self.loaded: dict[str, Any] | None = None

    def observe(self, event: StepEvent) -> FailureRisk | None:
        self.seen += 1
        return FailureRisk(
            event.episode_id,
            event.step_id,
            self._score,
            "healthy",  # type: ignore[arg-type]
            "healthy",
            event.timestamp,
        )

    def reset(self, episode_id: str) -> None:
        self.resets += 1

    def finalize(self, episode_id: str) -> FailureRisk | None:
        self.finalized += 1
        return FailureRisk(
            episode_id,
            "final",
            self._score,
            "healthy",  # type: ignore[arg-type]
            "finalized",
            0.0,
        )

    def dump_state(self) -> dict[str, Any]:
        return {"seen": self.seen}

    def load_state(self, state: dict[str, Any]) -> None:
        self.loaded = state


class _NoReset:
    """A minimal base detector exposing only ``observe``."""

    name = "no_reset"

    def observe(self, event: StepEvent) -> FailureRisk | None:
        return None


def _cfg() -> Config:
    return Config(ml_ensemble_enabled=True, ml_ensemble_score_threshold=0.5)


def _ev(step: str = "0", episode: str = "ep") -> StepEvent:
    return StepEvent(
        step_id=step,
        episode_id=episode,
        timestamp=1.0,
        action_type="tool_call",
        action_signature=make_signature("tool_call", "t", step),
        tool_name="t",
        latency_ms=10.0,
    )


def test_observe_isolates_a_raising_base_detector() -> None:
    healthy = _Healthy()
    orch = MLOrchestrator([_Boom(), healthy], config=_cfg())

    risk = orch.observe(_ev())

    assert healthy.seen == 1, "healthy base detector never saw the step"
    assert risk is not None
    assert risk.trigger == "ml_ensemble"


def test_observe_isolation_holds_for_every_base_position() -> None:
    """A raiser last must not swallow the risk a raiser first would."""
    for base in ([_Boom(), _Healthy()], [_Healthy(), _Boom()]):
        orch = MLOrchestrator(base, config=_cfg())
        assert orch.observe(_ev()) is not None


class _RecordingSink:
    """Captures every risk the Monitor dispatches."""

    def __init__(self) -> None:
        self.received: list[FailureRisk] = []

    def emit(self, risk: FailureRisk) -> None:
        self.received.append(risk)


def test_monitor_still_emits_when_one_base_detector_raises() -> None:
    sink = _RecordingSink()
    monitor = Monitor([MLOrchestrator([_Boom(), _Healthy()], config=_cfg())], [sink])

    monitor.ingest(_ev())

    assert len(sink.received) == 1, "the ensemble went silent instead of degrading"
    assert sink.received[0].trigger == "ml_ensemble"


def test_a_faulty_base_detector_is_logged_once_not_per_step(caplog) -> None:
    orch = MLOrchestrator([_Boom(), _Healthy()], config=_cfg())
    with caplog.at_level("WARNING", logger="snagline"):
        for i in range(25):
            orch.observe(_ev(str(i)))
    faults = [r for r in caplog.records if "boom observe raised" in r.getMessage()]
    assert len(faults) == 1


def test_reset_isolates_so_sibling_episode_state_is_still_freed() -> None:
    healthy = _Healthy()
    orch = MLOrchestrator([_Boom(), healthy], config=_cfg())

    orch.reset("ep")

    assert healthy.resets == 1, "sibling state leaked past a failed reset"


def test_reset_skips_a_base_detector_without_reset() -> None:
    healthy = _Healthy()
    orch = MLOrchestrator([_NoReset(), healthy], config=_cfg())

    orch.reset("ep")  # must not raise AttributeError

    assert healthy.resets == 1


def test_finalize_isolates_so_later_completion_checks_still_run() -> None:
    healthy = _Healthy()
    orch = MLOrchestrator([_Boom(), healthy], config=_cfg())

    risk = orch.finalize("ep")

    assert healthy.finalized == 1
    assert risk is not None and risk.detail == "finalized"


def test_dump_state_isolates_so_the_snapshot_still_carries_siblings() -> None:
    healthy = _Healthy()
    healthy.seen = 7
    orch = MLOrchestrator([_Boom(), healthy], config=_cfg())

    dumped = orch.dump_state()

    assert dumped == {"healthy": {"seen": 7}}


def test_load_state_isolates_so_siblings_are_still_restored() -> None:
    healthy = _Healthy()
    orch = MLOrchestrator([_Boom(), healthy], config=_cfg())

    orch.load_state({"boom": {"x": 1}, "healthy": {"seen": 42}})

    assert healthy.loaded == {"seen": 42}


def test_monitor_snapshot_round_trip_survives_a_faulty_base_detector(
    tmp_path,
) -> None:
    healthy = _Healthy()
    healthy.seen = 3
    monitor = Monitor([MLOrchestrator([_Boom(), healthy], config=_cfg())], [])
    path = str(tmp_path / "snap.json")

    monitor.snapshot(path)

    target = _Healthy()
    restored = Monitor([MLOrchestrator([_Boom(), target], config=_cfg())], [])
    restored.restore(path)

    assert target.loaded == {"seen": 3}
