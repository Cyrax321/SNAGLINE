"""Tests for the ML ensemble orchestrator (next phase, step 3)."""

from __future__ import annotations

from snagline.config import Config
from snagline.detectors.ml_ensemble import MLOrchestrator
from snagline.events import StepEvent
from snagline.risk import FailureRisk


class _Stub:
    """A detector stub returning a fixed score, used to exercise the ensemble."""

    def __init__(self, score: float) -> None:
        self._score = score
        self.resets = 0

    def observe(self, event: StepEvent) -> FailureRisk | None:
        if self._score <= 0:
            return None
        return FailureRisk(
            event.episode_id,
            event.step_id,
            self._score,
            "stub",  # type: ignore[arg-type]
            "stub",
            event.timestamp,
        )

    def reset(self, episode_id: str) -> None:
        self.resets += 1


def _ev(episode="ep"):
    return StepEvent(
        step_id="s",
        episode_id=episode,
        timestamp=1.0,
        action_type="tool_call",
        action_signature="sig",
        tool_name="t",
        latency_ms=100.0,
        error=False,
    )


def test_ml_ensemble_noop_without_scores():
    orch = MLOrchestrator([_Stub(0.0), _Stub(0.0)], config=Config())
    assert orch.observe(_ev()) is None


def test_ml_ensemble_noisy_or_combines():
    cfg = Config()
    cfg.ml_ensemble_score_threshold = 0.5
    orch = MLOrchestrator([_Stub(0.3), _Stub(0.4)], config=cfg)
    risk = orch.observe(_ev())
    assert risk is not None
    assert risk.trigger == "ml_ensemble"
    # noisy-OR of 0.3 and 0.4 = 1 - 0.7*0.6 = 0.58
    assert abs(risk.score - 0.58) < 1e-6


def test_ml_ensemble_uses_model_when_provided():
    cfg = Config()
    cfg.ml_ensemble_score_threshold = 0.1
    # model that simply takes the max of the base scores
    orch = MLOrchestrator([_Stub(0.2), _Stub(0.6)], config=cfg, model=max)
    risk = orch.observe(_ev())
    assert risk is not None
    assert abs(risk.score - 0.6) < 1e-6


def test_ml_ensemble_below_threshold_is_silent():
    cfg = Config()
    cfg.ml_ensemble_score_threshold = 0.99
    orch = MLOrchestrator([_Stub(0.3), _Stub(0.4)], config=cfg)
    assert orch.observe(_ev()) is None


def test_ml_ensemble_reset_propagates():
    a, b = _Stub(0.5), _Stub(0.5)
    orch = MLOrchestrator([a, b], config=Config())
    orch.reset("ep")
    assert a.resets == 1
    assert b.resets == 1


def test_monitor_default_with_ml_ensemble_builds():
    from snagline.monitor import Monitor

    cfg = Config()
    cfg.ml_ensemble_enabled = True
    mon = Monitor.default(config=cfg)
    # Exactly one orchestrator detector wraps the base detectors (no double count).
    assert len(mon._detectors) == 1
    assert mon._detectors[0].name == "ml_ensemble"
