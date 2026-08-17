"""ML ensemble detector (next phase, step 3).

An orchestrator that combines the signals of several base detectors into a
single, stronger failure risk.

The default combiner is a transparent, dependency-free *noisy-OR* over the
base detectors' scores (``1 - prod(1 - score_i)``), which boosts confidence
when multiple independent detectors agree. A real ML model can be slotted in
via the ``model`` argument: a callable ``(List[float]) -> float`` that maps
the per-detector scores to a combined score. Installing the ``ml`` extra and
passing a fitted scikit-learn pipeline through ``model`` is the intended
upgrade path; it is never imported here, so the zero-dep default holds.

When ``Config.ml_ensemble_enabled`` is set, ``Monitor.default()`` wraps the
base detectors in a single ``MLOrchestrator`` instead of exposing them
individually, so there is no double counting.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from snagline.config import Config
from snagline.events import StepEvent
from snagline.risk import FailureRisk


class MLOrchestrator:
    name = "ml_ensemble"

    def __init__(
        self,
        detectors: Sequence[Any],
        config: Config | None = None,
        model: Callable[[list[float]], float] | None = None,
    ) -> None:
        self._base = list(detectors)
        self._cfg = config or Config()
        self._model = model

    def observe(self, event: StepEvent) -> FailureRisk | None:
        scores: list[float] = []
        for det in self._base:
            risk = det.observe(event)
            if risk is not None and 0.0 < risk.score <= 1.0:
                scores.append(risk.score)
        if not scores:
            return None
        combined = self._combine(scores)
        if combined < self._cfg.ml_ensemble_score_threshold:
            return None
        return FailureRisk(
            event.episode_id,
            event.step_id,
            min(1.0, combined),
            "ml_ensemble",
            "ensemble of detector signals",
            event.timestamp,
        )

    def _combine(self, scores: list[float]) -> float:
        if self._model is not None:
            return float(self._model(scores))
        return self._noisy_or(scores)

    @staticmethod
    def _noisy_or(scores: list[float]) -> float:
        p_no_failure = 1.0
        for s in scores:
            p_no_failure *= 1.0 - s
        return 1.0 - p_no_failure

    def reset(self, episode_id: str) -> None:
        for det in self._base:
            det.reset(episode_id)
