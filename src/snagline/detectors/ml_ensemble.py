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

Fail-open isolation is this class's own responsibility. ``Monitor.ingest``
guards every detector slot individually, but wrapping the base detectors
collapses them into one slot, so the Monitor's guard has nothing left to
isolate: without the per-base guards below, one raising detector would block
all the others and the Monitor would swallow the evidence. Every delegation
loop here therefore logs the fault once and moves on to the next detector.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from typing import Any

from snagline.config import Config
from snagline.events import StepEvent
from snagline.risk import FailureRisk

logger = logging.getLogger("snagline")


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
        # A faulty base detector must not spam the logs on every step; mirror
        # Monitor._log_fault_once and report each distinct fault exactly once.
        self._fault_logged: set[str] = set()

    @staticmethod
    def _base_name(det: Any) -> str:
        return str(getattr(det, "name", type(det).__name__))

    def _log_fault_once(self, det: Any, phase: str) -> None:
        key = f"{self._base_name(det)}:{phase}"
        if key in self._fault_logged:
            return
        self._fault_logged.add(key)
        logger.warning(
            "snagline: ml_ensemble base detector %s %s raised; ignoring (fail-open)",
            self._base_name(det),
            phase,
        )

    def observe(self, event: StepEvent) -> FailureRisk | None:
        scores: list[float] = []
        for det in self._base:
            try:
                risk = det.observe(event)
            except Exception:
                self._log_fault_once(det, "observe")
                continue
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
            reset = getattr(det, "reset", None)
            if not callable(reset):
                continue
            try:
                reset(episode_id)
            except Exception:
                # A leaked reset strands that episode's state forever and
                # defeats the retention cap (issue #184); it must never also
                # strand every sibling's state.
                self._log_fault_once(det, "reset")

    def finalize(self, episode_id: str) -> FailureRisk | None:
        # Passthrough for EpisodeFinalizer bases (issue #86): Monitor.end_episode
        # only sees top-level detectors, so an orchestrated completion check
        # would otherwise never be consulted. First non-None verdict wins; a
        # finished episode yields at most one end-of-run judgment.
        for det in self._base:
            finalize = getattr(det, "finalize", None)
            if callable(finalize):
                try:
                    risk = finalize(episode_id)
                except Exception:
                    self._log_fault_once(det, "finalize")
                    continue
                if risk is not None:
                    return risk
        return None

    def dump_state(self) -> dict[str, Any] | None:
        # Delegate so snapshot()/restore() reach through the wrapper (issue #91).
        merged: dict[str, Any] = {}
        for det in self._base:
            dump = getattr(det, "dump_state", None)
            if not callable(dump):
                continue
            try:
                state = dump()
            except Exception:
                # Losing one base detector's state is recoverable (it restarts
                # cold); losing the whole snapshot is not.
                self._log_fault_once(det, "dump_state")
                continue
            if state is not None:
                merged[self._base_name(det)] = state
        return merged or None

    def load_state(self, state: dict[str, Any]) -> None:
        for det in self._base:
            load = getattr(det, "load_state", None)
            if not callable(load):
                continue
            sub = state.get(self._base_name(det))
            if sub is not None:
                try:
                    load(sub)
                except Exception:
                    # Skipping one entry leaves that detector cold rather than
                    # leaving every later sibling silently unrestored.
                    self._log_fault_once(det, "load_state")
