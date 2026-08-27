"""SNAGLINE -- lightweight, dependency-free failure detection for AI agents.

Public API (core, zero third-party dependencies):
  * ``Monitor`` -- orchestrator with the fail-open guarantee. ``Monitor.default()``
    returns a ready-to-use instance with the tier-1 detectors and console sink.
  * ``StepEvent`` / ``EpisodeMeta`` / ``make_signature`` -- canonical schema.
  * ``FailureRisk`` -- the signal detectors emit and sinks consume.
  * ``Config`` -- all tunable thresholds in one place.
  * ``watch`` -- the stdlib ``raw`` adapter for plain Python loops.
"""

from __future__ import annotations

from snagline.adapters.raw import watch
from snagline.baseline import (
    BaselineProfile,
    ToolBaseline,
    fit_baseline_from_jsonl,
    load_baseline,
    save_baseline,
)
from snagline.config import Config
from snagline.detectors.goal_drift import GoalDriftDetector
from snagline.detectors.ml_ensemble import MLOrchestrator
from snagline.events import EpisodeMeta, StepEvent, make_signature
from snagline.monitor import Monitor
from snagline.risk import FailureRisk, TriggerType

try:
    from importlib.metadata import version

    __version__ = version("snagline")
except Exception:
    __version__ = "0.1.0"

__all__ = [
    "__version__",
    "Monitor",
    "StepEvent",
    "EpisodeMeta",
    "make_signature",
    "FailureRisk",
    "TriggerType",
    "Config",
    "watch",
    "BaselineProfile",
    "ToolBaseline",
    "fit_baseline_from_jsonl",
    "save_baseline",
    "load_baseline",
    "GoalDriftDetector",
    "MLOrchestrator",
]
