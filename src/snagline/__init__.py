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
from snagline.baseline import BaselineProfile, ToolBaseline, fit_baseline_from_jsonl
from snagline.config import Config
from snagline.events import EpisodeMeta, StepEvent, make_signature
from snagline.monitor import Monitor
from snagline.risk import FailureRisk, TriggerType

__all__ = [
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
]
