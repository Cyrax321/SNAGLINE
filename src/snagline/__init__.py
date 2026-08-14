"""SNAGLINE -- lightweight, dependency-free failure detection for AI agents.

Public API (core, zero third-party dependencies):
  * ``Monitor`` -- orchestrator with the fail-open guarantee.
  * ``StepEvent`` / ``EpisodeMeta`` / ``make_signature`` -- canonical schema.
  * ``FailureRisk`` -- the signal detectors emit and sinks consume.

Adapter-specific helpers (e.g. ``watch``) are added by ``snagline.adapters``
and re-exported there; the core never imports a framework.
"""

from __future__ import annotations

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
]
