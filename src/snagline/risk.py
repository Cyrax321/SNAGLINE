"""Failure risk payload emitted by detectors and consumed by sinks.

A ``FailureRisk`` deliberately carries NO raw content (no prompt, no response,
no ``StepEvent.metadata``). This is what keeps alerting channels (webhook,
Slack) from becoming accidental data-exfiltration paths (project.md §11).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

TriggerType = Literal[
    "loop",
    "error_cascade",
    "latency_anomaly",
    "goal_drift",
    "ml_ensemble",
]


@dataclass(frozen=True, slots=True)
class FailureRisk:
    """A detected failure signal, dispatched to every registered sink."""

    episode_id: str
    step_id: str
    score: float  # 0.0 - 1.0
    trigger: TriggerType
    detail: str  # short human-readable explanation, no raw content
    timestamp: float
