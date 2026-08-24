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
    "stagnation",
    # Horizon detectors (issues #84/#85/#86). Trigger names are API: the
    # CONTINUUM risk-policy proposal maps them to recovery modes by string.
    "token_runaway",
    "budget_breach",
    "meltdown",
    "silent_abort",
]

# Severity ordering is informational only; sinks decide what to do with it.
SEVERITY_CRITICAL = "critical"
SEVERITY_WARNING = "warning"
SEVERITY_INFO = "info"


def severity_from_score(score: float) -> str:
    """Map a 0..1 risk score to a coarse severity for routing/display."""
    if score >= 0.8:
        return SEVERITY_CRITICAL
    if score >= 0.5:
        return SEVERITY_WARNING
    return SEVERITY_INFO


@dataclass(frozen=True, slots=True)
class FailureRisk:
    """A detected failure signal, dispatched to every registered sink."""

    episode_id: str
    step_id: str
    score: float  # 0.0 - 1.0
    trigger: TriggerType
    detail: str  # short human-readable explanation, no raw content
    timestamp: float
    severity: str = SEVERITY_WARNING  # auto-derived from score if left default

    def __post_init__(self) -> None:
        # If the caller did not set an explicit severity, derive one from the
        # score so every risk carries a useful routing hint. A caller that
        # passes `severity=` explicitly keeps that value.
        if self.severity == SEVERITY_WARNING:
            object.__setattr__(self, "severity", severity_from_score(self.score))
