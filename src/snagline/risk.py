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
    # Governance decay across context compactions (issue #90). Driven by the
    # adapter-defined action types "compaction" / "constraint_present"; see
    # detectors/compaction_tripwire.py for the contract.
    "governance_decay",
    # Horizon-scale time axis (issue #92). Derived by the Monitor from
    # StepEvent timestamps at the top of ingest() -- never from the wall
    # clock, so replay stays deterministic. "idle_gap" marks a silence
    # between consecutive ingests; "wall_clock_budget" covers both the
    # single pre-breach warning (score 0.7, kept inside the "warning" band so
    # it is not paged like the breach) and the breach itself (1.0).
    "idle_gap",
    "wall_clock_budget",
    # Loop hardening modes (issue #89) and the side-effect guard. These are
    # emitted alongside the plain "loop" trigger by detectors/loop.py and
    # detectors/side_effect_guard.py; like every other entry above, the
    # trigger names are API -- CONTINUUM's risk-policy table maps
    # "side_effect_duplicate" to ABORT + immediate reconcile by name.
    "near_duplicate_loop",
    "cycle",
    "stall",
    "side_effect_duplicate",
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
    # "" means "unset": derived from score in __post_init__. A literal
    # severity, including "warning", survives construction exactly as the
    # caller requested (issue #246). The default cannot use the string
    # "warning" itself -- it would be indistinguishable from an explicit
    # "warning" and silently re-derived, re-routing alerts in
    # min_severity-filtered sinks. "" is not a legal severity value, so it
    # can only mean "unset".
    severity: str = ""
    # Triggers whose earlier, coarser finding this risk replaces for the
    # episode. Empty (the default) means "standalone finding".
    supersedes: tuple[TriggerType, ...] = ()
    # Specific step_ids this risk retires (e.g. earlier coarse loop firings
    # that described the same repeated action).
    superseded_step_ids: tuple[str, ...] = ()
    # Specific triggers that supersede this risk (if emitted after a more
    # specific trigger already fired on the same repeated actions).
    superseded_by: tuple[TriggerType, ...] = ()

    def __post_init__(self) -> None:
        # Derive a routing hint from the score only when the caller left
        # severity unset. A caller that passes `severity=` explicitly keeps
        # that value -- all three legal values, warning included.
        if not self.severity:
            object.__setattr__(self, "severity", severity_from_score(self.score))
