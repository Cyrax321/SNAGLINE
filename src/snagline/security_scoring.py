"""Dedicated security risk scoring engine.

Evaluates pre-execution tool requests independently of runtime failure risk,
producing a deterministic composite security score (0-100 scale) with explainable
contributing factors and policy threshold integration.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Mapping, Optional

from snagline.authorize import Decision

@dataclass(frozen=True)
class SecurityFactor:
    """Individual component contributing to the composite security risk score."""
    name: str
    impact: float
    reason: str
    rule_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "impact": round(self.impact, 2),
            "reason": self.reason,
            "rule_id": self.rule_id,
        }

@dataclass(frozen=True)
class SecurityScoreResult:
    """Composite security evaluation outcome."""
    score: float
    decision: Decision
    factors: List[SecurityFactor] = field(default_factory=list)
    rule_ids: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "score": round(self.score, 2),
            "decision": self.decision,
            "factors": [f.to_dict() for f in self.factors],
            "rule_ids": self.rule_ids,
        }


DEFAULT_TOOL_SENSITIVITIES: Dict[str, float] = {
    "bash": 35.0,
    "terminal": 35.0,
    "sh": 35.0,
    "zsh": 35.0,
    "write": 20.0,
    "edit": 20.0,
    "patch": 20.0,
    "delete": 30.0,
    "rm": 30.0,
    "read": 5.0,
    "grep": 5.0,
    "find": 5.0,
    "ls": 5.0,
}

FINDING_SEVERITY_IMPACTS: Dict[str, float] = {
    "low": 10.0,
    "medium": 25.0,
    "high": 50.0,
    "critical": 80.0,
}


class SecurityScoringEngine:
    """Calculates deterministic composite security risk score on a 0-100 scale."""

    def __init__(
        self,
        ask_threshold: float = 40.0,
        deny_threshold: float = 75.0,
        tool_sensitivities: Optional[Mapping[str, float]] = None,
        unknown_tool_sensitivity: float = 25.0,
        fail_closed: bool = False,
    ) -> None:
        if not (0.0 <= ask_threshold <= deny_threshold <= 100.0):
            raise ValueError(
                f"Invalid thresholds: expected 0 <= ask ({ask_threshold}) <= deny ({deny_threshold}) <= 100"
            )
        self.ask_threshold = float(ask_threshold)
        self.deny_threshold = float(deny_threshold)
        self.unknown_tool_sensitivity = float(unknown_tool_sensitivity)
        self.fail_closed = fail_closed

        self.tool_sensitivities: Dict[str, float] = dict(DEFAULT_TOOL_SENSITIVITIES)
        if tool_sensitivities:
            for k, v in tool_sensitivities.items():
                self.tool_sensitivities[k.lower()] = float(v)

    def evaluate(
        self,
        tool_name: str,
        action: str = "unknown",
        resource: Optional[str] = None,
        findings: Optional[List[Dict[str, Any]]] = None,
    ) -> SecurityScoreResult:
        """Evaluate a tool invocation and calculate its composite security risk score."""
        factors: List[SecurityFactor] = []
        rule_ids: List[str] = []

        # 1. Base Tool Sensitivity Factor
        t_key = tool_name.strip().lower()
        if t_key in self.tool_sensitivities:
            t_impact = self.tool_sensitivities[t_key]
            factors.append(
                SecurityFactor(
                    name="tool_sensitivity",
                    impact=t_impact,
                    reason=f"Base sensitivity for tool {tool_name!r} is {t_impact:.1f}",
                )
            )
        else:
            factors.append(
                SecurityFactor(
                    name="unknown_tool_penalty",
                    impact=self.unknown_tool_sensitivity,
                    reason=f"Unrecognized tool {tool_name!r} assigned baseline sensitivity {self.unknown_tool_sensitivity:.1f}",
                )
            )

        # 2. Findings Impact
        if findings:
            for item in findings:
                r_id = item.get("rule_id", "unspecified_rule")
                sev = str(item.get("severity", "medium")).lower()
                impact = FINDING_SEVERITY_IMPACTS.get(sev, 25.0)
                reason = item.get("reason", f"Security finding from {r_id}")
                factors.append(
                    SecurityFactor(
                        name=f"finding_{r_id}",
                        impact=impact,
                        reason=reason,
                        rule_id=r_id,
                    )
                )
                if r_id not in rule_ids:
                    rule_ids.append(r_id)

        # Compute aggregate raw score with diminishing returns above 50
        raw_sum = sum(f.impact for f in factors)
        if raw_sum > 50.0:
            excess = raw_sum - 50.0
            composite_score = 50.0 + (excess * 0.75)
        else:
            composite_score = raw_sum

        composite_score = min(max(composite_score, 0.0), 100.0)

        # Determine threshold decision
        if composite_score >= self.deny_threshold:
            decision: Decision = "deny"
        elif composite_score >= self.ask_threshold:
            decision = "ask"
        else:
            decision = "allow"

        return SecurityScoreResult(
            score=composite_score,
            decision=decision,
            factors=factors,
            rule_ids=rule_ids,
        )

