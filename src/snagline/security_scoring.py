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
