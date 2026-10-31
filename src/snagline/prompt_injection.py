"""Prompt injection detection indicators for untrusted content.

Identifies common injection patterns (system prompt overrides, jailbreak roleplays,
secret exfiltration attempts, and tool manipulation) in untrusted content before
tool execution or step ingestion.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Any, Dict, List, Literal, Mapping, Optional, Set, Tuple

InjectionSeverity = Literal["low", "medium", "high", "critical"]

SEVERITY_ORDER: Dict[InjectionSeverity, int] = {
    "low": 10,
    "medium": 20,
    "high": 30,
    "critical": 40,
}

@dataclass(frozen=True)
class PromptInjectionFinding:
    """Structured indicator returned when an untrusted payload triggers a pattern."""
    rule_id: str
    severity: InjectionSeverity
    reason: str
    matched_indicator: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "severity": self.severity,
            "reason": self.reason,
            "matched_indicator": self.matched_indicator,
        }

@dataclass
class PromptInjectionConfig:
    """Configuration knobs for prompt injection detection."""
    disabled_rules: Set[str] = field(default_factory=set)
    min_severity: InjectionSeverity = "medium"
    trusted_keys: Set[str] = field(
        default_factory=lambda: {"trusted", "trusted_input", "system_prompt", "developer_message"}
    )
    custom_patterns: List[Tuple[str, InjectionSeverity, str, str]] = field(default_factory=list)
