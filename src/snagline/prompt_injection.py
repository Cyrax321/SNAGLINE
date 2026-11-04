"""Prompt injection detection indicators for untrusted content.

Identifies common injection patterns (system prompt overrides, jailbreak roleplays,
secret exfiltration attempts, and tool manipulation) in untrusted content before
tool execution or step ingestion.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

InjectionSeverity = Literal["low", "medium", "high", "critical"]

SEVERITY_ORDER: dict[InjectionSeverity, int] = {
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

    def to_dict(self) -> dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "severity": self.severity,
            "reason": self.reason,
            "matched_indicator": self.matched_indicator,
        }


@dataclass
class PromptInjectionConfig:
    """Configuration knobs for prompt injection detection."""

    disabled_rules: set[str] = field(default_factory=set)
    min_severity: InjectionSeverity = "medium"
    trusted_keys: set[str] = field(
        default_factory=lambda: {
            "trusted",
            "trusted_input",
            "system_prompt",
            "developer_message",
        }
    )
    custom_patterns: list[tuple[str, InjectionSeverity, str, str]] = field(
        default_factory=list
    )

    def __post_init__(self) -> None:
        if self.min_severity not in SEVERITY_ORDER:
            raise ValueError(
                f"invalid min_severity {self.min_severity!r}; "
                f"must be one of {sorted(SEVERITY_ORDER)}"
            )


DEFAULT_PATTERNS: list[tuple[str, InjectionSeverity, str, str]] = [
    (
        "rule_instruction_override",
        "critical",
        r"(?i)\b(?:ignore|disregard|forget|override|bypass)\s+(?:all\s+)?(?:previous|prior|existing|above)\s+(?:instructions|prompts|rules|guidelines)\b",
        "Attempt to override or disregard prior system instructions",
    ),
    (
        "rule_instruction_override",
        "high",
        r"(?i)\b(?:you\s+must\s+now\s+act\s+as|from\s+now\s+on\s+you\s+are\s+a\s+new\s+agent|new\s+instructions\s+follow)\b",
        "Directive to adopt an unrestricted identity or replace instructions",
    ),
    (
        "rule_jailbreak_roleplay",
        "critical",
        r"(?i)\b(?:dan\s+mode|developer\s+mode\s+enabled|unrestricted\s+mode|always\s+comply\s+without\s+constraints)\b",
        "Jailbreak or constraint-removal roleplay pattern detected",
    ),
    (
        "rule_secret_exfiltration",
        "critical",
        r"(?i)\b(?:print|reveal|output|display|dump|leak|exfiltrate)\s+(?:your\s+)?(?:system\s+prompt|initial\s+instructions|api\s*keys?|bearer\s+tokens?|credentials?|passwords?)\b",
        "Attempt to exfiltrate system instructions, tokens, or credentials",
    ),
    (
        "rule_tool_manipulation",
        "high",
        r"(?i)\b(?:force\s+execute|run\s+command\s+silently|bypass\s+safety\s+filter|execute\s+tool\s+without\s+asking)\b",
        "Directive attempting to bypass authorization gates or force tool execution",
    ),
]


def inspect_text(
    text: str, config: PromptInjectionConfig | None = None
) -> list[PromptInjectionFinding]:
    """Scan untrusted text against configured prompt injection heuristics."""
    if not text or not isinstance(text, str):
        return []

    cfg = config or PromptInjectionConfig()
    min_rank = SEVERITY_ORDER.get(cfg.min_severity, 20)
    findings: list[PromptInjectionFinding] = []

    all_patterns = list(DEFAULT_PATTERNS) + list(cfg.custom_patterns)

    for rule_id, severity, regex_str, reason in all_patterns:
        if rule_id in cfg.disabled_rules:
            continue
        if SEVERITY_ORDER.get(severity, 0) < min_rank:
            continue

        match = re.search(regex_str, text)
        if match:
            matched_span = match.group(0)
            indicator_snippet = matched_span[:48]
            findings.append(
                PromptInjectionFinding(
                    rule_id=rule_id,
                    severity=severity,
                    reason=reason,
                    matched_indicator=indicator_snippet,
                )
            )

    return findings


def inspect_tool_input(
    tool_input: Mapping[str, Any],
    config: PromptInjectionConfig | None = None,
) -> list[PromptInjectionFinding]:
    """Recursively scan tool input parameters for prompt injection indicators."""
    if not tool_input or not isinstance(tool_input, Mapping):
        return []

    cfg = config or PromptInjectionConfig()
    findings: list[PromptInjectionFinding] = []

    def _walk(obj: Any, key_name: str = "") -> None:
        if key_name in cfg.trusted_keys:
            return
        if isinstance(obj, str):
            findings.extend(inspect_text(obj, cfg))
        elif isinstance(obj, Mapping):
            for k, v in obj.items():
                _walk(v, str(k))
        elif isinstance(obj, (list, tuple, set)):
            for item in obj:
                _walk(item, key_name)

    _walk(tool_input)
    return findings


def as_authorize_rule(
    config: PromptInjectionConfig | None = None,
    on_match: Any = "deny",
) -> Any:
    """Return an authorization rule evaluating tool requests for prompt injections."""
    from snagline.authorize import RuleVerdict

    cfg = config or PromptInjectionConfig()

    def _rule(request: Any) -> RuleVerdict | None:
        findings: list[PromptInjectionFinding] = []
        if getattr(request, "resource", None):
            findings.extend(inspect_text(str(request.resource), cfg))
        if getattr(request, "tool_input", None):
            findings.extend(inspect_tool_input(request.tool_input, cfg))

        if not findings:
            return None

        highest = max(findings, key=lambda f: SEVERITY_ORDER.get(f.severity, 0))
        reason = (
            f"Prompt injection indicator detected ({highest.rule_id}): {highest.reason}"
        )
        return RuleVerdict(
            rule_id=f"prompt_injection_{highest.rule_id}",
            decision=on_match,
            reason=reason,
        )

    return _rule
