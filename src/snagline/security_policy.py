"""Security policy schema and validation layer.

Provides strict, deterministic schema definitions for security policies, validating
fields, types, tool restrictions, and detector parameters while rejecting unknown keys.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Literal, Optional, Set

from snagline.authorize import Decision

class PolicyValidationError(ValueError):
    """Raised when a security policy configuration fails schema validation."""
    pass

@dataclass
class AgentRuleSpec:
    """Per-agent execution constraints."""
    allowed_tools: List[str] = field(default_factory=list)
    denied_tools: List[str] = field(default_factory=list)
    ask_tools: List[str] = field(default_factory=list)

@dataclass
class SecurityPolicySchema:
    """Validated security policy specification."""
    version: str = "1.0"
    name: str = "default_security_policy"
    profile: Optional[str] = None
    default_decision: Decision = "allow"
    allowed_tools: List[str] = field(default_factory=list)
    denied_tools: List[str] = field(default_factory=list)
    ask_tools: List[str] = field(default_factory=list)
    allowed_roots: List[str] = field(default_factory=list)
    agent_rules: Dict[str, AgentRuleSpec] = field(default_factory=dict)
    detector_settings: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
