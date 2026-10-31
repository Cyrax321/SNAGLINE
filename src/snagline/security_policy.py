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


KNOWN_POLICY_FIELDS: Set[str] = {
    "version",
    "name",
    "profile",
    "default_decision",
    "allowed_tools",
    "denied_tools",
    "ask_tools",
    "allowed_roots",
    "agent_rules",
    "detector_settings",
}

KNOWN_AGENT_FIELDS: Set[str] = {
    "allowed_tools",
    "denied_tools",
    "ask_tools",
}


def validate_policy_dict(raw: Dict[str, Any]) -> SecurityPolicySchema:
    """Strictly validate a raw policy dictionary against the security schema."""
    if not isinstance(raw, dict):
        raise PolicyValidationError(f"Policy must be a mapping; got {type(raw).__name__}")

    # 1. Unknown fields check
    for k in raw:
        if k not in KNOWN_POLICY_FIELDS:
            raise PolicyValidationError(f"Unknown field in security policy: {k!r}")

    # 2. Version validation
    version = str(raw.get("version", "1.0"))
    if not version:
        raise PolicyValidationError("Policy 'version' cannot be empty")

    name = str(raw.get("name", "default_security_policy"))

    # 3. Profile & Default decision validation
    profile = raw.get("profile")
    if profile is not None and profile not in ("fail_closed", "permissive", "strict"):
        raise PolicyValidationError(f"Invalid profile {profile!r}; expected 'fail_closed', 'permissive', or 'strict'")

    default_decision = raw.get("default_decision", "allow")
    if default_decision not in ("allow", "ask", "deny"):
        raise PolicyValidationError(f"Invalid default_decision {default_decision!r}; expected 'allow', 'ask', or 'deny'")

    # 4. List fields validation
    def _validate_str_list(field_name: str) -> List[str]:
        val = raw.get(field_name, [])
        if not isinstance(val, list) or not all(isinstance(x, str) for x in val):
            raise PolicyValidationError(f"Field {field_name!r} must be a list of strings")
        return list(val)

    allowed_tools = _validate_str_list("allowed_tools")
    denied_tools = _validate_str_list("denied_tools")
    ask_tools = _validate_str_list("ask_tools")
    allowed_roots = _validate_str_list("allowed_roots")

    # 5. Conflict checks
    allow_set = set(allowed_tools)
    deny_set = set(denied_tools)
    ask_set = set(ask_tools)

    if overlap := allow_set.intersection(deny_set):
        raise PolicyValidationError(f"Conflicting tool rules: {sorted(overlap)} in both allowed and denied")
    if overlap := allow_set.intersection(ask_set):
        raise PolicyValidationError(f"Conflicting tool rules: {sorted(overlap)} in both allowed and ask")
    if overlap := deny_set.intersection(ask_set):
        raise PolicyValidationError(f"Conflicting tool rules: {sorted(overlap)} in both denied and ask")

    # 6. Agent rules validation
    agent_rules_raw = raw.get("agent_rules", {})
    if not isinstance(agent_rules_raw, dict):
        raise PolicyValidationError("'agent_rules' must be a mapping of agent_id to rules")

    agent_rules: Dict[str, AgentRuleSpec] = {}
    for agent_id, spec_dict in agent_rules_raw.items():
        if not isinstance(spec_dict, dict):
            raise PolicyValidationError(f"Rule for agent {agent_id!r} must be a mapping")
        for ak in spec_dict:
            if ak not in KNOWN_AGENT_FIELDS:
                raise PolicyValidationError(f"Unknown field in agent {agent_id!r} rule: {ak!r}")

        a_allowed = spec_dict.get("allowed_tools", [])
        a_denied = spec_dict.get("denied_tools", [])
        a_ask = spec_dict.get("ask_tools", [])

        if not all(isinstance(x, list) and all(isinstance(s, str) for s in x) for x in (a_allowed, a_denied, a_ask)):
            raise PolicyValidationError(f"Tool lists for agent {agent_id!r} must be lists of strings")

        if overlap := set(a_allowed).intersection(set(a_denied)):
            raise PolicyValidationError(f"Conflicting tools for agent {agent_id!r}: {sorted(overlap)}")

        agent_rules[agent_id] = AgentRuleSpec(
            allowed_tools=list(a_allowed),
            denied_tools=list(a_denied),
            ask_tools=list(a_ask),
        )

    # 7. Detector settings validation
    detector_settings = raw.get("detector_settings", {})
    if not isinstance(detector_settings, dict):
        raise PolicyValidationError("'detector_settings' must be a mapping")

    return SecurityPolicySchema(
        version=version,
        name=name,
        profile=profile,
        default_decision=default_decision,
        allowed_tools=allowed_tools,
        denied_tools=denied_tools,
        ask_tools=ask_tools,
        allowed_roots=allowed_roots,
        agent_rules=agent_rules,
        detector_settings=dict(detector_settings),
    )

