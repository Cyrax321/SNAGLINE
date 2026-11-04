"""Security policy schema and validation layer.

Provides strict, deterministic schema definitions for security policies, validating
fields, types, tool restrictions, and detector parameters while rejecting unknown keys.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from snagline.authorize import Decision


class PolicyValidationError(ValueError):
    """Raised when a security policy configuration fails schema validation."""

    pass


@dataclass
class AgentRuleSpec:
    """Per-agent execution constraints."""

    allowed_tools: list[str] = field(default_factory=list)
    denied_tools: list[str] = field(default_factory=list)
    ask_tools: list[str] = field(default_factory=list)


@dataclass
class SecurityPolicySchema:
    """Validated security policy specification."""

    version: str = "1.0"
    name: str = "default_security_policy"
    profile: str | None = None
    default_decision: Decision = "allow"
    allowed_tools: list[str] = field(default_factory=list)
    denied_tools: list[str] = field(default_factory=list)
    ask_tools: list[str] = field(default_factory=list)
    allowed_roots: list[str] = field(default_factory=list)
    agent_rules: dict[str, AgentRuleSpec] = field(default_factory=dict)
    detector_settings: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


KNOWN_POLICY_FIELDS: set[str] = {
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

KNOWN_AGENT_FIELDS: set[str] = {
    "allowed_tools",
    "denied_tools",
    "ask_tools",
}


def validate_policy_dict(raw: dict[str, Any]) -> SecurityPolicySchema:
    """Strictly validate a raw policy dictionary against the security schema."""
    if not isinstance(raw, dict):
        raise PolicyValidationError(
            f"Policy must be a mapping; got {type(raw).__name__}"
        )

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
        raise PolicyValidationError(
            f"Invalid profile {profile!r}; expected 'fail_closed', 'permissive', or 'strict'"
        )

    default_decision = raw.get("default_decision", "allow")
    if default_decision not in ("allow", "ask", "deny"):
        raise PolicyValidationError(
            f"Invalid default_decision {default_decision!r}; expected 'allow', 'ask', or 'deny'"
        )

    # 4. List fields validation
    def _validate_str_list(field_name: str) -> list[str]:
        val = raw.get(field_name, [])
        if not isinstance(val, list) or not all(isinstance(x, str) for x in val):
            raise PolicyValidationError(
                f"Field {field_name!r} must be a list of strings"
            )
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
        raise PolicyValidationError(
            f"Conflicting tool rules: {sorted(overlap)} in both allowed and denied"
        )
    if overlap := allow_set.intersection(ask_set):
        raise PolicyValidationError(
            f"Conflicting tool rules: {sorted(overlap)} in both allowed and ask"
        )
    if overlap := deny_set.intersection(ask_set):
        raise PolicyValidationError(
            f"Conflicting tool rules: {sorted(overlap)} in both denied and ask"
        )

    # 6. Agent rules validation
    agent_rules_raw = raw.get("agent_rules", {})
    if not isinstance(agent_rules_raw, dict):
        raise PolicyValidationError(
            "'agent_rules' must be a mapping of agent_id to rules"
        )

    agent_rules: dict[str, AgentRuleSpec] = {}
    for agent_id, spec_dict in agent_rules_raw.items():
        if not isinstance(spec_dict, dict):
            raise PolicyValidationError(
                f"Rule for agent {agent_id!r} must be a mapping"
            )
        for ak in spec_dict:
            if ak not in KNOWN_AGENT_FIELDS:
                raise PolicyValidationError(
                    f"Unknown field in agent {agent_id!r} rule: {ak!r}"
                )

        a_allowed = spec_dict.get("allowed_tools", [])
        a_denied = spec_dict.get("denied_tools", [])
        a_ask = spec_dict.get("ask_tools", [])

        if not all(
            isinstance(x, list) and all(isinstance(s, str) for s in x)
            for x in (a_allowed, a_denied, a_ask)
        ):
            raise PolicyValidationError(
                f"Tool lists for agent {agent_id!r} must be lists of strings"
            )

        if overlap := set(a_allowed).intersection(set(a_denied)):
            raise PolicyValidationError(
                f"Conflicting tools for agent {agent_id!r}: {sorted(overlap)}"
            )

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


def load_security_policy(source: str | Any) -> SecurityPolicySchema:
    """Load and strictly validate security policy from JSON/YAML or mapping."""
    import json
    from pathlib import Path

    raw: dict[str, Any]
    if isinstance(source, dict):
        raw = source
    elif isinstance(source, (str, Path)):
        p = Path(source)
        if p.exists() and p.is_file():
            content = p.read_text(encoding="utf-8")
        else:
            content = str(source)

        content = content.strip()
        if content.startswith("{"):
            try:
                raw = json.loads(content)
            except Exception as err:
                raise PolicyValidationError(
                    f"Invalid JSON in policy source: {err}"
                ) from err
        else:
            # Attempt yaml parsing if pyyaml is installed, otherwise lightweight key-value parser
            try:
                import yaml  # type: ignore

                raw = yaml.safe_load(content)
            except ImportError:
                # Basic key-value parser for simple YAML specs
                raw = {}
                for line in content.splitlines():
                    line = line.strip()
                    if line and not line.startswith("#") and ":" in line:
                        k, v = line.split(":", 1)
                        raw[k.strip()] = v.strip()
            except Exception as err:
                raise PolicyValidationError(
                    f"Invalid YAML in policy source: {err}"
                ) from err
    else:
        raise PolicyValidationError(
            f"Unsupported policy source type: {type(source).__name__}"
        )

    return validate_policy_dict(raw)


def compile_policy(schema: SecurityPolicySchema) -> Any:
    """Compile a validated SecurityPolicySchema into an executable Authorization Policy."""
    from snagline.authorize import (
        Policy,
        Rule,
        RuleVerdict,
        allow_tools,
        ask_tools,
        deny_tools,
    )

    rules: list[Rule] = []
    # 1. Per-agent rules take highest precedence
    if schema.agent_rules:
        for agent_id, a_spec in schema.agent_rules.items():
            if a_spec.denied_tools:
                d_names = frozenset(a_spec.denied_tools)

                def _make_agent_deny(aid: str, names: frozenset[str]) -> Rule:
                    return lambda req: RuleVerdict(
                        rule_id=f"agent_{aid}_deny",
                        decision="deny"
                        if req.agent_id == aid and req.tool_name in names
                        else None,
                        reason=f"Tool {req.tool_name!r} denied for agent {aid!r}",
                    )

                rules.append(_make_agent_deny(agent_id, d_names))
            if a_spec.ask_tools:
                ask_names = frozenset(a_spec.ask_tools)

                def _make_agent_ask(aid: str, names: frozenset[str]) -> Rule:
                    return lambda req: RuleVerdict(
                        rule_id=f"agent_{aid}_ask",
                        decision="ask"
                        if req.agent_id == aid and req.tool_name in names
                        else None,
                        reason=f"Tool {req.tool_name!r} requires approval for agent {aid!r}",
                    )

                rules.append(_make_agent_ask(agent_id, ask_names))
            if a_spec.allowed_tools:
                a_names = frozenset(a_spec.allowed_tools)

                def _make_agent_allow(aid: str, names: frozenset[str]) -> Rule:
                    return lambda req: RuleVerdict(
                        rule_id=f"agent_{aid}_allow",
                        decision="allow"
                        if req.agent_id == aid and req.tool_name in names
                        else None,
                        reason=f"Tool {req.tool_name!r} explicitly allowed for agent {aid!r}",
                    )

                rules.append(_make_agent_allow(agent_id, a_names))

    # 2. Filesystem confinement. `allowed_roots` is a validated schema field
    #    (the canonical example in docs/SECURITY_POLICY.md lists it), and
    #    validation accepting a field the compiler silently drops is a
    #    downgrade with no diagnostic -- so route it through the same
    #    path-security rule authorize's own policy loader builds instead of
    #    leaving the boundary unenforced.
    if schema.allowed_roots:
        from snagline.path_security import PathSecurityConfig, as_authorize_rule

        rules.append(
            as_authorize_rule(
                PathSecurityConfig(allowed_roots=tuple(schema.allowed_roots))
            )
        )

    # 3. Global explicit rules
    if schema.denied_tools:
        rules.append(deny_tools(*schema.denied_tools))
    if schema.ask_tools:
        rules.append(ask_tools(*schema.ask_tools))
    if schema.allowed_tools:
        rules.append(allow_tools(*schema.allowed_tools))

    return Policy(
        rules=tuple(rules),
        default_decision=schema.default_decision,
        profile=schema.profile,
        policy_id=schema.name,
    )
