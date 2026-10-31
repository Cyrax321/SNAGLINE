import json
from pathlib import Path
import pytest

from snagline.authorize import ToolRequest
from snagline.security_policy import (
    PolicyValidationError,
    SecurityPolicySchema,
    compile_policy,
    load_security_policy,
    validate_policy_dict,
)


def test_valid_security_policy_schema():
    raw = {
        "version": "1.0",
        "name": "prod_guard",
        "profile": "fail_closed",
        "default_decision": "deny",
        "allowed_tools": ["Read", "Grep"],
        "denied_tools": ["Bash", "Delete"],
        "ask_tools": ["Write"],
        "allowed_roots": ["/var/data"],
        "agent_rules": {
            "researcher": {
                "allowed_tools": ["Read"],
                "denied_tools": ["Bash"],
                "ask_tools": [],
            }
        },
        "detector_settings": {"max_samples": 500},
    }
    schema = validate_policy_dict(raw)
    assert schema.name == "prod_guard"
    assert schema.profile == "fail_closed"
    assert schema.default_decision == "deny"
    assert schema.allowed_tools == ["Read", "Grep"]
    assert "researcher" in schema.agent_rules


def test_reject_unknown_policy_fields():
    raw = {
        "version": "1.0",
        "name": "test",
        "unknown_extra_field": "danger",
    }
    with pytest.raises(PolicyValidationError, match="Unknown field in security policy"):
        validate_policy_dict(raw)


def test_reject_invalid_field_types():
    with pytest.raises(PolicyValidationError, match="must be a list of strings"):
        validate_policy_dict({"version": "1.0", "allowed_tools": "Read"})

    with pytest.raises(PolicyValidationError, match="Invalid default_decision"):
        validate_policy_dict({"version": "1.0", "default_decision": "invalid_choice"})

    with pytest.raises(PolicyValidationError, match="Invalid profile"):
        validate_policy_dict({"version": "1.0", "profile": "unsupported_profile"})


def test_reject_conflicting_tool_rules():
    raw = {
        "version": "1.0",
        "allowed_tools": ["Bash", "Read"],
        "denied_tools": ["Bash"],
    }
    with pytest.raises(PolicyValidationError, match="Conflicting tool rules"):
        validate_policy_dict(raw)


def test_compile_policy_execution():
    raw = {
        "version": "1.0",
        "default_decision": "ask",
        "allowed_tools": ["Read"],
        "denied_tools": ["Bash"],
        "ask_tools": ["Deploy"],
        "agent_rules": {
            "untrusted_agent": {
                "denied_tools": ["Read"],
                "allowed_tools": [],
                "ask_tools": [],
            }
        },
    }
    schema = validate_policy_dict(raw)
    policy = compile_policy(schema)

    # General requests
    assert policy.decide(ToolRequest("Read", "read")).decision == "allow"
    assert policy.decide(ToolRequest("Bash", "execute")).decision == "deny"
    assert policy.decide(ToolRequest("Deploy", "push")).decision == "ask"
    assert policy.decide(ToolRequest("Custom", "other")).decision == "ask"

    # Per-agent override: untrusted_agent is denied Read
    req_agent = ToolRequest("Read", "read", agent_id="untrusted_agent")
    assert policy.decide(req_agent).decision == "deny"


def test_load_security_policy_json(tmp_path: Path):
    data = {
        "version": "1.0",
        "name": "json_policy",
        "allowed_tools": ["Grep"],
        "default_decision": "allow",
    }
    file_path = tmp_path / "policy.json"
    file_path.write_text(json.dumps(data), encoding="utf-8")

    schema = load_security_policy(file_path)
    assert schema.name == "json_policy"
    assert schema.allowed_tools == ["Grep"]
