import json
from pathlib import Path

import pytest

from snagline.authorize import ToolRequest
from snagline.security_policy import (
    PolicyValidationError,
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


def test_reject_conflicting_agent_tool_rules():
    """Issue #741: the per-agent conflict check must reject all three pairs, like the global one."""
    with pytest.raises(PolicyValidationError, match=r"in both denied and ask"):
        validate_policy_dict(
            {
                "version": "1.0",
                "agent_rules": {
                    "worker": {"denied_tools": ["Bash"], "ask_tools": ["Bash"]}
                },
            }
        )

    with pytest.raises(PolicyValidationError, match=r"in both allowed and ask"):
        validate_policy_dict(
            {
                "version": "1.0",
                "agent_rules": {
                    "worker": {
                        "allowed_tools": ["Bash"],
                        "ask_tools": ["Bash"],
                    }
                },
            }
        )

    with pytest.raises(PolicyValidationError, match=r"in both allowed and denied"):
        validate_policy_dict(
            {
                "version": "1.0",
                "agent_rules": {
                    "worker": {
                        "allowed_tools": ["Bash"],
                        "denied_tools": ["Bash"],
                    }
                },
            }
        )

    # The message names the agent and both lists
    with pytest.raises(PolicyValidationError, match=r"'worker'.*'Bash'.*allowed.*ask"):
        validate_policy_dict(
            {
                "version": "1.0",
                "agent_rules": {
                    "worker": {
                        "allowed_tools": ["Bash"],
                        "ask_tools": ["Bash"],
                    }
                },
            }
        )


def test_agent_list_conflicts_with_global_list_still_valid():
    """Issue #741: the new checks are scoped per agent.

    A tool shared between an agent's list and a *global* list is not a
    conflict -- the agent rule intentionally overrides the global one.
    """
    raw = {
        "version": "1.0",
        "allowed_tools": ["Read"],
        "denied_tools": ["Bash"],
        "ask_tools": ["Write"],
        "agent_rules": {
            "researcher": {
                "allowed_tools": ["Bash"],
                "ask_tools": ["Read"],
                "denied_tools": ["Write"],
            }
        },
    }
    schema = validate_policy_dict(raw)
    assert schema.agent_rules["researcher"].allowed_tools == ["Bash"]
