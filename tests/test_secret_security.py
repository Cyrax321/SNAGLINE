"""Unit and regression tests for deterministic secret detection and redaction (issue #640)."""

from __future__ import annotations

import base64
import re

import pytest

from snagline.authorize import (
    Policy,
    PolicyError,
    ToolRequest,
    policy_from_dict,
)
from snagline.secret_security import (
    SecretFinding,
    SecretMatch,
    SecretSecurityConfig,
    as_authorize_rule,
    findings_at_or_above,
    inspect_text,
    inspect_tool_input,
    redact_mapping,
    redact_secrets,
    strongest_severity,
)

_MOCK_JWT_PAYLOAD = (
    base64.b64encode(b'{"alg":"HS256","typ":"JWT"}').decode().rstrip("=")
    + "."
    + base64.b64encode(b'{"sub":"1234567890"}').decode().rstrip("=")
    + ".dozG4B_p_kP_k_1234567890abcdef"
)

# ---------------------------------------------------------------------------
# Representative credential patterns
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("rule_id", "sample"),
    [
        ("aws_access_key", "export AWS_ACCESS_KEY_ID=AKIAIOSFODNN7EXAMPLE"),
        ("github_token", "token = 'ghp_" + "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'"),
        (
            "github_token",
            "github_pat_"
            + "11AAAAAAA00112233445566778899aabbccddeeffgghhiijjkkllmmnnooppqqrrssttuuvvwwxxyyzz012",
        ),
        (
            "slack_token",
            "SLACK_BOT_TOKEN=xoxb-mock-sample-bot-token-for-test-0123456789",
        ),
        (
            "openai_api_key",
            "api_key = 'sk-proj-1234567890abcdef1234567890abcdef'",
        ),
        (
            "anthropic_api_key",
            "client = Anthropic(api_key='sk-ant-api03-1234567890abcdef1234567890')",
        ),
        (
            "stripe_secret_key",
            "stripe_secret = 'sk_live_" + "1234567890abcdef12345678'",
        ),
        (
            "database_connection_uri",
            "DATABASE_URL=postgres://"
            + "dbuser:pAssW0rd123@db.internal:5432/production",
        ),
        ("google_api_key", "key: AIzaSyD-1234567890abcdefghijklmnopqrstuv"),
        (
            "private_key",
            "-----BEGIN RSA PRIVATE KEY-----\nMIIEowIBAAKCAQEA0...\n-----END RSA PRIVATE KEY-----",
        ),
        (
            "jwt_token",
            f"Authorization: Bearer {_MOCK_JWT_PAYLOAD}",
        ),
        (
            "uri_basic_auth",
            "git clone https://username:super_secret_pass123@github.com/org/repo.git",
        ),
        (
            "generic_credential_assignment",
            'DB_PASSWORD = "superSecretPassword123!"',
        ),
        (
            "generic_credential_assignment",
            "MY_API_KEY: 'my-super-secret-token-value'",
        ),
    ],
)
def test_detects_representative_credentials(rule_id: str, sample: str) -> None:
    findings = inspect_text(sample)
    assert any(f.rule_id == rule_id for f in findings)


def test_openai_pattern_does_not_claim_anthropic_key() -> None:
    sample = "client = Anthropic(api_key='sk-ant-api03-1234567890abcdef1234567890')"
    findings = inspect_text(sample)
    rule_ids = {f.rule_id for f in findings}
    assert "anthropic_api_key" in rule_ids
    assert "openai_api_key" not in rule_ids


# ---------------------------------------------------------------------------
# Zero secret leakage in findings or reasons
# ---------------------------------------------------------------------------


def test_zero_raw_secret_leakage_in_findings() -> None:
    secret = "AKIAIOSFODNN7EXAMPLE"
    text = f"The client key is {secret}."
    findings = inspect_text(text)
    assert len(findings) == 1
    f = findings[0]

    # Raw secret must NOT appear anywhere in the finding object's attributes or repr
    assert secret not in f.reason
    assert secret not in f.rule_id
    assert secret not in f.match_type
    assert secret not in f.secret_signature
    assert secret not in repr(f)
    assert len(f.secret_signature) == 64  # SHA-256 digest


# ---------------------------------------------------------------------------
# Benign strings and false positive controls
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "benign",
    [
        "hello world",
        "https://example.com/api/v1/users",
        "SELECT * FROM users WHERE active = true;",
        "const AKIA = 'short';",
        "git clone https://github.com/org/repo.git",
        "let x = 42;",
        "",
        "   \n\t  ",
    ],
)
def test_benign_content_yields_zero_findings(benign: str) -> None:
    assert inspect_text(benign) == ()


# ---------------------------------------------------------------------------
# Redaction utilities
# ---------------------------------------------------------------------------


def test_redact_secrets_default_mask() -> None:
    raw = "AWS key is AKIAIOSFODNN7EXAMPLE and slack is xoxb-mock-sample-bot-token-for-test-0123456789."
    redacted = redact_secrets(raw)
    assert "AKIAIOSFODNN7EXAMPLE" not in redacted
    assert "xoxb-mock-sample-bot-token-for-test-0123456789" not in redacted
    assert "[REDACTED:aws_access_key]" in redacted
    assert "[REDACTED:slack_token]" in redacted
    assert redacted.startswith(
        "AWS key is [REDACTED:aws_access_key] and slack is [REDACTED:slack_token]."
    )


def test_redact_secrets_custom_mask() -> None:
    raw = "key is AKIAIOSFODNN7EXAMPLE"
    redacted = redact_secrets(raw, mask="***{rule_id}***")
    assert redacted == "key is ***aws_access_key***"


def test_redact_mapping_recursive() -> None:
    payload = {
        "user": "alice",
        "env": {
            "AWS_KEY": "AKIAIOSFODNN7EXAMPLE",
            "RETRY": 3,
            "TOKENS": [
                "safe_item",
                "xoxb-mock-sample-bot-token-for-test-0123456789",
            ],
        },
        "meta": ("sk-proj-1234567890abcdef1234567890abcdef", 100),
    }
    redacted = redact_mapping(payload)
    assert redacted["user"] == "alice"
    assert redacted["env"]["AWS_KEY"] == "[REDACTED:aws_access_key]"
    assert redacted["env"]["RETRY"] == 3
    assert redacted["env"]["TOKENS"][0] == "safe_item"
    assert redacted["env"]["TOKENS"][1] == "[REDACTED:slack_token]"
    assert redacted["meta"][0] == "[REDACTED:openai_api_key]"
    assert redacted["meta"][1] == 100


# ---------------------------------------------------------------------------
# Tool argument scanning
# ---------------------------------------------------------------------------


def test_inspect_tool_input() -> None:
    tool_input = {
        "command": "curl -H 'Authorization: Bearer ghp_ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789' https://api.github.com",
        "nested": {"key": "AIzaSyD-1234567890abcdefghijklmnopqrstuv"},
        "count": 5,
    }
    findings = inspect_tool_input(tool_input)
    assert len(findings) == 2
    rule_ids = {f.rule_id for f in findings}
    assert "github_token" in rule_ids
    assert "google_api_key" in rule_ids


# ---------------------------------------------------------------------------
# Custom rules extension
# ---------------------------------------------------------------------------


def test_custom_rule_extension() -> None:
    def custom_secret_rule(text: str) -> list[SecretMatch]:
        pattern = re.compile(r"\bMYCORP-[0-9A-Z]{12}\b")
        hits: list[SecretMatch] = []
        for m in pattern.finditer(text):
            hits.append(
                SecretMatch(
                    rule_id="mycorp_token",
                    severity="critical",
                    start=m.start(),
                    end=m.end(),
                    raw_secret=m.group(0),
                    match_type="MyCorp Internal Token",
                )
            )
        return hits

    sample = "Connecting with MYCORP-ABCD1234EF56."
    findings = inspect_text(sample, extra_rules=(("mycorp_token", custom_secret_rule),))
    assert any(f.rule_id == "mycorp_token" for f in findings)
    assert any(f.severity == "critical" for f in findings)

    # Verify custom rule works in redaction
    redacted = redact_secrets(
        sample, extra_rules=(("mycorp_token", custom_secret_rule),)
    )
    assert redacted == "Connecting with [REDACTED:mycorp_token]."


# ---------------------------------------------------------------------------
# Config filtering, severity overrides, and helpers
# ---------------------------------------------------------------------------


def test_config_filtering_and_overrides() -> None:
    sample = "export AWS_ACCESS_KEY_ID=AKIAIOSFODNN7EXAMPLE"
    # Disabled rules
    cfg_disabled = SecretSecurityConfig(disabled_rules=frozenset({"aws_access_key"}))
    assert inspect_text(sample, config=cfg_disabled) == ()

    # Severity override
    cfg_override = SecretSecurityConfig(severity_overrides={"aws_access_key": "info"})
    findings = inspect_text(sample, config=cfg_override)
    assert len(findings) == 1
    assert findings[0].severity == "info"

    # Enabled rules filter
    cfg_enabled = SecretSecurityConfig(enabled_rules=frozenset({"slack_token"}))
    assert inspect_text(sample, config=cfg_enabled) == ()


def test_strongest_severity_and_threshold() -> None:
    f_info = SecretFinding("r1", "info", "info reason", "sig1", "type1")
    f_crit = SecretFinding("r2", "critical", "crit reason", "sig2", "type2")
    assert strongest_severity((f_info, f_crit)) == "critical"
    assert strongest_severity(()) is None

    above_warning = findings_at_or_above((f_info, f_crit), "warning")
    assert above_warning == (f_crit,)


# ---------------------------------------------------------------------------
# Authorization engine integration
# ---------------------------------------------------------------------------


def test_as_authorize_rule_scans_tool_request() -> None:
    rule = as_authorize_rule()
    # Tool request with secret in command/resource
    req = ToolRequest(
        tool_name="Bash",
        action="execute",
        resource="aws s3 cp s3://bucket/data.txt --access-key AKIAIOSFODNN7EXAMPLE",
    )
    verdict = rule(req)
    assert verdict is not None
    assert verdict.decision == "deny"
    assert "secret credential exposure" in verdict.reason
    assert "AKIAIOSFODNN7EXAMPLE" not in verdict.reason

    # Tool request with secret in tool_input using production from_tool_input
    req_input = ToolRequest.from_tool_input(
        "Bash",
        {"command": "echo xoxb-mock-sample-bot-token-for-test-0123456789"},
    )
    verdict_input = rule(req_input)
    assert verdict_input is not None
    assert verdict_input.decision == "deny"

    # Clean tool request
    clean_req = ToolRequest.from_tool_input(
        "Bash",
        {"command": "echo safe"},
    )
    assert rule(clean_req) is None


def test_as_authorize_rule_reports_highest_severity() -> None:
    # Warning finding in resource (jwt), critical in tool_input (aws)
    rule = as_authorize_rule()
    req = ToolRequest(
        tool_name="Bash",
        action="execute",
        resource=f"Authorization: Bearer {_MOCK_JWT_PAYLOAD}",
        tool_input={"key": "AKIAIOSFODNN7EXAMPLE"},
    )
    verdict = rule(req)
    assert verdict is not None
    assert "severity=critical" in verdict.reason


def test_as_authorize_rule_rejects_truncated_resource() -> None:
    rule = as_authorize_rule()
    req_trunc = ToolRequest(
        tool_name="Bash",
        action="execute",
        resource="echo secret...<truncated>",
    )
    verdict = rule(req_trunc)
    assert verdict is not None
    assert verdict.decision == "deny"
    assert "truncated" in verdict.reason


def test_redact_secrets_literal_braces_in_mask() -> None:
    sample = "key is AKIAIOSFODNN7EXAMPLE"
    redacted = redact_secrets(sample, mask="{CUSTOM_TAG}:{rule_id}")
    assert redacted == "key is {CUSTOM_TAG}:aws_access_key"


def test_redact_mapping_nested_containers() -> None:
    data = {
        "user": "alice",
        "nested_set": {"AKIAIOSFODNN7EXAMPLE", "benign"},
        "nested_frozenset": frozenset({"AKIAIOSFODNN7EXAMPLE"}),
        "nested_tuple": ("safe", ["AKIAIOSFODNN7EXAMPLE"]),
    }
    redacted = redact_mapping(data)
    assert redacted["user"] == "alice"
    assert "[REDACTED:aws_access_key]" in redacted["nested_set"]
    assert "[REDACTED:aws_access_key]" in redacted["nested_frozenset"]
    assert redacted["nested_tuple"][1][0] == "[REDACTED:aws_access_key]"


def test_as_authorize_rule_in_policy() -> None:
    rule = as_authorize_rule(SecretSecurityConfig(decision="ask"))
    policy = Policy(rules=(rule,))

    res_dirty = policy.decide(
        ToolRequest(
            tool_name="Bash",
            action="execute",
            resource="echo AIzaSyD-1234567890abcdefghijklmnopqrstuv",
        )
    )
    assert res_dirty.decision == "ask"

    res_clean = policy.decide(
        ToolRequest(tool_name="Bash", action="execute", resource="echo hello")
    )
    assert res_clean.decision == "allow"


# ---------------------------------------------------------------------------
# Declarative policy parsing
# ---------------------------------------------------------------------------


def test_declarative_policy_from_dict() -> None:
    raw = {
        "rules": [
            {
                "type": "secret_exposure",
                "decision": "ask",
                "min_severity": "critical",
                "disabled_rules": ["generic_credential_assignment"],
                "enabled_rules": ["aws_access_key", "github_token"],
                "severity_overrides": {"github_token": "critical"},
            }
        ]
    }
    pol = policy_from_dict(raw)
    assert len(pol.rules) == 1

    verdict_clean = pol.decide(
        ToolRequest(tool_name="Read", action="read", resource="src/index.ts")
    )
    assert verdict_clean.decision == "allow"

    verdict_secret = pol.decide(
        ToolRequest(
            tool_name="Read",
            action="read",
            resource="AKIAIOSFODNN7EXAMPLE",
        )
    )
    assert verdict_secret.decision == "ask"


def test_declarative_policy_schema_validation() -> None:
    with pytest.raises(PolicyError, match="unknown rule keys"):
        policy_from_dict({"rules": [{"type": "secret_exposure", "invalid_key": True}]})

    with pytest.raises(PolicyError, match="must be 'ask' or 'deny'"):
        policy_from_dict({"rules": [{"type": "secret_exposure", "decision": "allow"}]})

    with pytest.raises(PolicyError, match="disabled_rules must be a list"):
        policy_from_dict(
            {"rules": [{"type": "secret_exposure", "disabled_rules": 123}]}
        )

    with pytest.raises(PolicyError, match="enabled_rules has unknown ids"):
        policy_from_dict(
            {"rules": [{"type": "secret_exposure", "enabled_rules": ["bogus_rule"]}]}
        )

    with pytest.raises(PolicyError, match="is not a severity"):
        policy_from_dict(
            {
                "rules": [
                    {
                        "type": "secret_exposure",
                        "severity_overrides": {"aws_access_key": "bad_sev"},
                    }
                ]
            }
        )


# ---------------------------------------------------------------------------
# Monitor independence and documentation criteria
# ---------------------------------------------------------------------------


def test_secret_detection_independent_of_monitor() -> None:
    from snagline import Monitor, StepEvent

    m = Monitor.default()
    event = StepEvent(
        step_id="step_sec_1",
        episode_id="ep_sec_1",
        timestamp=100.0,
        action_type="tool_call",
        action_signature="sig_sec",
        tool_name="Bash",
    )
    m.ingest(event)
    assert m.metrics()["events_ingested"] == 1


def test_limitations_clearly_documented() -> None:
    import snagline.secret_security as sec

    doc = sec.__doc__ or ""
    assert "LIMITATIONS" in doc
    assert "Encoded secrets" in doc
    assert "Split" in doc or "constructed" in doc
    assert "Unseen" in doc or "proprietary" in doc


def test_edge_cases_and_coverage_branches() -> None:
    # Empty inputs
    assert inspect_text("") == ()
    assert inspect_text(None) == ()  # type: ignore[arg-type]
    assert inspect_tool_input("not_a_dict") == ()  # type: ignore[arg-type]
    assert redact_secrets("") == ""
    assert redact_secrets(None) is None  # type: ignore[arg-type]
    assert redact_mapping("not_a_dict") == {}  # type: ignore[arg-type]

    # Tool input with set
    findings = inspect_tool_input({"tokens": {"AKIAIOSFODNN7EXAMPLE"}})
    assert len(findings) == 1

    # Overlapping matches
    def r1(text: str) -> list[SecretMatch]:
        idx = text.find("SECRET_ONE")
        if idx != -1:
            return [SecretMatch("r1", "critical", idx, idx + 10, "SECRET_ONE", "R1")]
        return []

    def r2(text: str) -> list[SecretMatch]:
        idx = text.find("ONE_MORE")
        if idx != -1:
            return [SecretMatch("r2", "critical", idx, idx + 8, "ONE_MORE", "R2")]
        return []

    raw = "prefix SECRET_ONE_MORE suffix"
    redacted = redact_secrets(raw, extra_rules=(("r1", r1), ("r2", r2)))
    assert "SECRET_ONE" not in redacted
    assert "ONE_MORE" not in redacted
    assert "[REDACTED:r1]" in redacted

    # Extra rule filtering
    def dummy_rule(text: str) -> list[SecretMatch]:
        return [SecretMatch("dummy", "warning", 0, 5, text[:5], "Dummy")]

    cfg_disabled = SecretSecurityConfig(disabled_rules=frozenset({"dummy"}))
    assert (
        inspect_text(
            "hello world", config=cfg_disabled, extra_rules=(("dummy", dummy_rule),)
        )
        == ()
    )

    cfg_enabled = SecretSecurityConfig(enabled_rules=frozenset({"aws_access_key"}))
    assert (
        inspect_text(
            "hello world", config=cfg_enabled, extra_rules=(("dummy", dummy_rule),)
        )
        == ()
    )

    # as_authorize_rule with no resource and no tool input
    rule = as_authorize_rule()
    assert (
        rule(ToolRequest(tool_name="Read", action="read", resource=None, tool_input={}))
        is None
    )
