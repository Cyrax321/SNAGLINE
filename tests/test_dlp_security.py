"""Unit and integration tests for Data Loss Prevention (issue #641)."""

from __future__ import annotations

import pytest

from snagline.authorize import (
    PolicyError,
    ToolRequest,
    policy_from_dict,
)
from snagline.dlp_security import (
    DlpFinding,
    DlpMatch,
    DlpSecurityConfig,
    _luhn_checksum_valid,
    as_authorize_rule,
    findings_at_or_above,
    inspect_payload,
    mask_credit_card,
    mask_email,
    mask_phone,
    mask_ssn,
    redact_payload,
    redact_text,
    scan_text,
    strongest_severity,
)

# ---------------------------------------------------------------------------
# Checksum and Masking Tests
# ---------------------------------------------------------------------------


def test_luhn_checksum_valid() -> None:
    # Standard valid test card numbers (Luhn compliant)
    assert _luhn_checksum_valid("4111111111111111")
    assert _luhn_checksum_valid("4111 1111 1111 1111")
    assert _luhn_checksum_valid("4111-1111-1111-1111")
    assert _luhn_checksum_valid("5500000000000004")
    assert _luhn_checksum_valid("378282246310005")

    # Invalid checksum
    assert not _luhn_checksum_valid("4111111111111112")
    # Too short or too long
    assert not _luhn_checksum_valid("12345")
    assert not _luhn_checksum_valid("123456789012345678901234")


def test_masking_helpers() -> None:
    assert mask_ssn("123-45-6789") == "***-**-6789"
    assert mask_ssn("invalid") == "***-**-****"

    assert mask_credit_card("4111-1111-1111-1111") == "4111-****-****-1111"
    assert mask_credit_card("123") == "****-****-****-****"

    assert mask_email("alice@example.com") == "a***@example.com"
    assert mask_email("invalid") == "***@***"

    assert mask_phone("555-123-4567") == "***-***-4567"
    assert mask_phone("123") == "***-***-****"


# ---------------------------------------------------------------------------
# Pattern Detection Tests
# ---------------------------------------------------------------------------


def test_detect_ssn() -> None:
    # Valid SSNs
    hits = scan_text("User SSN is 123-45-6789 in records.")
    ssn_hits = [h for h in hits if h.rule_id == "us_ssn"]
    assert len(ssn_hits) == 1
    assert ssn_hits[0].raw_value == "123-45-6789"
    assert ssn_hits[0].redacted_value == "***-**-6789"
    assert ssn_hits[0].severity == "critical"

    # Invalid area numbers (000, 666, 900-999)
    assert not any(h.rule_id == "us_ssn" for h in scan_text("000-12-3456"))
    assert not any(h.rule_id == "us_ssn" for h in scan_text("666-12-3456"))
    assert not any(h.rule_id == "us_ssn" for h in scan_text("900-12-3456"))
    # Invalid group (00)
    assert not any(h.rule_id == "us_ssn" for h in scan_text("123-00-3456"))
    # Invalid serial (0000)
    assert not any(h.rule_id == "us_ssn" for h in scan_text("123-45-0000"))


def test_detect_credit_card() -> None:
    # Valid Visa with hyphens
    hits = scan_text("Card number: 4111-1111-1111-1111 exp 12/28")
    cc_hits = [h for h in hits if h.rule_id == "credit_card"]
    assert len(cc_hits) == 1
    assert cc_hits[0].raw_value == "4111-1111-1111-1111"
    assert cc_hits[0].redacted_value == "4111-****-****-1111"

    # Arbitrary 16 digit number failing Luhn
    hits_fail = scan_text("Transaction ID: 1234567890123456")
    assert not any(h.rule_id == "credit_card" for h in hits_fail)


def test_detect_email() -> None:
    hits = scan_text("Contact administrator at security-alerts@snagline.internal.")
    email_hits = [h for h in hits if h.rule_id == "email_address"]
    assert len(email_hits) == 1
    assert email_hits[0].raw_value == "security-alerts@snagline.internal"
    assert email_hits[0].redacted_value == "s***@snagline.internal"
    assert email_hits[0].severity == "warning"


def test_detect_phone() -> None:
    hits = scan_text("Customer phone: (555) 123-4567 today.")
    phone_hits = [h for h in hits if h.rule_id == "phone_number"]
    assert len(phone_hits) == 1
    assert phone_hits[0].raw_value == "(555) 123-4567"
    assert phone_hits[0].redacted_value == "***-***-4567"


# ---------------------------------------------------------------------------
# Structured Payload Inspection Tests
# ---------------------------------------------------------------------------


def test_inspect_payload_nested_structure() -> None:
    payload = {
        "destination": "https://api.external.com/v1/customer",
        "records": [
            {
                "id": "cust-001",
                "email": "customer@example.com",
                "ssn": "123-45-6789",
            },
            {
                "id": "cust-002",
                "payment": {
                    "card": "4111-1111-1111-1111",
                },
            },
        ],
    }

    findings = inspect_payload(payload)
    assert len(findings) == 3

    rule_by_path = {f.path: f.rule_id for f in findings}
    assert rule_by_path["records[0].email"] == "email_address"
    assert rule_by_path["records[0].ssn"] == "us_ssn"
    assert rule_by_path["records[1].payment.card"] == "credit_card"

    # Value signatures must be unique SHA-256 digests
    assert all(len(f.value_signature) == 64 for f in findings)
    # Raw values must not appear on findings
    for f in findings:
        assert "123-45-6789" not in f.reason
        assert "4111-1111-1111-1111" not in f.reason
        assert "customer@example.com" not in f.reason


def test_inspect_payload_benign() -> None:
    payload = {
        "action": "query_inventory",
        "sku": "SKU-998877",
        "count": 42,
        "tags": ["hardware", "storage"],
    }
    findings = inspect_payload(payload)
    assert findings == ()


# ---------------------------------------------------------------------------
# Redaction Utility Tests
# ---------------------------------------------------------------------------


def test_redact_text() -> None:
    text = "Send email to alice@example.com and verify SSN 123-45-6789."
    redacted = redact_text(text)
    assert "alice@example.com" not in redacted
    assert "123-45-6789" not in redacted
    assert "[REDACTED:email_address]" in redacted
    assert "[REDACTED:us_ssn]" in redacted

    # Custom mask template
    custom = redact_text(text, mask="<MASKED:{rule_id}>")
    assert "<MASKED:email_address>" in custom
    assert "<MASKED:us_ssn>" in custom


def test_redact_payload_recursive() -> None:
    data = {
        "user": "alice",
        "contacts": ["bob@example.com", "555-123-4567"],
        "nested": {
            "card": "4111-1111-1111-1111",
        },
    }
    redacted = redact_payload(data)
    assert redacted["user"] == "alice"
    assert redacted["contacts"][0] == "[REDACTED:email_address]"
    assert redacted["contacts"][1] == "[REDACTED:phone_number]"
    assert redacted["nested"]["card"] == "[REDACTED:credit_card]"


# ---------------------------------------------------------------------------
# Severity Filters and Config Tests
# ---------------------------------------------------------------------------


def test_severity_filtering() -> None:
    findings = (
        DlpFinding("phone_number", "info", "phone found", "sig1", "***"),
        DlpFinding("email_address", "warning", "email found", "sig2", "***"),
        DlpFinding("us_ssn", "critical", "ssn found", "sig3", "***"),
    )

    assert len(findings_at_or_above(findings, "info")) == 3
    assert len(findings_at_or_above(findings, "warning")) == 2
    assert len(findings_at_or_above(findings, "critical")) == 1
    assert strongest_severity(findings) == "critical"
    assert strongest_severity(()) is None


def test_config_enabled_disabled_rules() -> None:
    text = "SSN 123-45-6789 and email bob@example.com"

    # Only ssn enabled
    cfg_only_ssn = DlpSecurityConfig(enabled_rules=frozenset({"us_ssn"}))
    hits1 = scan_text(text, cfg_only_ssn)
    assert len(hits1) == 1
    assert hits1[0].rule_id == "us_ssn"

    # Disable ssn
    cfg_no_ssn = DlpSecurityConfig(disabled_rules=frozenset({"us_ssn"}))
    hits2 = scan_text(text, cfg_no_ssn)
    assert len(hits2) == 1
    assert hits2[0].rule_id == "email_address"

    # Override severity
    cfg_override = DlpSecurityConfig(severity_overrides={"email_address": "critical"})
    hits3 = scan_text(text, cfg_override)
    email_hit = next(h for h in hits3 if h.rule_id == "email_address")
    assert email_hit.severity == "critical"


def test_secrets_integration_toggle() -> None:
    text = "Credential is AKIAIOSFODNN7EXAMPLE and email alice@example.com"

    cfg_with_sec = DlpSecurityConfig(include_secrets=True)
    hits_with = scan_text(text, cfg_with_sec)
    assert any("secret" in h.rule_id for h in hits_with)

    cfg_no_sec = DlpSecurityConfig(include_secrets=False)
    hits_without = scan_text(text, cfg_no_sec)
    assert not any("secret" in h.rule_id for h in hits_without)


# ---------------------------------------------------------------------------
# Authorization Rule Integration Tests
# ---------------------------------------------------------------------------


def test_as_authorize_rule_egress_action() -> None:
    rule = as_authorize_rule()

    # Outbound tool request with SSN in tool_input
    req_egress = ToolRequest(
        tool_name="HttpEgress",
        action="network",
        resource="https://external.api/v1",
        tool_input={"customer_ssn": "123-45-6789"},
    )
    verdict = rule(req_egress)
    assert verdict is not None
    assert verdict.rule_id == "dlp"
    assert verdict.decision == "deny"
    assert "us_ssn" in verdict.reason
    assert "input.customer_ssn" in verdict.reason
    # Raw ssn must not be in reason
    assert "123-45-6789" not in verdict.reason

    # Benign outbound request
    req_benign = ToolRequest(
        tool_name="HttpEgress",
        action="network",
        resource="https://external.api/v1",
        tool_input={"status": "ok"},
    )
    assert rule(req_benign) is None

    # Inactive action (e.g. read)
    req_read = ToolRequest(
        tool_name="Read",
        action="read",
        resource="/tmp/file.txt",
        tool_input={"ssn": "123-45-6789"},
    )
    assert rule(req_read) is None


def test_as_authorize_rule_custom_decision_and_severity() -> None:
    cfg = DlpSecurityConfig(
        decision="ask",
        min_severity="critical",
    )
    rule = as_authorize_rule(cfg)

    # Warning severity hit (email) ignored by min_severity="critical"
    req_email = ToolRequest(
        tool_name="SendMessage",
        action="network",
        tool_input={"to": "alice@example.com"},
    )
    assert rule(req_email) is None

    # Critical severity hit (SSN) triggers ask decision
    req_ssn = ToolRequest(
        tool_name="SendMessage",
        action="network",
        tool_input={"ssn": "123-45-6789"},
    )
    verdict = rule(req_ssn)
    assert verdict is not None
    assert verdict.decision == "ask"


# ---------------------------------------------------------------------------
# Declarative Policy Parsing Tests
# ---------------------------------------------------------------------------


def test_policy_from_dict_dlp_rule() -> None:
    doc = {
        "rules": [
            {
                "type": "dlp",
                "decision": "deny",
                "min_severity": "warning",
                "include_secrets": False,
                "disabled_rules": ["phone_number"],
                "actions": ["network", "post"],
            }
        ]
    }
    policy = policy_from_dict(doc)
    assert len(policy.rules) == 1

    req = ToolRequest(
        tool_name="Post",
        action="post",
        tool_input={"email": "alice@example.com"},
    )
    verdict = policy.decide(req)
    assert verdict.decision == "deny"
    assert "email_address" in verdict.reason


def test_policy_from_dict_dlp_validation_errors() -> None:
    # Unknown key
    with pytest.raises(PolicyError, match="unknown rule keys"):
        policy_from_dict({"rules": [{"type": "dlp", "extra_bad_key": 123}]})

    # Bad decision
    with pytest.raises(PolicyError, match="must be 'ask' or 'deny'"):
        policy_from_dict({"rules": [{"type": "dlp", "decision": "invalid"}]})

    # Bad min_severity
    with pytest.raises(PolicyError, match="must be one of info"):
        policy_from_dict({"rules": [{"type": "dlp", "min_severity": "fatal"}]})

    # Bad include_secrets type
    with pytest.raises(PolicyError, match="include_secrets must be a boolean"):
        policy_from_dict({"rules": [{"type": "dlp", "include_secrets": "yes"}]})

    # Bad enabled_rules
    with pytest.raises(PolicyError, match="enabled_rules has unknown ids"):
        policy_from_dict(
            {"rules": [{"type": "dlp", "enabled_rules": ["non_existent_rule"]}]}
        )

    # Bad actions type
    with pytest.raises(PolicyError, match="actions must be a list of strings"):
        policy_from_dict({"rules": [{"type": "dlp", "actions": "network"}]})


def test_scan_text_empty_and_whitespace() -> None:
    assert scan_text("") == []
    assert scan_text("   \n\t  ") == []
    assert inspect_payload(None) == ()
    assert inspect_payload(12345) == ()
    assert inspect_payload(True) == ()


def test_credit_card_brand_variants() -> None:
    # Mastercard (16 digits starting with 55)
    hits_mc = scan_text("Mastercard: 5500-0000-0000-0004")
    assert any(h.rule_id == "credit_card" for h in hits_mc)

    # Amex (15 digits starting with 37)
    hits_amex = scan_text("Amex: 3782-822463-10005")
    assert any(h.rule_id == "credit_card" for h in hits_amex)

    # Card with invalid length
    hits_short = scan_text("Card: 4111-1111-1111")
    assert not any(h.rule_id == "credit_card" for h in hits_short)


def test_structured_payload_mixed_primitive_types() -> None:
    payload = {
        "active": True,
        "count": 100,
        "ratio": 3.14,
        "nothing": None,
        "emails": ["user1@example.com", None, 42],
        "sub": {
            "valid_ssn": "123-45-6789",
            "metadata": [False, {"deep_email": "deep@example.org"}],
        },
    }
    findings = inspect_payload(payload)
    rule_ids = {f.rule_id for f in findings}
    assert "email_address" in rule_ids
    assert "us_ssn" in rule_ids

    # Verify redaction on mixed payload works recursively
    redacted = redact_payload(payload)
    assert redacted["active"] is True
    assert redacted["count"] == 100
    assert redacted["emails"][0] == "[REDACTED:email_address]"
    assert redacted["emails"][1] is None
    assert redacted["sub"]["valid_ssn"] == "[REDACTED:us_ssn]"
    assert redacted["sub"]["metadata"][1]["deep_email"] == "[REDACTED:email_address]"


def test_custom_extra_rule_scan() -> None:
    def detect_custom(text: str) -> list[DlpMatch]:
        if "EMPLOYEE_BADGE" in text:
            return [
                DlpMatch(
                    rule_id="custom_badge",
                    severity="warning",
                    start=text.index("EMPLOYEE_BADGE"),
                    end=text.index("EMPLOYEE_BADGE") + 14,
                    raw_value="EMPLOYEE_BADGE",
                    redacted_value="[REDACTED:badge]",
                    match_type="Employee Badge",
                )
            ]
        return []

    hits = scan_text(
        "Payload includes EMPLOYEE_BADGE_9988",
        extra_rules=(("custom_badge", detect_custom),),
    )
    assert any(h.rule_id == "custom_badge" for h in hits)


def test_as_authorize_rule_reports_highest_severity() -> None:
    rule = as_authorize_rule()
    # Phone number is info, SSN is critical
    req = ToolRequest(
        tool_name="HttpEgress",
        action="network",
        resource="https://external.api/v1",
        tool_input={
            "phone": "555-123-4567",
            "ssn": "123-45-6789",
        },
    )
    verdict = rule(req)
    assert verdict is not None
    assert "severity=critical" in verdict.reason


def test_as_authorize_rule_rejects_truncated_resource() -> None:
    rule = as_authorize_rule()
    req_trunc = ToolRequest(
        tool_name="HttpEgress",
        action="network",
        resource="https://external.api/v1/data?query=params...<truncated>",
    )
    verdict = rule(req_trunc)
    assert verdict is not None
    assert verdict.decision == "deny"
    assert "truncated" in verdict.reason


def test_redact_text_literal_braces_in_mask() -> None:
    text = "User SSN is 123-45-6789"
    redacted = redact_text(text, mask="{CUSTOM_TAG}:{rule_id}")
    assert redacted == "User SSN is {CUSTOM_TAG}:us_ssn"


def test_redact_payload_nested_containers() -> None:
    data = {
        "user": "alice",
        "nested_set": {"123-45-6789", "benign"},
        "nested_frozenset": frozenset({"123-45-6789"}),
        "nested_tuple": ("safe", ["123-45-6789"]),
    }
    redacted = redact_payload(data)
    assert redacted["user"] == "alice"
    assert "[REDACTED:us_ssn]" in redacted["nested_set"]
    assert "[REDACTED:us_ssn]" in redacted["nested_frozenset"]
    assert redacted["nested_tuple"][1][0] == "[REDACTED:us_ssn]"


def test_inspect_payload_frozenset_and_set() -> None:
    payload = {
        "set_values": {"123-45-6789"},
        "frozenset_values": frozenset({"4111-1111-1111-1111"}),
    }
    findings = inspect_payload(payload)
    rules = {f.rule_id for f in findings}
    assert "us_ssn" in rules
    assert "credit_card" in rules


def test_secrets_integration_retains_distinct_signatures() -> None:
    token1 = "ghp_" + "111111111111111111111111111111111111"
    token2 = "ghp_" + "222222222222222222222222222222222222"
    payload = {
        "key1": token1,
        "key2": token2,
    }
    findings = inspect_payload(payload, DlpSecurityConfig(include_secrets=True))
    secret_findings = [f for f in findings if f.rule_id.startswith("secret:")]
    assert len(secret_findings) == 2
    signatures = [f.value_signature for f in secret_findings]
    assert len(set(signatures)) == 2
    assert all(len(sig) == 64 for sig in signatures)
