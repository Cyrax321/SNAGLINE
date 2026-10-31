import concurrent.futures
import json
import os
from pathlib import Path
import pytest

from snagline.authorize import Policy, ToolRequest, allow_tools, deny_tools
from snagline.security_audit import (
    AuditLogError,
    AuditRecord,
    GENESIS_HASH,
    SecurityAuditLog,
    verify_audit_log_file,
)


def test_audit_record_hash_determinism():
    rec1 = AuditRecord(
        index=0,
        timestamp="2026-10-31T12:00:00Z",
        decision="allow",
        tool_name="Read",
        policy_id="default",
        rule_findings=[],
        request_meta={"file": "test.txt"},
        previous_hash=GENESIS_HASH,
    )
    rec2 = AuditRecord(
        index=0,
        timestamp="2026-10-31T12:00:00Z",
        decision="allow",
        tool_name="Read",
        policy_id="default",
        rule_findings=[],
        request_meta={"file": "test.txt"},
        previous_hash=GENESIS_HASH,
    )
    assert rec1.compute_hash() == rec2.compute_hash()
    assert len(rec1.compute_hash()) == 64


def test_security_audit_log_append_and_verify():
    log = SecurityAuditLog()
    rec1 = log.append("allow", "Read", policy_id="p1")
    rec2 = log.append("deny", "Bash", policy_id="p1", rule_findings=[{"rule_id": "r1"}])
    rec3 = log.append("ask", "Write", policy_id="p1")

    assert len(log.records) == 3
    assert rec1.previous_hash == GENESIS_HASH
    assert rec2.previous_hash == rec1.entry_hash
    assert rec3.previous_hash == rec2.entry_hash

    is_valid, err = log.verify()
    assert is_valid is True
    assert err is None


def test_security_audit_log_tampering_detection():
    log = SecurityAuditLog()
    log.append("allow", "Read")
    log.append("deny", "Bash")
    log.append("allow", "Grep")

    # Tamper with second record's decision
    log.records[1].decision = "allow"
    # Note: log._records is the internal list
    log._records[1].decision = "allow"

    is_valid, err = log.verify()
    assert is_valid is False
    assert "Tampered content at entry 1" in err


def test_security_audit_log_file_persistence_and_verification(tmp_path: Path):
    log_file = tmp_path / "audit.jsonl"
    log = SecurityAuditLog(log_path=log_file)
    log.append("allow", "Read")
    log.append("deny", "Bash", rule_findings=[{"rule_id": "destructive"}])
    log.append("ask", "Deploy")

    assert log_file.exists()
    lines = log_file.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 3

    # Verify via standalone file verifier
    is_valid, err = verify_audit_log_file(log_file)
    assert is_valid is True
    assert err is None

    # Load existing in fresh log instance
    loaded_log = SecurityAuditLog(log_path=log_file)
    assert len(loaded_log.records) == 3
    is_valid, err = loaded_log.verify()
    assert is_valid is True

    # Tamper with file
    lines[1] = json.dumps({**json.loads(lines[1]), "decision": "allow"})
    tampered_file = tmp_path / "tampered.jsonl"
    tampered_file.write_text("\n".join(lines) + "\n", encoding="utf-8")

    is_valid, err = verify_audit_log_file(tampered_file)
    assert is_valid is False
    assert "Hash mismatch" in err or "Tampered" in err


def test_security_audit_log_secret_redaction():
    log = SecurityAuditLog()
    record = log.append(
        "allow",
        "Http",
        request_meta={
            "url": "https://api.internal/v1",
            "api_key": "sk-secret-123456",
            "auth_token": "bearer-xyz",
            "nested": {"password": "supersecretpassword", "safe_val": 42},
        },
    )
    meta = record.request_meta
    assert meta["url"] == "https://api.internal/v1"
    assert meta["api_key"] == "[REDACTED]"
    assert meta["auth_token"] == "[REDACTED]"
    assert meta["nested"]["password"] == "[REDACTED]"
    assert meta["nested"]["safe_val"] == 42


def test_security_audit_log_concurrency():
    log = SecurityAuditLog()

    def worker(i: int):
        for j in range(25):
            log.append("allow", f"Tool_{i}_{j}", policy_id="concurrent")

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(worker, i) for i in range(4)]
        concurrent.futures.wait(futures)

    assert len(log.records) == 100
    is_valid, err = log.verify()
    assert is_valid is True
    assert err is None


def test_security_audit_log_policy_integration():
    audit_log = SecurityAuditLog()
    policy = Policy(
        rules=(allow_tools("Read"), deny_tools("Bash")),
        default_decision="ask",
        audit_log=audit_log,
        policy_id="test-pol",
    )

    req1 = ToolRequest("Read", "read", {"path": "a.py"})
    req2 = ToolRequest("Bash", "execute", {"command": "rm -rf /"})
    req3 = ToolRequest("Grep", "search", {"pattern": "foo"})

    assert policy.decide(req1).decision == "allow"
    assert policy.decide(req2).decision == "deny"
    assert policy.decide(req3).decision == "ask"

    assert len(audit_log.records) == 3
    assert audit_log.records[0].decision == "allow"
    assert audit_log.records[0].tool_name == "Read"
    assert audit_log.records[1].decision == "deny"
    assert audit_log.records[1].tool_name == "Bash"
    assert audit_log.records[2].decision == "ask"
    assert audit_log.records[2].tool_name == "Grep"

    is_valid, err = audit_log.verify()
    assert is_valid is True
