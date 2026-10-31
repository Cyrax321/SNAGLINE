# Security Authorization Audit Log

The `SecurityAuditLog` provides an append-only, tamper-evident record of all pre-execution authorization decisions made by SNAGLINE's policy engine (`snagline.authorize`).

## Key Properties

- **SHA-256 Hash Chaining**: Every audit entry records a deterministic SHA-256 hash of its canonical JSON representation and cryptographically links to the preceding entry's hash.
- **Tamper Evidence**: Any modification to historical entries, reordering of records, or unauthorized deletion breaks the hash chain and is immediately flagged by `verify()` or `verify_audit_log_file()`.
- **Zero-Secret Persistence**: Sensitive credentials, auth headers, and API keys are automatically redacted prior to disk persistence.
- **Thread Safety**: Appends are guarded by re-entrant mutexes, supporting high-throughput concurrent agent loops.
- **Fail-Open / Fail-Closed Flexibility**: Operates in fail-open mode by default to avoid interrupting host processes, or fail-closed mode when strict compliance is enforced.

## Log Entry Schema

Each record contains:

```json
{
  "index": 0,
  "timestamp": "2026-10-31T22:30:00+00:00",
  "decision": "deny",
  "tool_name": "Bash",
  "policy_id": "production_default",
  "rule_findings": [
    {
      "rule_id": "rule_destructive",
      "decision": "deny",
      "reason": "Destructive filesystem operation detected"
    }
  ],
  "request_meta": {
    "action": "execute",
    "resource": "/var/data"
  },
  "previous_hash": "0000000000000000000000000000000000000000000000000000000000000000",
  "entry_hash": "a4f8...b12c"
}
```

## Security Boundary & Threat Model

> [!IMPORTANT]
> A local hash chain is **tamper-evident**, not **tamper-proof**. An adversary with local filesystem write access can modify records and recalculate the chain if they rewrite the entire file. For non-repudiation against host compromise, stream audit logs to an external append-only sink or cloud SIEM.

## Verification

Audit log integrity can be verified programmatically:

```python
from snagline.security_audit import verify_audit_log_file

is_valid, error = verify_audit_log_file("var/log/snagline_audit.jsonl")
if not is_valid:
    print(f"Audit log integrity violation: {error}")
```
