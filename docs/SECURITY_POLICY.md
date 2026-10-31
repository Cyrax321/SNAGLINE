# Security Policy Schema & Validation Specification

The `snagline.security_policy` module defines a strict, deterministic schema and validation layer for SNAGLINE authorization policies.

## Validation Principles

- **Strict Field Validation**: Any unknown fields in policy definitions raise `PolicyValidationError` immediately instead of being silently ignored.
- **Conflict Prevention**: A tool cannot be simultaneously assigned to multiple conflicting lists (e.g., both allowed and denied).
- **Per-Agent Granularity**: Policies can specify agent-level overrides that take precedence over global rules.
- **Fail-Closed Security**: Invalid configurations cannot silently downgrade security restrictions.

## Example Policy (YAML/JSON)

```yaml
version: "1.0"
name: "production_sandbox_policy"
profile: "fail_closed"
default_decision: "deny"

allowed_tools:
  - "Read"
  - "Grep"

ask_tools:
  - "Write"
  - "Edit"

denied_tools:
  - "Bash"
  - "Terminal"

allowed_roots:
  - "/workspace/app"

agent_rules:
  untrusted_subagent:
    allowed_tools:
      - "Read"
    denied_tools:
      - "Write"
      - "Edit"
      - "Bash"

detector_settings:
  max_samples: 1000
```

## Compilation & Usage

```python
from snagline.security_policy import load_security_policy, compile_policy

schema = load_security_policy("config/security_policy.yaml")
policy = compile_policy(schema)

# Evaluate incoming requests
result = policy.decide(request)
```
