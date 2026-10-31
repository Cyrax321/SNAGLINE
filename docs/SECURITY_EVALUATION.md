# Security Evaluation API and CLI

The security evaluation system enables offline and remote inspection of proposed tool requests against SNAGLINE's security policies, prompt injection heuristics, shell threat rules, and composite risk scoring without executing the requested tool.

## CLI Commands

### 1. Evaluate a Tool Request (`snagline eval`)

```bash
# Basic evaluation
snagline eval --tool Bash --action execute --resource "cat /etc/passwd"

# Evaluate with custom policy and JSON output
snagline eval --tool Read --action read --resource "src/main.py" --policy policy.yaml --json
```

**JSON Output Format:**
```json
{
  "tool_name": "Bash",
  "action": "execute",
  "decision": "deny",
  "security_score": 85.0,
  "reason": "Destructive filesystem operation detected",
  "matched_rules": ["rule_destructive"],
  "factors": [
    {
      "name": "tool_sensitivity",
      "impact": 35.0,
      "reason": "Base sensitivity for tool 'Bash' is 35.0"
    }
  ],
  "findings": []
}
```

### 2. Validate Security Policy Schema (`snagline validate-policy`)

```bash
snagline validate-policy config/security_policy.yaml
```

Exits `0` if the policy strictly satisfies the schema, or `2` with validation errors on unknown fields or conflicts.

## Sidecar HTTP Endpoint (`POST /security/eval`)

The SNAGLINE sidecar server exposes `POST /security/eval`:

**Request:**
```http
POST /security/eval HTTP/1.1
Host: 127.0.0.1:9876
Content-Type: application/json

{
  "tool_name": "Bash",
  "action": "execute",
  "resource": "rm -rf /tmp/test",
  "tool_input": {"command": "rm -rf /tmp/test"}
}
```

**Response (`200 OK`):**
Returns the same evaluation payload as `snagline eval --json`.
