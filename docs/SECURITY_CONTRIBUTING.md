# Contributing Security Rules & Detectors

This guide explains how to add new security rules, threat patterns, and inspectors to SNAGLINE.

## Architecture Principles

1. **Strict Non-Execution Invariant**:
   Inspectors must **never** execute, spawn, or shell out to untrusted strings. Inspection must remain purely lexical or AST-based.
2. **Zero-Secret Retention**:
   Rules must never echo, log, or persist raw matched credential strings or sensitive tokens. Always hash or redact matched findings (`rule_id`, `reason`, `matched_signature`).
3. **Boundary Discipline**:
   - Runtime failure detectors belong in `snagline.detectors` and adhere to the **hard fail-open guarantee** (`Monitor.ingest()` can never block).
   - Pre-execution security filters belong in `snagline.authorize` and support the **fail-closed security profile** (`Policy.decide()`).

## Requirements for New Security Rules

When introducing a new security rule or pattern:

- **Rule Identifier**: Assign a distinct, permanent `rule_id` (e.g. `rule_destructive_rm`, `rule_prompt_override`).
- **Severity Rating**: Classify under standard severity ranks (`low`, `medium`, `high`, `critical`).
- **Explainability**: Provide a succinct human-readable `reason` that does not leak secrets.
- **Test Coverage**:
  - Positive tests: verify targeted adversarial evasions trigger the rule.
  - Negative tests: verify ordinary benign developer commands/queries do **not** false-positive.
  - Non-execution test: verify the inspector performs no I/O or process spawning.
- **Document Limitations**: Explicitly document known bypasses or evasion strategies in the corresponding module documentation.
