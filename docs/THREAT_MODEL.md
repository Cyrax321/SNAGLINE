# SNAGLINE Security Threat Model & Trust Boundaries

This document defines the formal threat model, trust boundaries, attacker capabilities, and security invariants for SNAGLINE's pre-execution authorization and security inspection subsystems.

---

## 1. System Overview & Security Invariants

SNAGLINE operates two fundamentally distinct execution pipelines:

| Subsystem | Architectural Role | Reliability Contract | Primary Invariant |
|---|---|---|---|
| **Runtime Failure Monitor** (`snagline.monitor`) | Post-execution telemetry observer | **Hard Fail-Open** | Can never crash, stall, or intercept host execution steps. |
| **Pre-Execution Authorization Gate** (`snagline.authorize`) | Pre-execution security filter | **Configurable (Fail-Closed in Security Profiles)** | Evaluates proposed tool calls; denies or requests operator approval before execution. |

---

## 2. Protected Assets & Trust Boundaries

### Protected Assets
1. **Local Host Integrity**: Protection against destructive shell commands (`rm -rf /`, raw disk formatting, fork bombs).
2. **Filesystem Boundaries**: Confining tool reads and writes to explicit `allowed_roots`.
3. **Sensitive Credentials**: Preventing leakage of API tokens, bearer keys, passwords, and private keys in tool inputs, outputs, or logs.
4. **Audit Trail Integrity**: Guaranteeing tamper-evident traceability of all authorization decisions via SHA-256 hash chains.

### Trust Boundaries
- **Untrusted**: LLM outputs, external web responses, user prompts, third-party MCP tool requests, unverified agent tool inputs.
- **Trusted**: Host operating environment, SNAGLINE library code, local policy configuration files explicitly provisioned by system operators.

---

## 3. Attacker Capabilities & Threat Classes

| Threat Class | Attacker Capability | Mitigation Mechanism |
|---|---|---|
| **Direct Destructive Execution** | Agent attempts dangerous shell commands | `snagline.shell_security`: Deterministic regex rules for destructive patterns, unsafe redirects, and piping. |
| **Directory Traversal** | Agent references paths outside sandbox via `../` | `snagline.path_security`: Canonical path resolution and `allowed_roots` enforcement. |
| **Credential / Secret Exposure** | Prompts or tool arguments contain raw secrets | `snagline.secret_security` & `dlp_security`: Pattern detection and automatic redaction before persistence. |
| **Prompt Injection** | Untrusted external text instructs agent to override policy | `snagline.prompt_injection`: Heuristic indicators for instruction overrides and secret exfiltration. |
| **Audit Log Tampering** | Attacker attempts to modify past decisions | `snagline.security_audit`: SHA-256 hash-chained append-only ledger. |

---

## 4. Documented Non-Goals & Limitations

1. **Local Hash Chain vs Host Compromise**: A local hash chain is *tamper-evident*, not *tamper-proof*. An adversary who gains root access to the filesystem can rewrite the entire log.
2. **Regex & Heuristic Parsers**: Static heuristics cannot replace full AST shell interpreters or solve the halting problem. Complex shell string evaluations (e.g. `eval $(printf ...)` ) must be barred by restricting shell access altogether.
3. **Semantic Injections**: Advisory prompt injection detectors cannot guarantee complete elimination of indirect prompt injection. They must be combined with strict tool allowlists.
