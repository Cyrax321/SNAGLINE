# Security Risk Scoring Engine

The `snagline.security_scoring` module provides a dedicated pre-execution security risk scoring engine. It evaluates tool invocations across multiple orthogonal dimensions to produce an explainable composite security score on a deterministic 0 to 100 scale.

> [!NOTE]
> Security risk scoring is strictly independent of SNAGLINE's runtime failure detection (`FailureRisk.score`). Runtime failure risk tracks step loops, error cascades, and latency drift, while security risk scoring tracks privilege escalation, sensitive resource exposure, and malicious payloads.

## Score Calculation & Factors

1. **Base Tool Sensitivity**:
   Tools are assigned baseline sensitivity scores reflecting their execution capabilities:
   - Shell / System execution (`Bash`, `Terminal`): `35.0`
   - Filesystem modification (`Write`, `Edit`, `Delete`): `20.0`–`30.0`
   - Read-only queries (`Read`, `Grep`, `Find`): `5.0`
   - Unrecognized tools: `25.0` (configurable)
2. **Security Findings Impact**:
   Pre-execution detector findings (from shell security, DLP, secret exposure, or prompt injection) add severity-weighted impacts:
   - `low`: `+10.0`
   - `medium`: `+25.0`
   - `high`: `+50.0`
   - `critical`: `+80.0`
3. **Composite Normalization**:
   Scores scale with sub-linear diminishing returns above 50 and are clamped between `0.0` and `100.0`.

## Decision Thresholds

Policies configure two progressive thresholds:

| Composite Score | Default Threshold | Decision | Action |
|---|---|---|---|
| `< 40.0` | `< ask_threshold` | `allow` | Tool execution proceeds automatically |
| `40.0 – 74.9` | `ask_threshold` to `deny_threshold` | `ask` | Escalated to operator or human-in-the-loop approval |
| `>= 75.0` | `>= deny_threshold` | `deny` | Tool execution blocked immediately |

## Policy Integration Example

```python
from snagline.authorize import Policy
from snagline.security_scoring import SecurityScoringEngine

engine = SecurityScoringEngine(
    ask_threshold=40.0,
    deny_threshold=75.0,
    tool_sensitivities={"custom_deploy": 45.0},
)

policy = Policy(
    rules=(engine.as_authorize_rule(),),
    default_decision="allow",
)
```
