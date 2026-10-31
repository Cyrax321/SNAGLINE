"""Security evaluation engine for inspecting proposed tool requests.

Evaluates pre-execution requests across all security inspectors (shell security,
secret exposure, prompt injection, policy rules, and risk scoring) without executing
the underlying tool.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

from snagline.authorize import (
    Decision,
    Policy,
    ToolRequest,
)
from snagline.prompt_injection import inspect_text as inspect_injection, inspect_tool_input as inspect_input_injection
from snagline.security_policy import load_security_policy, compile_policy
from snagline.security_scoring import SecurityScoringEngine


def evaluate_request(
    tool_name: str,
    action: str = "unknown",
    resource: Optional[str] = None,
    tool_input: Optional[Mapping[str, Any]] = None,
    policy: Optional[Policy] = None,
    policy_path: Optional[str | Path] = None,
    scoring_engine: Optional[SecurityScoringEngine] = None,
) -> Dict[str, Any]:
    """Inspect and evaluate a proposed tool invocation against security controls."""
    if not tool_name:
        raise ValueError("tool_name is required for security evaluation")

    input_map = dict(tool_input or {})
    res_str = str(resource) if resource is not None else None

    # 1. Resolve Policy
    active_policy: Policy
    if policy is not None:
        active_policy = policy
    elif policy_path is not None:
        schema = load_security_policy(policy_path)
        active_policy = compile_policy(schema)
    else:
        active_policy = Policy(default_decision="allow")

    # 2. Run Inspector Findings
    findings: List[Dict[str, Any]] = []

    # Prompt injection inspection
    if res_str:
        for f in inspect_injection(res_str):
            findings.append(f.to_dict())
    if input_map:
        for f in inspect_input_injection(input_map):
            findings.append(f.to_dict())

    # Shell threat inspection
    try:
        from snagline.shell_security import inspect_command, ShellThreatConfig
        if action == "execute" and res_str:
            for sf in inspect_command(res_str):
                findings.append({
                    "rule_id": sf.rule_id,
                    "severity": sf.severity,
                    "reason": sf.reason,
                    "command_signature": sf.command_signature,
                })
    except ImportError:
        pass

    # Secret exposure inspection
    try:
        from snagline.secret_security import inspect_text as inspect_secret
        if res_str:
            for match in inspect_secret(res_str):
                findings.append({
                    "rule_id": match.rule_id,
                    "severity": match.severity,
                    "reason": f"Secret pattern matched: {match.rule_id}",
                })
    except ImportError:
        pass

    # 3. Authorization Verdict
    request = ToolRequest(
        tool_name=tool_name,
        action=action,
        resource=res_str,
        tool_input=input_map,
    )
    verdict = active_policy.decide(request)

    # 4. Security Risk Score
    engine = scoring_engine or SecurityScoringEngine()
    score_result = engine.evaluate(
        tool_name=tool_name,
        action=action,
        resource=res_str,
        findings=findings,
    )

    # Combine decisions: strictest wins
    rank = {"allow": 0, "ask": 1, "deny": 2}
    final_decision: Decision
    if rank.get(verdict.decision, 0) >= rank.get(score_result.decision, 0):
        final_decision = verdict.decision
        decision_reason = verdict.reason
    else:
        final_decision = score_result.decision
        decision_reason = f"Security risk score ({score_result.score:.1f}/100) triggered {score_result.decision}"

    return {
        "tool_name": tool_name,
        "action": action,
        "decision": final_decision,
        "policy_decision": verdict.decision,
        "score_decision": score_result.decision,
        "security_score": score_result.score,
        "reason": decision_reason,
        "matched_rules": list(verdict.matched_rules) + score_result.rule_ids,
        "factors": [f.to_dict() for f in score_result.factors],
        "findings": findings,
    }
