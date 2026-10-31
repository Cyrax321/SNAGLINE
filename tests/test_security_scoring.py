import pytest

from snagline.authorize import Policy, ToolRequest
from snagline.security_scoring import (
    SecurityFactor,
    SecurityScoreResult,
    SecurityScoringEngine,
)


def test_security_scoring_baseline_sensitivity():
    engine = SecurityScoringEngine(ask_threshold=40.0, deny_threshold=75.0)

    # Benign read tool
    res_read = engine.evaluate(tool_name="Read")
    assert res_read.score == 5.0
    assert res_read.decision == "allow"
    assert len(res_read.factors) == 1
    assert res_read.factors[0].name == "tool_sensitivity"

    # Shell execution tool
    res_bash = engine.evaluate(tool_name="Bash")
    assert res_bash.score == 35.0
    assert res_bash.decision == "allow"


def test_security_scoring_findings_impact():
    engine = SecurityScoringEngine(ask_threshold=40.0, deny_threshold=75.0)

    # Bash tool (35) + medium finding (25) = 60 -> ask
    res = engine.evaluate(
        tool_name="Bash",
        findings=[{"rule_id": "rule_suspicious_redirect", "severity": "medium", "reason": "Redirect detected"}],
    )
    assert res.score >= 50.0
    assert res.decision == "ask"
    assert "rule_suspicious_redirect" in res.rule_ids

    # Bash tool (35) + critical finding (80) = raw 115 -> clamped/scaled > 75 -> deny
    res_crit = engine.evaluate(
        tool_name="Bash",
        findings=[{"rule_id": "rule_destructive", "severity": "critical", "reason": "Destructive rm command"}],
    )
    assert res_crit.score >= 75.0
    assert res_crit.decision == "deny"


def test_security_scoring_threshold_boundaries():
    with pytest.raises(ValueError, match="Invalid thresholds"):
        SecurityScoringEngine(ask_threshold=80.0, deny_threshold=40.0)

    with pytest.raises(ValueError, match="Invalid thresholds"):
        SecurityScoringEngine(ask_threshold=-5.0)

    engine = SecurityScoringEngine(ask_threshold=30.0, deny_threshold=70.0)
    assert engine.ask_threshold == 30.0
    assert engine.deny_threshold == 70.0


def test_security_scoring_explainability():
    engine = SecurityScoringEngine()
    res = engine.evaluate(
        tool_name="CustomTool",
        findings=[
            {"rule_id": "r1", "severity": "low", "reason": "Minor note"},
            {"rule_id": "r2", "severity": "high", "reason": "Dangerous operation"},
        ],
    )
    d = res.to_dict()
    assert "score" in d
    assert "factors" in d
    assert len(d["factors"]) == 3
    assert any(f["name"] == "unknown_tool_penalty" for f in d["factors"])
    assert any(f["rule_id"] == "r1" for f in d["factors"])
    assert any(f["rule_id"] == "r2" for f in d["factors"])


def test_security_scoring_policy_integration():
    engine = SecurityScoringEngine(ask_threshold=30.0, deny_threshold=70.0)
    policy = Policy(rules=(engine.as_authorize_rule(),), default_decision="allow")

    # Read tool scores 5 -> below ask threshold (30) -> allow
    req_read = ToolRequest("Read", "read", {"path": "test.py"})
    assert policy.decide(req_read).decision == "allow"

    # Bash tool scores 35 -> above ask threshold (30) -> ask
    req_bash = ToolRequest("Bash", "execute", {"command": "ls -la"})
    res_bash = policy.decide(req_bash)
    assert res_bash.decision == "ask"
    assert "Security risk score" in res_bash.reason
