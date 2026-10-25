"""Unit and regression tests for deterministic path security (issue #639)."""

from __future__ import annotations

from pathlib import Path

import pytest

from snagline.authorize import (
    Policy,
    PolicyError,
    ToolRequest,
    policy_from_dict,
)
from snagline.path_security import (
    PathFinding,
    PathSecurityConfig,
    as_authorize_rule,
    findings_at_or_above,
    inspect_path,
    normalize_lexical_path,
    strongest_severity,
)

# ---------------------------------------------------------------------------
# Lexical path normalization tests
# ---------------------------------------------------------------------------


def test_normalize_lexical_path_standard() -> None:
    assert normalize_lexical_path("foo/bar/baz") == "foo/bar/baz"
    assert normalize_lexical_path("./foo/./bar/../baz") == "foo/baz"
    assert normalize_lexical_path("/var/log/../log/syslog") == "/var/log/syslog"
    assert normalize_lexical_path("///app///src///main.py") == "/app/src/main.py"
    assert normalize_lexical_path("") == ""
    assert normalize_lexical_path("   ") == ""


def test_normalize_lexical_path_windows_and_backslashes() -> None:
    assert normalize_lexical_path(r"foo\bar\baz") == "foo/bar/baz"
    assert normalize_lexical_path(r"C:\Windows\System32") == "C:/Windows/System32"
    assert normalize_lexical_path(r"d:\app\..\secret.txt") == "d:/secret.txt"


def test_normalize_lexical_path_with_base_dir() -> None:
    assert (
        normalize_lexical_path("config/app.json", base_dir="/workspace/project")
        == "/workspace/project/config/app.json"
    )
    # Absolute paths ignore base_dir
    assert (
        normalize_lexical_path("/etc/hosts", base_dir="/workspace/project")
        == "/etc/hosts"
    )


# ---------------------------------------------------------------------------
# Path traversal tests
# ---------------------------------------------------------------------------


def test_traversal_detection_raw_dotdot() -> None:
    findings = inspect_path("../../etc/passwd")
    assert any(f.rule_id == "path_traversal" for f in findings)
    assert any(f.severity == "critical" for f in findings)

    findings = inspect_path("/app/../../etc/shadow")
    assert any(f.rule_id == "path_traversal" for f in findings)


def test_traversal_detection_with_base_dir() -> None:
    cfg = PathSecurityConfig(base_dir="/workspace/app")
    findings = inspect_path("../other/secret.txt", config=cfg)
    assert any(f.rule_id == "path_traversal" for f in findings)


def test_traversal_null_byte_injection() -> None:
    findings = inspect_path("safe/path\0/etc/passwd")
    assert len(findings) == 1
    assert findings[0].rule_id == "path_traversal"
    assert findings[0].severity == "critical"
    assert "null byte" in findings[0].reason


# ---------------------------------------------------------------------------
# Sensitive locations tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "path",
    [
        ".env",
        ".env.local",
        ".env.production",
        ".env.test",
        "/app/.env.development",
        "~/.ssh/id_rsa",
        "/home/user/.ssh/id_ed25519",
        "~/.ssh/authorized_keys",
        "~/.aws/credentials",
        "/root/.azure/accessTokens.json",
        "/var/run/secrets/kubernetes.io/serviceaccount/token",
        "/etc/shadow",
        "/etc/sudoers",
        "/etc/ssl/private/key.pem",
        "/workspace/server.key",
        "/workspace/cert.pfx",
        "/home/user/.bash_history",
        "/home/user/.git/config",
        "/home/user/.git-credentials",
        "credentials.json",
        "service_account.json",
    ],
)
def test_sensitive_locations_detected(path: str) -> None:
    findings = inspect_path(path)
    assert any(f.rule_id == "sensitive_locations" for f in findings)
    assert any(f.severity == "critical" for f in findings)


# ---------------------------------------------------------------------------
# Allowed roots tests
# ---------------------------------------------------------------------------


def test_allowed_roots_allows_subpaths() -> None:
    cfg = PathSecurityConfig(allowed_roots=("/workspace/project",))
    # Legitimate subpath
    findings = inspect_path("/workspace/project/src/index.ts", config=cfg)
    assert not any(f.rule_id == "allowed_roots" for f in findings)

    # Outside root
    findings = inspect_path("/etc/passwd", config=cfg)
    assert any(f.rule_id == "allowed_roots" for f in findings)

    # Prefix mismatch: /workspace/project_evil is not inside /workspace/project
    findings = inspect_path("/workspace/project_evil/file.txt", config=cfg)
    assert any(f.rule_id == "allowed_roots" for f in findings)


def test_allowed_roots_multiple() -> None:
    cfg = PathSecurityConfig(allowed_roots=("/workspace", "/data/storage"))
    assert not any(
        f.rule_id == "allowed_roots"
        for f in inspect_path("/workspace/app.py", config=cfg)
    )
    assert not any(
        f.rule_id == "allowed_roots"
        for f in inspect_path("/data/storage/dataset.csv", config=cfg)
    )
    assert any(
        f.rule_id == "allowed_roots"
        for f in inspect_path("/opt/forbidden.txt", config=cfg)
    )


# ---------------------------------------------------------------------------
# Denied paths and patterns tests
# ---------------------------------------------------------------------------


def test_denied_paths() -> None:
    cfg = PathSecurityConfig(
        denied_paths=("/workspace/confidential",),
    )
    findings = inspect_path("/workspace/confidential/report.pdf", config=cfg)
    assert any(f.rule_id == "denied_paths" for f in findings)

    findings = inspect_path("/workspace/public/report.pdf", config=cfg)
    assert not any(f.rule_id == "denied_paths" for f in findings)


def test_denied_patterns() -> None:
    cfg = PathSecurityConfig(
        denied_patterns=("*.confidential", "*_private.*"),
    )
    findings = inspect_path("user_private.csv", config=cfg)
    assert any(f.rule_id == "denied_patterns" for f in findings)

    findings = inspect_path("doc.confidential", config=cfg)
    assert any(f.rule_id == "denied_patterns" for f in findings)

    findings = inspect_path("user_public.csv", config=cfg)
    assert not any(f.rule_id == "denied_patterns" for f in findings)


# ---------------------------------------------------------------------------
# Benign paths remain usable
# ---------------------------------------------------------------------------


def test_benign_paths_remain_usable() -> None:
    cfg = PathSecurityConfig(
        allowed_roots=("/workspace",),
        base_dir="/workspace",
    )
    assert inspect_path("/workspace/src/app.py", config=cfg) == ()
    assert inspect_path("src/utils.py", config=cfg) == ()
    assert inspect_path("docs/README.md", config=cfg) == ()
    assert inspect_path("tests/test_app.py", config=cfg) == ()


# ---------------------------------------------------------------------------
# Non-invasive invariant (never touches disk)
# ---------------------------------------------------------------------------


def test_path_validation_does_not_modify_files(tmp_path: Path) -> None:
    nonexistent = tmp_path / "does_not_exist_ever" / "file.txt"
    assert not nonexistent.exists()

    findings = inspect_path(str(nonexistent))
    assert isinstance(findings, tuple)
    assert not nonexistent.exists()
    assert not (tmp_path / "does_not_exist_ever").exists()


# ---------------------------------------------------------------------------
# Config rule toggling and overrides
# ---------------------------------------------------------------------------


def test_config_rule_toggling() -> None:
    # Disable sensitive_locations rule
    cfg = PathSecurityConfig(disabled_rules=frozenset({"sensitive_locations"}))
    findings = inspect_path(".env", config=cfg)
    assert not any(f.rule_id == "sensitive_locations" for f in findings)

    # Severity override
    cfg_override = PathSecurityConfig(
        severity_overrides={"sensitive_locations": "info"}
    )
    findings = inspect_path(".env", config=cfg_override)
    finding = next(f for f in findings if f.rule_id == "sensitive_locations")
    assert finding.severity == "info"


def test_strongest_severity_and_filter() -> None:
    f1 = PathFinding("r1", "info", "reason1", "sig1")
    f2 = PathFinding("r2", "critical", "reason2", "sig2")
    assert strongest_severity((f1, f2)) == "critical"
    assert strongest_severity(()) is None

    above_warning = findings_at_or_above((f1, f2), "warning")
    assert above_warning == (f2,)


# ---------------------------------------------------------------------------
# Authorize rule integration
# ---------------------------------------------------------------------------


def test_as_authorize_rule_denies_path_violations() -> None:
    rule = as_authorize_rule()
    # Read tool request on sensitive file
    req_deny = ToolRequest(tool_name="Read", action="read", resource=".env")
    verdict = rule(req_deny)
    assert verdict is not None
    assert verdict.decision == "deny"
    assert "filesystem security" in verdict.reason
    assert "sig=" in verdict.reason
    # Raw path is not exposed in the reason
    assert ".env" not in verdict.reason

    # Safe resource passes
    req_allow = ToolRequest(tool_name="Read", action="read", resource="src/index.ts")
    assert rule(req_allow) is None

    # Non-filesystem action is ignored by path security
    req_bash = ToolRequest(tool_name="Bash", action="execute", resource="ls -la")
    assert rule(req_bash) is None


def test_as_authorize_rule_in_policy() -> None:
    rule = as_authorize_rule(PathSecurityConfig(allowed_roots=("/app",)))
    policy = Policy(rules=(rule,))

    res_allowed = policy.decide(
        ToolRequest(tool_name="Read", action="read", resource="/app/main.py")
    )
    assert res_allowed.decision == "allow"

    res_denied = policy.decide(
        ToolRequest(tool_name="Read", action="read", resource="/etc/shadow")
    )
    assert res_denied.decision == "deny"


# ---------------------------------------------------------------------------
# Declarative policy parsing
# ---------------------------------------------------------------------------


def test_declarative_policy_from_dict() -> None:
    raw = {
        "rules": [
            {
                "type": "path_security",
                "decision": "ask",
                "min_severity": "warning",
                "allowed_roots": ["/workspace"],
                "denied_paths": ["/workspace/confidential"],
                "denied_patterns": ["*.key"],
            }
        ]
    }
    pol = policy_from_dict(raw)
    assert len(pol.rules) == 1

    # Safe path
    verdict = pol.decide(
        ToolRequest(tool_name="Read", action="read", resource="/workspace/safe.py")
    )
    assert verdict.decision == "allow"

    # Outside root -> triggers ask
    verdict = pol.decide(
        ToolRequest(tool_name="Read", action="read", resource="/etc/hosts")
    )
    assert verdict.decision == "ask"


def test_declarative_policy_schema_validation() -> None:
    with pytest.raises(PolicyError, match="unknown rule keys"):
        policy_from_dict({"rules": [{"type": "path_security", "bad_key": True}]})

    with pytest.raises(PolicyError, match="must be 'ask' or 'deny'"):
        policy_from_dict({"rules": [{"type": "path_security", "decision": "invalid"}]})

    with pytest.raises(PolicyError, match="allowed_roots must be a list"):
        policy_from_dict(
            {"rules": [{"type": "path_security", "allowed_roots": "not_a_list"}]}
        )

    with pytest.raises(PolicyError, match="enabled_rules has unknown ids"):
        policy_from_dict(
            {"rules": [{"type": "path_security", "enabled_rules": ["nonexistent"]}]}
        )


# ---------------------------------------------------------------------------
# Monitor independence and documentation criteria
# ---------------------------------------------------------------------------


def test_filesystem_checks_independent_of_monitor() -> None:
    """The monitoring pipeline operates completely decoupled from path security."""
    from snagline import Monitor, StepEvent

    m = Monitor.default()
    event = StepEvent(
        step_id="step_fs_1",
        episode_id="ep_fs_1",
        timestamp=100.0,
        action_type="tool_call",
        action_signature="sig_fs",
        tool_name="Read",
    )
    m.ingest(event)
    assert m.metrics()["events_ingested"] == 1


def test_limitations_clearly_documented() -> None:
    """Module documentation explicitly mentions symlinks, TOCTOU, and lexical limits."""
    import snagline.path_security as ps

    doc = ps.__doc__ or ""
    assert "LIMITATIONS" in doc
    assert "Symlinks" in doc or "symlink" in doc
    assert "TOCTOU" in doc or "race" in doc
    assert "Lexical path checks alone cannot guarantee" in doc


def test_normalize_lexical_path_edge_branches() -> None:
    # Drive with relative path without leading slash
    assert normalize_lexical_path("C:foo/bar") == "C:/foo/bar"
    # Leading double slash normalized to single slash
    assert normalize_lexical_path("//server/share") == "/server/share"
    # Exact match on root
    cfg = PathSecurityConfig(allowed_roots=("/workspace",))
    assert inspect_path("/workspace", config=cfg) == ()
    assert inspect_path("/workspace/", config=cfg) == ()


def test_inspect_path_empty_and_custom_extra_rules() -> None:
    assert inspect_path("") == ()
    assert inspect_path("   ") == ()

    def custom_rule(raw: str, norm: str, cfg: PathSecurityConfig):
        if "bad_word" in norm:
            return PathFinding("custom_rule", "warning", "found bad word", "sig_bad")
        return None

    # Extra rule hit
    hits = inspect_path("bad_word.txt", extra_rules=(("custom_rule", custom_rule),))
    assert any(h.rule_id == "custom_rule" for h in hits)

    # Extra rule filtered by enabled_rules
    cfg_enabled = PathSecurityConfig(enabled_rules=frozenset({"path_traversal"}))
    hits = inspect_path(
        "bad_word.txt",
        config=cfg_enabled,
        extra_rules=(("custom_rule", custom_rule),),
    )
    assert not any(h.rule_id == "custom_rule" for h in hits)

    # Extra rule filtered by disabled_rules
    cfg_disabled = PathSecurityConfig(disabled_rules=frozenset({"custom_rule"}))
    hits = inspect_path(
        "bad_word.txt",
        config=cfg_disabled,
        extra_rules=(("custom_rule", custom_rule),),
    )
    assert not any(h.rule_id == "custom_rule" for h in hits)


def test_as_authorize_rule_empty_or_none_resource() -> None:
    rule = as_authorize_rule()
    req = ToolRequest(tool_name="Read", action="read", resource=None)
    assert rule(req) is None

    req_empty = ToolRequest(tool_name="Read", action="read", resource="")
    assert rule(req_empty) is None


def test_empty_allowed_roots_denies_all_paths() -> None:
    cfg = PathSecurityConfig(allowed_roots=())
    hits = inspect_path("/workspace/file.txt", config=cfg)
    assert any(h.rule_id == "allowed_roots" for h in hits)


def test_relative_allowed_roots_and_denied_paths_resolved_with_base_dir() -> None:
    cfg = PathSecurityConfig(
        base_dir="/workspace/project",
        allowed_roots=("src", "docs"),
        denied_paths=("src/private",),
    )
    # Inside allowed relative root
    assert inspect_path("src/main.py", config=cfg) == ()
    # Outside allowed relative roots
    hits_outside = inspect_path("scripts/deploy.sh", config=cfg)
    assert any(h.rule_id == "allowed_roots" for h in hits_outside)
    # Inside denied relative path
    hits_denied = inspect_path("src/private/key.pem", config=cfg)
    assert any(h.rule_id == "denied_paths" for h in hits_denied)


def test_windows_case_insensitivity_on_denied_and_patterns() -> None:
    cfg = PathSecurityConfig(
        denied_paths=(r"C:\workspace\Confidential",),
        denied_patterns=("*.SECRET",),
    )
    hits_path = inspect_path(r"c:\workspace\confidential\doc.txt", config=cfg)
    assert any(h.rule_id == "denied_paths" for h in hits_path)

    hits_pat = inspect_path(r"C:\workspace\data.secret", config=cfg)
    assert any(h.rule_id == "denied_patterns" for h in hits_pat)


def test_as_authorize_rule_unknown_action_path_inspected() -> None:
    cfg = PathSecurityConfig(allowed_roots=("/workspace",))
    rule = as_authorize_rule(config=cfg)
    # Unknown action but resource is a path
    req = ToolRequest(tool_name="CustomTool", action="unknown", resource="/etc/passwd")
    verdict = rule(req)
    assert verdict is not None
    assert verdict.decision == "deny"


def test_as_authorize_rule_truncated_or_overlong_resource_rejected() -> None:
    rule = as_authorize_rule()
    req_trunc = ToolRequest(
        tool_name="Read",
        action="read",
        resource="/path/to/some/file...<truncated>",
    )
    verdict = rule(req_trunc)
    assert verdict is not None
    assert verdict.decision == "deny"
    assert "truncated" in verdict.reason

    req_overlong = ToolRequest(
        tool_name="Read",
        action="read",
        resource="/workspace/" + "a" * 4100,
    )
    verdict_overlong = rule(req_overlong)
    assert verdict_overlong is not None
    assert verdict_overlong.decision == "deny"
