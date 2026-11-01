"""Regression tests for issue #676: content keys blinding the security rules.

``ToolRequest.from_tool_input`` used to replace the values of every
content-bearing key (``body``, ``message``, ``content``, ...) with
``"<content redacted>"`` before the request reached the rules. The DLP and
secret rules scan ``request.tool_input``, so an outbound POST carrying an SSN
in ``body`` -- exactly the traffic those rules exist to guard -- was reduced
to a marker and permitted without a verdict.

The fix splits the two jobs that markering conflated: ``tool_input`` now
retains bounded content for in-process rule inspection, and the no-leak
invariant moved to ``repr``, which is the surface that reaches logs.
"""

from __future__ import annotations

from snagline.authorize import Policy, ToolRequest
from snagline.dlp_security import as_authorize_rule as dlp_authorize_rule
from snagline.secret_security import (
    SecretSecurityConfig,
)
from snagline.secret_security import (
    as_authorize_rule as secret_authorize_rule,
)
from snagline.shell_security import ShellThreatConfig

# ---------------------------------------------------------------------------
# The reported bug: payload content reaches the rules
# ---------------------------------------------------------------------------


def test_676_dlp_rule_catches_a_secret_in_body() -> None:
    """The issue's standalone repro: an SSN in ``body`` must be denied."""
    req = ToolRequest.from_tool_input(
        "post_api",
        {"url": "https://api.internal/upload", "body": "My SSN is 123-45-6789"},
    )
    verdict = dlp_authorize_rule()(req)
    assert verdict is not None
    assert verdict.decision == "deny"
    assert "us_ssn" in verdict.reason


def test_676_secret_rule_catches_a_key_in_message() -> None:
    """A notification tool sending a credential in ``message`` is denied."""
    req = ToolRequest.from_tool_input(
        "notify_slack",
        {"channel": "#ops", "message": "deploy key: AKIAIOSFODNN7EXAMPLE"},
    )
    verdict = secret_authorize_rule()(req)
    assert verdict is not None
    assert verdict.decision == "deny"
    assert "aws_access_key" in verdict.reason


def test_676_dlp_rule_catches_a_card_in_data() -> None:
    """``data`` is a content key too -- an outbound card number is denied."""
    req = ToolRequest.from_tool_input(
        "post_api",
        {"url": "https://api.internal/charge", "data": "4111 1111 1111 1111"},
    )
    verdict = dlp_authorize_rule()(req)
    assert verdict is not None
    assert verdict.decision == "deny"
    assert "credit_card" in verdict.reason


def test_676_nested_content_is_inspected_too() -> None:
    """Content under a structural key survives capped, so it is scanned."""
    req = ToolRequest.from_tool_input(
        "SomeNewTool",
        {"notebook": {"content": "ssn 123-45-6789"}},
    )
    verdict = dlp_authorize_rule()(req)
    assert verdict is not None
    assert verdict.decision == "deny"
    assert "us_ssn" in verdict.reason


def test_676_benign_payload_still_has_no_verdict() -> None:
    """Retaining content does not make the rules fire on clean traffic."""
    req = ToolRequest.from_tool_input(
        "post_api", {"url": "https://api.internal/upload", "body": "hello world"}
    )
    verdict = dlp_authorize_rule()(req)
    assert verdict is None


def test_676_policy_decision_denies_the_leaky_request() -> None:
    """End to end through Policy.decide, the shape hosts actually drive."""
    policy = Policy(rules=(dlp_authorize_rule(), secret_authorize_rule()))
    leaky = ToolRequest.from_tool_input(
        "post_api",
        {"url": "https://api.internal/upload", "body": "My SSN is 123-45-6789"},
    )
    assert policy.decide(leaky).decision == "deny"


# ---------------------------------------------------------------------------
# The privacy invariant moves to repr, it does not disappear
# ---------------------------------------------------------------------------


def test_676_repr_never_prints_content_keys() -> None:
    secret = "AWS_SECRET_ACCESS_KEY=hunter2"
    req = ToolRequest.from_tool_input("Write", {"file_path": "/a", "content": secret})
    assert secret not in repr(req)
    assert "<content redacted>" in repr(req)


def test_676_repr_never_prints_nested_content() -> None:
    body = "TOP SECRET FILE BODY"
    req = ToolRequest.from_tool_input("SomeNewTool", {"notebook": {"content": body}})
    assert body not in repr(req)


def test_676_repr_of_a_truncated_content_value_is_marked() -> None:
    """A content string past the cap is truncated in tool_input AND repr."""
    huge = "x" * 5000
    req = ToolRequest.from_tool_input("post_api", {"body": huge})
    retained = req.tool_input["body"]
    assert isinstance(retained, str)
    assert retained.endswith("...<truncated>")
    assert len(retained) < len(huge)
    assert huge not in repr(req)


def test_676_content_values_are_bounded_in_memory() -> None:
    """Retaining content must not retain an unbounded body."""
    huge = "y" * 100_000
    req = ToolRequest.from_tool_input("post_api", {"body": huge})
    retained = req.tool_input["body"]
    assert isinstance(retained, str)
    assert len(retained) <= 4096 + len("...<truncated>")


def test_676_deep_content_collapses_to_the_marker() -> None:
    """Past the depth limit a container still collapses, not walks forever."""
    deep: dict = {"leaf": "SECRET-DEEP"}
    for _ in range(6):
        deep = {"level": deep}
    req = ToolRequest.from_tool_input("SomeNewTool", {"payload": deep})
    assert "SECRET-DEEP" not in repr(req)


def test_676_resource_extraction_is_unaffected() -> None:
    """url/path/command still become the resource, capped as before."""
    req = ToolRequest.from_tool_input(
        "post_api", {"url": "https://api.internal/upload", "body": "text"}
    )
    assert req.resource == "https://api.internal/upload"
    assert req.action == "unknown"


# ---------------------------------------------------------------------------
# Rules that never cared about content are untouched
# ---------------------------------------------------------------------------


def test_676_shell_rule_still_acts_on_the_command() -> None:
    """``command`` is not a content key: the shell rule verdicts as before."""
    from snagline.shell_security import as_authorize_rule as shell_rule

    req = ToolRequest.from_tool_input("Bash", {"command": "rm -rf /"})
    verdict = shell_rule(ShellThreatConfig())(req)
    assert verdict is not None
    assert verdict.decision == "deny"


def test_676_secret_config_decision_is_respected() -> None:
    """A non-deny decision is still honoured on inspected content."""
    cfg = SecretSecurityConfig(decision="ask")
    req = ToolRequest.from_tool_input(
        "notify_slack",
        {"channel": "#ops", "message": "key AKIAIOSFODNN7EXAMPLE"},
    )
    verdict = secret_authorize_rule(cfg)(req)
    assert verdict is not None
    assert verdict.decision == "ask"
