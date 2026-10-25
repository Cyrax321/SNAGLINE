"""Pre-execution authorization gate (``snagline.authorize``).

Fail-open is the contract, so most of these tests exist to prove the gate
CANNOT wedge a host by accident: an unset policy allows, a crashing rule
allows, an unknown tool allows, and a malformed *request* allows. The strict
postures are tested alongside them so the opt-ins are known to work rather
than assumed.

The precedence tests are the load-bearing ones. Strictest-verdict-wins is
what makes an allowlist above a denylist harmless; if that ever regresses to
first-match-wins, a policy that reads as safe silently stops being safe.
"""

from __future__ import annotations

import json

import pytest

from snagline.authorize import (
    PROFILE_FAIL_CLOSED,
    Policy,
    PolicyError,
    ToolRequest,
    allow_agent_tools,
    allow_tools,
    ask_tools,
    decide,
    deny_command_glob,
    deny_resource_glob,
    deny_tools,
    policy_from_dict,
    policy_from_file,
    policy_from_sequence,
    security_policy,
)

# --- the fail-open guarantee ----------------------------------------------


def test_empty_policy_allows() -> None:
    """The default construction permits everything, which is the whole point."""
    assert Policy().decide(ToolRequest.from_tool_input("Bash", {"command": "rm -rf /"}))


def test_unset_policy_decision_is_allow() -> None:
    result = Policy().decide(ToolRequest.from_tool_input("Write", {"file_path": "/x"}))
    assert result.decision == "allow"
    assert result.allowed


def test_a_crashing_rule_cannot_wedge_the_host() -> None:
    def boom(request: ToolRequest) -> None:
        raise RuntimeError("rule is broken")

    result = Policy(rules=(boom,)).decide(ToolRequest.from_tool_input("Read", {}))
    assert result.decision == "allow", "a crashing rule must fall back to the default"
    assert "RuntimeError" in result.reason


def test_rule_error_decision_opts_into_strictness() -> None:
    def boom(request: ToolRequest) -> None:
        raise RuntimeError("rule is broken")

    result = Policy(rules=(boom,), rule_error_decision="deny").decide(
        ToolRequest.from_tool_input("Read", {})
    )
    assert result.decision == "deny"


def test_malformed_tool_input_does_not_raise() -> None:
    """A host sending the wrong shape is not a reason to crash the gate."""
    malformed: tuple[object, ...] = (None, [], "a string", 7)
    for bad in malformed:
        request = ToolRequest.from_tool_input("Read", bad)  # type: ignore[arg-type]
        assert Policy().decide(request).decision == "allow"


def test_result_is_truthy_only_when_no_approval_is_needed() -> None:
    """``ask`` must be falsy, or ``if decide(req): run()`` skips the approval."""
    allowed = Policy().decide(ToolRequest.from_tool_input("Read", {}))
    asked = Policy(default_decision="ask").decide(
        ToolRequest.from_tool_input("Read", {})
    )
    denied = Policy(default_decision="deny").decide(
        ToolRequest.from_tool_input("Read", {})
    )
    assert bool(allowed) is True
    assert bool(asked) is False, "ask is not a proceed"
    assert bool(denied) is False
    assert asked.allowed is False


# --- strictness is opt-in --------------------------------------------------


def test_allowlist_denies_everything_unnamed() -> None:
    policy = Policy(rules=(allow_tools("Read", "Grep"),), default_decision="deny")
    assert policy.decide(ToolRequest.from_tool_input("Read", {})).decision == "allow"
    assert policy.decide(ToolRequest.from_tool_input("Glob", {})).decision == "deny"


def test_default_deny_reason_names_the_default() -> None:
    """An operator seeing a deny they did not write needs the 'why' inline."""
    policy = Policy(rules=(allow_tools("Read"),), default_decision="deny")
    result = policy.decide(ToolRequest.from_tool_input("Bash", {}))
    assert result.decision == "deny"
    assert "default_decision" in result.reason
    assert "deny" in result.reason


def test_unknown_tool_decision_overrides_rules() -> None:
    policy = Policy(rules=(allow_tools("Read"),), unknown_tool_decision="ask")
    result = policy.decide(ToolRequest.from_tool_input("SomeFutureTool", {}))
    assert result.decision == "ask"
    assert "unknown" in result.reason


def test_explicit_rule_still_applies_to_an_unknown_tool() -> None:
    """Our ignorance of the tool set must not outrank an explicit rule."""
    policy = Policy(rules=(deny_tools("SomeFutureTool"),))
    assert (
        policy.decide(ToolRequest.from_tool_input("SomeFutureTool", {})).decision
        == "deny"
    )


def test_unknown_tool_defaults_to_allow_without_the_opt_in() -> None:
    policy = Policy()
    assert (
        policy.decide(ToolRequest.from_tool_input("SomeFutureTool", {})).decision
        == "allow"
    )


# --- strictest verdict wins, regardless of rule order ---------------------


def test_allow_before_deny_still_denies() -> None:
    policy = Policy(rules=(allow_tools("Write"), deny_tools("Write")))
    assert policy.decide(ToolRequest.from_tool_input("Write", {})).decision == "deny"


def test_deny_before_allow_still_denies() -> None:
    policy = Policy(rules=(deny_tools("Write"), allow_tools("Write")))
    assert policy.decide(ToolRequest.from_tool_input("Write", {})).decision == "deny"


def test_rule_order_cannot_change_the_outcome() -> None:
    """The regression this whole design exists to prevent (issue-worthy)."""
    rules = (allow_tools("Write"), ask_tools("Write"), deny_tools("Write"))
    forward = Policy(rules=rules).decide(ToolRequest.from_tool_input("Write", {}))
    backward = Policy(rules=tuple(reversed(rules))).decide(
        ToolRequest.from_tool_input("Write", {})
    )
    assert forward.decision == backward.decision == "deny"


def test_ask_wins_over_allow() -> None:
    policy = Policy(rules=(allow_tools("Write"), ask_tools("Write")))
    assert policy.decide(ToolRequest.from_tool_input("Write", {})).decision == "ask"


def test_matched_rules_reports_every_opinion_not_just_the_winner() -> None:
    policy = Policy(rules=(allow_tools("Write"), deny_tools("Write")))
    result = policy.decide(ToolRequest.from_tool_input("Write", {}))
    assert set(result.matched_rules) == {"allow_tools", "deny_tools"}
    assert result.matched_rules[0] == "deny_tools", "strictest reported first"


# --- request normalization -------------------------------------------------


@pytest.mark.parametrize(
    ("tool", "tool_input", "action", "resource"),
    [
        ("Bash", {"command": "ls -la"}, "execute", "ls -la"),
        ("Read", {"file_path": "/etc/passwd"}, "read", "/etc/passwd"),
        ("Write", {"file_path": "/tmp/x"}, "write", "/tmp/x"),
        ("WebFetch", {"url": "https://x"}, "network", "https://x"),
        ("Mystery", {"path": "/a"}, "unknown", "/a"),
        ("Mystery", {}, "unknown", None),
    ],
)
def test_from_tool_input_extracts_action_and_resource(
    tool: str, tool_input: dict, action: str, resource: str | None
) -> None:
    request = ToolRequest.from_tool_input(tool, tool_input)
    assert request.action == action
    assert request.resource == resource


def test_non_string_resource_is_treated_as_absent() -> None:
    """Stringifying a guess would match a policy against something unasked."""
    request = ToolRequest.from_tool_input("Read", {"file_path": ["a", "b"]})
    assert request.resource is None


def test_signature_is_stable_and_content_free() -> None:
    """Two identical requests agree; the digest reveals no resource."""
    one = ToolRequest.from_tool_input("Read", {"file_path": "/secret/path"})
    two = ToolRequest.from_tool_input("Read", {"file_path": "/secret/path"})
    other = ToolRequest.from_tool_input("Read", {"file_path": "/other"})
    assert one.signature == two.signature
    assert one.signature != other.signature
    assert "/secret/path" not in one.signature
    assert len(one.signature) == 64


# --- resource and command rules --------------------------------------------


def test_deny_resource_glob_matches_paths() -> None:
    policy = Policy(rules=(deny_resource_glob("*.pem", "/etc/ssh/*"),))
    assert (
        policy.decide(
            ToolRequest.from_tool_input("Read", {"file_path": "/k.pem"})
        ).decision
        == "deny"
    )
    assert (
        policy.decide(
            ToolRequest.from_tool_input("Read", {"file_path": "/etc/ssh/id_rsa"})
        ).decision
        == "deny"
    )
    assert (
        policy.decide(
            ToolRequest.from_tool_input("Read", {"file_path": "/tmp/ok"})
        ).decision
        == "allow"
    )


def test_deny_resource_glob_ignores_a_request_with_no_resource() -> None:
    policy = Policy(rules=(deny_resource_glob("*"),))
    assert policy.decide(ToolRequest.from_tool_input("Mystery", {})).decision == "allow"


def test_deny_command_glob_only_fires_on_execute() -> None:
    policy = Policy(rules=(deny_command_glob("rm -rf*"),))
    assert (
        policy.decide(
            ToolRequest.from_tool_input("Bash", {"command": "rm -rf /"})
        ).decision
        == "deny"
    )
    read = policy.decide(ToolRequest.from_tool_input("Read", {"file_path": "rm -rf /"}))
    assert read.decision == "allow", "a path is not a command"


def test_non_matching_rule_has_no_opinion() -> None:
    policy = Policy(rules=(deny_tools("Bash"),))
    result = policy.decide(ToolRequest.from_tool_input("Read", {}))
    assert result.decision == "allow"
    assert result.matched_rules == ()


# --- declarative documents -------------------------------------------------


def test_policy_from_dict_round_trips_a_full_document() -> None:
    policy = policy_from_dict(
        {
            "description": "read-only by default",
            "default_decision": "allow",
            "rules": [
                {"type": "allow_tools", "tools": ["Read", "Grep"]},
                {"type": "deny_tools", "tools": ["Write"]},
                {"type": "deny_resource", "patterns": ["*.pem"]},
                {"type": "ask_tools", "tools": ["WebFetch"]},
            ],
        }
    )
    assert policy.decide(ToolRequest.from_tool_input("Read", {})).decision == "allow"
    assert policy.decide(ToolRequest.from_tool_input("Write", {})).decision == "deny"
    assert (
        policy.decide(
            ToolRequest.from_tool_input("Read", {"file_path": "/a.pem"})
        ).decision
        == "deny"
    )
    assert (
        policy.decide(ToolRequest.from_tool_input("WebFetch", {"url": "u"})).decision
        == "ask"
    )
    assert policy.decide(ToolRequest.from_tool_input("Glob", {})).decision == "allow"


def test_policy_from_dict_rejects_unknown_keys() -> None:
    with pytest.raises(PolicyError, match="unknown policy keys"):
        policy_from_dict({"default_decision": "allow", "deny": ["Write"]})


def test_policy_from_dict_rejects_an_unknown_rule_type() -> None:
    with pytest.raises(PolicyError, match="not a known rule type"):
        policy_from_dict({"rules": [{"type": "deny_everything"}]})


def test_policy_from_dict_rejects_a_bad_decision() -> None:
    with pytest.raises(PolicyError, match="is not a decision"):
        policy_from_dict({"default_decision": "maybe"})


def test_policy_from_dict_rejects_a_malformed_rule_list() -> None:
    with pytest.raises(PolicyError, match="rules must be a list"):
        policy_from_dict({"rules": {"type": "deny_tools"}})
    with pytest.raises(PolicyError, match="list of strings"):
        policy_from_dict({"rules": [{"type": "deny_tools", "tools": [1, 2]}]})


def test_resource_glob_is_case_sensitive_on_every_os() -> None:
    """The verdict must be a property of the policy, not of the host OS."""
    policy = Policy(rules=(deny_resource_glob("*.pem"),))
    upper = ToolRequest.from_tool_input("Read", {"file_path": "/secrets/CERT.PEM"})
    lower = ToolRequest.from_tool_input("Read", {"file_path": "/secrets/cert.pem"})
    assert policy.decide(upper).decision == "allow"
    assert policy.decide(lower).decision == "deny"


def test_command_glob_is_case_sensitive_on_every_os() -> None:
    policy = Policy(rules=(deny_command_glob("rm -rf*"),))
    upper = ToolRequest.from_tool_input("Bash", {"command": "RM -RF /"})
    lower = ToolRequest.from_tool_input("Bash", {"command": "rm -rf /"})
    assert policy.decide(upper).decision == "allow"
    assert policy.decide(lower).decision == "deny"


def test_policy_file_round_trip_json(tmp_path) -> None:
    path = tmp_path / "policy.json"
    path.write_text(
        json.dumps(
            {
                "default_decision": "deny",
                "rules": [{"type": "allow_tools", "tools": ["Read"]}],
            }
        ),
        encoding="utf-8",
    )
    policy = policy_from_file(str(path))
    assert policy.decide(ToolRequest.from_tool_input("Read", {})).decision == "allow"
    assert policy.decide(ToolRequest.from_tool_input("Bash", {})).decision == "deny"


def test_policy_file_rejects_non_finite_floats(tmp_path) -> None:
    """Same guard as Config.load_file: a value no comparison can order."""
    path = tmp_path / "policy.json"
    path.write_text(
        '{"default_decision": "allow", "weight": Infinity}', encoding="utf-8"
    )
    with pytest.raises(PolicyError, match="non-finite"):
        policy_from_file(str(path))


def test_policy_from_sequence_builds_from_code() -> None:
    policy = policy_from_sequence([deny_tools("Bash")])
    assert policy.decide(ToolRequest.from_tool_input("Bash", {})).decision == "deny"


def test_module_level_decide_shorthand() -> None:
    result = decide(
        ToolRequest.from_tool_input("Bash", {}), Policy(rules=(deny_tools("Bash"),))
    )
    assert result.decision == "deny"


def test_policy_file_round_trip_toml(tmp_path) -> None:
    try:
        import tomllib  # noqa: F401
    except ModuleNotFoundError:  # pragma: no cover - exercised only on <3.11
        pytest.skip("tomllib requires Python 3.11+")
    path = tmp_path / "policy.toml"
    path.write_text(
        'default_decision = "deny"\n'
        '[[rules]]\ntype = "allow_tools"\ntools = ["Read"]\n',
        encoding="utf-8",
    )
    policy = policy_from_file(str(path))
    assert policy.decide(ToolRequest.from_tool_input("Read", {})).decision == "allow"
    assert policy.decide(ToolRequest.from_tool_input("Write", {})).decision == "deny"


def test_policy_file_rejects_a_non_mapping_document(tmp_path) -> None:
    """A policy that parsed to a list permits everything silently otherwise."""
    path = tmp_path / "policy.json"
    path.write_text("[1, 2, 3]", encoding="utf-8")
    with pytest.raises(PolicyError, match="mapping at the top level"):
        policy_from_file(str(path))


def test_policy_from_dict_rejects_a_non_object_rule() -> None:
    with pytest.raises(PolicyError, match="rules\\[0\\] must be an object"):
        policy_from_dict({"rules": ["deny_tools"]})


def test_rule_with_typo_d_key_raises_instead_of_building_an_empty_rule() -> None:
    """A typo'd key used to be ignored, leaving a rule that matches nothing."""
    with pytest.raises(PolicyError, match=r"rules\[0\]\.pattern"):
        policy_from_dict(
            {"rules": [{"type": "deny_resource", "pattern": ["/etc/shadow"]}]}
        )


def test_rule_with_typo_d_tools_key_raises() -> None:
    with pytest.raises(PolicyError, match=r"rules\[0\]\.tool"):
        policy_from_dict({"rules": [{"type": "deny_tools", "tool": ["Bash"]}]})


def test_rule_with_unknown_extra_key_raises() -> None:
    with pytest.raises(PolicyError, match=r"rules\[0\]\.sevrity"):
        policy_from_dict(
            {"rules": [{"type": "deny_tools", "tools": ["Bash"], "sevrity": "high"}]}
        )


def test_rule_without_its_list_raises() -> None:
    with pytest.raises(PolicyError, match=r"rules\[0\]\.tools is missing"):
        policy_from_dict({"rules": [{"type": "deny_tools"}]})


def test_rule_with_explicit_empty_list_is_still_allowed() -> None:
    policy = policy_from_dict({"rules": [{"type": "deny_tools", "tools": []}]})
    assert len(policy.rules) == 1


def test_shell_threats_rule_with_unknown_key_raises() -> None:
    with pytest.raises(PolicyError, match=r"rules\[0\]\.decisio"):
        policy_from_dict({"rules": [{"type": "shell_threats", "decisio": "deny"}]})


def test_unknown_tool_decision_survives_a_rule_that_opines() -> None:
    """Unknown-tool handling must not be reachable only via the empty path."""
    policy = Policy(rules=(deny_resource_glob("*"),), unknown_tool_decision="deny")
    result = policy.decide(ToolRequest.from_tool_input("Mystery", {}))
    assert result.decision == "deny"
    assert "unknown_tool_decision" in result.reason


# --- Grep/Glob target the searched path, not the pattern ---------------------


def test_grep_resource_is_the_searched_path_not_the_pattern() -> None:
    """Regression: a path rule must see the directory, not the expression.

    Grep sends both ``path`` and ``pattern``. Matching the pattern let a search
    scoped to a protected directory walk past a rule denying that directory.
    """
    policy = Policy(rules=(deny_resource_glob("/etc/ssh*"),))
    result = policy.decide(
        ToolRequest.from_tool_input("Grep", {"pattern": "id_rsa", "path": "/etc/ssh"})
    )
    assert result.decision == "deny"


def test_grep_falls_back_to_pattern_when_no_path_is_given() -> None:
    policy = Policy(rules=(deny_resource_glob("*.pem"),))
    request = ToolRequest.from_tool_input("Grep", {"pattern": "x.pem"})
    assert policy.decide(request).decision == "deny"


def test_glob_resource_is_the_path_not_the_glob_pattern() -> None:
    policy = Policy(rules=(deny_resource_glob("/workspace/secrets*"),))
    result = policy.decide(
        ToolRequest.from_tool_input(
            "Glob", {"pattern": "**/*.env", "path": "/workspace/secrets"}
        )
    )
    assert result.decision == "deny"


# --- content redaction ------------------------------------------------------


def test_content_bearing_input_is_not_retained() -> None:
    """``Write.content`` is a file body; it must not sit in the request."""
    secret = "AWS_SECRET_ACCESS_KEY=hunter2"
    request = ToolRequest.from_tool_input(
        "Write", {"file_path": "/a", "content": secret}
    )
    assert secret not in repr(request)
    assert secret not in str(request.tool_input)
    assert request.tool_input["file_path"] == "/a", "structure is kept"
    assert "content" in request.tool_input, "the key still answers 'was it supplied?'"
    assert request.tool_input["content"] != secret


def test_edit_input_is_redacted_but_the_path_survives() -> None:
    request = ToolRequest.from_tool_input(
        "Edit", {"file_path": "/a", "new_string": "top secret"}
    )
    assert request.resource == "/a"
    assert "top secret" not in repr(request)


def test_bash_command_is_not_treated_as_content() -> None:
    """``command`` is the resource a policy matches; redacting it would be useless."""
    request = ToolRequest.from_tool_input("Bash", {"command": "rm -rf /etc"})
    assert request.resource == "rm -rf /etc"
    assert request.tool_input["command"] == "rm -rf /etc"


def test_nested_content_is_redacted() -> None:
    """Content under a structural key must not survive either."""
    request = ToolRequest.from_tool_input(
        "SomeNewTool",
        {
            "notebook": {"content": "TOP SECRET FILE BODY"},
            "payload": {"new_string": "ANOTHER SECRET"},
        },
    )
    assert "TOP SECRET FILE BODY" not in repr(request)
    assert "ANOTHER SECRET" not in repr(request)
    assert request.tool_input["notebook"]["content"] == "<content redacted>"
    assert request.tool_input["payload"]["new_string"] == "<content redacted>"


def test_content_nested_in_a_list_is_redacted() -> None:
    """A MultiEdit-style list of edits keeps no bodies."""
    request = ToolRequest.from_tool_input(
        "Scratchpad",
        {"changes": [{"old_string": "SECRET-A", "new_string": "SECRET-B"}]},
    )
    assert "SECRET-A" not in repr(request)
    assert "SECRET-B" not in repr(request)
    edits = request.tool_input["changes"]
    assert edits[0]["old_string"] == "<content redacted>"


def test_oversized_nested_strings_are_truncated() -> None:
    huge = "b" * 10_000
    request = ToolRequest.from_tool_input("SomeNewTool", {"payload": {"query": huge}})
    retained = request.tool_input["payload"]["query"]
    assert len(retained) < len(huge)
    assert retained.endswith("<truncated>")


def test_values_below_the_depth_limit_collapse_to_the_marker() -> None:
    """Deeper than ``_REDACT_DEPTH`` is dropped, not walked forever."""
    deep: dict = {"leaf": "SECRET-DEEP"}
    for _ in range(6):
        deep = {"level": deep}
    request = ToolRequest.from_tool_input("SomeNewTool", {"payload": deep})
    assert "SECRET-DEEP" not in repr(request)


def test_oversized_structural_values_are_truncated() -> None:
    """The resource is read from the redacted copy, so the cap applies to it.

    Reading the resource out of the raw mapping would leave it unbounded while
    ``tool_input`` was truncated -- truncation bypassed by the exact field a
    policy matches on.
    """
    huge = "a" * 10_000
    request = ToolRequest.from_tool_input("Read", {"file_path": huge})
    resource = request.resource
    assert resource is not None
    assert len(resource) < len(huge)
    assert resource.endswith("<truncated>")
    assert request.tool_input["file_path"] == resource


# --- config errors must not fail open ---------------------------------------


def test_explicit_null_decision_is_rejected_not_read_as_absent() -> None:
    """A ``"default_decision": null`` must not silently become ``allow``."""
    with pytest.raises(PolicyError, match="is null"):
        policy_from_dict({"default_decision": None})
    with pytest.raises(PolicyError, match="is null"):
        policy_from_dict({"rules": [], "rule_error_decision": None})


def test_absent_decision_key_is_fine() -> None:
    assert policy_from_dict({"rules": []}).default_decision == "allow"


def test_duplicate_json_keys_are_rejected(tmp_path) -> None:
    """A duplicate ``rules`` key would otherwise erase a deny list silently."""
    path = tmp_path / "policy.json"
    path.write_text(
        '{"rules": [{"type": "deny_tools", "tools": ["Bash"]}], "rules": []}',
        encoding="utf-8",
    )
    with pytest.raises(PolicyError, match="duplicate key"):
        policy_from_file(str(path))


def test_duplicate_keys_inside_a_rule_are_rejected(tmp_path) -> None:
    path = tmp_path / "policy.json"
    path.write_text(
        '{"rules": [{"type": "deny_tools", "tools": ["A"], "tools": []}]}',
        encoding="utf-8",
    )
    with pytest.raises(PolicyError, match="duplicate key"):
        policy_from_file(str(path))


# --- fail-closed security profile and per-agent allowlists (issue #637) ---


def test_security_profile_fails_closed_by_default() -> None:
    p = Policy(profile=PROFILE_FAIL_CLOSED)
    assert p.default_decision == "deny"
    assert p.rule_error_decision == "deny"
    assert p.unknown_tool_decision == "deny"

    # Unmatched known tool request denies
    verdict = p.decide(ToolRequest.from_tool_input("Read", {"file_path": "/tmp/test"}))
    assert verdict.decision == "deny"
    assert not verdict.allowed
    assert "no rule matched; policy default_decision is 'deny'" in verdict.reason


def test_security_profile_denies_unknown_tools() -> None:
    p = Policy(profile=PROFILE_FAIL_CLOSED)
    verdict = p.decide(ToolRequest.from_tool_input("CompletelyUnknownTool", {}))
    assert verdict.decision == "deny"
    assert not verdict.allowed
    assert "tool 'CompletelyUnknownTool' is unknown" in verdict.reason


def test_security_profile_denies_on_rule_error() -> None:
    def crashing_rule(req: ToolRequest):
        raise RuntimeError("evaluation error")

    p = Policy(profile=PROFILE_FAIL_CLOSED, rules=(crashing_rule,))
    verdict = p.decide(ToolRequest.from_tool_input("Read", {"file_path": "/safe"}))
    assert verdict.decision == "deny"
    assert not verdict.allowed
    assert "rule 0 raised RuntimeError: evaluation error" in verdict.reason


def test_security_policy_helper_factory() -> None:
    p = security_policy(
        allowed_tools=["Read", "Grep"],
        rules=(deny_resource_glob("*.pem"),),
    )
    assert p.profile == PROFILE_FAIL_CLOSED
    assert p.default_decision == "deny"
    assert p.rule_error_decision == "deny"
    assert p.unknown_tool_decision == "deny"

    # Allowed tool on non-denied resource allows
    assert p.decide(
        ToolRequest.from_tool_input("Read", {"file_path": "/tmp/data.txt"})
    ).allowed
    # Allowed tool on denied resource denies
    assert not p.decide(
        ToolRequest.from_tool_input("Read", {"file_path": "/tmp/secret.pem"})
    ).allowed
    # Unallowed tool denies
    assert not p.decide(
        ToolRequest.from_tool_input("Bash", {"command": "echo 1"})
    ).allowed


def test_per_agent_allowlists_in_code() -> None:
    rule = allow_agent_tools(
        {
            "researcher": ["Read", "WebSearch", "Grep"],
            "engineer": ["Read", "Write", "Bash"],
        }
    )
    p = Policy(rules=(rule,), default_decision="deny")

    # Researcher running Read -> allowed
    req_r_read = ToolRequest.from_tool_input(
        "Read", {"file_path": "/a"}, agent_id="researcher"
    )
    assert p.decide(req_r_read).allowed

    # Researcher running Bash -> denied
    req_r_bash = ToolRequest.from_tool_input(
        "Bash", {"command": "ls"}, agent_id="researcher"
    )
    res_r_bash = p.decide(req_r_bash)
    assert not res_r_bash.allowed
    assert "tool 'Bash' is not allowed for agent 'researcher'" in res_r_bash.reason

    # Engineer running Bash -> allowed
    req_e_bash = ToolRequest.from_tool_input(
        "Bash", {"command": "ls"}, agent_id="engineer"
    )
    assert p.decide(req_e_bash).allowed

    # Unconfigured agent running Read -> denied
    req_guest = ToolRequest.from_tool_input(
        "Read", {"file_path": "/a"}, agent_id="guest"
    )
    res_guest = p.decide(req_guest)
    assert not res_guest.allowed
    assert "agent 'guest' has no configured tool allowlist" in res_guest.reason

    # Request without agent_id passes through to default
    req_no_agent = ToolRequest.from_tool_input("Read", {"file_path": "/a"})
    res_no_agent = p.decide(req_no_agent)
    assert not res_no_agent.allowed
    assert "no rule matched" in res_no_agent.reason


def test_agent_id_parsed_from_tool_input_dict() -> None:
    req = ToolRequest.from_tool_input(
        "Read",
        {"agent_id": "bot_42", "file_path": "/etc/hosts"},
    )
    assert req.agent_id == "bot_42"
    assert req.resource == "/etc/hosts"
    plain_req = ToolRequest.from_tool_input("Read", {"file_path": "/etc/hosts"})
    assert req.signature != plain_req.signature
    assert len(req.signature) == 64


def test_declarative_security_profile_and_agent_allowlists() -> None:
    doc = {
        "profile": "fail_closed",
        "agent_allowlists": {"auditor": ["Read", "Glob"]},
        "rules": [{"type": "deny_resource", "patterns": ["/etc/*"]}],
    }
    p = policy_from_dict(doc)
    assert p.profile == "fail_closed"
    assert p.default_decision == "deny"
    assert p.unknown_tool_decision == "deny"
    assert p.rule_error_decision == "deny"

    # Auditor accessing /tmp/foo -> allowed
    assert p.decide(
        ToolRequest.from_tool_input(
            "Read", {"file_path": "/tmp/foo"}, agent_id="auditor"
        )
    ).allowed
    # Auditor accessing /etc/shadow -> denied (deny_resource wins over allow)
    assert not p.decide(
        ToolRequest.from_tool_input(
            "Read", {"file_path": "/etc/shadow"}, agent_id="auditor"
        )
    ).allowed
    # Auditor running Write -> denied
    assert not p.decide(
        ToolRequest.from_tool_input(
            "Write", {"file_path": "/tmp/foo"}, agent_id="auditor"
        )
    ).allowed


def test_declarative_rule_type_agent_allowlists() -> None:
    doc = {
        "rules": [
            {
                "type": "agent_allowlists",
                "allowlists": {"tester": ["Read"]},
            }
        ],
        "default_decision": "deny",
    }
    p = policy_from_dict(doc)
    assert p.decide(ToolRequest.from_tool_input("Read", {}, agent_id="tester")).allowed
    assert not p.decide(
        ToolRequest.from_tool_input("Bash", {}, agent_id="tester")
    ).allowed


def test_invalid_security_profile_and_allowlists_fail_at_load_time() -> None:
    # Invalid profile name
    with pytest.raises(PolicyError, match="is not a valid security profile"):
        policy_from_dict({"profile": "bogus"})

    # Profile cannot be null
    with pytest.raises(PolicyError, match="profile is null"):
        policy_from_dict({"profile": None})

    # agent_allowlists must be an object
    with pytest.raises(PolicyError, match="agent_allowlists must be an object"):
        policy_from_dict({"agent_allowlists": ["Read"]})

    # agent_allowlists cannot be null
    with pytest.raises(PolicyError, match="agent_allowlists is null"):
        policy_from_dict({"agent_allowlists": None})

    # Invalid agent id
    with pytest.raises(PolicyError, match="agent_id '' must be a non-empty string"):
        policy_from_dict({"agent_allowlists": {"": ["Read"]}})

    # Invalid tools type
    with pytest.raises(PolicyError, match="tools for agent 'bot' must be a list"):
        policy_from_dict({"agent_allowlists": {"bot": "Read"}})

    # Empty rule allowlists
    with pytest.raises(PolicyError, match=r"rules\[0\]\.allowlists is missing"):
        policy_from_dict({"rules": [{"type": "agent_allowlists"}]})

    # Policy construction invalid profile
    with pytest.raises(ValueError, match="is not a valid security profile"):
        Policy(profile="invalid")


def test_monitor_pipeline_remains_fail_open_independent_of_authorize() -> None:
    """The monitoring pipeline never imports or enforces authorize policies."""
    from snagline import Monitor, StepEvent

    m = Monitor.default()
    # Sending unexpected step event never raises or fails closed
    event = StepEvent(
        step_id="step_1",
        episode_id="ep_1",
        timestamp=1000.0,
        action_type="tool_call",
        action_signature="sig_test",
        tool_name="UnknownTool",
    )
    m.ingest(event)
    assert m.metrics()["events_ingested"] == 1


def test_fail_closed_profile_explicit_default_decision_respected() -> None:
    """When default_decision is explicitly provided with fail_closed, it is respected."""
    pol = Policy(profile="fail_closed", default_decision="allow")
    assert pol.default_decision == "allow"
    res = pol.decide(ToolRequest(tool_name="Read", action="read", resource="safe.txt"))
    assert res.decision == "allow"
    # Unnamed error/unknown decisions remain fail-closed
    assert pol.rule_error_decision == "deny"
    assert pol.unknown_tool_decision == "deny"
