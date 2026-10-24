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
    Policy,
    PolicyError,
    ToolRequest,
    allow_tools,
    ask_tools,
    decide,
    deny_command_glob,
    deny_resource_glob,
    deny_tools,
    policy_from_dict,
    policy_from_file,
    policy_from_sequence,
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
        policy_from_dict({"rules": [{"type": "deny_resource", "pattern": ["/etc/shadow"]}]})


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
