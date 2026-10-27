"""Regression tests for the unattended-issue batch: #677, #678, #679, #686, #687.

Each test pins the contract the corresponding issue reported broken. #680 is
covered by its own repaired test
(``tests/server/test_read_timeout_130.py::test_cli_mentions_read_timeout_next_to_max_body``,
now invoked through ``python -m snagline`` like every other CLI test), so it
has no extra case here.
"""

from __future__ import annotations

import pytest

from snagline.config import Config
from snagline.detectors.silent_abort import SilentAbortDetector
from snagline.dlp_security import (
    DlpSecurityConfig,
    inspect_payload,
    redact_payload,
    redact_text,
    scan_text,
)
from snagline.path_security import (
    PathSecurityConfig,
    inspect_path,
)
from snagline.secret_security import (
    inspect_tool_input,
    redact_mapping,
    scan_secret_spans,
)

_GITHUB_TOKEN = "ghp_" + "a" * 36
_ANTHROPIC_KEY = "sk-ant-" + "b" * 20


# ---------------------------------------------------------------------------
# #677: redact_text silently dropped every secret match
# ---------------------------------------------------------------------------


def test_677_scan_text_gives_secret_matches_real_spans() -> None:
    """Secret hits must carry (start, end) offsets, not the 0/0 sentinel."""
    text = f"contact {_GITHUB_TOKEN} please"
    matches = [m for m in scan_text(text) if m.rule_id.startswith("secret:")]
    assert matches, "secret rule must run when include_secrets is True (the default)"
    for m in matches:
        assert 0 < m.start < m.end <= len(text)
        assert text[m.start : m.end] == _GITHUB_TOKEN


def test_677_redact_text_redacts_secrets_alongside_pii() -> None:
    text = f"Contact me at alice@example.com with key {_GITHUB_TOKEN}"
    redacted = redact_text(text)
    assert _GITHUB_TOKEN not in redacted
    assert "alice@example.com" not in redacted
    assert "REDACTED" in redacted


def test_677_redact_text_leaves_secrets_when_include_secrets_is_false() -> None:
    text = f"Contact me at alice@example.com with key {_GITHUB_TOKEN}"
    redacted = redact_text(text, DlpSecurityConfig(include_secrets=False))
    assert _GITHUB_TOKEN in redacted, "secrets must be untouched when disabled"
    assert "alice@example.com" not in redacted


def test_677_redact_payload_redacts_secrets_in_containers() -> None:
    payload = {"body": f"api_key={_ANTHROPIC_KEY}", "note": "harmless"}
    redacted = redact_payload(payload)
    assert _ANTHROPIC_KEY not in str(redacted)
    assert redacted["note"] == "harmless"


def test_677_nested_pii_inside_a_credential_span_does_not_corrupt_text() -> None:
    """A PII span nested inside a secret span must not double-replace."""
    text = "password: 'alice@example.com'"
    redacted = redact_text(text)
    # The outer credential assignment wins; the text is rebuilt exactly once.
    assert redacted == "[REDACTED:secret:generic_credential_assignment]"


def test_677_secret_spans_never_carry_the_raw_value() -> None:
    """The privacy invariant: findings carry a signature, never the secret."""
    for m in scan_text(f"token {_GITHUB_TOKEN}"):
        assert m.raw_value == "" or not m.rule_id.startswith("secret:")
    from snagline.secret_security import SecretMatch

    for sm in scan_secret_spans(f"token {_GITHUB_TOKEN}"):
        assert isinstance(sm, SecretMatch)


# ---------------------------------------------------------------------------
# #678: rule_path_traversal rejected valid absolute paths inside allowed_roots
# ---------------------------------------------------------------------------


def test_678_absolute_path_inside_allowed_roots_is_not_traversal() -> None:
    cfg = PathSecurityConfig(
        base_dir="/workspace", allowed_roots=("/workspace", "/tmp")
    )
    findings = inspect_path("/tmp/scratch.txt", cfg)
    assert [f.rule_id for f in findings] == []


def test_678_relative_path_escaping_base_dir_is_still_traversal() -> None:
    cfg = PathSecurityConfig(
        base_dir="/workspace", allowed_roots=("/workspace", "/tmp")
    )
    findings = inspect_path("../../etc/shadow", cfg)
    assert any(f.rule_id == "path_traversal" for f in findings)


def test_678_relative_path_inside_base_dir_is_clean() -> None:
    cfg = PathSecurityConfig(base_dir="/workspace", allowed_roots=None)
    findings = inspect_path("logs/app.txt", cfg)
    assert [f.rule_id for f in findings] == []


def test_678_absolute_path_outside_roots_is_flagged_by_the_roots_rule() -> None:
    """Containment of absolute paths belongs to allowed_roots, not base_dir."""
    cfg = PathSecurityConfig(base_dir="/workspace", allowed_roots=("/workspace",))
    findings = inspect_path("/tmp/scratch.txt", cfg)
    assert any(f.rule_id == "allowed_roots" for f in findings)
    assert not any(f.rule_id == "path_traversal" for f in findings)


# ---------------------------------------------------------------------------
# #679: cyclic / deeply nested payloads crashed with RecursionError
# ---------------------------------------------------------------------------


def _cyclic_payload() -> dict[str, object]:
    payload: dict[str, object] = {"ssn": "123-45-6789"}
    payload["self"] = payload
    payload["siblings"] = [payload, "plain"]
    return payload


def _deep_payload(depth: int = 200) -> dict[str, object]:
    root: dict[str, object] = {}
    node = root
    for _ in range(depth):
        nxt: dict[str, object] = {}
        node["nested"] = nxt
        node = nxt
    return root


def test_679_dlp_inspect_payload_survives_a_cycle() -> None:
    findings = inspect_payload(_cyclic_payload())
    assert [f.rule_id for f in findings] == ["us_ssn"]
    assert findings[0].path == "ssn"


def test_679_dlp_inspect_payload_survives_deep_nesting() -> None:
    assert inspect_payload(_deep_payload()) == ()


def test_679_dlp_redact_payload_survives_a_cycle() -> None:
    redacted = redact_payload(_cyclic_payload())
    assert "123-45-6789" not in str(redacted)
    assert "<cycle>" in str(redacted), "the cycle is truncated, not rebuilt"


def test_679_dlp_redact_payload_acyclic_round_trip() -> None:
    payload = {"a": ["123-45-6789"], "b": {"c": "alice@example.com"}}
    redacted = redact_payload(payload)
    assert redacted == {
        "a": ["[REDACTED:us_ssn]"],
        "b": {"c": "[REDACTED:email_address]"},
    }


def test_679_secret_inspect_tool_input_survives_a_cycle() -> None:
    cyclic: dict[str, object] = {"token": _GITHUB_TOKEN}
    cyclic["self"] = cyclic
    cyclic["siblings"] = [cyclic, 42]
    findings = inspect_tool_input(cyclic)
    assert [f.rule_id for f in findings] == ["github_token"]


def test_679_secret_redact_mapping_survives_a_cycle() -> None:
    cyclic: dict[str, object] = {"token": _GITHUB_TOKEN}
    cyclic["self"] = cyclic
    cyclic["siblings"] = [cyclic, 42]
    redacted = redact_mapping(cyclic)
    assert _GITHUB_TOKEN not in str(redacted)
    assert "<cycle>" in str(redacted)


def test_679_shared_acyclic_containers_are_still_scanned_at_every_path() -> None:
    """The visited set must cut cycles, not de-duplicate honest DAG sharing."""
    shared = {"token": _GITHUB_TOKEN}
    findings = inspect_payload({"one": shared, "two": shared})
    assert sorted(f.path for f in findings) == ["one.token", "two.token"]


# ---------------------------------------------------------------------------
# #686: SilentAbortDetector deleted its config and ignored the #578 field
# ---------------------------------------------------------------------------


def test_686_config_field_is_honoured() -> None:
    cfg = Config(silent_abort_output_action_types=frozenset({"tool_call"}))
    detector = SilentAbortDetector(config=cfg)
    assert detector.output_action_types == frozenset({"tool_call"})


def test_686_explicit_argument_still_wins_over_config() -> None:
    cfg = Config(silent_abort_output_action_types=frozenset({"tool_call"}))
    detector = SilentAbortDetector(
        output_action_types={"node_run", "agent_step"}, config=cfg
    )
    assert detector.output_action_types == frozenset({"node_run", "agent_step"})


def test_686_default_matches_the_config_default() -> None:
    detector = SilentAbortDetector()
    assert detector.output_action_types == Config().silent_abort_output_action_types
    assert detector.output_action_types == frozenset({"message", "plan_step"})


# ---------------------------------------------------------------------------
# #687: pytest lacked pythonpath, so a bare `pytest` failed to import
# ---------------------------------------------------------------------------


def test_687_pyproject_declares_the_import_roots() -> None:
    tomllib = pytest.importorskip("tomllib", reason="needs Python 3.11+ stdlib")
    from pathlib import Path

    pyproject = Path(__file__).resolve().parent.parent / "pyproject.toml"
    with pyproject.open("rb") as fh:
        data = tomllib.load(fh)
    pythonpath = data["tool"]["pytest"]["ini_options"].get("pythonpath")
    assert pythonpath is not None, "pythonpath must be configured (issue #687)"
    assert "src" in pythonpath and "." in pythonpath


def test_687_snagline_imports_from_the_local_src_tree() -> None:
    """The pythonpath entries put the checkout's own src/ first on sys.path."""
    import snagline

    assert "src" in snagline.__file__.replace("\\", "/"), snagline.__file__
