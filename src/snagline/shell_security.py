"""Deterministic shell command threat detection (issue #638).

Inspects a command *string* before execution and returns structured findings.
It never runs the command, never shells out, and never changes Monitor
runtime behaviour. Findings are meant to feed ``snagline.authorize.Policy``
so an operator can map them to allow / ask / deny.

Design constraints
------------------
* Deterministic: same command + config always yields the same findings.
* Modular: each rule is an independently testable callable.
* Privacy: findings carry a one-way ``command_signature``, not the raw
  command text. Reasons name the rule and a short pattern class only.
* Not a shell parser: see ``LIMITATIONS`` below and docs/SHELL_SECURITY.md.

LIMITATIONS / known bypass classes
----------------------------------
False negatives (commands that may slip through):
* Nested / mismatched quotes that hide a token from simple regex.
* Multi-stage indirection (``x=rm; $x -rf /``) after assignment.
* Encoding layers beyond the common ``base64 -d | sh`` shape
  (xxd, od, printf octal chains, custom decoders).
* Unicode lookalikes / homoglyphs in command names.
* Non-POSIX shells (cmd.exe, PowerShell) -- patterns target sh/bash.
* Commands split across several tool calls (state is not tracked).

False positives (benign commands that may still match):
* Documentation strings that literally contain ``rm -rf`` in an echo/heredoc
  without quotes the heuristic trusts.
* Legitimate ``curl ... | jq`` pipelines that share the pipe shape with
  ``curl | sh`` (the pipe-to-interpreter rule is deliberately narrow).
* Test fixtures and dry-run scripts that mention destructive tokens.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Literal

from snagline.events import make_signature

ShellSeverity = Literal["info", "warning", "critical"]

#: Strictness order used when picking the strongest finding.
_SEVERITY_RANK: dict[ShellSeverity, int] = {
    "info": 0,
    "warning": 1,
    "critical": 2,
}

#: A detection rule: (command_text) -> finding or None.
ShellRuleFn = Callable[[str], "ShellFinding | None"]


@dataclass(frozen=True, slots=True)
class ShellFinding:
    """One structured hit from :func:`inspect_command`.

    ``command_signature`` is a SHA-256 digest suitable for logs. The raw
    command is intentionally absent so sinks and audit trails do not retain
    shell contents by default.
    """

    rule_id: str
    severity: ShellSeverity
    reason: str
    command_signature: str


@dataclass(frozen=True, slots=True)
class ShellThreatConfig:
    """Which rules run and how findings map to an authorize decision.

    ``enabled_rules``:
        ``None`` means all built-in rules. A frozenset restricts to those ids.
    ``disabled_rules``:
        Always subtracted after ``enabled_rules``.
    ``severity_overrides``:
        Optional per-rule severity replacement.
    ``min_severity``:
        Findings below this level are ignored by :func:`as_authorize_rule`.
    ``decision``:
        Verdict an authorize rule emits when at least one finding survives
        the severity filter (``deny`` or ``ask``).
    """

    enabled_rules: frozenset[str] | None = None
    disabled_rules: frozenset[str] = frozenset()
    severity_overrides: Mapping[str, ShellSeverity] = field(default_factory=dict)
    min_severity: ShellSeverity = "warning"
    decision: Literal["ask", "deny"] = "deny"


# ---------------------------------------------------------------------------
# Built-in rules (each independently testable)
# ---------------------------------------------------------------------------

# Word-ish token: command name at start or after a shell separator.
_SEP = r"(?:^|[;|&`\n]|\$\(|\(\s*)\s*(?:sudo\s+)?"

_DESTRUCTIVE = re.compile(
    _SEP + r"(?:"
    r"rm\s+-(?:[a-zA-Z]*f[a-zA-Z]*r[a-zA-Z]*|[a-zA-Z]*r[a-zA-Z]*f[a-zA-Z]*)\b"
    r"|mkfs(?:\.\w+)?\b"
    r"|dd\s+.*\bof\s*=\s*/dev/"
    r"|shred\b"
    r"|wipefs\b"
    r"|:\(\)\s*\{\s*:\|:&\s*\}\s*;:"  # classic fork bomb
    r")",
    re.IGNORECASE | re.DOTALL,
)

_PIPE_TO_SHELL = re.compile(
    r"(?:curl|wget|fetch)\b[^|;]*\|\s*(?:ba)?sh\b"
    r"|(?:curl|wget|fetch)\b[^|;]*\|\s*(?:ba)?sh\s+-s\b"
    r"|\b(?:ba)?sh\s+-c\s+.*(?:curl|wget)\b",
    re.IGNORECASE,
)

_CHAINING = re.compile(
    r"(?:;|\n|&&|\|\||`|\$\([^)]+\)|\|\s*(?:ba)?sh\b)",
)

_UNSAFE_REDIRECT = re.compile(
    r"(?:^|[\s;|&])(?:>>?|Tee-Object)\s*/(?:dev/sd|dev/nvme|etc/|boot/)",
    re.IGNORECASE,
)

_OBFUSCATION = re.compile(
    r"(?:"
    r"base64\s+-d\b[^|;]*\|\s*(?:ba)?sh\b"
    r"|base64\s+--decode\b[^|;]*\|\s*(?:ba)?sh\b"
    r"|\beval\b"
    r"|\$\{IFS\}"
    r"|\\x[0-9a-fA-F]{2}"
    r"|\$\(\s*(?:printf|echo)\b"
    r")",
    re.IGNORECASE,
)


def rule_destructive(command: str) -> ShellFinding | None:
    """Flag classic destructive / disk-wiping command shapes."""
    if not _DESTRUCTIVE.search(command):
        return None
    return _finding(
        "destructive_command",
        "critical",
        "destructive or disk-wiping command pattern",
        command,
    )


def rule_pipe_to_shell(command: str) -> ShellFinding | None:
    """Flag download-and-execute via curl/wget piped to a shell."""
    if not _PIPE_TO_SHELL.search(command):
        return None
    return _finding(
        "pipe_to_shell",
        "critical",
        "remote content piped to a shell interpreter",
        command,
    )


def rule_chaining(command: str) -> ShellFinding | None:
    """Flag multi-command chaining and command substitution.

    Benign one-liners with ``&&`` are common; severity stays at warning so
    operators can ask rather than hard-deny. Combine with other rules for
    critical paths.
    """
    if not _CHAINING.search(command):
        return None
    return _finding(
        "command_chaining",
        "warning",
        "command chaining or substitution construct",
        command,
    )


def rule_unsafe_redirect(command: str) -> ShellFinding | None:
    """Flag redirection toward device nodes or system paths."""
    if not _UNSAFE_REDIRECT.search(command):
        return None
    return _finding(
        "unsafe_redirect",
        "critical",
        "redirection toward a device or system path",
        command,
    )


def rule_obfuscation(command: str) -> ShellFinding | None:
    """Flag common obfuscation / dynamic-eval shapes."""
    if not _OBFUSCATION.search(command):
        return None
    return _finding(
        "obfuscation",
        "critical",
        "obfuscation or dynamic-eval construct",
        command,
    )


def _finding(
    rule_id: str,
    severity: ShellSeverity,
    reason: str,
    command: str,
) -> ShellFinding:
    return ShellFinding(
        rule_id=rule_id,
        severity=severity,
        reason=reason,
        command_signature=make_signature("shell_command", None, command),
    )


#: Default rule table. Order is stable for deterministic output ordering.
BUILTIN_RULES: tuple[tuple[str, ShellRuleFn], ...] = (
    ("destructive_command", rule_destructive),
    ("pipe_to_shell", rule_pipe_to_shell),
    ("command_chaining", rule_chaining),
    ("unsafe_redirect", rule_unsafe_redirect),
    ("obfuscation", rule_obfuscation),
)

BUILTIN_RULE_IDS: frozenset[str] = frozenset(rule_id for rule_id, _ in BUILTIN_RULES)


def inspect_command(
    command: str,
    config: ShellThreatConfig | None = None,
    *,
    extra_rules: Sequence[tuple[str, ShellRuleFn]] = (),
) -> tuple[ShellFinding, ...]:
    """Inspect ``command`` and return zero or more findings.

    Pure function: no subprocess, no ``os.system``, no file writes. Empty or
    whitespace-only input yields no findings.
    """
    if not isinstance(command, str) or not command.strip():
        return ()
    cfg = config or ShellThreatConfig()
    active_builtin = _active_rule_ids(cfg)
    findings: list[ShellFinding] = []
    for rule_id, fn in (*BUILTIN_RULES, *extra_rules):
        if rule_id in BUILTIN_RULE_IDS:
            if rule_id not in active_builtin:
                continue
        else:
            if cfg.enabled_rules is not None and rule_id not in cfg.enabled_rules:
                continue
            if rule_id in cfg.disabled_rules:
                continue
        hit = fn(command)
        if hit is None:
            continue
        override = cfg.severity_overrides.get(hit.rule_id)
        if override is not None:
            hit = ShellFinding(
                rule_id=hit.rule_id,
                severity=override,
                reason=hit.reason,
                command_signature=hit.command_signature,
            )
        findings.append(hit)
    # Stable order: severity desc, then rule_id asc (deterministic).
    findings.sort(key=lambda f: (-_SEVERITY_RANK[f.severity], f.rule_id))
    return tuple(findings)


def _active_rule_ids(cfg: ShellThreatConfig) -> frozenset[str]:
    base = (
        BUILTIN_RULE_IDS if cfg.enabled_rules is None else frozenset(cfg.enabled_rules)
    )
    return frozenset(base - frozenset(cfg.disabled_rules))


def strongest_severity(
    findings: Sequence[ShellFinding],
) -> ShellSeverity | None:
    """Highest severity among findings, or ``None`` when empty."""
    if not findings:
        return None
    return max(findings, key=lambda f: _SEVERITY_RANK[f.severity]).severity


def findings_at_or_above(
    findings: Sequence[ShellFinding],
    minimum: ShellSeverity,
) -> tuple[ShellFinding, ...]:
    """Filter findings to those at least as severe as ``minimum``."""
    floor = _SEVERITY_RANK[minimum]
    return tuple(f for f in findings if _SEVERITY_RANK[f.severity] >= floor)


def as_authorize_rule(config: ShellThreatConfig | None = None):
    """Build a ``snagline.authorize.Rule`` from shell threat findings.

    Lazy-imports authorize so this module stays usable without pulling the
    whole policy engine. Non-execute ``ToolRequest`` values are ignored.
    """
    from snagline.authorize import RuleVerdict, ToolRequest

    cfg = config or ShellThreatConfig()

    def _rule(request: ToolRequest) -> RuleVerdict | None:
        if request.action != "execute" or not request.resource:
            return None
        hits = findings_at_or_above(
            inspect_command(request.resource, cfg),
            cfg.min_severity,
        )
        if not hits:
            return None
        top = hits[0]
        # Reason names rule ids and signature only -- never the raw command.
        ids = ",".join(h.rule_id for h in hits)
        return RuleVerdict(
            rule_id="shell_threats",
            decision=cfg.decision,
            reason=(
                f"shell threat ({ids}) severity={top.severity} "
                f"sig={top.command_signature[:12]}"
            ),
        )

    return _rule
