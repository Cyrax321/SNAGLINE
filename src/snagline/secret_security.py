"""Deterministic secret detection and redaction utilities (issue #640).

Inspects text, tool arguments, or structured mappings for exposed credentials,
API keys, access tokens, and private keys before execution.

Provides reusable redaction utilities that mask detected secrets while preserving
sufficient context (such as the secret category or pattern name) for operational
debugging.

Design constraints
------------------
* Deterministic: same text + config always yields the exact same findings.
* Safe reporting: findings carry a one-way ``secret_signature`` and rule metadata.
  The detected raw secret is never included in findings, error messages, or logs.
* Non-invasive: pure pattern-based inspection with zero external network calls.
  Secrets are never collected or transmitted.
* Decoupled: operates independently of authorization policies and monitoring.

LIMITATIONS / known bypass classes
----------------------------------
Pattern-based secret detection has inherent limitations:
1. Encoded secrets: Secrets encoded in base64, hex, rot13, or URL encoding may
   not match standard token prefixes unless decoded prior to inspection.
2. Split or constructed tokens: Credentials dynamically assembled across
   multiple variables, string concatenations, or separate tool calls cannot be
   detected in isolation.
3. Unseen or proprietary formats: Custom tokens, high-entropy random strings,
   or newly introduced API key formats lacking distinct structural prefixes
   may not match built-in heuristics.
4. False positives: Synthetic test fixtures, example documentation strings
   (such as placeholder keys), or random hashes sharing pattern shapes may
   trigger detection.

For defense in depth, combine pattern inspection with secret scanning at the
repository, CI, and identity-provider levels.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

from snagline.events import make_signature

SecretSeverity = Literal["info", "warning", "critical"]

#: Strictness order used when picking the strongest finding.
_SEVERITY_RANK: dict[SecretSeverity, int] = {
    "info": 0,
    "warning": 1,
    "critical": 2,
}


@dataclass(frozen=True, slots=True)
class SecretMatch:
    """Internal representation of a matched secret span."""

    rule_id: str
    severity: SecretSeverity
    start: int
    end: int
    raw_secret: str
    match_type: str


#: A detection rule function: (text) -> list of SecretMatch.
SecretRuleFn = Callable[[str], list[SecretMatch]]


@dataclass(frozen=True, slots=True)
class SecretFinding:
    """One structured hit from secret inspection.

    ``secret_signature`` is a SHA-256 digest suitable for audit trails.
    The raw credential value is intentionally excluded to ensure secrets
    are not retained or leaked in audit logs.
    """

    rule_id: str
    severity: SecretSeverity
    reason: str
    secret_signature: str
    match_type: str


@dataclass(frozen=True, slots=True)
class SecretSecurityConfig:
    """Configuration for secret detection and redaction.

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
    severity_overrides: Mapping[str, SecretSeverity] = field(default_factory=dict)
    min_severity: SecretSeverity = "warning"
    decision: Literal["ask", "deny"] = "deny"


# ---------------------------------------------------------------------------
# Built-in detection patterns
# ---------------------------------------------------------------------------

_PATTERNS: tuple[tuple[str, SecretSeverity, str, re.Pattern[str]], ...] = (
    (
        "aws_access_key",
        "critical",
        "AWS Access Key ID",
        re.compile(r"\b(AKIA[0-9A-Z]{16})\b"),
    ),
    (
        "github_token",
        "critical",
        "GitHub Personal Access Token",
        re.compile(
            r"\b((?:ghp|gho|ghu|ghs|ghr)_[0-9a-zA-Z]{36}|github_pat_[0-9a-zA-Z_]{80,90})\b"
        ),
    ),
    (
        "slack_token",
        "critical",
        "Slack Token",
        re.compile(r"\b(xox[baprs]-[0-9a-zA-Z-]{10,80})\b"),
    ),
    (
        "openai_api_key",
        "critical",
        "OpenAI API Key",
        re.compile(r"\b(sk-(?!ant-)[a-zA-Z0-9_-]{20,})\b"),
    ),
    (
        "anthropic_api_key",
        "critical",
        "Anthropic API Key",
        re.compile(r"\b(sk-ant-[a-zA-Z0-9_-]{20,})\b"),
    ),
    (
        "stripe_secret_key",
        "critical",
        "Stripe Secret Key",
        re.compile(r"\b(sk_live_[0-9a-zA-Z]{24,})\b"),
    ),
    (
        "google_api_key",
        "critical",
        "Google Cloud / Firebase API Key",
        re.compile(r"\b(AIza[0-9A-Za-z_-]{35,36})\b"),
    ),
    (
        "private_key",
        "critical",
        "Cryptographic Private Key",
        re.compile(
            r"-----BEGIN (?:RSA |EC |DSA |OPENSSH |PGP )?PRIVATE KEY-----[\s\S]*?-----END (?:RSA |EC |DSA |OPENSSH |PGP )?PRIVATE KEY-----"
        ),
    ),
    (
        "jwt_token",
        "warning",
        "JSON Web Token (JWT)",
        re.compile(
            r"\b(ey[A-Za-z0-9_-]{10,}\.ey[A-Za-z0-9_-]{10,}\.[A-Za-z0-9-_+/=]{10,})\b"
        ),
    ),
    (
        "database_connection_uri",
        "critical",
        "Database Connection URI with Credentials",
        re.compile(
            r"\b(?:postgres|postgresql|mysql|mongodb|redis)://[^:\s/@]+:[^@\s/]+@\S+"
        ),
    ),
    (
        "uri_basic_auth",
        "critical",
        "URI Basic Auth Credentials",
        re.compile(r"https?://(?:[^:\s/@]+):([^@\s/]+)@"),
    ),
    (
        "generic_credential_assignment",
        "warning",
        "Generic Password or API Key Assignment",
        re.compile(
            r"(?i)(?<![a-zA-Z0-9])(?:[a-zA-Z0-9_-]*?(?:password|passwd|secret|api_?key|auth_?token))\s*[:=]\s*['\"]([^'\"]{6,})['\"]"
        ),
    ),
)

BUILTIN_RULE_IDS: frozenset[str] = frozenset(rule_id for rule_id, _, _, _ in _PATTERNS)


def _scan_matches(
    text: str,
    cfg: SecretSecurityConfig,
    extra_rules: Sequence[tuple[str, SecretRuleFn]] = (),
) -> list[SecretMatch]:
    if not text:
        return []

    active_builtin = _active_rule_ids(cfg)
    matches: list[SecretMatch] = []

    for rule_id, default_sev, desc, pattern in _PATTERNS:
        if rule_id not in active_builtin:
            continue
        sev = cfg.severity_overrides.get(rule_id, default_sev)
        for m in pattern.finditer(text):
            # If there is a capture group, use group 1 as the secret; otherwise entire match
            raw_secret = m.group(1) if m.groups() else m.group(0)
            matches.append(
                SecretMatch(
                    rule_id=rule_id,
                    severity=sev,
                    start=m.start(),
                    end=m.end(),
                    raw_secret=raw_secret,
                    match_type=desc,
                )
            )

    for rule_id, fn in extra_rules:
        if cfg.enabled_rules is not None and rule_id not in cfg.enabled_rules:
            continue
        if rule_id in cfg.disabled_rules:
            continue
        for hit in fn(text):
            if cfg.enabled_rules is not None and hit.rule_id not in cfg.enabled_rules:
                continue
            if hit.rule_id in cfg.disabled_rules:
                continue
            sev = cfg.severity_overrides.get(hit.rule_id, hit.severity)
            matches.append(
                SecretMatch(
                    rule_id=hit.rule_id,
                    severity=sev,
                    start=hit.start,
                    end=hit.end,
                    raw_secret=hit.raw_secret,
                    match_type=hit.match_type,
                )
            )

    # Sort matches by start index
    matches.sort(key=lambda m: m.start)
    return matches


def _active_rule_ids(cfg: SecretSecurityConfig) -> frozenset[str]:
    base = (
        BUILTIN_RULE_IDS if cfg.enabled_rules is None else frozenset(cfg.enabled_rules)
    )
    return frozenset(base - frozenset(cfg.disabled_rules))


# ---------------------------------------------------------------------------
# Public inspection functions
# ---------------------------------------------------------------------------


def inspect_text(
    text: str,
    config: SecretSecurityConfig | None = None,
    *,
    extra_rules: Sequence[tuple[str, SecretRuleFn]] = (),
) -> tuple[SecretFinding, ...]:
    """Inspect ``text`` for secrets and return zero or more structured findings.

    The raw secret is never included in the returned findings. A one-way
    SHA-256 hash digest is provided instead.
    """
    if not isinstance(text, str) or not text.strip():
        return ()

    cfg = config or SecretSecurityConfig()
    matches = _scan_matches(text, cfg, extra_rules)
    findings: list[SecretFinding] = []

    for m in matches:
        sig = make_signature("secret_credential", None, m.raw_secret)
        findings.append(
            SecretFinding(
                rule_id=m.rule_id,
                severity=m.severity,
                reason=f"exposed {m.match_type} pattern detected",
                secret_signature=sig,
                match_type=m.match_type,
            )
        )

    # Deterministic sort: severity desc, then rule_id asc, then signature asc
    findings.sort(
        key=lambda f: (
            -_SEVERITY_RANK[f.severity],
            f.rule_id,
            f.secret_signature,
        )
    )
    return tuple(findings)


def inspect_tool_input(
    tool_input: Mapping[str, Any],
    config: SecretSecurityConfig | None = None,
    *,
    extra_rules: Sequence[tuple[str, SecretRuleFn]] = (),
) -> tuple[SecretFinding, ...]:
    """Recursively scan string values inside ``tool_input`` for exposed secrets."""
    if not isinstance(tool_input, Mapping):
        return ()

    cfg = config or SecretSecurityConfig()
    findings: list[SecretFinding] = []

    def _walk(val: Any) -> None:
        if isinstance(val, str):
            findings.extend(inspect_text(val, cfg, extra_rules=extra_rules))
        elif isinstance(val, Mapping):
            try:
                items = list(val.values())
            except (RuntimeError, TypeError):
                items = []
            for v in items:
                _walk(v)
        elif isinstance(val, (list, tuple, set, frozenset)):
            try:
                items = list(val)
            except (RuntimeError, TypeError):
                items = []
            for item in items:
                _walk(item)

    _walk(tool_input)
    # Deduplicate findings by secret_signature and rule_id
    seen: set[tuple[str, str]] = set()
    deduped: list[SecretFinding] = []
    for f in findings:
        key = (f.rule_id, f.secret_signature)
        if key not in seen:
            seen.add(key)
            deduped.append(f)

    deduped.sort(
        key=lambda f: (
            -_SEVERITY_RANK[f.severity],
            f.rule_id,
            f.secret_signature,
        )
    )
    return tuple(deduped)


# ---------------------------------------------------------------------------
# Redaction utilities
# ---------------------------------------------------------------------------


def redact_secrets(
    text: str,
    mask: str | None = None,
    config: SecretSecurityConfig | None = None,
    *,
    extra_rules: Sequence[tuple[str, SecretRuleFn]] = (),
) -> str:
    """Redact detected secrets from ``text``.

    If ``mask`` is ``None``, defaults to ``"[REDACTED:{rule_id}]"``.
    Preserves surrounding text and non-secret structure.
    """
    if not isinstance(text, str) or not text:
        return text

    cfg = config or SecretSecurityConfig()
    matches = _scan_matches(text, cfg, extra_rules)
    if not matches:
        return text

    # Merge overlapping or adjacent matches
    merged: list[SecretMatch] = []
    for m in matches:
        if not merged:
            merged.append(m)
            continue
        prev = merged[-1]
        if m.start < prev.end:
            # Overlapping match: extend end if necessary
            if m.end > prev.end:
                merged[-1] = SecretMatch(
                    rule_id=prev.rule_id,
                    severity=prev.severity,
                    start=prev.start,
                    end=m.end,
                    raw_secret=text[prev.start : m.end],
                    match_type=prev.match_type,
                )
        else:
            merged.append(m)

    # Build redacted string from back to front to preserve slice offsets
    result = text
    for m in reversed(merged):
        replacement = (
            mask.replace("{rule_id}", m.rule_id)
            if mask is not None
            else f"[REDACTED:{m.rule_id}]"
        )
        result = result[: m.start] + replacement + result[m.end :]

    return result


def _redact_value(val: Any, mask: str | None, cfg: SecretSecurityConfig) -> Any:
    if isinstance(val, str):
        return redact_secrets(val, mask, cfg)
    if isinstance(val, Mapping):
        return {k: _redact_value(v, mask, cfg) for k, v in val.items()}
    if isinstance(val, list):
        return [_redact_value(item, mask, cfg) for item in val]
    if isinstance(val, tuple):
        return tuple(_redact_value(item, mask, cfg) for item in val)
    if isinstance(val, set):
        return {_redact_value(item, mask, cfg) for item in val}
    if isinstance(val, frozenset):
        return frozenset(_redact_value(item, mask, cfg) for item in val)
    return val


def redact_mapping(
    mapping: Mapping[str, Any],
    mask: str | None = None,
    config: SecretSecurityConfig | None = None,
) -> dict[str, Any]:
    """Recursively redact secrets across dictionary values."""
    if not isinstance(mapping, Mapping):
        return {}

    cfg = config or SecretSecurityConfig()
    return {k: _redact_value(v, mask, cfg) for k, v in mapping.items()}


def strongest_severity(
    findings: Sequence[SecretFinding],
) -> SecretSeverity | None:
    """Highest severity among findings, or ``None`` when empty."""
    if not findings:
        return None
    return max(findings, key=lambda f: _SEVERITY_RANK[f.severity]).severity


def findings_at_or_above(
    findings: Sequence[SecretFinding],
    minimum: SecretSeverity,
) -> tuple[SecretFinding, ...]:
    """Filter findings to those at least as severe as ``minimum``."""
    floor = _SEVERITY_RANK[minimum]
    return tuple(f for f in findings if _SEVERITY_RANK[f.severity] >= floor)


# ---------------------------------------------------------------------------
# Authorize engine integration
# ---------------------------------------------------------------------------


def as_authorize_rule(config: SecretSecurityConfig | None = None):
    """Build a ``snagline.authorize.Rule`` from secret detection findings.

    Scans both ``request.resource`` and any string values in ``request.tool_input``.
    """
    from snagline.authorize import RuleVerdict, ToolRequest

    cfg = config or SecretSecurityConfig()

    def _rule(request: ToolRequest) -> RuleVerdict | None:
        if request.resource and (
            request.resource.endswith("...<truncated>") or len(request.resource) > 4096
        ):
            return RuleVerdict(
                rule_id="secret_exposure",
                decision=cfg.decision,
                reason="resource exceeds maximum length or is truncated",
            )

        hits: list[SecretFinding] = []
        if request.resource:
            hits.extend(inspect_text(request.resource, cfg))
        if request.tool_input:
            hits.extend(inspect_tool_input(request.tool_input, cfg))

        filtered = findings_at_or_above(hits, cfg.min_severity)
        if not filtered:
            return None

        top = max(filtered, key=lambda f: _SEVERITY_RANK[f.severity])
        ids = ",".join(sorted({h.rule_id for h in filtered}))
        return RuleVerdict(
            rule_id="secret_exposure",
            decision=cfg.decision,
            reason=(
                f"secret credential exposure ({ids}) severity={top.severity} "
                f"sig={top.secret_signature[:12]}"
            ),
        )

    return _rule
