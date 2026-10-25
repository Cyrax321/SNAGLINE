"""Data Loss Prevention (DLP) checks for outbound tool payloads (issue #641).

Inspects content and structured parameters supplied to outbound tool requests
(such as network egress, API requests, emails, messages, or external writes)
for personal identifiers (PII), payment card information, and sensitive data.

Design constraints
------------------
* Deterministic: same payload and config always produce the exact same findings.
* Safe reporting: findings carry a SHA-256 value signature and redacted snippet.
  The detected raw sensitive values are never logged or exposed.
* In-process inspection: evaluates only content explicitly passed to the tool
  request. It does not provide network-level packet capture or packet inspection.
* Decoupled: operates independently of monitoring and general authorization.

LIMITATIONS / known boundary classes
------------------------------------
1. In-process scope: this detector only inspects payload content explicitly
   passed to tool requests. It cannot inspect raw OS socket traffic, encrypted
   TLS streams below the application layer, or side-channel data.
2. Encoded data: base64, compressed, or custom-obfuscated data will not match
   plaintext patterns unless decoded prior to inspection.
3. Natural false positives: numeric sequences or identifiers resembling credit
   card or SSN patterns may trigger warnings. High-confidence rules (like credit
   cards) apply checksum validation (Luhn algorithm) to minimize noise.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

from snagline.events import make_signature

DlpSeverity = Literal["info", "warning", "critical"]

_SEVERITY_RANK: dict[DlpSeverity, int] = {
    "info": 0,
    "warning": 1,
    "critical": 2,
}


@dataclass(frozen=True, slots=True)
class DlpMatch:
    """Internal representation of a matched DLP span."""

    rule_id: str
    severity: DlpSeverity
    start: int
    end: int
    raw_value: str
    redacted_value: str
    match_type: str
    value_signature: str = ""


DlpRuleFn = Callable[[str], list[DlpMatch]]


@dataclass(frozen=True, slots=True)
class DlpFinding:
    """One structured hit from data loss prevention inspection.

    ``value_signature`` is a SHA-256 digest suitable for audit trails.
    The raw sensitive value is never retained on findings to preserve
    the privacy invariant.
    """

    rule_id: str
    severity: DlpSeverity
    reason: str
    value_signature: str
    redacted_value: str
    path: str = ""


@dataclass(frozen=True, slots=True)
class DlpSecurityConfig:
    """Configuration for Data Loss Prevention inspection.

    ``enabled_rules``:
        None means all built-in rules. A frozenset restricts to those rule IDs.
    ``disabled_rules``:
        Always subtracted after enabled_rules.
    ``severity_overrides``:
        Optional per-rule severity replacement.
    ``min_severity``:
        Findings below this level are ignored by :func:`as_authorize_rule`.
    ``decision``:
        Verdict an authorize rule emits when at least one finding survives
        the severity filter (deny or ask).
    ``include_secrets``:
        Whether to run credential detection patterns alongside PII patterns.
    """

    enabled_rules: frozenset[str] | None = None
    disabled_rules: frozenset[str] = frozenset()
    severity_overrides: Mapping[str, DlpSeverity] = field(default_factory=dict)
    min_severity: DlpSeverity = "warning"
    decision: Literal["ask", "deny"] = "deny"
    include_secrets: bool = True


# ---------------------------------------------------------------------------
# Checksum and Validation Helpers
# ---------------------------------------------------------------------------


def _luhn_checksum_valid(number_str: str) -> bool:
    """Validate numeric string using the Luhn mod-10 algorithm."""
    digits = [int(c) for c in number_str if c.isdigit()]
    if len(digits) < 13 or len(digits) > 19:
        return False
    checksum = 0
    reverse_digits = digits[::-1]
    for i, digit in enumerate(reverse_digits):
        if i % 2 == 1:
            doubled = digit * 2
            checksum += doubled - 9 if doubled > 9 else doubled
        else:
            checksum += digit
    return checksum % 10 == 0


def mask_ssn(ssn: str) -> str:
    """Mask SSN to ***-**-XXXX."""
    digits = re.sub(r"\D", "", ssn)
    if len(digits) == 9:
        return f"***-**-{digits[5:]}"
    return "***-**-****"


def mask_credit_card(cc: str) -> str:
    """Mask credit card to XXXX-****-****-XXXX."""
    digits = re.sub(r"\D", "", cc)
    if len(digits) >= 13:
        return f"{digits[:4]}-****-****-{digits[-4:]}"
    return "****-****-****-****"


def mask_email(email: str) -> str:
    """Mask email address to j***@example.com."""
    parts = email.split("@", 1)
    if len(parts) == 2:
        name, domain = parts
        masked_name = name[0] + "***" if name else "***"
        return f"{masked_name}@{domain}"
    return "***@***"


def mask_phone(phone: str) -> str:
    """Mask phone number to ***-***-XXXX."""
    digits = re.sub(r"\D", "", phone)
    if len(digits) >= 7:
        return f"***-***-{digits[-4:]}"
    return "***-***-****"


# ---------------------------------------------------------------------------
# Detection Rule Handlers
# ---------------------------------------------------------------------------

_SSN_PATTERN = re.compile(
    r"\b(?!000|666|9\d{2})(\d{3})[- ](?!00)(\d{2})[- ](?!0000)(\d{4})\b"
)

_EMAIL_PATTERN = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")

_PHONE_PATTERN = re.compile(
    r"(?:(?:\+?1[-. ]?)?(?:\([0-9]{3}\)|[0-9]{3})[-. ]?[0-9]{3}[-. ]?[0-9]{4})\b"
)

# Potential card patterns: 13-19 digits with optional single space or hyphen between digits
_CC_CANDIDATE_PATTERN = re.compile(r"\b\d(?:[ -]?\d){12,18}\b")


def _detect_ssn(text: str) -> list[DlpMatch]:
    matches: list[DlpMatch] = []
    for m in _SSN_PATTERN.finditer(text):
        raw = m.group(0)
        matches.append(
            DlpMatch(
                rule_id="us_ssn",
                severity="critical",
                start=m.start(),
                end=m.end(),
                raw_value=raw,
                redacted_value=mask_ssn(raw),
                match_type="US Social Security Number",
            )
        )
    return matches


def _detect_credit_card(text: str) -> list[DlpMatch]:
    matches: list[DlpMatch] = []
    for m in _CC_CANDIDATE_PATTERN.finditer(text):
        raw = m.group(0)
        clean = re.sub(r"[ -]", "", raw)
        # Check standard card brand lengths (Visa, MC, Amex, Discover)
        if len(clean) in {13, 15, 16, 19} and _luhn_checksum_valid(clean):
            matches.append(
                DlpMatch(
                    rule_id="credit_card",
                    severity="critical",
                    start=m.start(),
                    end=m.end(),
                    raw_value=raw,
                    redacted_value=mask_credit_card(raw),
                    match_type="Payment Card Number",
                )
            )
    return matches


def _detect_email(text: str) -> list[DlpMatch]:
    matches: list[DlpMatch] = []
    for m in _EMAIL_PATTERN.finditer(text):
        raw = m.group(0)
        matches.append(
            DlpMatch(
                rule_id="email_address",
                severity="warning",
                start=m.start(),
                end=m.end(),
                raw_value=raw,
                redacted_value=mask_email(raw),
                match_type="Email Address",
            )
        )
    return matches


def _detect_phone(text: str) -> list[DlpMatch]:
    matches: list[DlpMatch] = []
    for m in _PHONE_PATTERN.finditer(text):
        raw = m.group(0)
        matches.append(
            DlpMatch(
                rule_id="phone_number",
                severity="info",
                start=m.start(),
                end=m.end(),
                raw_value=raw,
                redacted_value=mask_phone(raw),
                match_type="Phone Number",
            )
        )
    return matches


_BUILTIN_RULES: tuple[tuple[str, DlpSeverity, str, DlpRuleFn], ...] = (
    ("us_ssn", "critical", "US Social Security Number", _detect_ssn),
    ("credit_card", "critical", "Payment Card Number", _detect_credit_card),
    ("email_address", "warning", "Email Address", _detect_email),
    ("phone_number", "info", "Phone Number", _detect_phone),
)

BUILTIN_DLP_RULE_IDS: frozenset[str] = frozenset(
    rule_id for rule_id, _, _, _ in _BUILTIN_RULES
)


# ---------------------------------------------------------------------------
# Core Scanner
# ---------------------------------------------------------------------------


def _active_rules(cfg: DlpSecurityConfig) -> list[tuple[str, DlpSeverity, DlpRuleFn]]:
    active: list[tuple[str, DlpSeverity, DlpRuleFn]] = []
    for rule_id, default_sev, _, fn in _BUILTIN_RULES:
        if cfg.enabled_rules is not None and rule_id not in cfg.enabled_rules:
            continue
        if rule_id in cfg.disabled_rules:
            continue
        sev = cfg.severity_overrides.get(rule_id, default_sev)
        active.append((rule_id, sev, fn))
    return active


def scan_text(
    text: str,
    config: DlpSecurityConfig | None = None,
    *,
    extra_rules: Sequence[tuple[str, DlpRuleFn]] = (),
) -> list[DlpMatch]:
    """Scan string for DLP pattern matches."""
    if not text:
        return []

    cfg = config or DlpSecurityConfig()
    rules = _active_rules(cfg)
    matches: list[DlpMatch] = []

    for _rule_id, sev, fn in rules:
        for match in fn(text):
            if sev != match.severity:
                match = DlpMatch(
                    rule_id=match.rule_id,
                    severity=sev,
                    start=match.start,
                    end=match.end,
                    raw_value=match.raw_value,
                    redacted_value=match.redacted_value,
                    match_type=match.match_type,
                    value_signature=match.value_signature,
                )
            matches.append(match)

    for extra_id, extra_fn in extra_rules:
        if cfg.enabled_rules is not None and extra_id not in cfg.enabled_rules:
            continue
        if extra_id in cfg.disabled_rules:
            continue
        matches.extend(extra_fn(text))

    # Also run credentials detection if configured
    if cfg.include_secrets:
        from snagline.secret_security import SecretSecurityConfig, inspect_text

        sec_cfg = SecretSecurityConfig()
        sec_findings = inspect_text(text, sec_cfg)
        for sf in sec_findings:
            matches.append(
                DlpMatch(
                    rule_id=f"secret:{sf.rule_id}",
                    severity=sf.severity,
                    start=0,
                    end=0,
                    raw_value="",
                    redacted_value="[REDACTED:secret]",
                    match_type=sf.match_type,
                    value_signature=sf.secret_signature,
                )
            )

    return matches


def inspect_payload(
    payload: Any,
    config: DlpSecurityConfig | None = None,
    *,
    path: str = "",
    extra_rules: Sequence[tuple[str, DlpRuleFn]] = (),
) -> tuple[DlpFinding, ...]:
    """Recursively scan an arbitrary payload (string, dict, list) for DLP hits."""
    cfg = config or DlpSecurityConfig()
    findings: list[DlpFinding] = []

    if isinstance(payload, str):
        matches = scan_text(payload, cfg, extra_rules=extra_rules)
        for m in matches:
            sig = (
                m.value_signature
                if m.value_signature
                else make_signature(f"dlp:{m.rule_id}", None, m.raw_value)
            )
            findings.append(
                DlpFinding(
                    rule_id=m.rule_id,
                    severity=m.severity,
                    reason=f"outbound payload contains {m.match_type}",
                    value_signature=sig,
                    redacted_value=m.redacted_value,
                    path=path,
                )
            )
    elif isinstance(payload, Mapping):
        for k, v in list(payload.items()):
            sub_path = f"{path}.{k}" if path else str(k)
            findings.extend(
                inspect_payload(
                    v,
                    cfg,
                    path=sub_path,
                    extra_rules=extra_rules,
                )
            )
    elif isinstance(payload, (list, tuple, set, frozenset)):
        for idx, item in enumerate(payload):
            sub_path = f"{path}[{idx}]" if path else f"[{idx}]"
            findings.extend(
                inspect_payload(
                    item,
                    cfg,
                    path=sub_path,
                    extra_rules=extra_rules,
                )
            )

    return tuple(findings)


# ---------------------------------------------------------------------------
# Redaction Utilities
# ---------------------------------------------------------------------------


def redact_text(
    text: str,
    config: DlpSecurityConfig | None = None,
    *,
    mask: str | None = None,
) -> str:
    """Replace sensitive spans in string with safe redacted markers."""
    cfg = config or DlpSecurityConfig()
    # For text redaction, only span-based rules apply directly
    matches = [m for m in scan_text(text, cfg) if m.start < m.end]
    if not matches:
        return text

    # Sort descending by start position to replace without shifting offsets
    matches.sort(key=lambda m: (m.start, m.end - m.start), reverse=True)

    result = list(text)
    for m in matches:
        marker = (
            mask.replace("{rule_id}", m.rule_id) if mask else f"[REDACTED:{m.rule_id}]"
        )
        result[m.start : m.end] = list(marker)

    return "".join(result)


def redact_payload(
    payload: Any,
    config: DlpSecurityConfig | None = None,
    *,
    mask: str | None = None,
) -> Any:
    """Recursively redact sensitive data from nested mappings and lists."""
    cfg = config or DlpSecurityConfig()

    if isinstance(payload, str):
        return redact_text(payload, cfg, mask=mask)
    if isinstance(payload, Mapping):
        return {k: redact_payload(v, cfg, mask=mask) for k, v in payload.items()}
    if isinstance(payload, tuple):
        return tuple(redact_payload(item, cfg, mask=mask) for item in payload)
    if isinstance(payload, list):
        return [redact_payload(item, cfg, mask=mask) for item in payload]
    if isinstance(payload, frozenset):
        return frozenset(redact_payload(item, cfg, mask=mask) for item in payload)
    if isinstance(payload, set):
        return {redact_payload(item, cfg, mask=mask) for item in payload}
    return payload


# ---------------------------------------------------------------------------
# Severity Filters
# ---------------------------------------------------------------------------


def findings_at_or_above(
    findings: Sequence[DlpFinding],
    minimum: DlpSeverity,
) -> tuple[DlpFinding, ...]:
    """Filter DLP findings to those at least as severe as minimum."""
    floor = _SEVERITY_RANK[minimum]
    return tuple(f for f in findings if _SEVERITY_RANK[f.severity] >= floor)


def strongest_severity(findings: Sequence[DlpFinding]) -> DlpSeverity | None:
    """Return the highest severity among findings, or None if empty."""
    if not findings:
        return None
    return max(findings, key=lambda f: _SEVERITY_RANK[f.severity]).severity


# ---------------------------------------------------------------------------
# Authorization Engine Integration
# ---------------------------------------------------------------------------

_EGRESS_ACTIONS = frozenset({"network", "egress", "post", "send", "write", "unknown"})


def as_authorize_rule(
    config: DlpSecurityConfig | None = None,
    *,
    actions: frozenset[str] = _EGRESS_ACTIONS,
):
    """Build a ``snagline.authorize.Rule`` from data loss prevention findings."""
    from snagline.authorize import RuleVerdict, ToolRequest

    cfg = config or DlpSecurityConfig()

    def _rule(request: ToolRequest) -> RuleVerdict | None:
        if request.action not in actions:
            return None

        if request.resource and (
            request.resource.endswith("...<truncated>") or len(request.resource) > 4096
        ):
            return RuleVerdict(
                rule_id="dlp",
                decision=cfg.decision,
                reason="resource exceeds maximum length or is truncated",
            )

        # Inspect tool_input as payload; also inspect resource if present
        findings: list[DlpFinding] = []
        if request.tool_input:
            findings.extend(inspect_payload(request.tool_input, cfg, path="input"))
        if request.resource:
            findings.extend(inspect_payload(request.resource, cfg, path="resource"))

        hits = findings_at_or_above(findings, cfg.min_severity)
        if not hits:
            return None

        top = max(hits, key=lambda f: _SEVERITY_RANK[f.severity])
        rule_ids = ",".join(sorted({h.rule_id for h in hits}))
        paths = ",".join(sorted({h.path for h in hits if h.path}))
        return RuleVerdict(
            rule_id="dlp",
            decision=cfg.decision,
            reason=(
                f"outbound data loss prevention violation ({rule_ids}) "
                f"severity={top.severity} at [{paths}]"
            ),
        )

    return _rule
