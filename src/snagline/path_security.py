"""Deterministic filesystem path security checks (issue #639).

Inspects a file path *string* or path-like object before execution and returns
structured findings. It never accesses the physical filesystem, never checks
file attributes, never creates or modifies files, and never changes Monitor
runtime behaviour. Findings are meant to feed ``snagline.authorize.Policy``
so an operator can map them to allow / ask / deny.

Design constraints
------------------
* Deterministic: same path + config always yields the exact same findings.
* Non-invasive: pure lexical path inspection. Zero filesystem reads or writes.
* Modular: each check is an independently testable rule function.
* Privacy: findings carry a one-way ``path_signature``, not the raw path.
  Reasons name the rule and a short pattern classification only.
* Independent: filesystem checks are completely decoupled from the monitoring
  event ingestion pipeline.

LIMITATIONS / known bypass classes
----------------------------------
Lexical path checks alone cannot guarantee filesystem confinement:
1. Symlinks: Lexical analysis inspects the path text, not the underlying inode.
   A symlink inside an allowed root pointing to an out-of-bounds target will
   appear benign to lexical checks.
2. Filesystem races (TOCTOU): The file target may change or be replaced by a
   symlink between the time this authorization decision is made and the time
   the runtime actually performs the file operation.
3. Hard links: Hard links to sensitive files residing within allowed directories
   cannot be distinguished from regular files without filesystem metadata.
4. Alternate data streams and platform quirks: Platform-specific tricks (such
   as Windows ADS ``file:stream`` or DOS short names 8.3) may refer to files
   under names not recognized by standard lexical patterns.

For strict confinement, pair these lexical policy rules with kernel-level or
container-level filesystem sandboxing.
"""

from __future__ import annotations

import fnmatch
import os
import posixpath
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Literal

from snagline.events import make_signature

PathSeverity = Literal["info", "warning", "critical"]

#: Strictness order used when picking the strongest finding.
_SEVERITY_RANK: dict[PathSeverity, int] = {
    "info": 0,
    "warning": 1,
    "critical": 2,
}

#: A detection rule: (raw_path, normalized_path, config) -> finding or None.
PathRuleFn = Callable[[str, str, "PathSecurityConfig"], "PathFinding | None"]


@dataclass(frozen=True, slots=True)
class PathFinding:
    """One structured hit from :func:`inspect_path`.

    ``path_signature`` is a SHA-256 digest suitable for audit logs. The raw
    path is intentionally absent from the finding so sinks and audit trails do
    not retain sensitive file paths by default.
    """

    rule_id: str
    severity: PathSeverity
    reason: str
    path_signature: str


@dataclass(frozen=True, slots=True)
class PathSecurityConfig:
    """Configuration for lexical path security inspection.

    ``allowed_roots``:
        If provided, any path that falls outside these directory trees triggers
        an ``allowed_roots`` finding.
    ``denied_paths``:
        Paths or directory trees that are explicitly forbidden.
    ``denied_patterns``:
        Glob patterns (e.g. ``*.pem``, ``.env*``) matched against the filename
        or normalized path.
    ``base_dir``:
        Working directory used to resolve relative paths lexically.
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

    allowed_roots: tuple[str, ...] | None = None
    denied_paths: tuple[str, ...] = ()
    denied_patterns: tuple[str, ...] = ()
    base_dir: str | None = None
    enabled_rules: frozenset[str] | None = None
    disabled_rules: frozenset[str] = frozenset()
    severity_overrides: Mapping[str, PathSeverity] = field(default_factory=dict)
    min_severity: PathSeverity = "warning"
    decision: Literal["ask", "deny"] = "deny"


# ---------------------------------------------------------------------------
# Lexical path normalization
# ---------------------------------------------------------------------------

_WINDOWS_DRIVE_RE = re.compile(r"^[a-zA-Z]:")


def normalize_lexical_path(
    path: str | os.PathLike[str],
    base_dir: str | None = None,
) -> str:
    """Normalize a path lexically without accessing the filesystem.

    Converts backslashes to forward slashes, collapses redundant separators
    and ``.`` / ``..`` segments, and resolves relative paths against
    ``base_dir`` if provided.
    """
    raw = os.fspath(path)
    if not raw or not raw.strip():
        return ""

    # Replace backslashes with standard forward slashes
    clean = raw.replace("\\", "/")

    # Preserve leading Windows drive letters if present (e.g., C:/)
    drive = ""
    match = _WINDOWS_DRIVE_RE.match(clean)
    if match:
        drive = clean[:2]
        clean = clean[2:]
        if not clean.startswith("/"):
            clean = "/" + clean

    # Resolve relative paths against base_dir if base_dir is supplied
    if base_dir and not clean.startswith("/") and not drive:
        clean_base = os.fspath(base_dir).replace("\\", "/")
        clean = posixpath.join(clean_base, clean)

    # Lexically normalize the path
    norm = posixpath.normpath(clean)

    # If posixpath.normpath collapsed multiple leading slashes to '//', fix it
    if norm.startswith("//") and not norm.startswith("///"):
        norm = "/" + norm.lstrip("/")

    return drive + norm


def _is_windows_path(path: str) -> bool:
    return os.name == "nt" or bool(_WINDOWS_DRIVE_RE.match(path))


def _is_subpath(target: str, root: str, *, case_insensitive: bool = False) -> bool:
    """Check if normalized ``target`` is inside or equal to normalized ``root``."""
    t = target.lower() if case_insensitive else target
    r = root.lower() if case_insensitive else root
    target_clean = t.rstrip("/")
    root_clean = r.rstrip("/")
    if target_clean == root_clean:
        return True
    return t.startswith(root_clean + "/")


# ---------------------------------------------------------------------------
# Built-in detection rules
# ---------------------------------------------------------------------------

_SENSITIVE_BASENAMES = frozenset(
    {
        ".env",
        ".env.local",
        ".env.production",
        ".env.development",
        ".env.staging",
        ".env.test",
        ".git-credentials",
        ".netrc",
        ".pgpass",
        "id_rsa",
        "id_ecdsa",
        "id_ed25519",
        "id_dsa",
        "authorized_keys",
        "known_hosts",
        "credentials.json",
        "service_account.json",
        "service-account.json",
        "shadow",
        "master.passwd",
        "sudoers",
        ".bash_history",
        ".zsh_history",
        ".sh_history",
        ".python_history",
        "sam",
    }
)

_SENSITIVE_PREFIXES = (
    "/etc/shadow",
    "/etc/sudoers",
    "/etc/master.passwd",
    "/etc/ssl/private",
    "/var/run/secrets/kubernetes.io",
    "/run/secrets",
    "/proc/self/environ",
    "/proc/kcore",
)

_SENSITIVE_DIR_SEGMENTS = (
    "/.ssh/",
    "/.aws/",
    "/.azure/",
    "/.gcp/",
    "/.config/gcloud/",
    "/.kube/",
    "/windows/system32/config/",
)

_SENSITIVE_EXTENSIONS = (
    ".pem",
    ".key",
    ".pfx",
    ".p12",
    ".pkcs12",
)


def rule_path_traversal(
    path: str,
    norm_path: str,
    config: PathSecurityConfig,
) -> PathFinding | None:
    """Detect directory traversal sequences attempting to escape boundaries."""
    # Check for raw traversal patterns (dot-dot sequences)
    clean_raw = path.replace("\\", "/")
    has_traversal_token = (
        clean_raw.startswith("../")
        or "/../" in clean_raw
        or clean_raw.endswith("/..")
        or clean_raw == ".."
    )
    escapes_norm = norm_path.startswith("../") or norm_path == ".."

    # If base_dir is configured, check if normalized path escapes base_dir
    escapes_base = False
    if config.base_dir:
        norm_base = normalize_lexical_path(config.base_dir)
        escapes_base = not _is_subpath(
            norm_path,
            norm_base,
            case_insensitive=_is_windows_path(norm_path),
        )

    if has_traversal_token or escapes_norm or escapes_base:
        return _finding(
            "path_traversal",
            "critical",
            "path traversal sequence detected",
            norm_path,
        )
    return None


def rule_sensitive_locations(
    path: str,
    norm_path: str,
    config: PathSecurityConfig,
) -> PathFinding | None:
    """Detect attempts to access known credentials, secrets, and system paths."""
    lower_norm = norm_path.lower()
    basename = posixpath.basename(norm_path).lower()

    # Basename matches
    if basename in _SENSITIVE_BASENAMES or basename.startswith(".env."):
        return _finding(
            "sensitive_locations",
            "critical",
            "access to sensitive credential or environment file",
            norm_path,
        )

    # Extension matches (keys and certificates)
    for ext in _SENSITIVE_EXTENSIONS:
        if basename.endswith(ext):
            return _finding(
                "sensitive_locations",
                "critical",
                "access to cryptographic secret or key file",
                norm_path,
            )

    # Sensitive path prefixes
    for prefix in _SENSITIVE_PREFIXES:
        if _is_subpath(lower_norm, prefix):
            return _finding(
                "sensitive_locations",
                "critical",
                "access to restricted system location",
                norm_path,
            )

    # Sensitive directory segments (e.g., ~/.ssh/, ~/.aws/)
    for seg in _SENSITIVE_DIR_SEGMENTS:
        if seg in lower_norm or lower_norm.startswith(seg.lstrip("/")):
            return _finding(
                "sensitive_locations",
                "critical",
                "access to sensitive user configuration or key store",
                norm_path,
            )

    # Git internal config files
    if "/.git/config" in lower_norm or lower_norm.endswith("/.git/config"):
        return _finding(
            "sensitive_locations",
            "critical",
            "access to git configuration store",
            norm_path,
        )

    return None


def rule_allowed_roots(
    path: str,
    norm_path: str,
    config: PathSecurityConfig,
) -> PathFinding | None:
    """Verify that path resides strictly inside one of the configured allowed roots."""
    if config.allowed_roots is None:
        return None

    if not config.allowed_roots:
        return _finding(
            "allowed_roots",
            "critical",
            "path is outside configured allowed roots",
            norm_path,
        )

    is_win = _is_windows_path(norm_path)
    # Check against each allowed root
    for root in config.allowed_roots:
        norm_root = normalize_lexical_path(root, base_dir=config.base_dir)
        if _is_subpath(norm_path, norm_root, case_insensitive=is_win):
            return None

    return _finding(
        "allowed_roots",
        "critical",
        "path is outside configured allowed roots",
        norm_path,
    )


def rule_denied_paths(
    path: str,
    norm_path: str,
    config: PathSecurityConfig,
) -> PathFinding | None:
    """Detect attempts to access explicitly denied directory paths."""
    if not config.denied_paths:
        return None

    is_win = _is_windows_path(norm_path)
    for denied in config.denied_paths:
        norm_denied = normalize_lexical_path(denied, base_dir=config.base_dir)
        if _is_subpath(norm_path, norm_denied, case_insensitive=is_win):
            return _finding(
                "denied_paths",
                "critical",
                "access to explicitly denied path",
                norm_path,
            )
    return None


def rule_denied_patterns(
    path: str,
    norm_path: str,
    config: PathSecurityConfig,
) -> PathFinding | None:
    """Detect files matching explicitly denied glob patterns."""
    if not config.denied_patterns:
        return None

    basename = posixpath.basename(norm_path)
    is_win = _is_windows_path(norm_path)
    target_base = basename.lower() if is_win else basename
    target_path = norm_path.lower() if is_win else norm_path

    for pattern in config.denied_patterns:
        pat = pattern.lower() if is_win else pattern
        if fnmatch.fnmatch(target_base, pat) or fnmatch.fnmatch(target_path, pat):
            return _finding(
                "denied_patterns",
                "critical",
                "path matches denied pattern",
                norm_path,
            )
    return None


def _finding(
    rule_id: str,
    severity: PathSeverity,
    reason: str,
    normalized_path: str,
) -> PathFinding:
    return PathFinding(
        rule_id=rule_id,
        severity=severity,
        reason=reason,
        path_signature=make_signature("filesystem_path", None, normalized_path),
    )


# ---------------------------------------------------------------------------
# Rule registry and inspection engine
# ---------------------------------------------------------------------------

BUILTIN_RULES: tuple[tuple[str, PathRuleFn], ...] = (
    ("path_traversal", rule_path_traversal),
    ("sensitive_locations", rule_sensitive_locations),
    ("allowed_roots", rule_allowed_roots),
    ("denied_paths", rule_denied_paths),
    ("denied_patterns", rule_denied_patterns),
)

BUILTIN_RULE_IDS: frozenset[str] = frozenset(rule_id for rule_id, _ in BUILTIN_RULES)


def inspect_path(
    path: str | os.PathLike[str],
    config: PathSecurityConfig | None = None,
    *,
    extra_rules: Sequence[tuple[str, PathRuleFn]] = (),
) -> tuple[PathFinding, ...]:
    """Inspect ``path`` and return zero or more findings.

    Pure function: no filesystem access, no stat calls, no subprocesses.
    Empty or whitespace-only input yields no findings.
    """
    raw_str = os.fspath(path) if not isinstance(path, str) else path
    if not raw_str or not raw_str.strip():
        return ()

    cfg = config or PathSecurityConfig()

    # Reject embedded null bytes immediately as a critical traversal/security violation
    if "\0" in raw_str:
        return (
            _finding(
                "path_traversal",
                "critical",
                "null byte injection in file path",
                "null_byte_detected",
            ),
        )

    norm_path = normalize_lexical_path(raw_str, base_dir=cfg.base_dir)
    active_builtin = _active_rule_ids(cfg)

    findings: list[PathFinding] = []
    for rule_id, fn in (*BUILTIN_RULES, *extra_rules):
        if rule_id in BUILTIN_RULE_IDS:
            if rule_id not in active_builtin:
                continue
        else:
            if cfg.enabled_rules is not None and rule_id not in cfg.enabled_rules:
                continue
            if rule_id in cfg.disabled_rules:
                continue

        hit = fn(raw_str, norm_path, cfg)
        if hit is None:
            continue

        override = cfg.severity_overrides.get(hit.rule_id)
        if override is not None:
            hit = PathFinding(
                rule_id=hit.rule_id,
                severity=override,
                reason=hit.reason,
                path_signature=hit.path_signature,
            )
        findings.append(hit)

    # Stable deterministic order: severity desc, then rule_id asc.
    findings.sort(key=lambda f: (-_SEVERITY_RANK[f.severity], f.rule_id))
    return tuple(findings)


def _active_rule_ids(cfg: PathSecurityConfig) -> frozenset[str]:
    base = (
        BUILTIN_RULE_IDS if cfg.enabled_rules is None else frozenset(cfg.enabled_rules)
    )
    return frozenset(base - frozenset(cfg.disabled_rules))


def strongest_severity(
    findings: Sequence[PathFinding],
) -> PathSeverity | None:
    """Highest severity among findings, or ``None`` when empty."""
    if not findings:
        return None
    return max(findings, key=lambda f: _SEVERITY_RANK[f.severity]).severity


def findings_at_or_above(
    findings: Sequence[PathFinding],
    minimum: PathSeverity,
) -> tuple[PathFinding, ...]:
    """Filter findings to those at least as severe as ``minimum``."""
    floor = _SEVERITY_RANK[minimum]
    return tuple(f for f in findings if _SEVERITY_RANK[f.severity] >= floor)


# ---------------------------------------------------------------------------
# Authorization engine integration
# ---------------------------------------------------------------------------

_FILE_ACTIONS = frozenset({"read", "write", "filesystem"})


def _looks_like_path(val: str) -> bool:
    return (
        val.startswith(("/", "./", "../", "~"))
        or "\\" in val
        or bool(_WINDOWS_DRIVE_RE.match(val))
    )


def as_authorize_rule(
    config: PathSecurityConfig | None = None,
    *,
    actions: frozenset[str] = _FILE_ACTIONS,
):
    """Build a ``snagline.authorize.Rule`` from path security findings.

    Lazy-imports authorize so this module stays usable without pulling the
    whole policy engine. Non-filesystem ``ToolRequest`` values are ignored.
    """
    from snagline.authorize import RuleVerdict, ToolRequest

    cfg = config or PathSecurityConfig()

    def _rule(request: ToolRequest) -> RuleVerdict | None:
        if not request.resource:
            return None

        is_targeted_action = request.action in actions or (
            request.action == "unknown" and _looks_like_path(request.resource)
        )
        if not is_targeted_action:
            return None

        if len(request.resource) > 4096 or request.resource.endswith("...<truncated>"):
            return RuleVerdict(
                rule_id="path_security",
                decision=cfg.decision,
                reason="resource path exceeds maximum length or is truncated",
            )

        hits = findings_at_or_above(
            inspect_path(request.resource, cfg),
            cfg.min_severity,
        )
        if not hits:
            return None

        top = hits[0]
        ids = ",".join(h.rule_id for h in hits)
        return RuleVerdict(
            rule_id="path_security",
            decision=cfg.decision,
            reason=(
                f"filesystem security ({ids}) severity={top.severity} "
                f"sig={top.path_signature[:12]}"
            ),
        )

    return _rule
