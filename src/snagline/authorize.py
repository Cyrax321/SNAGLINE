"""Pre-execution authorization: decide allow/ask/deny BEFORE a tool runs.

SNAGLINE's whole promise is that it watches and never interferes
(README: "Hard fail-open guarantee: it can never crash or stall the host").
That guarantee covers the *detection* path -- ``Monitor.ingest()`` observes a
step after it happened and dispatches a ``FailureRisk`` to sinks. Nothing in
the core has ever been able to say "no" to a tool call, and that is by
design: a monitor that can block is a policy engine, and a policy engine
that fails open is a security control with a hole in it.

This module is that policy engine, kept deliberately outside the core:

* It is NOT reached from ``Monitor.ingest()``. The detection path cannot call
  it, so ``fail_open=True`` remains a structural guarantee rather than a
  promise about behaviour.
* It is reached only at an adapter boundary, where a host already asks us
  "may I run this tool?" -- the Claude Code ``PreToolUse`` hook, or any host
  that calls ``decide()`` itself before dispatching a tool call.
* It is fail-OPEN by default. An unknown tool, a policy that fails to load,
  or a rule that raises all resolve to ``allow`` unless the operator opts into
  strictness, because an unused gate that breaks the host is worse than no
  gate. Opting in is explicit and per-policy, never global.

What it deliberately does NOT do, so the boundary is not misread: it does not
inspect content for secrets or PII, does not score risk, and does not persist
an audit trail. A policy is a set of rules over *which tool is being called on
what resource*; adding detection heuristics to it would blur the line between
the monitor and the firewall, and would put content handling code on a path
that runs before a user has approved anything.
"""

from __future__ import annotations

import fnmatch
import json
import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

from snagline.events import make_signature

Decision = Literal["allow", "ask", "deny"]

#: Every legal decision, most-permissive first. Ordering is load-bearing:
#: :func:`_rank` uses it so the strictest verdict wins regardless of rule order.
DECISIONS: tuple[Decision, ...] = ("allow", "ask", "deny")

#: What a policy does when no rule has an opinion. Fail-open, and named here so
#: the intent is greppable rather than buried at the constructor default.
DEFAULT_DECISION: Decision = "allow"

#: Predefined security profiles.
PROFILE_FAIL_CLOSED = "fail_closed"
PROFILE_PERMISSIVE = "permissive"
VALID_PROFILES = frozenset({PROFILE_FAIL_CLOSED, PROFILE_PERMISSIVE, "strict"})

#: tool_name -> the key in ``tool_input`` holding the thing being acted on.
#: tool_name -> (resource keys, first match wins, action).
#:
#: The key list is ordered because a tool can carry several plausible
#: resources and only one of them is what a path policy is actually about.
#: ``Grep`` and ``Glob`` send both ``path`` and ``pattern``: ``pattern`` is the
#: search EXPRESSION, and ``path`` is the directory being searched. Taking
#: ``pattern`` let a Grep scoped to ``path=/etc/ssh`` with ``pattern=*.pem``
#: walk straight past a ``deny_resource_glob("/etc/ssh/*")`` rule, because the
#: rule was matching the expression rather than the protected directory. The
#: searched directory is the resource; ``pattern`` is only a fallback for a
#: bare search with no path, where it is still the best thing to match on.
#:
#: Mirrors Claude Code's own tool set; an unlisted tool still gets a decision,
#: it just arrives with ``action="unknown"``.
_TOOL_RESOURCES: dict[str, tuple[tuple[str, ...], str]] = {
    "Bash": (("command",), "execute"),
    "Read": (("file_path",), "read"),
    "Write": (("file_path",), "write"),
    "Edit": (("file_path",), "write"),
    "MultiEdit": (("file_path",), "write"),
    "NotebookEdit": (("notebook_path",), "write"),
    "Glob": (("path", "pattern"), "read"),
    "Grep": (("path", "pattern"), "read"),
    "WebFetch": (("url",), "network"),
    "WebSearch": (("query",), "network"),
}


def _rank(decision: Decision) -> int:
    """Strictness order: deny beats ask beats allow.

    Used instead of first-match-wins so that rule ORDER cannot change the
    outcome. An allowlist written above a denylist is a bug that silently
    re-opens whatever the denylist was closing; taking the strictest verdict
    makes that class of mistake unrepresentable.
    """
    return DECISIONS.index(decision)


@dataclass(frozen=True, slots=True)
class ToolRequest:
    """One tool call, reduced to what a policy is allowed to reason about.

    ``resource`` is the path / command / URL the tool was pointed at, or
    ``None`` when it cannot be identified. It is a *string the host already
    sent us*, not content SNAGLINE derived.

    ``tool_input`` is a REDACTED copy (:func:`_redact`): content-bearing keys
    (``Write.content``, ``Edit.new_string``, ``prompt``, ...) survive as keys
    with their values replaced by a marker, and oversized structural values are
    truncated. Keeping the raw mapping would put file bodies in memory and in
    this object's ``repr``, which would break the same no-content-retention
    invariant that governs ``FailureRisk``. A rule that genuinely needs a
    field this module redacts should receive it through a host-side rule that
    does not retain it.

    ``signature`` is a one-way SHA-256 digest of the request, offered for
    logging an allow/deny decision without recording what was requested. It
    reuses :func:`snagline.events.make_signature` so there is exactly one
    hashing convention in the codebase.
    """

    tool_name: str
    action: str = "unknown"
    resource: str | None = None
    tool_input: Mapping[str, Any] = field(default_factory=dict)
    episode_id: str | None = None
    agent_id: str | None = None

    @property
    def signature(self) -> str:
        """One-way digest of (tool, action, resource, agent). Safe to log."""
        parts = [self.action, self.resource or ""]
        if self.agent_id:
            parts.append(self.agent_id)
        return make_signature("tool_request", self.tool_name, *parts)

    @classmethod
    def from_tool_input(
        cls,
        tool_name: str,
        tool_input: Mapping[str, Any] | None,
        *,
        episode_id: str | None = None,
        agent_id: str | None = None,
    ) -> ToolRequest:
        """Build a request from a host's raw ``(tool_name, tool_input)``.

        Best-effort by construction: a tool this module has never seen, an
        input that is not a mapping, or a resource that is not a string all
        degrade to ``action="unknown"`` / ``resource=None`` rather than
        raising. The gate must not be the reason a host crashes.
        """
        data: Mapping[str, Any] = tool_input if isinstance(tool_input, Mapping) else {}
        # Redact ONCE and read the resource out of the redacted copy. Reading
        # it from the raw mapping would leave an oversized value unbounded in
        # ``resource`` while ``tool_input`` was truncated, so the truncation
        # would be trivially bypassed by the very field a policy matches on.
        safe = _redact(data)
        if agent_id is None:
            raw_agent = data.get("agent_id")
            if isinstance(raw_agent, str) and raw_agent.strip():
                agent_id = raw_agent.strip()
        known = _TOOL_RESOURCES.get(tool_name)
        if known is None:
            return cls(
                tool_name=tool_name,
                action="unknown",
                resource=_first_str(safe, ("url", "path", "file_path", "command")),
                tool_input=safe,
                episode_id=episode_id,
                agent_id=agent_id,
            )
        keys, action = known
        return cls(
            tool_name=tool_name,
            action=action,
            resource=_first_str(safe, keys),
            tool_input=safe,
            episode_id=episode_id,
            agent_id=agent_id,
        )


#: Input keys whose values are CONTENT the agent is about to send somewhere,
#: not a structural description of the request. Their values are replaced with
#: a marker so a rule can still see that content was supplied, without this
#: object carrying a file body in memory or in its repr.
_CONTENT_KEYS = frozenset(
    {
        "content",
        "contents",
        "new_string",
        "old_string",
        "text",
        "body",
        "message",
        "prompt",
        "patch",
        "edits",
        "script",
        "code",
        "input",
        "data",
        "value",
        "replacement",
        "insert_text",
    }
)

#: Cap on a retained structural value. A policy matches on paths and patterns;
#: nobody writes one against a 4 MB search expression, and this object is built
#: once per tool call, so an unbounded value is a per-call memory hazard.
_MAX_RESOURCE_CHARS = 4096

_CONTENT_MARKER = "<content redacted>"


#: How deep :func:`_redact` walks into nested dicts and lists. Hosts nest
#: content under structural keys (``{"notebook": {"content": ...}}``,
#: ``{"changes": [{"new_string": ...}]}``); anything deeper is collapsed to the
#: marker instead of being retained.
_REDACT_DEPTH = 3


def _redact_value(value: Any, depth: int) -> Any:
    """Redact one tool-input value, walking dicts/lists up to ``depth`` levels."""
    if isinstance(value, Mapping):
        if depth <= 0:
            return _CONTENT_MARKER
        return {
            key: _CONTENT_MARKER
            if key in _CONTENT_KEYS
            else _redact_value(item, depth - 1)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        if depth <= 0:
            return _CONTENT_MARKER
        items = [_redact_value(item, depth - 1) for item in value]
        return tuple(items) if isinstance(value, tuple) else items
    if isinstance(value, str) and len(value) > _MAX_RESOURCE_CHARS:
        return value[:_MAX_RESOURCE_CHARS] + "...<truncated>"
    return value


def _redact(data: Mapping[str, Any]) -> dict[str, Any]:
    """Copy ``data`` with content-bearing values replaced by a marker.

    SNAGLINE's privacy invariant is that detectors reason about hashes, timings
    and counts -- never prompt or response content (README, "No content
    retention"). Holding a whole ``tool_input`` verbatim on a request object
    that a custom rule can inspect would quietly break that: ``Write.content``
    and ``Edit.new_string`` are the file bodies a host is about to write, and
    they would sit in ``ToolRequest.tool_input`` and in its ``repr``. Keys are
    preserved and values replaced, so ``"content" in request.tool_input`` still
    answers "was content supplied?" without answering "what was it?".

    The walk is recursive (dicts and lists, bounded by :data:`_REDACT_DEPTH`):
    content nested under a structural key -- an unknown tool's
    ``{"notebook": {"content": ...}}``, an ``edits`` list, an MCP-served tool --
    is redacted too, and the size cap applies to nested strings as well. At the
    depth limit a container collapses to the marker rather than being kept.
    """
    return {
        key: _CONTENT_MARKER
        if key in _CONTENT_KEYS
        else _redact_value(value, _REDACT_DEPTH)
        for key, value in data.items()
    }


def _first_str(data: Mapping[str, Any], keys: Iterable[str]) -> str | None:
    """First key in ``keys`` whose value is a non-empty string, else ``None``.

    A non-string value (``file_path`` arriving as a list, say) is treated as
    absent rather than stringified: guessing at a resource makes a policy
    match against something the host never asked for.
    """
    for key in keys:
        value = data.get(key)
        if isinstance(value, str) and value:
            return value
    return None


@dataclass(frozen=True, slots=True)
class RuleVerdict:
    """One rule's answer. ``decision=None`` means "I have no opinion"."""

    rule_id: str
    decision: Decision | None = None
    reason: str = ""


#: A rule is any callable over one request. Returning ``None`` (or a verdict
#: with no decision) means "no opinion" and does not constrain the outcome.
Rule = Callable[[ToolRequest], RuleVerdict | None]


def deny_tools(*tool_names: str) -> Rule:
    """Deny these exact tool names."""
    names = frozenset(tool_names)
    return lambda request: RuleVerdict(
        "deny_tools",
        "deny" if request.tool_name in names else None,
        f"tool {request.tool_name!r} is denied by policy",
    )


def ask_tools(*tool_names: str) -> Rule:
    """Ask the operator about these exact tool names."""
    names = frozenset(tool_names)
    return lambda request: RuleVerdict(
        "ask_tools",
        "ask" if request.tool_name in names else None,
        f"tool {request.tool_name!r} needs operator approval",
    )


def allow_tools(*tool_names: str) -> Rule:
    """Explicitly allow these tools.

    Only meaningful under a ``default_decision`` stricter than ``allow``: it is
    how an allowlist is expressed, and it documents the intent even when it is
    currently redundant.
    """
    names = frozenset(tool_names)
    return lambda request: RuleVerdict(
        "allow_tools",
        "allow" if request.tool_name in names else None,
        f"tool {request.tool_name!r} is explicitly allowed",
    )


def deny_resource_glob(*patterns: str) -> Rule:
    """Deny when the requested resource matches any shell-style glob.

    ``fnmatchcase`` semantics: case-sensitive on every OS, so the same policy
    gives the same verdict on Windows, Linux and macOS. ``*.pem`` and
    ``/etc/ssh/*`` behave the way an operator reading the policy expects.
    Matching is against ``request.resource`` only; ``None`` never matches, so a
    request with no identifiable resource is decided by the rest of the policy
    rather than by this rule.
    """
    return lambda request: RuleVerdict(
        "deny_resource_glob",
        "deny"
        if request.resource is not None
        and any(fnmatch.fnmatchcase(request.resource, pat) for pat in patterns)
        else None,
        f"resource {request.resource!r} matches a denied pattern",
    )


def deny_command_glob(*patterns: str) -> Rule:
    """Deny when an ``execute`` request's command matches any glob.

    Narrower than :func:`deny_resource_glob` on purpose: a command string and a
    file path live in different risk classes, and one rule that covers both
    invites a pattern written for the wrong one. Matching is case-sensitive on
    every OS, like :func:`deny_resource_glob`.
    """
    return lambda request: RuleVerdict(
        "deny_command_glob",
        "deny"
        if request.action == "execute"
        and request.resource is not None
        and any(fnmatch.fnmatchcase(request.resource, pat) for pat in patterns)
        else None,
        f"command matches a denied pattern: {request.resource!r}",
    )


def allow_agent_tools(allowlists: Mapping[str, Sequence[str]]) -> Rule:
    """Authorize tools according to per-agent allowlists.

    Schema: ``{"agent_name": ["ToolA", "ToolB"]}``.

    When a request carries an ``agent_id``:
    * If the agent is in ``allowlists`` and the requested tool is in that
      agent's allowed tools, returns an 'allow' verdict.
    * If the agent is in ``allowlists`` but the requested tool is not allowed,
      returns a 'deny' verdict.
    * If the agent is not in ``allowlists``, returns a 'deny' verdict (an
      unconfigured agent cannot execute tools under this rule).

    When a request carries no ``agent_id``:
    * Returns None (passes through to subsequent rules).
    """
    if not isinstance(allowlists, Mapping):
        raise ValueError(
            "agent_allowlists must be an object mapping agent_id to tool list"
        )

    cleaned: dict[str, frozenset[str]] = {}
    for agent, tools in allowlists.items():
        if not isinstance(agent, str) or not agent.strip():
            raise ValueError(f"agent_id {agent!r} must be a non-empty string")
        if not isinstance(tools, (list, tuple, set, frozenset)) or not all(
            isinstance(t, str) and t.strip() for t in tools
        ):
            raise ValueError(
                f"tools for agent {agent!r} must be a list of non-empty strings"
            )
        cleaned[agent.strip()] = frozenset(t.strip() for t in tools)

    def _rule(request: ToolRequest) -> RuleVerdict | None:
        agent_id = request.agent_id
        if agent_id is None:
            return None
        allowed = cleaned.get(agent_id)
        if allowed is None:
            return RuleVerdict(
                "agent_allowlists",
                "deny",
                f"agent {agent_id!r} has no configured tool allowlist",
            )
        if request.tool_name in allowed:
            return RuleVerdict(
                "agent_allowlists",
                "allow",
                f"tool {request.tool_name!r} is allowed for agent {agent_id!r}",
            )
        return RuleVerdict(
            "agent_allowlists",
            "deny",
            f"tool {request.tool_name!r} is not allowed for agent {agent_id!r}",
        )

    return _rule


@dataclass(frozen=True, slots=True)
class DecisionResult:
    """The outcome of one :func:`decide` call, with the reasoning kept."""

    decision: Decision
    reason: str
    #: Ids of every rule that had an opinion, strictest first. Kept so an
    #: operator can see *all* the reasons, not just the winning one -- a deny
    #: that another rule also flagged is worth knowing about.
    matched_rules: tuple[str, ...] = ()

    @property
    def allowed(self) -> bool:
        """True only for an outright ``allow`` -- ``ask`` is deliberately False.

        ``ask`` means "a human has not approved this yet". Folding it into
        ``allowed`` because it is merely "not a denial" made the obvious
        caller mistake -- ``if policy.decide(req): run_the_tool()`` -- skip
        the very approval the policy asked for. An unapproved request now reads
        as not-proceedable, and a caller who wants the approval flow has to
        say so explicitly.
        """
        return self.decision == "allow"

    def __bool__(self) -> bool:
        """``if result:`` reads as "proceed with no further approval needed"."""
        return self.allowed


_DEFAULT_UNSET: Any = object()


@dataclass(frozen=True, slots=True)
class Policy:
    """A set of rules plus what to do when they have no opinion.

    Fail-open by default, and that default is the safe one to ship:

    * ``default_decision="allow"`` -- no rule matched, so allow.
    * ``rule_error_decision=None`` -- a rule raised, so behave as if nothing
      matched. A crashing rule must not be able to wedge the host.
    * ``unknown_tool_decision=None`` -- a tool this module does not know was
      requested, so behave as if nothing matched.

    The three strict postures are opt-in and spelled out, never defaults::

        Policy(rules=(allow_tools("Read", "Grep"),), default_decision="deny")

    which is an allowlist -- anything not named is denied, including every tool
    released after the policy was written. That is the fail-closed mode, and
    it fails closed by design rather than by accident.

    For security-sensitive deployments, the fail-closed security profile
    ensures:
    * Unknown tools are denied.
    * Rule errors fail closed to deny.
    * Unmatched requests are denied.

        Policy(profile="fail_closed", rules=(allow_tools("Read", "Grep"),))
    """

    rules: tuple[Rule, ...] = ()
    default_decision: Decision = _DEFAULT_UNSET
    rule_error_decision: Decision | None = None
    unknown_tool_decision: Decision | None = None
    profile: str | None = None

    def __post_init__(self) -> None:
        if self.profile is not None and self.profile not in VALID_PROFILES:
            raise ValueError(
                f"profile={self.profile!r} is not a valid security profile; "
                f"expected one of {sorted(VALID_PROFILES)}"
            )
        if self.default_decision is _DEFAULT_UNSET:
            resolved_default: Decision = (
                "deny"
                if self.profile in (PROFILE_FAIL_CLOSED, "strict")
                else DEFAULT_DECISION
            )
            object.__setattr__(self, "default_decision", resolved_default)
        elif self.default_decision not in DECISIONS:
            raise ValueError(
                f"default_decision={self.default_decision!r} is not a valid decision; "
                f"expected one of {list(DECISIONS)}"
            )
        if (
            self.rule_error_decision is not None
            and self.rule_error_decision not in DECISIONS
        ):
            raise ValueError(
                f"rule_error_decision={self.rule_error_decision!r} is not a valid decision; "
                f"expected one of {list(DECISIONS)}"
            )
        if (
            self.unknown_tool_decision is not None
            and self.unknown_tool_decision not in DECISIONS
        ):
            raise ValueError(
                f"unknown_tool_decision={self.unknown_tool_decision!r} is not a valid decision; "
                f"expected one of {list(DECISIONS)}"
            )
        if self.profile in (PROFILE_FAIL_CLOSED, "strict"):
            if self.rule_error_decision is None:
                object.__setattr__(self, "rule_error_decision", "deny")
            if self.unknown_tool_decision is None:
                object.__setattr__(self, "unknown_tool_decision", "deny")

    def decide(self, request: ToolRequest) -> DecisionResult:
        """Evaluate every rule and return the strictest verdict."""
        if request.action == "unknown":
            override = self.unknown_tool_decision
            if override is not None:
                # Short-circuits the rules: the operator has said what to do
                # with tools we do not recognise, and that outranks whatever
                # the rules would have made of it.
                return DecisionResult(
                    override,
                    f"tool {request.tool_name!r} is unknown; "
                    f"policy unknown_tool_decision is {override!r}",
                    (),
                )
            # Otherwise fall through and run the rules: an operator may have
            # named that tool explicitly, and an explicit rule is better
            # knowledge than our own ignorance of the tool set.

        opinions: list[tuple[Decision, RuleVerdict]] = []
        for index, rule in enumerate(self.rules):
            try:
                verdict = rule(request)
            except Exception as exc:  # noqa: BLE001 - a rule must not wedge the host
                fallback = (
                    self.rule_error_decision
                    if self.rule_error_decision is not None
                    else self.default_decision
                )
                return DecisionResult(
                    fallback,
                    f"rule {index} raised {type(exc).__name__}: {exc}",
                    (),
                )
            if verdict is not None and verdict.decision is not None:
                # Paired with the narrowed decision so the strictness sort
                # below never has to re-narrow an Optional field.
                opinions.append((verdict.decision, verdict))

        if not opinions:
            # Name the default in the reason. "no rule matched" reads like a
            # bug to an operator staring at a deny they did not write; "no rule
            # matched; policy default_decision is 'deny'" tells them this is an
            # allowlist and the tool is simply not on it. (The unknown-tool
            # case already returned at the top of this method.)
            return DecisionResult(
                self.default_decision,
                f"no rule matched; policy default_decision is "
                f"{self.default_decision!r}",
                (),
            )

        # Strictest wins, so rule order cannot change the outcome. ``max`` keeps
        # first-seen order among ties, which makes the reported reason stable
        # across runs instead of depending on sort implementation details.
        best_decision, best = max(opinions, key=lambda pair: _rank(pair[0]))
        ordered = tuple(
            verdict.rule_id
            for _, verdict in sorted(opinions, key=lambda pair: -_rank(pair[0]))
        )
        return DecisionResult(best_decision, best.reason, ordered)


def decide(request: ToolRequest, policy: Policy) -> DecisionResult:
    """Module-level shorthand for ``policy.decide(request)``."""
    return policy.decide(request)


def security_policy(
    rules: Sequence[Rule] = (),
    *,
    allowed_tools: Sequence[str] | None = None,
    agent_allowlists: Mapping[str, Sequence[str]] | None = None,
    default_decision: Decision = "deny",
    rule_error_decision: Decision = "deny",
    unknown_tool_decision: Decision = "deny",
) -> Policy:
    """Build a fail-closed security policy.

    In a security policy:
    * Unconfigured tools are denied (fails closed).
    * Rule evaluation errors result in deny rather than falling open.
    * Requests matching no rule are denied by default.
    * Optional ``allowed_tools`` configures global tool permissions.
    * Optional ``agent_allowlists`` configures per-agent tool permissions.
    """
    built_rules: list[Rule] = []
    if agent_allowlists is not None:
        built_rules.append(allow_agent_tools(agent_allowlists))
    if allowed_tools is not None:
        built_rules.append(allow_tools(*allowed_tools))
    built_rules.extend(rules)
    return Policy(
        rules=tuple(built_rules),
        default_decision=default_decision,
        rule_error_decision=rule_error_decision,
        unknown_tool_decision=unknown_tool_decision,
        profile=PROFILE_FAIL_CLOSED,
    )


# --------------------------------------------------------------------------
# Declarative policy documents
# --------------------------------------------------------------------------


class PolicyError(ValueError):
    """A policy document could not be understood. Always fatal at load time.

    Deliberately NOT swallowed into a fail-open allow. A policy that silently
    fails to parse and then permits everything is indistinguishable from no
    policy at all, which is the exact failure mode an operator is trying to
    avoid by writing one. ``Policy.load_file`` raises; the caller decides what
    an unloadable policy means for their host.
    """


_ALLOWED_KEYS = frozenset(
    {
        "rules",
        "default_decision",
        "rule_error_decision",
        "unknown_tool_decision",
        "profile",
        "agent_allowlists",
        "description",
    }
)


def policy_from_dict(data: Mapping[str, Any]) -> Policy:
    """Build a :class:`Policy` from a plain mapping.

    Schema (the ``description`` key is ignored, so a policy file can be
    self-documenting)::

        {
          "profile": "fail_closed",
          "default_decision": "deny",
          "agent_allowlists": {
            "researcher": ["Read", "WebSearch"]
          },
          "rules": [
            {"type": "deny_tools",     "tools": ["Write", "Edit"]},
            {"type": "allow_tools",    "tools": ["Read", "Grep"]},
            {"type": "deny_resource",  "patterns": ["*.pem", "/etc/ssh/*"]},
            {"type": "deny_command",   "patterns": ["rm -rf*"]},
            {"type": "shell_threats",  "decision": "deny", "min_severity": "warning"}
          ]
        }

    Unknown top-level keys and unknown rule ``type`` values raise rather than
    being skipped: a typo'd rule that silently does nothing is the same
    invisible-failure problem :class:`PolicyError` exists to prevent.
    """
    unknown = set(data) - _ALLOWED_KEYS
    if unknown:
        raise PolicyError(
            f"unknown policy keys {sorted(unknown)}; expected {sorted(_ALLOWED_KEYS)}"
        )

    raw_profile: str | None = None
    if "profile" in data:
        val = data["profile"]
        if val is None:
            raise PolicyError(
                "profile is null. Omit the key to leave it unset; use one of "
                f"{sorted(VALID_PROFILES)} to set it explicitly."
            )
        if not isinstance(val, str) or val not in VALID_PROFILES:
            raise PolicyError(
                f"profile={val!r} is not a valid security profile; "
                f"expected one of {sorted(VALID_PROFILES)}"
            )
        raw_profile = val

    agent_rules: list[Rule] = []
    if "agent_allowlists" in data:
        raw_allowlists = data["agent_allowlists"]
        if raw_allowlists is None:
            raise PolicyError(
                "agent_allowlists is null. Omit the key or provide an object mapping agent_id to tool list."
            )
        if not isinstance(raw_allowlists, Mapping):
            raise PolicyError(
                f"agent_allowlists must be an object; got {type(raw_allowlists).__name__}"
            )
        try:
            agent_rules.append(allow_agent_tools(raw_allowlists))
        except ValueError as exc:
            raise PolicyError(str(exc)) from exc

    def decision(key: str) -> Decision | None:
        """Read one decision key.

        Absent means "not configured" and yields ``None``. An explicit
        ``null`` is a DIFFERENT thing -- a policy store or a YAML-to-JSON
        conversion that emits every field, nulling the unset ones -- and it is
        rejected rather than read as absent. Silently resolving an operator's
        ``"default_decision": null`` to ``allow`` would turn a typo into a
        policy that permits everything, which is the exact silent-failure this
        loader refuses to produce anywhere else.
        """
        if key not in data:
            return None
        value = data[key]
        if value is None:
            raise PolicyError(
                f"{key} is null. Omit the key to leave it unset; use one of "
                f"{list(DECISIONS)} to set it explicitly."
            )
        if value not in DECISIONS:
            raise PolicyError(
                f"{key}={value!r} is not a decision; expected one of {list(DECISIONS)}"
            )
        return value

    raw_rules = data.get("rules", [])
    if not isinstance(raw_rules, list):
        raise PolicyError(f"rules must be a list; got {type(raw_rules).__name__}")
    parsed_rules = tuple(_rule_from_dict(raw, i) for i, raw in enumerate(raw_rules))
    rules = tuple(agent_rules) + parsed_rules

    default = decision("default_decision")
    rule_error = decision("rule_error_decision")
    unknown_tool = decision("unknown_tool_decision")

    if raw_profile in (PROFILE_FAIL_CLOSED, "strict"):
        default = default or "deny"
        rule_error = rule_error or "deny"
        unknown_tool = unknown_tool or "deny"
    else:
        default = default or DEFAULT_DECISION

    return Policy(
        rules=rules,
        default_decision=default,
        rule_error_decision=rule_error,
        unknown_tool_decision=unknown_tool,
        profile=raw_profile,
    )


def _rule_from_dict(raw: Any, index: int) -> Rule:
    if not isinstance(raw, Mapping):
        raise PolicyError(f"rules[{index}] must be an object; got {type(raw).__name__}")
    kind = raw.get("type")
    # Shell threat detection (issue #638) is a structured rule, not a
    # tools/patterns list. Handled before the simple builders so a typo'd
    # shell_threats entry still gets a precise schema error.
    if kind == "shell_threats":
        return _shell_threats_rule_from_dict(raw, index)
    if kind == "path_security":
        return _path_security_rule_from_dict(raw, index)
    if kind == "agent_allowlists":
        allowed_keys = {"type", "allowlists"}
        unknown_keys = set(raw) - allowed_keys
        if unknown_keys:
            paths = ", ".join(f"rules[{index}].{k}" for k in sorted(unknown_keys))
            raise PolicyError(
                f"unknown rule keys {paths}; expected one of {sorted(allowed_keys)}"
            )
        if "allowlists" not in raw:
            raise PolicyError(
                f"rules[{index}].allowlists is missing; expected an object mapping agent_id to tool list"
            )
        allowlists = raw["allowlists"]
        if not isinstance(allowlists, Mapping):
            raise PolicyError(
                f"rules[{index}].allowlists must be an object; got {type(allowlists).__name__}"
            )
        try:
            return allow_agent_tools(allowlists)
        except ValueError as exc:
            raise PolicyError(f"rules[{index}]: {exc}") from exc
    builders: dict[str, Callable[..., Rule]] = {
        "deny_tools": deny_tools,
        "ask_tools": ask_tools,
        "allow_tools": allow_tools,
        "deny_resource": deny_resource_glob,
        "deny_command": deny_command_glob,
    }
    if kind not in builders:
        raise PolicyError(
            f"rules[{index}].type={kind!r} is not a known rule type; "
            f"expected one of {sorted([*builders, 'shell_threats', 'path_security', 'agent_allowlists'])}"
        )
    key = "patterns" if kind.startswith("deny_") and kind != "deny_tools" else "tools"
    allowed = {"type", key}
    unknown = set(raw) - allowed
    if unknown:
        # A typo'd key used to be ignored, which built a rule with an empty
        # list: one that silently matches nothing and denies nothing.
        paths = ", ".join(f"rules[{index}].{k}" for k in sorted(unknown))
        raise PolicyError(
            f"unknown rule keys {paths}; expected one of {sorted(allowed)}"
        )
    if key not in raw:
        raise PolicyError(
            f"rules[{index}].{key} is missing; expected a list of strings "
            f"(use an explicit [] for an intentionally empty rule)"
        )
    values = raw[key]
    if not isinstance(values, list) or not all(isinstance(v, str) for v in values):
        raise PolicyError(f"rules[{index}].{key} must be a list of strings")
    return builders[kind](*values)


def _shell_threats_rule_from_dict(raw: Mapping[str, Any], index: int) -> Rule:
    """Build a shell-threat authorize rule from a declarative policy entry.

    Schema::

        {"type": "shell_threats", "decision": "deny", "min_severity": "warning",
         "disabled_rules": ["command_chaining"]}

    Optional keys only; defaults match :class:`snagline.shell_security.ShellThreatConfig`.
    """
    from snagline.shell_security import (
        BUILTIN_RULE_IDS,
        ShellThreatConfig,
        as_authorize_rule,
    )

    allowed = {
        "type",
        "decision",
        "min_severity",
        "disabled_rules",
        "enabled_rules",
        "severity_overrides",
    }
    unknown = set(raw) - allowed
    if unknown:
        paths = ", ".join(f"rules[{index}].{k}" for k in sorted(unknown))
        raise PolicyError(
            f"unknown rule keys {paths}; expected one of {sorted(allowed)}"
        )

    decision = raw.get("decision", "deny")
    if decision not in ("ask", "deny"):
        raise PolicyError(
            f"rules[{index}].decision={decision!r} must be 'ask' or 'deny'"
        )
    min_severity = raw.get("min_severity", "warning")
    if min_severity not in ("info", "warning", "critical"):
        raise PolicyError(
            f"rules[{index}].min_severity={min_severity!r} must be "
            f"one of info|warning|critical"
        )
    disabled = raw.get("disabled_rules", [])
    if not isinstance(disabled, list) or not all(isinstance(v, str) for v in disabled):
        raise PolicyError(f"rules[{index}].disabled_rules must be a list of strings")
    enabled = raw.get("enabled_rules")
    enabled_set: frozenset[str] | None
    if enabled is None:
        enabled_set = None
    elif isinstance(enabled, list) and all(isinstance(v, str) for v in enabled):
        unknown = set(enabled) - BUILTIN_RULE_IDS
        if unknown:
            raise PolicyError(
                f"rules[{index}].enabled_rules has unknown ids {sorted(unknown)}"
            )
        enabled_set = frozenset(enabled)
    else:
        raise PolicyError(f"rules[{index}].enabled_rules must be a list of strings")
    overrides_raw = raw.get("severity_overrides", {})
    if not isinstance(overrides_raw, Mapping):
        raise PolicyError(f"rules[{index}].severity_overrides must be an object")
    overrides: dict[str, Any] = {}
    for key, value in overrides_raw.items():
        if value not in ("info", "warning", "critical"):
            raise PolicyError(
                f"rules[{index}].severity_overrides[{key!r}]={value!r} is not a severity"
            )
        overrides[str(key)] = value
    cfg = ShellThreatConfig(
        enabled_rules=enabled_set,
        disabled_rules=frozenset(disabled),
        severity_overrides=overrides,
        min_severity=min_severity,
        decision=decision,
    )
    return as_authorize_rule(cfg)


def _path_security_rule_from_dict(raw: Mapping[str, Any], index: int) -> Rule:
    """Build a path-security authorize rule from a declarative policy entry.

    Schema::

        {"type": "path_security", "decision": "deny", "min_severity": "warning",
         "allowed_roots": ["/workspace"], "denied_paths": ["/workspace/secret"]}

    Optional keys only; defaults match :class:`snagline.path_security.PathSecurityConfig`.
    """
    from snagline.path_security import (
        BUILTIN_RULE_IDS,
        PathSecurityConfig,
        as_authorize_rule,
    )

    allowed = {
        "type",
        "decision",
        "min_severity",
        "allowed_roots",
        "denied_paths",
        "denied_patterns",
        "base_dir",
        "disabled_rules",
        "enabled_rules",
        "severity_overrides",
    }
    unknown = set(raw) - allowed
    if unknown:
        paths = ", ".join(f"rules[{index}].{k}" for k in sorted(unknown))
        raise PolicyError(
            f"unknown rule keys {paths}; expected one of {sorted(allowed)}"
        )

    decision = raw.get("decision", "deny")
    if decision not in ("ask", "deny"):
        raise PolicyError(
            f"rules[{index}].decision={decision!r} must be 'ask' or 'deny'"
        )
    min_severity = raw.get("min_severity", "warning")
    if min_severity not in ("info", "warning", "critical"):
        raise PolicyError(
            f"rules[{index}].min_severity={min_severity!r} must be "
            f"one of info|warning|critical"
        )

    allowed_roots = raw.get("allowed_roots")
    allowed_roots_tuple: tuple[str, ...] | None
    if allowed_roots is None:
        allowed_roots_tuple = None
    elif isinstance(allowed_roots, list) and all(
        isinstance(v, str) for v in allowed_roots
    ):
        allowed_roots_tuple = tuple(allowed_roots)
    else:
        raise PolicyError(f"rules[{index}].allowed_roots must be a list of strings")

    denied_paths = raw.get("denied_paths", [])
    if not isinstance(denied_paths, list) or not all(
        isinstance(v, str) for v in denied_paths
    ):
        raise PolicyError(f"rules[{index}].denied_paths must be a list of strings")

    denied_patterns = raw.get("denied_patterns", [])
    if not isinstance(denied_patterns, list) or not all(
        isinstance(v, str) for v in denied_patterns
    ):
        raise PolicyError(f"rules[{index}].denied_patterns must be a list of strings")

    base_dir = raw.get("base_dir")
    if base_dir is not None and not isinstance(base_dir, str):
        raise PolicyError(f"rules[{index}].base_dir must be a string")

    disabled = raw.get("disabled_rules", [])
    if not isinstance(disabled, list) or not all(isinstance(v, str) for v in disabled):
        raise PolicyError(f"rules[{index}].disabled_rules must be a list of strings")

    enabled = raw.get("enabled_rules")
    enabled_set: frozenset[str] | None
    if enabled is None:
        enabled_set = None
    elif isinstance(enabled, list) and all(isinstance(v, str) for v in enabled):
        unknown_rules = set(enabled) - BUILTIN_RULE_IDS
        if unknown_rules:
            raise PolicyError(
                f"rules[{index}].enabled_rules has unknown ids {sorted(unknown_rules)}"
            )
        enabled_set = frozenset(enabled)
    else:
        raise PolicyError(f"rules[{index}].enabled_rules must be a list of strings")

    overrides_raw = raw.get("severity_overrides", {})
    if not isinstance(overrides_raw, Mapping):
        raise PolicyError(f"rules[{index}].severity_overrides must be an object")
    overrides: dict[str, Any] = {}
    for key, value in overrides_raw.items():
        if value not in ("info", "warning", "critical"):
            raise PolicyError(
                f"rules[{index}].severity_overrides[{key!r}]={value!r} is not a severity"
            )
        overrides[str(key)] = value

    cfg = PathSecurityConfig(
        allowed_roots=allowed_roots_tuple,
        denied_paths=tuple(denied_paths),
        denied_patterns=tuple(denied_patterns),
        base_dir=base_dir,
        enabled_rules=enabled_set,
        disabled_rules=frozenset(disabled),
        severity_overrides=overrides,
        min_severity=min_severity,
        decision=decision,
    )
    return as_authorize_rule(cfg)


def _no_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """``object_pairs_hook`` that rejects duplicate keys instead of dropping one.

    ``json.loads`` keeps the LAST occurrence of a duplicate key and says
    nothing. In a policy document that is a denial being erased: two ``rules``
    keys means the second, possibly empty, list silently replaces the first and
    the affected calls proceed under the default. Every object level is checked
    by the hook, including each rule, so the failure names the rule that was
    duplicated rather than the file that merely contains it.
    """
    seen: dict[str, Any] = {}
    for key, value in pairs:
        if key in seen:
            raise PolicyError(f"duplicate key {key!r} in policy document")
        seen[key] = value
    return seen


def _load_toml(text: str) -> dict[str, Any]:
    try:
        import tomllib
    except ModuleNotFoundError:  # pragma: no cover - exercised only on <3.11
        raise PolicyError(
            "TOML policy files require Python 3.11+ (tomllib); use JSON or upgrade."
        ) from None
    return tomllib.loads(text)


def policy_from_file(path: str) -> Policy:
    """Load a policy from a ``.json`` or ``.toml`` document.

    Mirrors :meth:`snagline.config.Config.load_file`: JSON via the stdlib, TOML
    via ``tomllib`` on 3.11+. Non-finite floats are rejected for the same
    reason ``Config`` rejects them -- a policy is not the place to accept a
    value no comparison can order.
    """
    with open(path, encoding="utf-8") as fh:
        text = fh.read()
    data = (
        _load_toml(text)
        if path.endswith(".toml")
        else json.loads(text, object_pairs_hook=_no_duplicate_keys)
    )
    # Type-check BEFORE iterating: a top-level JSON array has no ``.items()``,
    # so the guard below would never be reached and the caller would get an
    # AttributeError instead of a message naming the policy file.
    if not isinstance(data, Mapping):
        raise PolicyError(f"{path}: policy must be a mapping at the top level")
    for key, value in data.items():
        if isinstance(value, float) and not math.isfinite(value):
            raise PolicyError(f"{key}: non-finite float value {value!r} in {path}")
    return policy_from_dict(data)


def policy_from_sequence(items: Sequence[Rule]) -> Policy:
    """``Policy`` from a plain sequence, for callers that build rules in code."""
    return Policy(rules=tuple(items))
