"""Model Context Protocol (MCP) security gateway for SNAGLINE.

Normalizes MCP tool requests and evaluates them through the pre-execution security
pipeline before the host executes them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional

from snagline.authorize import ToolRequest

ACTION_MAP: Dict[str, str] = {
    "read_file": "read",
    "read": "read",
    "view": "read",
    "cat": "read",
    "write_file": "write",
    "write": "write",
    "edit": "edit",
    "create": "write",
    "bash": "execute",
    "sh": "execute",
    "terminal": "execute",
    "execute_command": "execute",
    "run_command": "execute",
    "delete_file": "delete",
    "remove": "delete",
}

RESOURCE_KEYS = ("command", "path", "file_path", "uri", "url", "file", "target")


@dataclass
class McpToolRequest:
    """Standardized MCP tool request container."""
    name: str
    arguments: Mapping[str, Any] = field(default_factory=dict)
    agent_id: Optional[str] = None
    context: Mapping[str, Any] = field(default_factory=dict)

    def to_tool_request(self) -> ToolRequest:
        """Normalize MCP tool request into canonical SNAGLINE ToolRequest."""
        lower_name = self.name.lower().replace("-", "_")
        action = ACTION_MAP.get(lower_name, "unknown")

        # Extract primary resource string from arguments
        resource: Optional[str] = None
        for key in RESOURCE_KEYS:
            if key in self.arguments and isinstance(self.arguments[key], str):
                resource = self.arguments[key]
                break

        return ToolRequest(
            tool_name=self.name,
            action=action,
            resource=resource,
            tool_input=dict(self.arguments),
            agent_id=self.agent_id,
        )


class McpSecurityGateway:
    """Evaluates incoming MCP tool calls against SNAGLINE security authorization controls."""

    def __init__(
        self,
        policy: Any = None,
        policy_path: Any = None,
        scoring_engine: Any = None,
        fail_closed: bool = True,
    ) -> None:
        self.policy = policy
        self.policy_path = policy_path
        self.scoring_engine = scoring_engine
        self.fail_closed = fail_closed

    def evaluate(
        self,
        name: str,
        arguments: Mapping[str, Any],
        agent_id: Optional[str] = None,
        context: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Evaluate MCP tool call and return structured decision."""
        from snagline.security_evaluation import evaluate_request

        mcp_req = McpToolRequest(
            name=name,
            arguments=arguments,
            agent_id=agent_id,
            context=context or {},
        )
        tool_req = mcp_req.to_tool_request()

        try:
            eval_res = evaluate_request(
                tool_name=tool_req.tool_name,
                action=tool_req.action,
                resource=tool_req.resource,
                tool_input=tool_req.tool_input,
                policy=self.policy,
                policy_path=self.policy_path,
                scoring_engine=self.scoring_engine,
            )
            decision = eval_res["decision"]
            allowed = (decision == "allow")
            return {
                "decision": decision,
                "allowed": allowed,
                "reason": eval_res["reason"],
                "tool_name": name,
                "security_score": eval_res["security_score"],
                "findings": eval_res["findings"],
                "matched_rules": eval_res["matched_rules"],
            }
        except Exception as err:
            if self.fail_closed:
                return {
                    "decision": "deny",
                    "allowed": False,
                    "reason": f"Security gateway evaluation error (fail-closed): {err}",
                    "tool_name": name,
                    "security_score": 100.0,
                    "findings": [],
                    "matched_rules": ["gateway_error_fail_closed"],
                }
            return {
                "decision": "allow",
                "allowed": True,
                "reason": f"Security gateway evaluation error (fail-open): {err}",
                "tool_name": name,
                "security_score": 0.0,
                "findings": [],
                "matched_rules": [],
            }

