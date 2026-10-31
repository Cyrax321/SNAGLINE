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
