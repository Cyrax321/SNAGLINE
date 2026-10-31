# MCP Security Gateway Integration

The `snagline.adapters.mcp_gateway` module integrates SNAGLINE's pre-execution authorization controls into Model Context Protocol (MCP) ecosystems.

## Overview

MCP servers and hosts handle requests from LLM agents across diverse tool sets. The `McpSecurityGateway` normalizes incoming tool calls and evaluates them against configured security policies, shell threat rules, prompt injection heuristics, and risk scoring without executing the tools.

```
                    ┌─────────────────────────┐
                    │   MCP Agent / Client    │
                    └────────────┬────────────┘
                                 │ tools/call
                                 ▼
                    ┌─────────────────────────┐
                    │   McpSecurityGateway    │
                    └────────────┬────────────┘
                                 │
                 ┌───────────────┴───────────────┐
                 ▼                               ▼
       Authorization Policy             Security Scoring &
        allow / ask / deny              Detection Indicators
                 │                               │
                 └───────────────┬───────────────┘
                                 ▼
                     Decision: allow / ask / deny
```

## Normalization Mapping

The gateway automatically maps common MCP tool conventions:
- Tools such as `read_file`, `view`, `cat` map to action `"read"`.
- Tools such as `write_file`, `create`, `edit` map to action `"write"`.
- Tools such as `bash`, `execute_command`, `run_command` map to action `"execute"`.
- Resource targets are extracted from `path`, `command`, `uri`, or `url` arguments.

## Usage Example

```python
from snagline.adapters.mcp_gateway import McpSecurityGateway
from snagline.authorize import Policy, deny_tools

# Configure policy
policy = Policy(
    rules=(deny_tools("bash"),),
    default_decision="allow",
)

gateway = McpSecurityGateway(policy=policy, fail_closed=True)

# Evaluate tool request before execution
result = gateway.evaluate(
    name="bash",
    arguments={"command": "rm -rf /"},
    agent_id="autonomous_worker",
)

if not result["allowed"]:
    raise PermissionError(f"MCP tool call blocked: {result['reason']}")
```

## Host Enforcement Contract

> [!IMPORTANT]
> The `McpSecurityGateway` evaluates whether a tool call should proceed, but does **not** invoke the tool. The hosting MCP server or agent framework is responsible for enforcing the returned `allow`, `ask`, or `deny` decision before dispatching execution.
