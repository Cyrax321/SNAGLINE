import pytest

from snagline.adapters.mcp_gateway import (
    McpSecurityGateway,
    McpToolRequest,
    get_mcp_eval_tool_schema,
)
from snagline.authorize import Policy, allow_tools, deny_tools, ask_tools


def test_mcp_tool_request_normalization():
    req = McpToolRequest(
        name="read_file",
        arguments={"path": "/workspace/main.py"},
        agent_id="coder",
    )
    t_req = req.to_tool_request()
    assert t_req.tool_name == "read_file"
    assert t_req.action == "read"
    assert t_req.resource == "/workspace/main.py"
    assert t_req.agent_id == "coder"


def test_mcp_gateway_allow_benign():
    gateway = McpSecurityGateway()
    res = gateway.evaluate(
        name="read_file",
        arguments={"path": "README.md"},
    )
    assert res["decision"] == "allow"
    assert res["allowed"] is True
    assert res["security_score"] <= 20.0


def test_mcp_gateway_deny_destructive():
    gateway = McpSecurityGateway()
    res = gateway.evaluate(
        name="bash",
        arguments={"command": "rm -rf /"},
    )
    assert res["decision"] == "deny"
    assert res["allowed"] is False
    assert res["security_score"] >= 75.0


def test_mcp_gateway_ask_decision():
    policy = Policy(
        rules=(ask_tools("write_file"),),
        default_decision="allow",
    )
    gateway = McpSecurityGateway(policy=policy)
    res = gateway.evaluate(
        name="write_file",
        arguments={"path": "out.txt", "content": "data"},
    )
    assert res["decision"] == "ask"
    assert res["allowed"] is False


def test_mcp_gateway_fail_closed_on_error(monkeypatch):
    gateway = McpSecurityGateway(fail_closed=True)

    def crashing_eval(*args, **kwargs):
        raise RuntimeError("Internal crash during inspection")

    monkeypatch.setattr("snagline.security_evaluation.evaluate_request", crashing_eval)

    res = gateway.evaluate(name="custom_tool", arguments={})
    assert res["decision"] == "deny"
    assert res["allowed"] is False
    assert "fail-closed" in res["reason"]


def test_mcp_tool_schema_export():
    schema = get_mcp_eval_tool_schema()
    assert schema["name"] == "snagline_evaluate_tool"
    assert "inputSchema" in schema
    assert "name" in schema["inputSchema"]["required"]
    assert "arguments" in schema["inputSchema"]["required"]
