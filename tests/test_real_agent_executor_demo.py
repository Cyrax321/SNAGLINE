"""Regression test for issue #614: real_agent_executor_demo error mode.

Verifies that ``examples/real_agent_executor_demo.py --mode error`` survives
the tool failure, emits the expected ``error_cascade`` detection risk, and
exits cleanly with 0 rather than crashing on an unhandled exception.
"""

from __future__ import annotations

import os
import subprocess
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEMO = os.path.join(REPO, "examples", "real_agent_executor_demo.py")


def _langchain_agent_available() -> bool:
    try:
        from langchain.agents import create_agent  # noqa: F401

        return True
    except Exception:
        return False


@pytest.mark.skipif(
    not _langchain_agent_available(),
    reason="langchain.agents.create_agent not installed",
)
@pytest.mark.parametrize(
    ("mode", "expected_trigger", "expect_detected"),
    [
        ("error", '"trigger": "error_cascade"', True),
        ("loop", '"trigger": "loop"', True),
        ("healthy", '"trigger"', False),
    ],
)
def test_real_agent_executor_demo_modes(
    mode: str, expected_trigger: str, expect_detected: bool
) -> None:
    env = dict(os.environ)
    src = os.path.join(REPO, "src")
    env["PYTHONPATH"] = os.pathsep.join(
        p for p in [*env.get("PYTHONPATH", "").split(os.pathsep), src] if p
    )

    result = subprocess.run(
        [sys.executable, DEMO, "--mode", mode],
        capture_output=True,
        text=True,
        env=env,
        timeout=60,
    )
    assert result.returncode == 0, (
        f"Demo exited with {result.returncode}: {result.stderr}"
    )
    output = result.stdout + result.stderr
    if expect_detected:
        assert expected_trigger in output, (
            f"Expected {expected_trigger} in output:\n{output}"
        )
    else:
        assert expected_trigger not in output, (
            f"Expected no {expected_trigger} in output:\n{output}"
        )
