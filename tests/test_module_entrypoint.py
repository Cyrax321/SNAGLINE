"""``python -m snagline`` package entry point (issue #487)."""

from __future__ import annotations

import subprocess
import sys


def _run_module(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "snagline", *args],
        capture_output=True,
        text=True,
    )


def test_python_dash_m_snagline_help_runs_and_exits_zero() -> None:
    # Pre-fix this raised "No module named snagline.__main__; 'snagline' is a
    # package and cannot be directly executed" with a non-zero exit.
    result = _run_module("--help")
    assert result.returncode == 0, result.stderr
    assert "usage" in result.stdout.lower()


def test_python_dash_m_snagline_bad_command_still_dispatches() -> None:
    # A recognized-but-invalid invocation reaches the parser (exit 2), proving
    # the module actually dispatched into cli.main rather than failing to load.
    result = _run_module("no-such-command")
    assert result.returncode == 2
