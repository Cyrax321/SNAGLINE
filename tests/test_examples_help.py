"""Issue #297: every runnable example must honour --help instead of running its demo.

``examples/baseline_to_monitor.py`` and ``examples/replay_offline_trajectory.py``
once built no argparse parser at all, so ``--help`` was silently ignored and the
script ran its full demo. The guard here runs each of them with ``--help`` and
asserts it prints usage and exits before doing any work.
"""

from __future__ import annotations

import os
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXAMPLES = os.path.join(REPO, "examples")

# Scripts that previously ignored --help, plus the one the issue cites as the
# reference behaviour; the reference is asserted first so a failure points at
# the outliers rather than at the whole directory.
CASES = [
    "raw_loop_example.py",
    "baseline_to_monitor.py",
    "replay_offline_trajectory.py",
]

# Output the demos emit when they actually run; none of it should appear for --help.
DEMO_MARKERS = [
    "[demo] replaying",
    "Fitted baseline for tools:",
    "Healthy episode risks:",
]


def _run_help(script: str) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    # Make the import work whether or not the package is installed.
    src = os.path.join(REPO, "src")
    env["PYTHONPATH"] = os.pathsep.join(
        p for p in [*env.get("PYTHONPATH", "").split(os.pathsep), src] if p
    )
    return subprocess.run(
        [sys.executable, os.path.join(EXAMPLES, script), "--help"],
        capture_output=True,
        text=True,
        env=env,
        timeout=120,
    )


def test_raw_loop_example_help_is_reference() -> None:
    result = _run_help("raw_loop_example.py")
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout


def test_baseline_to_monitor_help_prints_usage() -> None:
    result = _run_help("baseline_to_monitor.py")
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout
    assert "--healthy" in result.stdout
    for marker in DEMO_MARKERS:
        assert marker not in result.stdout, f"--help ran the demo: {marker!r} printed"
        assert marker not in result.stderr, f"--help ran the demo: {marker!r} printed"


def test_replay_offline_trajectory_help_prints_usage() -> None:
    result = _run_help("replay_offline_trajectory.py")
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout
    for marker in DEMO_MARKERS:
        assert marker not in result.stdout, f"--help ran the demo: {marker!r} printed"
        assert marker not in result.stderr, f"--help ran the demo: {marker!r} printed"


def test_every_runnable_example_accepts_help() -> None:
    # Catch future outliers: any pure-stdlib example with a ``main()`` entry
    # point must define a parser, otherwise --help silently runs the demo again.
    # Examples needing an optional extra (langchain/langgraph) are skipped:
    # they cannot even be imported without the extra installed.
    import ast

    optional_markers = ("snagline[", "langchain", "langgraph")
    failures: list[str] = []
    checked: list[str] = []

    for name in sorted(os.listdir(EXAMPLES)):
        if not name.endswith(".py"):
            continue
        with open(os.path.join(EXAMPLES, name), encoding="utf-8") as fh:
            source = fh.read()
        if any(marker in source for marker in optional_markers):
            continue
        tree = ast.parse(source, name)
        if not any(
            isinstance(node, ast.FunctionDef) and node.name == "main"
            for node in tree.body
        ):
            continue
        checked.append(name)
        result = _run_help(name)
        if result.returncode != 0 or "usage:" not in result.stdout:
            failures.append(
                f"{name}: rc={result.returncode} stdout={result.stdout!r} stderr={result.stderr!r}"
            )

    assert checked, "no pure-stdlib examples found; the guard matched nothing"
    assert not failures, "--help did not print usage for: " + "\n".join(failures)
