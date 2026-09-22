"""Issue #297: every runnable example must honour --help instead of running its demo.

``examples/baseline_to_monitor.py``, ``examples/replay_offline_trajectory.py`` and
``examples/langchain_example.py`` built no argparse parser at all, so every
argument -- including ``--help`` -- was silently ignored and the script ran its
full demo. Each is now run with ``--help`` and asserted to print usage and exit
before doing any work. The guard at the bottom walks the whole ``examples/``
directory so future outliers cannot slip back in.
"""

from __future__ import annotations

import ast
import importlib.util
import os
import subprocess
import sys
from collections.abc import Iterator

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXAMPLES = os.path.join(REPO, "examples")

# Optional extras an example may import. The import *location* decides whether
# --help is testable: an import at module scope is evaluated while the module is
# being built, so the module cannot even be imported without the extra and
# --help has no chance to run. An import inside main() sits after
# argparse.parse_args(), so --help exits first and needs no extra at all.
OPTIONAL_EXTRA_ROOTS = frozenset({"langchain", "langgraph", "openai", "anthropic"})

# Output the demos emit when they actually run; none of it should appear for --help.
DEMO_MARKERS = [
    "[demo] replaying",
    "[demo] invoking the model",
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


def _assert_help_not_demo(script: str) -> subprocess.CompletedProcess[str]:
    """Run ``script --help``; assert it prints usage and never runs the demo."""
    result = _run_help(script)
    assert result.returncode == 0, f"{script}: {result.stderr}"
    assert "usage:" in result.stdout, f"{script}: --help printed no usage line"
    for marker in DEMO_MARKERS:
        assert marker not in result.stdout, (
            f"{script}: --help ran the demo ({marker!r} on stdout)"
        )
        assert marker not in result.stderr, (
            f"{script}: --help ran the demo ({marker!r} on stderr)"
        )
    return result


def test_raw_loop_example_help_is_reference() -> None:
    # The issue cites this one as the reference behaviour; it is checked first so
    # a failure points at the outliers rather than the whole directory.
    result = _assert_help_not_demo("raw_loop_example.py")
    assert "--healthy" in result.stdout


def test_baseline_to_monitor_help_prints_usage() -> None:
    result = _assert_help_not_demo("baseline_to_monitor.py")
    assert "--healthy" in result.stdout


def test_replay_offline_trajectory_help_prints_usage() -> None:
    _assert_help_not_demo("replay_offline_trajectory.py")


def test_langchain_example_help_prints_usage() -> None:
    # langchain-core is imported inside main(), after argparse, so --help must
    # work here even though the extra is not installed.
    _assert_help_not_demo("langchain_example.py")


def _imports_in(node: ast.AST) -> Iterator[ast.Import | ast.ImportFrom]:
    """Import nodes inside ``node``, without descending into function/class scope."""
    if isinstance(node, (ast.Import, ast.ImportFrom)):
        yield node
        return
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return
    for child in ast.iter_child_nodes(node):
        yield from _imports_in(child)


def _extra_of_root(root: str) -> str | None:
    """The optional extra a root module belongs to, if any.

    LangChain ships several importable packages (``langchain``, ``langchain_core``,
    ``langchain_openai``, ...) that all back the single ``langchain`` extra.
    """
    for prefix in OPTIONAL_EXTRA_ROOTS:
        if root == prefix or root.startswith(prefix + "_"):
            return prefix
    return None


def _imported_optional_modules(node: ast.Import | ast.ImportFrom) -> list[str]:
    """Fully-qualified optional-extra modules brought in by one import node."""
    if isinstance(node, ast.Import):
        names = [alias.name for alias in node.names]
    elif node.level == 0 and node.module:  # absolute ``from x import y``
        names = [node.module]
    else:  # relative import: belongs to the package, never an optional extra
        return []
    return [name for name in names if _extra_of_root(name.split(".")[0]) is not None]


def _module_scope_optional_imports(tree: ast.Module) -> list[str]:
    """Optional-extra modules this file imports at import time."""
    modules: list[str] = []
    for node in tree.body:
        for imp in _imports_in(node):
            modules.extend(_imported_optional_modules(imp))
    return modules


def _is_importable(module: str) -> bool:
    """Whether ``module`` resolves in the current interpreter.

    ``find_spec`` raises rather than returning None when a *parent* package is
    missing (e.g. ``langchain.agents`` without ``langchain`` installed).
    """
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


def _defines_parser(tree: ast.AST) -> bool:
    return any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "argparse"
        and node.func.attr == "ArgumentParser"
        for node in ast.walk(tree)
    )


def test_every_runnable_example_accepts_help() -> None:
    # Catch future outliers: any example with a main() entry point must define a
    # parser, and --help must print usage instead of running the demo.
    no_parser: list[str] = []
    needs_extra: list[str] = []
    help_failures: list[str] = []
    checked: list[str] = []

    for name in sorted(os.listdir(EXAMPLES)):
        if not name.endswith(".py"):
            continue
        with open(os.path.join(EXAMPLES, name), encoding="utf-8") as fh:
            source = fh.read()
        tree = ast.parse(source, name)
        if not any(
            isinstance(node, ast.FunctionDef) and node.name == "main"
            for node in tree.body
        ):
            continue

        # Applies to the optional-extra examples too: the parser itself is
        # statically verifiable, which is exactly the bug from #297.
        if not _defines_parser(tree):
            no_parser.append(name)
            continue

        result = _run_help(name)
        if result.returncode == 0 and "usage:" in result.stdout:
            checked.append(name)
            continue

        # --help failed. If the example imports an optional extra at module scope
        # that this interpreter does not have, that is the cause: the module
        # cannot even be imported, so --help never gets to run (e.g.
        # real_agent_demo.py without langchain-core, whose module-level TOOLS list
        # references the Tool class the failed import leaves undefined). Skip it,
        # naming the missing module so every skip stays auditable. Anything else
        # is a real regression.
        missing = [
            module
            for module in _module_scope_optional_imports(tree)
            if not _is_importable(module)
        ]
        if missing:
            needs_extra.append(f"{name} (missing {', '.join(missing)})")
            continue

        help_failures.append(
            f"{name}: rc={result.returncode} stdout={result.stdout!r} stderr={result.stderr!r}"
        )

    assert not no_parser, "examples with no argparse parser: " + ", ".join(no_parser)
    assert checked, "no examples were runnable; the guard matched nothing"
    assert not help_failures, "--help did not print usage for: " + "\n".join(
        help_failures
    )
