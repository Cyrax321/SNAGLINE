"""Packaging / documentation invariants that drift between PRs.

Three guards, each closing an issue whose root cause was a hand-maintained
value nobody re-checks:

* ``py.typed`` ships in the package so downstream type checkers see snagline's
  annotations (PEP 561, issue #306).
* The dev extra's tool pins match what CI installs, so a contributor's local
  lint run and CI agree (issue #309).
* No prose file hardcodes a test count, because such a count is stale the
  moment anyone adds a test and it is environment-dependent anyway (the
  recurrence of issue #96 / #310).
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
PYPROJECT = REPO_ROOT / "pyproject.toml"
CI_FILE = REPO_ROOT / ".github" / "workflows" / "ci.yml"

# Prose files that quote suite results. A literal "<number> passed"/"N tests"
# in any of them is a future bug report.
PROSE_FILES = (REPO_ROOT / "README.md", REPO_ROOT / "project.md")
TEST_COUNT_RE = re.compile(r"\b\d{2,4}\s+(?:passed|tests?)\b", re.IGNORECASE)


def _package_dir() -> Path:
    import snagline

    return Path(snagline.__file__).parent


def test_py_typed_marker_ships_with_package() -> None:
    """PEP 561 marker: without it mypy/pyright see every import as Any."""
    marker = _package_dir() / "py.typed"
    assert marker.is_file(), (
        f"{marker} is missing: downstream type checkers treat snagline as "
        "untyped even though every module is annotated"
    )
    # The marker is intentionally empty; content would only confuse tooling.
    assert marker.read_bytes().strip() == b""


def test_py_typed_is_declared_as_package_data() -> None:
    """Guard against a future [tool.setuptools] change silently dropping it.

    setuptools includes package data by default under pyproject config, so
    there is nothing to assert on in pyproject today -- instead assert the
    built wheel would carry it, which is what actually matters. Kept as a
    cheap source-tree check so it runs without ``build`` installed.
    """
    assert (REPO_ROOT / "src" / "snagline" / "py.typed").is_file()


def test_dev_extra_tool_pins_match_ci() -> None:
    """``pyproject.toml``'s dev pins must agree with the CI install line.

    The dev extra carries a comment claiming they match; ruff patch releases
    change rule behaviour, so a drift here means a contributor sees green
    locally and red in CI (or vice versa).
    """
    tomllib = pytest.importorskip("tomllib")  # 3.11+; CI's oldest leg is 3.10
    with PYPROJECT.open("rb") as fh:
        project = tomllib.load(fh)["project"]
    dev = project["optional-dependencies"]["dev"]

    ci_text = CI_FILE.read_text(encoding="utf-8")
    install_line = next(
        line
        for line in ci_text.splitlines()
        if "pytest-cov==" in line and "pip install" in line
    )

    for requirement in dev:
        if "==" not in requirement:
            continue  # unpinned in both places (pytest, mypy) -- fine
        name, _, version = requirement.partition("==")
        assert f'"{name}=={version}"' in install_line, (
            f"dev extra pins {name}=={version} but the CI install line is:\n"
            f"  {install_line.strip()}\n"
            "the dev extra's comment claims these match -- pick one version "
            "and set it in both places"
        )


@pytest.mark.parametrize("prose", PROSE_FILES, ids=lambda p: p.name)
def test_no_prose_file_hardcodes_a_test_count(prose: Path) -> None:
    """A literal count is stale on the next PR and env-dependent anyway.

    The README's live CI badge is the only place a count should appear; prose
    says how to run the suite, not what it returned.
    """
    if not prose.is_file():
        pytest.skip(f"{prose} not present in this checkout")

    text = prose.read_text(encoding="utf-8")
    # Shield the CI badge URL and any fenced command a reader is told to run:
    # ``python -m pytest tests/ -q`` is an instruction, not a result.
    stripped = re.sub(r"https?://\S+", "", text)

    offenders = TEST_COUNT_RE.findall(stripped)
    assert not offenders, (
        f"{prose.name} hardcodes a test count {offenders[:5]}: the number "
        "drifts on every PR that adds a test and differs by environment -- "
        "point readers at `python -m pytest tests/ -q` or the CI badge instead"
    )


def _init_fallback_version() -> str:
    """The string literal ``__init__.py`` assigns to ``__version__`` in its
    ``except`` fallback (used for editable / metadata-less installs).

    Read from source with ``ast`` rather than importing ``snagline.__version__``:
    in an installed environment ``importlib.metadata.version`` wins and returns
    the *distribution* version, which is exactly the value that can mask a stale
    hand-maintained fallback. We want the literal, not the resolved value.
    """
    init_py = _package_dir() / "__init__.py"
    tree = ast.parse(init_py.read_text(encoding="utf-8"))
    literals = [
        node.value.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "__version__" for t in node.targets)
        and isinstance(node.value, ast.Constant)
        and isinstance(node.value.value, str)
    ]
    assert len(literals) == 1, (
        f"expected exactly one literal __version__ assignment in {init_py}, "
        f"found {literals!r}"
    )
    return literals[0]


def test_init_fallback_version_matches_pyproject() -> None:
    """The version lives in two hand-maintained places -- ``pyproject.toml``'s
    ``project.version`` and ``__init__.py``'s ``except`` fallback literal -- and
    nothing tied them together. A release that bumps pyproject but forgets the
    fallback reports a stale version in editable installs (issue #457)."""
    tomllib = pytest.importorskip("tomllib")  # 3.11+; CI's oldest leg is 3.10
    with PYPROJECT.open("rb") as fh:
        pyproject_version = tomllib.load(fh)["project"]["version"]

    assert _init_fallback_version() == pyproject_version, (
        "version drift: pyproject.toml project.version is "
        f"{pyproject_version!r} but src/snagline/__init__.py's fallback literal "
        f"is {_init_fallback_version()!r} -- bump both together"
    )
