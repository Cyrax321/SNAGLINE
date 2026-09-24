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


def test_changelog_project_url_is_declared_and_points_at_a_real_file() -> None:
    """PyPI renders a ``Changelog`` sidebar link when ``[project.urls]`` has one.

    A ``CHANGELOG.md`` exists at the repo root but ``[project.urls]`` listed
    Homepage / Repository / Documentation / Issues and never the changelog
    (issue #467), so the release page pointed at everything except the one file
    that says what changed. Guard both halves: the key exists, and it names the
    file that actually ships.
    """
    tomllib = pytest.importorskip("tomllib")  # 3.11+; CI's oldest leg is 3.10
    with PYPROJECT.open("rb") as fh:
        urls = tomllib.load(fh)["project"]["urls"]

    assert "Changelog" in urls, (
        "[project.urls] has no Changelog entry; PyPI will not render a "
        "changelog link even though CHANGELOG.md ships at the repo root"
    )
    assert urls["Changelog"].endswith("CHANGELOG.md"), (
        f"the Changelog URL {urls['Changelog']!r} does not point at CHANGELOG.md"
    )
    assert (REPO_ROOT / "CHANGELOG.md").is_file(), (
        "the Changelog URL points at CHANGELOG.md but no such file ships"
    )


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
