"""The release-notes extraction step must fail closed on a missing section.

``.github/workflows/release.yml`` calls ``.github/scripts/extract_release_notes.py``
to pull the ``## [VERSION]`` section out of ``CHANGELOG.md``, and the result
becomes the GitHub release body. The step used to fall back to writing the
*whole* changelog when ``VERSION`` matched no heading, so a one-character
version mismatch published the entire file -- ``[Unreleased]`` included -- as
that release's notes, on a green run (issue #601). The inline ``python -c``
form it replaced was also unrunnable: YAML block scalars need every line
indented past ``run:`` while a ``python -c`` program must start at column 0,
so the parse fix in #585 traded a YAML error for an ``IndentationError``.
Both properties are pinned here, by running the script itself the way the
workflow runs it.
"""

from __future__ import annotations

import os
import pathlib
import re
import subprocess
import sys

import pytest

_CHANGELOG = """\
# Changelog

## [Unreleased]

- Unreleased work.

## [0.1.1] - 2026-10-26

- The release being cut.

## [0.1.0] - 2026-08-27

- Initial release.
"""

_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
_SCRIPT = _REPO_ROOT / ".github" / "scripts" / "extract_release_notes.py"
_WORKFLOW = _REPO_ROOT / ".github" / "workflows" / "release.yml"


def _extract(
    changelog: str, version: str, tmp_path: pathlib.Path
) -> subprocess.CompletedProcess[str]:
    """Run the extraction script in a scratch checkout, as the workflow does.

    The script reads ``CHANGELOG.md`` and writes ``RELEASE_NOTES.md`` relative
    to its working directory, so a temp dir stands in for the repo root.
    """
    (tmp_path / "CHANGELOG.md").write_text(changelog)
    env = {**os.environ, "VERSION": version}
    return subprocess.run(
        [sys.executable, str(_SCRIPT)],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def test_matching_version_writes_only_that_section(tmp_path: pathlib.Path) -> None:
    result = _extract(_CHANGELOG, "0.1.1", tmp_path)
    assert result.returncode == 0, result.stderr
    body = (tmp_path / "RELEASE_NOTES.md").read_text()
    assert "## [0.1.1]" in body
    assert "The release being cut." in body
    # Exactly one section: neither the neighbours nor the title leak in.
    assert "Unreleased" not in body
    assert "Initial release." not in body
    assert "# Changelog" not in body


def test_unmatched_version_fails_rather_than_publishing_everything(
    tmp_path: pathlib.Path,
) -> None:
    # 0.2.0 has no section. Before #601 this wrote all of CHANGELOG.md into
    # RELEASE_NOTES.md and the publish still went green.
    result = _extract(_CHANGELOG, "0.2.0", tmp_path)
    assert result.returncode != 0
    # The message names the version an operator would act on.
    assert "0.2.0" in result.stderr
    # Nothing partial: the artifact that feeds the release body is not left
    # behind for a later step to upload.
    assert not (tmp_path / "RELEASE_NOTES.md").exists()


def test_version_mismatch_by_one_character_also_fails(
    tmp_path: pathlib.Path,
) -> None:
    result = _extract(_CHANGELOG, "0.11", tmp_path)
    assert result.returncode != 0
    assert not (tmp_path / "RELEASE_NOTES.md").exists()


@pytest.mark.parametrize("version", ["0.1.0", "0.1.1", "UNRELEASED"])
def test_exact_heading_is_required_not_a_substring(
    tmp_path: pathlib.Path, version: str
) -> None:
    """A heading that merely *contains* the version must not match."""
    changelog = "## [Unreleased]\n\n- work.\n\n## [0.10.0] - 2026-11-01\n\n- other.\n"
    # "0.1.0" is a substring of the [0.10.0] heading; the pattern is anchored
    # on the bracketed heading, so it -- and junk like "UNRELEASED" -- misses.
    result = _extract(changelog, version, tmp_path)
    assert result.returncode != 0
    assert not (tmp_path / "RELEASE_NOTES.md").exists()


def test_extraction_is_pinned_to_the_shipped_changelog(tmp_path: pathlib.Path) -> None:
    """The shipped changelog's headings still anchor the extraction.

    A pattern that drifted away from the file it extracts would turn every
    release into the failure path, so pin the repository's own changelog: the
    0.1.0 section must extract cleanly from the real file.
    """
    changelog = (_REPO_ROOT / "CHANGELOG.md").read_text()
    headings = re.findall(r"^## \[([^\]]+)\]", changelog, re.M)
    assert "0.1.0" in headings
    result = _extract(changelog, "0.1.0", tmp_path)
    assert result.returncode == 0, result.stderr
    assert "## [0.1.0]" in (tmp_path / "RELEASE_NOTES.md").read_text()


def test_release_workflow_calls_the_script() -> None:
    """The script and the workflow stay wired together.

    A workflow that still points at the inline ``python -c`` form would pass
    every script test and then fail (or, pre-#585, not parse) at release
    time. Pin that the step invokes the script, and that no indented inline
    program -- the shape that cannot run -- remains anywhere in the file.
    """
    workflow = _WORKFLOW.read_text()
    assert "python .github/scripts/extract_release_notes.py" in workflow
    # A *multi-line* inline program is the shape that cannot run: every line
    # past the first must be indented for YAML and unindented for Python. A
    # single-line ``python -c "..."`` (the version-determination step uses
    # one) is fine, so only pin the broken form.
    assert 'python -c "\n' not in workflow


def test_extraction_script_has_no_indented_program() -> None:
    """The script is plain Python: no YAML indentation games to trip over."""
    source = _SCRIPT.read_text()
    compile(source, str(_SCRIPT), "exec")
    # The script must not grow a shebang-relative cwd dependence beyond the
    # documented CHANGELOG.md read; keep the contract explicit.
    assert "CHANGELOG.md" in source
    assert "RELEASE_NOTES.md" in source
