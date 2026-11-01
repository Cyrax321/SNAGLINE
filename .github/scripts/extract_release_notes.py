#!/usr/bin/env python3
"""Extract one release's notes out of ``CHANGELOG.md``.

Called from ``.github/workflows/release.yml`` (build job). The extracted
section is written to ``RELEASE_NOTES.md``, uploaded as an artifact, and used
verbatim as the GitHub release body.

Failing closed matters here: ``RELEASE_NOTES.md`` **is** the published release
notes, so a version whose heading does not exist must fail the build. The
inline form this replaces fell back to writing the *entire* changelog when
``VERSION`` matched no heading, which turned a one-character version mismatch
into a green publish of the whole file -- the entire ``[Unreleased]`` section
included (issue #601). The fix for a missing section is to add the section,
not to publish everything.

Why this is a script rather than inline ``python -c`` in the workflow: the
two requirements are unsatisfiable together. YAML needs every line of a block
scalar indented past ``run:``, while a ``python -c`` program must start at
column 0 or die with ``IndentationError``. The inline form was committed
unindented (unparseable YAML), then fixed by indenting it (#585), which
traded the parse error for an ``IndentationError`` at run time -- so the
extraction step had not actually run once since. A script is plain Python on
both axes and directly testable from ``tests/``.

Exit codes: ``0`` with ``RELEASE_NOTES.md`` written on a match; ``1`` with a
message naming the missing version and no artifact written otherwise.
"""

from __future__ import annotations

import os
import pathlib
import re
import sys

# The fallback the issue is about. Kept as an explicit constant so the
# contract is readable in one place: no match -> fail, never publish all.
_NOTES_PATH = "RELEASE_NOTES.md"


def section_for(changelog: str, version: str) -> str | None:
    """Return the ``## [version]`` section, or ``None`` when there is none.

    The lookahead stops at the next ``## [`` heading so the section is exactly
    its own body; ``re.escape`` keeps a dotted version from matching "any
    character" inside a heading (``0.1.0`` must not hook onto ``[0.1.0-rc1]``
    -- or, worse, onto ``[0.10.0]``).
    """
    pattern = rf"## \[{re.escape(version)}\].*?(?=\n## \[|\Z)"
    match = re.search(pattern, changelog, re.S)
    return None if match is None else match.group(0).strip()


def main() -> int:
    version = os.environ.get("VERSION", "0.1.0")
    changelog = pathlib.Path("CHANGELOG.md").read_text()
    section = section_for(changelog, version)
    if section is None:
        # Do not write RELEASE_NOTES.md: a partial or stale artifact can be
        # picked up by a later step and published.
        print(
            f"no '## [{version}]' section in CHANGELOG.md; refusing to "
            f"publish the whole changelog as {_NOTES_PATH} (issue #601)",
            file=sys.stderr,
        )
        return 1
    pathlib.Path(_NOTES_PATH).write_text(section + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
