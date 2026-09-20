"""Guard against hand-maintained test counts in prose (issue #310).

Counts quoted in README.md or project.md drift as soon as the suite
changes. The README badge and the CI workflow (ci.yml) are the sources
of truth; prose must not hardcode numbers. This test fails if a
documented test count like "715 passed" or "823 tests" appears again.

The regex deliberately requires two to four digits followed by
"passed" or "tests" so unrelated numbers (versions, ports, sample
values) are not flagged.
"""

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PATTERN = re.compile(r"\b\d{2,4}\s+(?:passing|passed|tests?)\b", re.IGNORECASE)
ALLOWED = (
    # The scanner's own docstring above mentions the shape of the pattern.
    "tests/test_no_hardcoded_test_counts.py",
)


def test_no_hardcoded_test_counts_in_docs():
    offenders = []
    for name in ("README.md", "project.md"):
        path = REPO_ROOT / name
        assert path.exists(), f"{name} missing from repo root"
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            match = PATTERN.search(line)
            if match:
                offenders.append(f"{name}:{lineno}: {line.strip()[:120]}")
    assert not offenders, (
        "Hand-maintained test counts found in docs (issue #310). Quote the "
        "command, not the number:\n" + "\n".join(offenders)
    )
