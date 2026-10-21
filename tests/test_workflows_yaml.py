"""Tests for GitHub Actions workflow definitions (issue #600).

Validates that all workflow files in .github/workflows/*.yml parse as valid YAML
and follow standard GitHub Actions workflow structure.
"""

from __future__ import annotations

import pathlib

import pytest
import yaml

_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
_WORKFLOWS_DIR = _REPO_ROOT / ".github" / "workflows"


def test_all_workflows_parse_as_valid_yaml():
    yaml_files = sorted(_WORKFLOWS_DIR.glob("*.yml")) + sorted(
        _WORKFLOWS_DIR.glob("*.yaml")
    )
    assert len(yaml_files) >= 15, (
        f"Expected at least 15 workflows, found {len(yaml_files)}"
    )
    for path in yaml_files:
        content = path.read_text(encoding="utf-8")
        parsed = yaml.safe_load(content)
        assert isinstance(parsed, dict), f"{path.name} did not parse into a mapping"
        assert "name" in parsed or "on" in parsed, f"{path.name} missing 'name' or 'on'"
        assert "jobs" in parsed, f"{path.name} missing 'jobs' section"
        assert isinstance(parsed["jobs"], dict), f"{path.name} jobs must be a mapping"
        assert len(parsed["jobs"]) > 0, f"{path.name} has no jobs defined"


def test_workflow_lint_job_present_in_ci():
    """Ensure the workflow-lint actionlint gate is declared in ci.yml (#600)."""
    ci_path = _WORKFLOWS_DIR / "ci.yml"
    parsed = yaml.safe_load(ci_path.read_text(encoding="utf-8"))
    assert "workflow-lint" in parsed.get("jobs", {}), (
        "ci.yml missing 'workflow-lint' job"
    )
    job = parsed["jobs"]["workflow-lint"]
    step_actions = [
        step.get("uses", "")
        for step in job.get("steps", [])
        if isinstance(step, dict) and "uses" in step
    ]
    assert any("actionlint" in action for action in step_actions), (
        "workflow-lint job must use actionlint"
    )


def test_malformed_yaml_fails_validation(tmp_path):
    """Simulate the release.yml col 0 syntax error that motivated #600."""
    broken_content = """\
name: Broken
on: [push]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - run: |
        python -c "
import os
"
"""
    broken_file = tmp_path / "broken.yml"
    broken_file.write_text(broken_content, encoding="utf-8")
    with pytest.raises(yaml.YAMLError):
        yaml.safe_load(broken_file.read_text(encoding="utf-8"))
