"""Loud validation of ``Config.log_format`` / ``SNAGLINE_LOG_FORMAT`` (issue #119).

Values outside {"text", "json"} raise ValueError at construction or resolve
time instead of silently doing nothing; case and surrounding whitespace are
normalized so operators are not punished for spelling.
"""

from __future__ import annotations

import json

import pytest

from snagline.config import Config


def test_log_format_defaults_to_text():
    assert Config().log_format == "text"


def test_constructor_rejects_invalid_log_format():
    with pytest.raises(ValueError, match="log_format must be one of"):
        Config(log_format="yaml")


def test_constructor_normalizes_case_and_whitespace():
    assert Config(log_format=" JSON ").log_format == "json"


@pytest.mark.parametrize("value", ["xml", "JSONL", "", "text,json"])
def test_invalid_values_are_rejected_everywhere(value, tmp_path):
    with pytest.raises(ValueError, match="log_format"):
        Config(log_format=value)
    with pytest.raises(ValueError, match="log_format"):
        Config.from_env({"SNAGLINE_LOG_FORMAT": value})
    with pytest.raises(ValueError, match="log_format"):
        Config.resolve(environ={"SNAGLINE_LOG_FORMAT": value})
    path = tmp_path / "cfg.json"
    path.write_text(json.dumps({"log_format": value}), encoding="utf-8")
    with pytest.raises(ValueError, match="log_format"):
        Config.load_file(str(path))


def test_resolve_env_layering_still_applies_valid_json():
    cfg = Config.resolve(environ={"SNAGLINE_LOG_FORMAT": " json "})
    assert cfg.log_format == "json"
