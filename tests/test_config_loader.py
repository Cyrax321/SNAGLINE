"""Tests for the 12-factor Config loaders (ATTACH_ANY_SYSTEM P0)."""

from __future__ import annotations

import json

from snagline.config import Config


def test_from_env_overrides_scalar_fields():
    env = {
        "SNAGLINE_FAIL_OPEN": "false",
        "SNAGLINE_CUSUM_K": "0.9",
        "SNAGLINE_LOOP_REPEAT_THRESHOLD": "7",
        "SNAGLINE_GOAL_DRIFT_ENABLED": "true",
    }
    cfg = Config.from_env(environ=env)
    assert cfg.fail_open is False
    assert cfg.cusum_k == 0.9
    assert cfg.loop_repeat_threshold == 7
    assert cfg.goal_drift_enabled is True


def test_from_env_ignores_unknown_prefix_and_keys():
    env = {
        "PATH": "/usr/bin",
        "SNAGLINE_BOGUS_FIELD": "1",
        "SNAGLINE_GOAL_DRIFT_BASELINE": "cannot-load-this",  # complex type -> skipped
    }
    cfg = Config.from_env(environ=env)
    assert cfg.fail_open is True  # default unchanged
    assert cfg.goal_drift_enabled is False


def test_from_env_skips_uncoercible_values():
    env = {"SNAGLINE_CUSUM_K": "not-a-float"}
    cfg = Config.from_env(environ=env)
    assert cfg.cusum_k == 0.5  # default unchanged, no raise


def test_load_file_json(tmp_path):
    path = tmp_path / "cfg.json"
    path.write_text(
        json.dumps({"cusum_k": 1.2, "loop_repeat_threshold": 9, "extra_ignored": "x"})
    )
    cfg = Config.load_file(str(path))
    assert cfg.cusum_k == 1.2
    assert cfg.loop_repeat_threshold == 9
    assert cfg.fail_open is True  # untouched default


def test_load_file_toml(tmp_path):
    try:
        import tomllib  # noqa: F401
    except ModuleNotFoundError:
        import pytest

        pytest.skip("tomllib requires Python 3.11+")
    path = tmp_path / "cfg.toml"
    path.write_text("cusum_h = 7.0\nloop_window_size = 20\n")
    cfg = Config.load_file(str(path))
    assert cfg.cusum_h == 7.0
    assert cfg.loop_window_size == 20


def test_resolve_layers_file_then_env(tmp_path):
    path = tmp_path / "cfg.json"
    path.write_text(json.dumps({"cusum_k": 1.2, "loop_repeat_threshold": 9}))
    env = {"SNAGLINE_LOOP_REPEAT_THRESHOLD": "15"}
    cfg = Config.resolve(path=str(path), environ=env)
    # From file:
    assert cfg.cusum_k == 1.2
    # From env, overriding the file:
    assert cfg.loop_repeat_threshold == 15
    # Untouched default:
    assert cfg.fail_open is True


def test_resolve_env_only():
    env = {"SNAGLINE_FAIL_OPEN": "false"}
    cfg = Config.resolve(environ=env)
    assert cfg.fail_open is False
    assert cfg.cusum_k == 0.5  # default unchanged
