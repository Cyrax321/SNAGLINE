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


def test_resolve_env_equal_to_default_still_beats_the_file(tmp_path):
    # Issue #66: resolve() inferred "this env var was set" by comparing an
    # env-derived Config against a default instance, so a variable whose value
    # happened to equal the built-in default looked unset and the file won --
    # inverting the documented env > file > defaults precedence.
    defaults = Config()
    path = tmp_path / "cfg.json"
    path.write_text(json.dumps({"cascade_error_threshold": 99, "fail_open": False}))
    env = {
        # Deliberately the built-in default value: an operator resetting a
        # shared config file back to stock behaviour from the environment.
        "SNAGLINE_CASCADE_ERROR_THRESHOLD": str(defaults.cascade_error_threshold),
        "SNAGLINE_FAIL_OPEN": "true",
    }
    cfg = Config.resolve(path=str(path), environ=env)
    assert cfg.cascade_error_threshold == defaults.cascade_error_threshold
    # fail_open is the sharpest case: an operator restoring the fail-open
    # guarantee via environment must not stay in strict mode.
    assert cfg.fail_open is True


def test_from_env_overrides_reports_only_keys_that_were_set():
    # The set of present keys is what resolve() needs; a value equal to the
    # default must still appear, and unrelated environment must not.
    overrides = Config.from_env_overrides(
        environ={
            "SNAGLINE_CUSUM_K": str(Config().cusum_k),  # == default
            "SNAGLINE_UNKNOWN_FIELD": "x",
            "PATH": "/usr/bin",
        }
    )
    assert overrides == {"cusum_k": Config().cusum_k}


def test_from_env_still_returns_a_full_config():
    # from_env's public behaviour is unchanged by the refactor.
    cfg = Config.from_env(environ={"SNAGLINE_LOOP_WINDOW_SIZE": "42"})
    assert cfg.loop_window_size == 42
    assert cfg.cusum_k == Config().cusum_k
