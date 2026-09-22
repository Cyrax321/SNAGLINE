"""Tests for the 12-factor Config loaders (ATTACH_ANY_SYSTEM P0)."""

from __future__ import annotations

import json
import logging

import pytest

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


def test_from_env_skips_uncoercible_boolean_values():
    env = {"SNAGLINE_FAIL_OPEN": "flase"}
    overrides = Config.from_env_overrides(environ=env)
    assert "fail_open" not in overrides
    assert Config.from_env(environ=env).fail_open is True


def test_from_env_accepts_false_boolean_aliases():
    for value in ("0", "false", "no", "off", "f"):
        assert Config.from_env(environ={"SNAGLINE_FAIL_OPEN": value}).fail_open is False


def test_resolve_invalid_boolean_env_keeps_file_value(tmp_path):
    path = tmp_path / "cfg.json"
    path.write_text(json.dumps({"fail_open": False}))
    cfg = Config.resolve(
        path=str(path), environ={"SNAGLINE_FAIL_OPEN": "definitely-not-a-bool"}
    )
    assert cfg.fail_open is False


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


def test_readme_configuration_snippet_matches_config_defaults():
    # Mechanical guard for issue #153: the README Configuration snippet must
    # never drift from the shipped Config dataclass defaults (cusum_min_samples
    # once claimed 20 while the code shipped 5). Parse every literal
    # key=value kwarg out of the snippet and compare against the dataclass.
    import ast
    import dataclasses
    import re
    from pathlib import Path

    readme = Path(__file__).resolve().parent.parent / "README.md"
    text = readme.read_text(encoding="utf-8")
    start = text.index("## Configuration")
    open_fence = text.index("```python", start) + len("```python")
    snippet = text[open_fence : text.index("```", open_fence)]
    defaults = {f.name: f.default for f in dataclasses.fields(Config)}

    checked: set[str] = set()
    for match in re.finditer(r"^    (\w+)=(.+?)(?:#.*)?$", snippet, re.MULTILINE):
        name, raw = match.group(1), match.group(2).strip().rstrip(",")
        if name not in defaults:
            continue  # lines like monitor = Monitor.default(...) are skipped
        try:
            value = ast.literal_eval(raw)
        except (ValueError, SyntaxError):
            continue  # non-literal args (e.g. baseline objects) are skipped
        assert value == defaults[name], (
            f"README Configuration snippet {name}={raw!r} does not match "
            f"Config.{name} default {defaults[name]!r}; update one of them"
        )
        checked.add(name)

    assert "cusum_min_samples" in checked  # the field that motivated this guard


def test_every_scalar_config_field_is_documented_in_readme_env_table():
    # Mechanical guard for issue #183: the README promises
    # "Every scalar field is settable via SNAGLINE_<UPPER_SNAKE>"
    # but the env-var table is prose and Config is code, so they drift.
    # This is the third occurrence after #153 (cusum_min_samples) and #133
    # (StagnationDetector). At f7857d1, 11 scalar fields were absent:
    # semantic_drift 6, side_effect 3, compaction_tripwire 2.
    import ast
    import dataclasses
    import re
    from pathlib import Path
    from typing import get_type_hints

    from snagline.config import Config, _coercible_hint

    # Determine scalar settable fields via the same coercion logic the
    # runtime uses: bool/int/float/str and Optional[scalar] are settable;
    # BaselineProfile and other object types are not.
    hints = get_type_hints(Config)
    scalar_fields: set[str] = set()
    for f in dataclasses.fields(Config):
        hint = hints.get(f.name, f.type)
        # _coercible_hint unwraps X|None where X is scalar; non-scalar
        # unions stay as-is and will not match the scalar set.
        coerced = _coercible_hint(hint)
        if coerced in (bool, int, float, str):
            scalar_fields.add(f.name)

    # Parse the README env-var table: only the first table under
    # "### Environment variables and config keys", not the
    # "A handful of variables sit outside" second table.
    readme = Path(__file__).resolve().parent.parent / "README.md"
    text = readme.read_text(encoding="utf-8")
    start = text.index("### Environment variables and config keys")
    end = text.index("A handful of variables sit outside", start)
    section = text[start:end]
    # Each row is `| `SNAGLINE_X` | `field` | default |`
    rows = re.findall(
        r"\|\s*`SNAGLINE_([A-Z0-9_]+)`\s*\|\s*`(\w+)`\s*\|\s*([^|]+?)\s*\|",
        section,
    )
    documented: dict[str, str] = {field: default.strip() for _, field, default in rows}

    missing = sorted(scalar_fields - set(documented))
    extra = sorted(set(documented) - scalar_fields)
    assert not missing, (
        f"README env-var table is missing {len(missing)} scalar Config field(s): "
        f"{missing}; add rows for SNAGLINE_<UPPER_SNAKE> with correct defaults"
    )
    assert not extra, (
        f"README env-var table documents {len(extra)} field(s) that are not "
        f"scalar Config fields: {extra}; remove or correct them"
    )

    # Spot-check defaults for a few representative fields, and fully verify
    # that every documented default matches the dataclass default (with
    # None shown as *(unset)* or None in the table).
    defaults = {f.name: f.default for f in dataclasses.fields(Config)}
    for field, raw_default in documented.items():
        cfg_default = defaults[field]
        readme_raw = raw_default.strip()
        # Normalize README representation of None
        if cfg_default is None:
            assert readme_raw in ("None", "*(unset)*", "*(unset)*"), (
                f"README default for {field} should be *(unset)* or None "
                f"when Config.{field} is None, got {readme_raw!r}"
            )
            continue
        # For other scalars, try literal evaluation
        # README shows booleans as True/False, numbers as 12/0.5, strings as
        # text/manual/prometheus without quotes.
        cleaned = readme_raw.strip("`")
        try:
            readme_value = ast.literal_eval(cleaned)
        except (ValueError, SyntaxError):
            # String defaults without quotes (e.g. all-MiniLM-L6-v2, text)
            readme_value = cleaned
        assert readme_value == cfg_default, (
            f"README default for {field} is {readme_raw!r} but "
            f"Config.{field} default is {cfg_default!r}"
        )

    # The three historic drift groups must all be present (would have caught
    # f7857d1). Keep them as an explicit regression anchor, not just via the
    # generic missing check, so a future refactor that accidentally narrows
    # scalar_fields does not silently hide the gap.
    for anchor in [
        "semantic_drift_enabled",
        "side_effect_guard_enabled",
        "compaction_tripwire_enabled",
    ]:
        assert anchor in documented, (
            f"historic drift anchor {anchor} missing from README"
        )


def test_stagnation_env_override_out_of_range_aborts_startup():
    """Issue #132 behavior B: SNAGLINE_STAGNATION_MIN_NOVELTY=99 coerces
    cleanly but is out of range. The contract (option 1) is that invalid
    monitoring config aborts startup loudly at construction/resolve time with
    a clear error naming the knob, instead of crashing later inside
    StagnationDetector or running silently mis-configured."""
    env = {"SNAGLINE_STAGNATION_MIN_NOVELTY": "99"}
    with pytest.raises(ValueError, match="stagnation_min_novelty"):
        Config.from_env(environ=env)
    with pytest.raises(ValueError, match="stagnation_min_novelty"):
        Config.resolve(environ=env)
    # The detector's own guard stays as defense in depth for direct use.
    from snagline.detectors.stagnation import StagnationDetector

    with pytest.raises(ValueError, match="min_novelty"):
        StagnationDetector(config=Config(stagnation_min_novelty=0.5), min_novelty=99)


def test_monitor_default_raises_on_invalid_config_but_not_stock():
    """Pins the Monitor.default() raising contract for issue #132.

    Construction-time validation means Monitor.default() MAY raise ValueError
    when handed an invalid monitoring config: fail-fast startup beats a
    silently disabled safety net. Runtime detection remains fail-open; only
    broken startup config fails. Stock config must never raise."""
    from snagline import Monitor

    # An out-of-range env value is already rejected when the Config itself is
    # built (see test_stagnation_env_override_out_of_range_aborts_startup).
    # The path this test pins is a config that slips past construction (e.g.
    # mutated afterwards) and reaches detector wiring inside default().
    with pytest.raises(ValueError):
        Config.from_env(environ={"SNAGLINE_STAGNATION_MIN_NOVELTY": "0"})
    bad = Config(stagnation_enabled=True)
    bad.stagnation_min_novelty = 0.0  # mutation bypasses __post_init__
    with pytest.raises(ValueError):
        Monitor.default(config=bad)
    Monitor.default(config=Config())  # stock: must not raise


def test_from_env_overrides_still_drops_uncoercible_values():
    """Issue #66 semantics unchanged by the stagnation range checks (#132):
    values that fail *coercion* are logged and dropped, not fatal. Only
    cleanly-coerced out-of-range values are configuration errors."""
    env = {"SNAGLINE_STAGNATION_MIN_NOVELTY": "not-a-number"}
    overrides = Config.from_env_overrides(environ=env)
    assert "stagnation_min_novelty" not in overrides
    cfg = Config.from_env(environ=env)  # default survives
    assert cfg.stagnation_min_novelty == 0.05


def test_from_env_warns_when_a_key_names_an_object_typed_field(caplog):
    """Issue #355: a present key naming a non-scalar field used to be skipped
    with no log line at all, so ``SNAGLINE_GOAL_DRIFT_BASELINE`` /
    ``SNAGLINE_CALIBRATION_BASELINE`` were silent no-ops -- discoverable only
    by noticing the detector staying inert. Both now log, like a malformed
    value always did."""
    caplog.set_level(logging.WARNING, logger="snagline.config")
    env = {
        "SNAGLINE_GOAL_DRIFT_BASELINE": "/tmp/x.json",
        "SNAGLINE_CALIBRATION_BASELINE": "/tmp/y.json",
    }
    overrides = Config.from_env_overrides(environ=env)
    assert overrides == {}, "an object-typed field must never be set from env"
    message = "\n".join(r.getMessage() for r in caplog.records)
    assert "SNAGLINE_GOAL_DRIFT_BASELINE" in message
    assert "SNAGLINE_CALIBRATION_BASELINE" in message
    assert "goal_drift_baseline" in message
    assert "calibration_baseline" in message


def test_object_typed_field_advice_names_the_path_variant_when_one_exists(caplog):
    """Only ``calibration_baseline`` has a ``*_path`` form; the advice for
    ``goal_drift_baseline`` must not invent one."""
    caplog.set_level(logging.WARNING, logger="snagline.config")

    Config.from_env_overrides(environ={"SNAGLINE_CALIBRATION_BASELINE": "/tmp/y.json"})
    assert any(
        "SNAGLINE_CALIBRATION_BASELINE_PATH" in r.getMessage() for r in caplog.records
    )

    caplog.clear()
    Config.from_env_overrides(environ={"SNAGLINE_GOAL_DRIFT_BASELINE": "/tmp/x.json"})
    assert any(
        "goal_drift_baseline" in r.getMessage()
        and "SNAGLINE_GOAL_DRIFT_BASELINE_PATH" not in r.getMessage()
        for r in caplog.records
    ), "no goal_drift_baseline_path exists; the advice must say to pass it in code"


def test_object_typed_field_warning_survives_the_resolve_layer(caplog, tmp_path):
    """``resolve`` is the entrypoint ``serve``/``watch``/``replay`` all use, so
    the operator has to see the warning there too, not only via ``from_env``."""
    caplog.set_level(logging.WARNING, logger="snagline.config")
    cfg_path = tmp_path / "conf.json"
    cfg_path.write_text("{}", encoding="utf-8")
    Config.resolve(
        path=str(cfg_path), environ={"SNAGLINE_GOAL_DRIFT_BASELINE": "/tmp/x.json"}
    )
    assert any("SNAGLINE_GOAL_DRIFT_BASELINE" in r.getMessage() for r in caplog.records)


def test_unknown_env_keys_stay_silent(caplog):
    """The new warning is scoped to keys that name a real but non-scalar field;
    a truly unknown key is still silent (it may belong to a sibling tool)."""
    caplog.set_level(logging.WARNING, logger="snagline.config")
    Config.from_env_overrides(environ={"SNAGLINE_BOGUS_FIELD": "1"})
    assert not [
        r for r in caplog.records if "SNAGLINE_BOGUS_FIELD" in r.getMessage()
    ], "unknown keys must not warn"


def test_every_shipped_detector_is_in_readme_detector_table():
    """Guard for issue #206: README detector table must list every detector wired in Monitor.default()."""
    import pathlib
    import re

    readme = pathlib.Path(__file__).resolve().parent.parent / "README.md"
    text = readme.read_text(encoding="utf-8")
    start = text.index("## What it detects")
    table_text = text[start : text.index("Detection is deterministic", start)]
    # Extract detector names from table rows: | **Name** |
    documented_raw = re.findall(
        r"\|\s*\*\*([A-Za-z\- ]+?)(?:\s*\(opt-in\))?\s*\*\*", table_text
    )

    # Normalize: lower, hyphens to spaces, collapse whitespace
    def norm(s: str) -> str:
        s = s.lower().replace("-", " ")
        s = re.sub(r"\s+", " ", s).strip()
        return s

    documented_norm = {norm(d) for d in documented_raw}
    detectors_dir = (
        pathlib.Path(__file__).resolve().parent.parent / "src/snagline/detectors"
    )
    shipped = set()
    for fp in detectors_dir.glob("*.py"):
        if fp.name in ("__init__.py", "base.py", "windowing.py"):
            continue
        src = fp.read_text(encoding="utf-8")
        for m in re.finditer(r"class\s+(\w+Detector)\b", src):
            name = m.group(1)
            words = re.sub(r"([a-z])([A-Z])", r"\1 \2", name).lower()
            words = words.replace("-", " ")
            shipped.add(words)
    # The README also lists Horizon-scale time axis and ML ensemble which are not Detector subclasses in the same sense;
    # we only check that every shipped Detector appears in the table, not the converse.
    for det in shipped:
        assert det in documented_norm, (
            f"Detector {det!r} not found in README detector table; add a row under '## What it detects'"
        )


def test_token_runaway_envelope_out_of_range_aborts_startup():
    """Issue #317: the token-budget envelope knobs coerce cleanly from env but
    are out of range, and both fail in the false-positive direction --
    episode_token_budget=0 fires a score-1.0 budget_breach on the first
    token-bearing step, and token_budget_warn_fraction=0.0 a score-0.8 warning.
    Same contract as the stagnation knobs (#132): invalid monitoring config
    aborts startup loudly at construction/resolve time with a clear error
    naming the knob, instead of paging at critical severity later."""
    from snagline.detectors.token_runaway import TokenRunawayDetector

    for budget in (0, -1):
        with pytest.raises(ValueError, match="episode_token_budget"):
            Config(token_runaway_enabled=True, episode_token_budget=budget)
    for fraction in (0.0, -0.5, 1.5):
        with pytest.raises(ValueError, match="token_budget_warn_fraction"):
            Config(token_runaway_enabled=True, token_budget_warn_fraction=fraction)

    # Env layering bypasses __post_init__ via setattr, so resolve() must
    # re-check too -- a typo'd SNAGLINE_EPISODE_TOKEN_BUDGET=0 must not page.
    with pytest.raises(ValueError, match="episode_token_budget"):
        Config.resolve(environ={"SNAGLINE_EPISODE_TOKEN_BUDGET": "0"})
    with pytest.raises(ValueError, match="token_budget_warn_fraction"):
        Config.resolve(environ={"SNAGLINE_TOKEN_BUDGET_WARN_FRACTION": "0"})

    # The detector's own guard stays as defense in depth for direct use.
    with pytest.raises(ValueError, match="budget_total_tokens"):
        TokenRunawayDetector(budget_total_tokens=0)
    with pytest.raises(ValueError, match="warn_fraction"):
        TokenRunawayDetector(budget_total_tokens=1000, warn_fraction=1.5)

    # Valid values -- including the documented "None disables the envelope" --
    # must survive every layer.
    cfg = Config.resolve(
        environ={"SNAGLINE_EPISODE_TOKEN_BUDGET": "1000"},
    )
    assert cfg.episode_token_budget == 1000
    assert cfg.token_budget_warn_fraction == 0.8
    assert Config(token_runaway_enabled=True).episode_token_budget is None


def test_semantic_drift_cusum_knobs_out_of_range_abort_startup():
    """Issue #370: semantic_drift_cusum_h is the denominator of the alarm
    score and semantic_drift_cusum_k is the per-step subtraction, so both
    coerce cleanly from env but fail loudly in opposite directions -- h=0
    deadens the detector with a swallowed ZeroDivisionError logged once per
    step, and a negative h or k storms a false positive on nearly every step.
    Same contract as the stagnation knobs (#132) and the deterministic CUSUM
    bars (#331): a configuration error at startup, not a broken detector
    discovered once steps are flowing."""
    for h in (0.0, -0.5):
        with pytest.raises(ValueError, match="semantic_drift_cusum_h"):
            Config(semantic_drift_enabled=True, semantic_drift_cusum_h=h)
    for k in (-0.1, -1.0):
        with pytest.raises(ValueError, match="semantic_drift_cusum_k"):
            Config(semantic_drift_enabled=True, semantic_drift_cusum_k=k)

    # Env layering bypasses __post_init__ via setattr, so resolve() must
    # re-check too -- a typo'd SNAGLINE_SEMANTIC_DRIFT_CUSUM_H=0 must not
    # deaden the detector after a clean startup.
    with pytest.raises(ValueError, match="semantic_drift_cusum_h"):
        Config.resolve(environ={"SNAGLINE_SEMANTIC_DRIFT_CUSUM_H": "0"})
    with pytest.raises(ValueError, match="semantic_drift_cusum_k"):
        Config.resolve(environ={"SNAGLINE_SEMANTIC_DRIFT_CUSUM_K": "-0.1"})

    # The defaults are valid, so a stock config (and the opt-in path with its
    # shipped values) must survive every layer without tripping the checks.
    cfg = Config.resolve(environ={"SNAGLINE_SEMANTIC_DRIFT_ENABLED": "true"})
    assert cfg.semantic_drift_cusum_h == 0.5
    assert cfg.semantic_drift_cusum_k == 0.05
    assert Config(semantic_drift_enabled=True).semantic_drift_cusum_h == 0.5
    # A zero slack is legitimate (no debt decay), so it must be accepted.
    assert (
        Config(
            semantic_drift_enabled=True, semantic_drift_cusum_k=0.0
        ).semantic_drift_cusum_k
        == 0.0
    )
