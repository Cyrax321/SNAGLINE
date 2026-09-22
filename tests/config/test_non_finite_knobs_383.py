"""Issue #383: non-finite float knobs silently deaden their detector.

``Config._coerce`` accepted ``inf`` / ``-inf`` / ``nan`` (and overflows like
``1e400``) from both ``SNAGLINE_*`` env vars and JSON config files. A
non-finite CUSUM knob does not crash -- ``cusum > inf`` is never true and
``max(0.0, nan)`` is ``0.0`` -- so the detector goes inert for the whole run
with the process healthy and nothing in the logs. The value is now rejected
where it enters.
"""

from __future__ import annotations

import json
import math

import pytest

from snagline.config import Config
from snagline.detectors.latency_anomaly import LatencyAnomalyDetector
from snagline.events import StepEvent

# Tokens that float() parses without complaint but that poison a detector.
_NON_FINITE_ENV: list[tuple[str, str]] = [
    ("inf", "Infinity"),
    ("-inf", "-Infinity"),
    ("nan", "NaN"),
    ("1e400", "an overflow to inf"),
]


@pytest.mark.parametrize("token,label", _NON_FINITE_ENV)
def test_from_env_rejects_a_non_finite_float_knob(token: str, label: str) -> None:
    # The un-coercible-value contract from issue #66 still holds: a value that
    # fails coercion is logged and dropped, the default survives, startup does
    # not abort. Non-finite is now in that class rather than being accepted.
    env = {"SNAGLINE_CUSUM_H": token}
    overrides = Config.from_env_overrides(environ=env)
    assert "cusum_h" not in overrides, f"{label} must not reach the Config"
    cfg = Config.from_env(environ=env)
    assert cfg.cusum_h == Config().cusum_h


def test_from_env_still_accepts_a_finite_float_knob() -> None:
    # Positive control: the rejection is on non-finiteness, not on the string
    # form. A large-but-finite value still lands.
    cfg = Config.from_env(environ={"SNAGLINE_CUSUM_H": "1e3"})
    assert cfg.cusum_h == 1000.0


def test_inf_cusum_h_no_longer_deadows_the_latency_detector() -> None:
    # The issue's reproduction: a 10 ms baseline followed by 20 consecutive
    # 10-second latencies produced zero risks, because cusum > inf is never
    # true. The knob is now rejected at the env boundary, so the detector
    # falls back to the default bar and keeps working.
    cfg = Config.from_env(
        environ={"SNAGLINE_CUSUM_H": "inf", "SNAGLINE_CUSUM_MIN_SAMPLES": "3"}
    )
    d = LatencyAnomalyDetector(config=cfg)
    for i in range(3):
        d.observe(StepEvent(f"s{i}", "ep", 0.0, "tool_call", "sig", latency_ms=10.0))
    fired = [
        d.observe(StepEvent(f"b{i}", "ep", 0.0, "tool_call", "sig", latency_ms=10000.0))
        for i in range(20)
    ]
    assert any(r is not None for r in fired), "the detector must still alarm"


def test_a_finite_cusum_h_still_alarms_on_the_same_burst() -> None:
    # Positive control: with the default threshold the same burst does fire,
    # so the test above is asserting the knob's effect, not a detector that
    # never alarms regardless. It passes both before and after the fix.
    cfg = Config.from_env(environ={"SNAGLINE_CUSUM_MIN_SAMPLES": "3"})
    d = LatencyAnomalyDetector(config=cfg)
    for i in range(3):
        d.observe(StepEvent(f"s{i}", "ep", 0.0, "tool_call", "sig", latency_ms=10.0))
    fired = [
        d.observe(StepEvent(f"b{i}", "ep", 0.0, "tool_call", "sig", latency_ms=10000.0))
        for i in range(20)
    ]
    assert any(r is not None for r in fired)


def test_nan_sigma_floor_does_not_pin_the_cusum_at_zero() -> None:
    # ``max(0.0, nan)`` is ``0.0`` in CPython, so a NaN sigma floor made
    # ``_floored_sigma`` return 0 and the accumulator could never climb. A
    # zero-mean baseline is what reaches that path (no relative floor term).
    cfg = Config.from_env(
        environ={
            "SNAGLINE_CUSUM_SIGMA_FLOOR_ABS": "nan",
            "SNAGLINE_CUSUM_MIN_SAMPLES": "3",
        }
    )
    d = LatencyAnomalyDetector(config=cfg)
    for i in range(3):
        d.observe(StepEvent(f"s{i}", "ep", 0.0, "tool_call", "sig", latency_ms=0.0))
    fired = [
        d.observe(StepEvent(f"b{i}", "ep", 0.0, "tool_call", "sig", latency_ms=5000.0))
        for i in range(20)
    ]
    assert any(r is not None for r in fired)


# --- JSON config files: json.loads accepts Infinity/NaN by default ---------


def test_load_file_rejects_a_json_infinity_token(tmp_path) -> None:
    p = tmp_path / "cfg.json"
    p.write_text(json.dumps({"cusum_h": math.inf}), encoding="utf-8")
    with pytest.raises(ValueError, match="non-finite"):
        Config.load_file(str(p))


def test_load_file_rejects_a_json_nan_token(tmp_path) -> None:
    # json.dumps emits the bare NaN token; json.loads accepts it back.
    p = tmp_path / "cfg.json"
    p.write_text('{"min_severity_for_halt": NaN}', encoding="utf-8")
    with pytest.raises(ValueError, match="non-finite"):
        Config.load_file(str(p))


def test_load_file_rejects_a_json_overflow_literal(tmp_path) -> None:
    # 1e400 is a syntactically valid JSON number that overflows to inf.
    p = tmp_path / "cfg.json"
    p.write_text('{"halt_timeout_s": 1e400}', encoding="utf-8")
    with pytest.raises(ValueError, match="non-finite"):
        Config.load_file(str(p))


def test_load_file_reports_which_key_is_bad(tmp_path) -> None:
    p = tmp_path / "cfg.json"
    p.write_text('{"cusum_h": Infinity}', encoding="utf-8")
    with pytest.raises(ValueError, match="cusum_h"):
        Config.load_file(str(p))


def test_load_file_still_accepts_finite_floats(tmp_path) -> None:
    # Positive control: an ordinary JSON float is untouched.
    p = tmp_path / "cfg.json"
    p.write_text(json.dumps({"cusum_h": 7.5}), encoding="utf-8")
    assert Config.load_file(str(p)).cusum_h == 7.5


def test_load_file_rejects_non_finite_even_for_unknown_keys(tmp_path) -> None:
    # Unknown keys are ignored for *selection*, but a file carrying a bare
    # Infinity is malformed regardless of which section it sits in, and
    # silently accepting it would leave the guard trivially bypassable by
    # nesting. Reject the whole file.
    p = tmp_path / "cfg.json"
    p.write_text('{"not_a_field": Infinity, "cusum_h": 7.5}', encoding="utf-8")
    with pytest.raises(ValueError, match="not_a_field"):
        Config.load_file(str(p))


def test_resolve_rejects_a_non_finite_file_value(tmp_path) -> None:
    # resolve() layers the file; the guard must fire there too, not only on a
    # direct load_file call.
    p = tmp_path / "cfg.json"
    p.write_text('{"cusum_h": Infinity}', encoding="utf-8")
    with pytest.raises(ValueError, match="non-finite"):
        Config.resolve(path=str(p), environ={})


def test_resolve_env_cannot_revive_a_non_finite_value() -> None:
    # The env path is logged-and-dropped (issue #66), so resolve() must fall
    # back to the built-in default rather than carrying inf through.
    cfg = Config.resolve(environ={"SNAGLINE_CUSUM_H": "inf"})
    assert math.isfinite(cfg.cusum_h)
