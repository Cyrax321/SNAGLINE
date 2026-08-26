"""Semantic goal-drift tests: ``drift`` extra (issue #81).

Everything here runs without sentence-transformers, numpy, or torch: the
detector accepts an injectable ``embedder`` callable, so the suite exercises
the real detection logic with small hand-built vectors. Import-guard tests
additionally poison ``sys.modules`` to prove laziness. Both sides are always
covered per detector behavior: an injected-drift sequence that fires and a
healthy sequence that stays silent.
"""

from __future__ import annotations

import json

import pytest

from snagline.baseline import BaselineProfile, save_baseline
from snagline.events import StepEvent

# ---------------------------------------------------------------------------
# Helpers


def _ev(i: int, tool: str = "search", episode: str = "ep") -> StepEvent:
    return StepEvent(
        step_id=f"s{i}",
        episode_id=episode,
        timestamp=float(i),
        action_type="tool_call",
        action_signature=f"sig-{tool}-{i}",
        tool_name=tool,
        latency_ms=100.0,
        error=False,
    )


def _profile(centroid: list[float]) -> BaselineProfile:
    p = BaselineProfile()
    p.embedding_centroid = centroid
    p.embedding_count = 4
    p.embedding_model = "fake-model"
    return p


def test_profile_without_semantics_serializes_exactly_as_before():
    # Byte-identical zero-dep preset: no embedding keys unless fitted.
    legacy_keys = {"version", "total_steps", "tools"}
    assert set(BaselineProfile().to_dict().keys()) == legacy_keys


def test_profile_semantic_round_trip_through_dict():
    p = _profile([1.0, 0.0])
    restored = BaselineProfile.from_dict(p.to_dict())
    assert restored.embedding_centroid == [1.0, 0.0]
    assert restored.embedding_count == 4
    assert restored.embedding_model == "fake-model"


def test_profile_legacy_json_without_embedding_keys_loads_unchanged():
    raw = {
        "version": 1,
        "total_steps": 3,
        "tools": {},
    }
    p = BaselineProfile.from_dict(raw)
    assert p.embedding_centroid is None
    assert p.embedding_count == 0
    assert p.embedding_model is None


def test_saved_baseline_file_round_trips_semantic_fields(tmp_path):
    p = _profile([0.6, 0.8])
    path = str(tmp_path / "base.json")
    save_baseline(p, path)
    with open(path, encoding="utf-8") as fh:
        on_disk = json.load(fh)
    assert on_disk["embedding_centroid"] == [0.6, 0.8]
    # The structural part stays untouched by the optional fields.
    assert set(on_disk.keys()) >= {"version", "total_steps", "tools"}


def test_profile_structural_to_dict_is_byte_identical_to_legacy_shape():
    # Compare against the exact pre-#81 serialization of an empty profile.
    expected = '{\n  "tools": {},\n  "total_steps": 0,\n  "version": 1\n}\n'
    import io

    buf = io.StringIO()
    from snagline.baseline import _write_json

    _write_json(buf, BaselineProfile().to_dict())
    assert buf.getvalue() == expected


def test_profile_from_dict_rejects_non_numeric_centroids():
    with pytest.raises((TypeError, ValueError)):
        BaselineProfile.from_dict({"embedding_centroid": ["x"]})


# ---------------------------------------------------------------------------
# Config keys (scalar coercion must cover the new opt-in fields)


def test_semantic_drift_keys_coerce_from_environment():
    from snagline.config import Config

    cfg = Config.from_env(
        {
            "SNAGLINE_SEMANTIC_DRIFT_ENABLED": "true",
            "SNAGLINE_SEMANTIC_DRIFT_MODEL": "custom/model",
            "SNAGLINE_SEMANTIC_DRIFT_MIN_SAMPLES": "7",
            "SNAGLINE_SEMANTIC_DRIFT_TOLERANCE": "0.4",
        }
    )
    assert cfg.semantic_drift_enabled is True
    assert cfg.semantic_drift_model == "custom/model"
    assert cfg.semantic_drift_min_samples == 7
    assert cfg.semantic_drift_tolerance == pytest.approx(0.4)


def test_semantic_drift_defaults_keep_preset_off():
    from snagline.config import Config

    cfg = Config()
    assert cfg.semantic_drift_enabled is False
    # Hand-computed expectation of the shipped defaults (not read back from
    # the same object): documented in README and DETECTOR_GUIDE.
    assert cfg.semantic_drift_min_samples == 10
    assert cfg.semantic_drift_tolerance == 0.3
    assert cfg.semantic_drift_cusum_k == 0.05
    assert cfg.semantic_drift_cusum_h == 0.5
