"""Import-guard tests: the ``ml`` extra must be truly optional (issue #80).

These tests never require numpy. They simulate its absence by poisoning
``sys.modules`` so they pass in both the zero-dependency CI legs and a real
no-extras environment.
"""

from __future__ import annotations

import importlib
import sys

from snagline.config import Config
from snagline.events import StepEvent
from snagline.monitor import Monitor


def _ev(i: int) -> StepEvent:
    return StepEvent(
        step_id=f"s{i}",
        episode_id="ep",
        timestamp=float(i),
        action_type="tool_call",
        action_signature=f"sig{i}",
        tool_name="t",
        latency_ms=100.0,
        error=False,
    )


def _block_numpy(monkeypatch) -> None:
    """Make any ``import numpy`` raise as if the extra were not installed."""
    monkeypatch.setitem(sys.modules, "numpy", None)
    monkeypatch.delitem(sys.modules, "snagline.ml.esn_ensemble", raising=False)


def test_core_import_succeeds_without_numpy(monkeypatch):
    _block_numpy(monkeypatch)
    snagline = importlib.import_module("snagline")
    assert snagline is not None


def test_esn_module_raises_helpful_importerror_without_numpy(monkeypatch):
    _block_numpy(monkeypatch)
    try:
        importlib.import_module("snagline.ml.esn_ensemble")
    except ImportError as exc:
        assert "snagline[ml]" in str(exc)
    else:
        raise AssertionError("esn_ensemble imported without numpy")


def test_ml_flag_without_numpy_falls_back_to_plain_orchestrator(monkeypatch):
    _block_numpy(monkeypatch)
    cfg = Config()
    cfg.ml_ensemble_enabled = True
    mon = Monitor.default(config=cfg)
    # Byte-identical fallback: exactly one noisy-OR orchestrator, no crash.
    assert [d.name for d in mon._detectors] == ["ml_ensemble"]
    mon.ingest(_ev(0))
    mon.end_episode("ep")


def test_zero_dep_preset_has_no_ml_detectors():
    mon = Monitor.default(config=Config())
    names = sorted(d.name for d in mon._detectors)
    assert names == ["error_cascade", "latency_anomaly", "loop"]
