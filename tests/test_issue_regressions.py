"""Regression tests pinning the audit issues that were already fixed in code.

These guard the fixes for issues #4, #7, #9, #10, #12, #14, #15, #16, and #20
so they cannot silently regress. Each test maps directly to a filed GitHub
issue.
"""

from __future__ import annotations

import logging

import pytest

from snagline.adapters.raw import watch
from snagline.config import Config
from snagline.detectors.error_cascade import ErrorCascadeDetector
from snagline.detectors.latency_anomaly import LatencyAnomalyDetector
from snagline.detectors.loop import LoopDetector
from snagline.events import EpisodeMeta, StepEvent, make_signature
from snagline.monitor import Monitor


def _ev(action_type, sig, episode="ep", error=False, latency_ms=None, tool_name=None):
    return StepEvent(
        step_id="s",
        episode_id=episode,
        timestamp=1.0,
        action_type=action_type,
        action_signature=sig,
        tool_name=tool_name,
        latency_ms=latency_ms,
        error=error,
    )


def test_loop_detector_emits_once_not_every_step():
    # Issue #4: a sustained loop must not spam one risk per step.
    det = LoopDetector(config=Config())
    risks = [det.observe(_ev("tool_call", "SAME")) for _ in range(6)]
    fired = [r for r in risks if r is not None]
    assert len(fired) == 1


def test_error_cascade_only_counts_tool_errors_by_default():
    # Issue #16: LLM/chain errors (action_type != tool_call) must NOT inflate
    # the cascade by default; only tool failures count.
    det = ErrorCascadeDetector(config=Config())
    llm_errors = [det.observe(_ev("message", "m", error=True)) for _ in range(5)]
    assert all(r is None for r in llm_errors)
    tool_errors = [det.observe(_ev("tool_call", "t", error=True)) for _ in range(3)]
    assert any(r is not None for r in tool_errors)


def test_make_signature_returns_full_digest_and_avoids_ambiguity():
    # Issue #15: full 64-char SHA-256 digest, and the "||" separator must not
    # let two distinct actions collide.
    a = make_signature("tool_call", "x", "a", "b")
    b = make_signature("tool_call", "x", "a|b")
    assert len(a) == 64 and len(b) == 64
    assert a != b  # ambiguity guarded by JSON-encoding the parts


def test_monitor_logs_fault_once_per_key(caplog):
    # Issue #14: under fail_open, a broken detector is logged once, not per step.
    class BadDet:
        name = "bad"

        def observe(self, event):
            raise RuntimeError("boom")

        def reset(self, episode_id):
            raise RuntimeError("boom")

    caplog.set_level(logging.ERROR, logger="snagline")
    mon = Monitor([BadDet()], [], fail_open=True)
    for _ in range(5):
        mon.ingest(_ev("tool_call", "sig"))
    keys = [
        r.getMessage()
        for r in caplog.records
        if "ignoring (fail-open)" in r.getMessage()
    ]
    assert len(keys) == 1


def test_default_docstring_lists_three_detectors():
    # Issue #20: the default Monitor wires loop, error-cascade, and latency.
    doc = Monitor.default.__doc__ or ""
    assert "loop" in doc
    assert "error-cascade" in doc
    assert "latency" in doc


def test_pyproject_declares_no_empty_extras():
    # Issue #12: no optional extra may ship an empty dependency list that
    # installs nothing and advertises unsupported integrations.
    path = "pyproject.toml"
    tomllib = pytest.importorskip("tomllib")  # 3.11+; skips on 3.10
    with open(path, "rb") as fh:
        data = tomllib.load(fh)
    extras = data["project"].get("optional-dependencies", {})
    assert extras, "expected at least the documented extras"
    for name, deps in extras.items():
        assert deps, f"extra {name!r} is empty; remove it or gate it as preview"


def test_latency_detector_ignores_plan_step_events():
    # Issue #10: whole-chain / plan_step durations must not be treated as a
    # single-tool latency sample, or deep agents false-positive constantly.
    det = LatencyAnomalyDetector(config=Config())
    # Five identical very long plan_step latencies: must NOT alarm (only
    # leaf tool_call/message latencies feed the CUSUM detector).
    for i in range(5):
        ev = _ev("plan_step", "chain", latency_ms=50000.0 + i, tool_name="planner")
        assert det.observe(ev) is None


def test_latency_detector_watches_low_frequency_tool():
    # Issue #9: a tool called only a handful of times (well under the old
    # threshold of 20) should still get latency protection after the short
    # warm-up (cusum_min_samples default is now 5).
    det = LatencyAnomalyDetector(config=Config())
    for i in range(5):
        ev = _ev("tool_call", "search", latency_ms=100.0, tool_name="search")
        det.observe(ev)
    spike = _ev("tool_call", "search", latency_ms=100000.0, tool_name="search")
    assert det.observe(spike) is not None


def test_raw_watch_context_manager_and_episodemeta_importable():
    # Issue #7: raw.watch must work and the dead EpisodeMeta construction must
    # be gone; the type is still part of the public schema and importable.
    assert EpisodeMeta.__name__ == "EpisodeMeta"

    class _Sink:
        def emit(self, risk):
            pass

    mon = Monitor.default(sinks=[_Sink()])
    with watch(mon, "ep-raw") as step:
        step("tool_call", tool_name="x", args="a")
    # The context manager tore down per-episode state without raising.
