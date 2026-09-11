"""Scaled-window snapshot/restore round-trips (issue #268).

``load_state`` rebuilt every detector window at the BASE ``window_size``
even when auto-scaling (issue #92) had grown the live window to the
effective size. The oldest items -- exactly the history scaling exists to
preserve -- were silently truncated, leaving the restored detector partially
blind mid-episode.
"""

from __future__ import annotations

import json

from snagline.config import Config
from snagline.detectors.error_cascade import ErrorCascadeDetector
from snagline.detectors.loop import LoopDetector
from snagline.detectors.meltdown import MeltdownDetector
from snagline.detectors.windowing import effective_window_size
from snagline.events import StepEvent, make_signature


def _event(step: int, sig: str, error: bool = False, tool: str = "t") -> StepEvent:
    return StepEvent(
        step_id=str(step),
        episode_id="ep",
        timestamp=float(step),
        action_type="tool_call",
        action_signature=make_signature("tool_call", tool, sig),
        tool_name=tool,
        latency_ms=100.0,
        error=error,
    )


def _scaling_cfg(**overrides) -> Config:
    kw = dict(window_scale_steps=10, max_window=512)
    kw.update(overrides)
    return Config(**kw)


def test_error_cascade_restore_keeps_scaled_window():
    cfg = _scaling_cfg()
    det = ErrorCascadeDetector(window_size=4, config=cfg)
    for i in range(30):
        det.observe(_event(i, f"u{i}", error=(i % 3 == 0)))
    live_len = len(det._windows["ep"])
    assert live_len > 4, "window must have scaled past its base"
    dumped = json.loads(json.dumps(det.dump_state()))
    restored = ErrorCascadeDetector(window_size=4, config=cfg)
    restored.load_state(dumped)
    assert len(restored._windows["ep"]) == live_len
    assert list(restored._windows["ep"]) == list(det._windows["ep"])
    assert len(restored._windows["ep"]) == effective_window_size(
        4, det._counts["ep"], cfg.window_scale_steps, cfg.max_window
    )


def test_loop_restore_keeps_scaled_windows():
    cfg = _scaling_cfg()
    det = LoopDetector(window_size=4, config=cfg)
    for i in range(30):
        det.observe(_event(i, f"u{i}"))
    live_len = len(det._windows["ep"])
    assert live_len > 4, "window must have scaled past its base"
    dumped = json.loads(json.dumps(det.dump_state()))
    restored = LoopDetector(window_size=4, config=cfg)
    restored.load_state(dumped)
    assert len(restored._windows["ep"]) == live_len
    assert list(restored._windows["ep"]) == list(det._windows["ep"])


def test_meltdown_restore_keeps_scaled_window_and_still_scores():
    cfg = _scaling_cfg()
    det = MeltdownDetector(window_size=4, config=cfg)
    tools = [f"tool{i % 6}" for i in range(30)]
    for i, tool in enumerate(tools):
        det.observe(_event(i, f"s{i}", tool=tool))
    live_len = len(det._eps["ep"].window)
    assert live_len > 4, "window must have scaled past its base"
    dumped = json.loads(json.dumps(det.dump_state()))
    restored = MeltdownDetector(window_size=4, config=cfg)
    restored.load_state(dumped)
    assert len(restored._eps["ep"].window) == live_len
    # A restored full window scores immediately instead of going blind for
    # (effective - base) refill steps.
    assert list(restored._eps["ep"].window) == list(det._eps["ep"].window)


def test_restore_without_counts_stays_backward_compatible():
    """Pre-#92 snapshots carry no scaler positions: restore from the window
    the payload implies."""
    cfg = _scaling_cfg()
    det = ErrorCascadeDetector(window_size=4, config=cfg)
    det.load_state({"windows": {"ep": [True, False]}})
    assert list(det._windows["ep"]) == [True, False]
