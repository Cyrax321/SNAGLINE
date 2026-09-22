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


def _scaling_cfg(scale_steps: int = 10, max_window: int = 512) -> Config:
    return Config(
        window_scale_steps=scale_steps,
        max_window=max_window,
    )


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


# --- the inferred position must seed the scaler, not just size the window ----
# (#403). A snapshot whose ``counts`` lacks an episode present in ``windows``
# restored at the right width and then lost it on the next observe, because
# observe() reads the counts dict to pick the next target.


def _strip_counts(dumped: dict, ep: str = "ep") -> dict:
    """A snapshot as an older release would have written it, or a partial
    payload: the window is present, the scaler position is not."""
    d = json.loads(json.dumps(dumped))
    if "counts" in d:
        d["counts"] = {k: v for k, v in d["counts"].items() if k != ep}
    return d


def test_error_cascade_keeps_history_on_first_observe_without_counts():
    """The restored window survived the first observe before #403 only by
    accident, when its width happened to sit at the base. Scaled past it, the
    missing position reset the scaler and one observe threw the history away.
    """
    cfg = _scaling_cfg()
    det = ErrorCascadeDetector(window_size=4, config=cfg)
    for i in range(30):
        det.observe(_event(i, f"u{i}", error=(i % 3 == 0)))
    live_len = len(det._windows["ep"])
    assert live_len > 4

    restored = ErrorCascadeDetector(window_size=4, config=cfg)
    stripped = _strip_counts(det.dump_state())
    restored.load_state(stripped)
    # The shipped window is the only evidence of progress, so the restored
    # width is what it implies -- smaller than live, and that is correct:
    # the position cannot be invented from a payload that lost it.
    implied = effective_window_size(
        4, len(stripped["windows"]["ep"]), cfg.window_scale_steps, cfg.max_window
    )
    assert len(restored._windows["ep"]) == implied
    # But the scaler is seeded from that inference instead of starting at 0,
    # so the first observe targets the implied size rather than refitting the
    # window down to the base of 4 and throwing the history away.
    restored.observe(_event(30, "u30"))
    assert len(restored._windows["ep"]) == implied


def test_meltdown_shrinks_back_to_target_when_oversized():
    """``push`` popped at most one item per call, so an oversized window --
    any restore whose inferred position disagreed with the live target --
    stayed the wrong size for the rest of the episode and entropy was
    computed over a sample the thresholds were not tuned for."""
    cfg = _scaling_cfg(max_window=24)
    det = MeltdownDetector(window_size=6, config=cfg)
    for i, tool in enumerate(f"tool{i % 6}" for i in range(40)):
        det.observe(_event(i, f"s{i}", tool=tool))
    assert len(det._eps["ep"].window) == 24

    restored = MeltdownDetector(window_size=6, config=cfg)
    restored.load_state(_strip_counts(det.dump_state()))
    restored_width = len(restored._eps["ep"].window)
    assert restored_width < 24, "fixture: the window must arrive oversized"

    # It recovers: enough steps to bring the scaler back up, and the window
    # is back at its target rather than pinned at the restored width.
    for i in range(60):
        restored.observe(_event(40 + i, f"s{i}", tool=f"tool{i % 6}"))
    assert len(restored._eps["ep"].window) == 24


def test_loop_keeps_scaled_window_on_first_observe_without_counts():
    cfg = _scaling_cfg()
    det = LoopDetector(window_size=4, config=cfg)
    for i in range(30):
        det.observe(_event(i, f"u{i}"))
    live_len = len(det._windows["ep"])
    assert live_len > 4

    restored = LoopDetector(window_size=4, config=cfg)
    stripped = _strip_counts(det.dump_state())
    restored.load_state(stripped)
    implied = effective_window_size(
        4, len(stripped["windows"]["ep"]), cfg.window_scale_steps, cfg.max_window
    )
    assert len(restored._windows["ep"]) == implied
    restored.observe(_event(30, "u30"))
    assert len(restored._windows["ep"]) == implied
    # The near/cycle families carry their own positions and need the same
    # treatment -- the near window is what the escalated re-arm scan reads.
    assert restored._near_counts.get("ep", 0) == len(
        stripped.get("near_windows", {}).get("ep", [])
    )


def test_position_without_window_survives_restore():
    """A position may exist with no window: the history expired but the
    scaler should not forget how far the episode has come. ``setdefault``
    keeps the snapshot's own value authoritative when both are present."""
    cfg = _scaling_cfg()
    det = ErrorCascadeDetector(window_size=4, config=cfg)
    det.load_state({"counts": {"ep": 40}, "windows": {}})
    assert det._counts["ep"] == 40
    det.observe(_event(0, "u0"))
    # The position survives into the counts the scaler reads, so the window
    # reaches the effective capacity on the step after the first observe
    # rather than restarting the episode from zero progress.
    assert det._counts["ep"] == 41
    det.observe(_event(1, "u1"))
    assert det._windows["ep"].maxlen == effective_window_size(
        4, 42, cfg.window_scale_steps, cfg.max_window
    )


def test_oversized_window_can_shrink_back_to_target():
    """``push`` pops while over rather than once.

    A call that appends one and pops at most one can only ever hold a
    too-wide window at its current width. Any caller that hands ``push`` a
    window already wider than its target -- a restore whose inferred
    position disagrees with the live one, or a reconfigure between snapshot
    and restore -- would then compute entropy over the wrong sample size for
    the rest of the episode. The loop is what keeps that from being
    permanent: the window returns to its target the moment one is applied.
    """
    from snagline.detectors.meltdown import _EpisodeWindow

    w = _EpisodeWindow()
    # Build a window of 18 against a target of 12, then drop the target: the
    # next push must bring the width down to the new target in one step.
    for i in range(18):
        w.push(f"t{i % 6}", 18)
    assert len(w.window) == 18
    w.push("t0", 12)
    assert len(w.window) == 12, "an oversized window must shrink to its target"
    # And the counts stay consistent with the surviving items.
    assert sum(w.counts.values()) == 12
    assert len(w.counts) <= 6


def test_snapshot_position_wins_over_inference():
    """When the snapshot carries a position for the episode it is
    authoritative: the inferred width is a fallback, not an override."""
    cfg = _scaling_cfg()
    det = ErrorCascadeDetector(window_size=4, config=cfg)
    det.load_state({"counts": {"ep": 40}, "windows": {"ep": [True, False]}})
    assert det._counts["ep"] == 40
