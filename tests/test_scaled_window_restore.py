"""Scaled-window snapshot/restore round-trip tests (issue #268).

``load_state`` used to rebuild every detector window at the BASE
``window_size`` even when auto-scaling (issue #92) had grown the live window
to the effective size. The oldest items -- exactly the history scaling exists
to preserve -- were silently truncated away, leaving the restored detector
partially blind mid-episode: meltdown's full-window gate returned ``None``
for the (effective - base) refill steps, and loop/error-cascade windows lost
occurrences that lived inside the scaled window.

These tests hold with ``window_scale_steps=0`` too (the default), but the
defect only manifests with scaling on, so every detector is constructed with
a scaling config and a steady-state episode long enough that the live window
has grown past the base.
"""

from __future__ import annotations

import json

from snagline.config import Config
from snagline.detectors.error_cascade import ErrorCascadeDetector
from snagline.detectors.loop import LoopDetector
from snagline.detectors.meltdown import MeltdownDetector
from snagline.events import StepEvent


def _cfg() -> Config:
    # base 12, one scale step per 12 events, capped at 40: after 120+ events
    # the effective window is 40 while the base is 12.
    return Config(window_scale_steps=12, max_window=40)


def _ev(
    step: int,
    *,
    signature: str,
    tool_name: str | None = "t",
    action_type: str = "tool_call",
    error: bool = False,
) -> StepEvent:
    return StepEvent(
        step_id=str(step),
        episode_id="ep",
        timestamp=float(step),
        action_type=action_type,
        action_signature=signature,
        tool_name=tool_name,
        error=error,
    )


def _observe_all(det, events):
    return [r for e in events if (r := det.observe(e)) is not None]


def _round_trip(det):
    det.load_state(json.loads(json.dumps(det.dump_state())))


# --- meltdown: entropy collapsed at restore time -----------------------------


def test_meltdown_scaled_window_survives_restore():
    """A collapsed window is fully restored: the next step fires exactly as
    the never-restarted twin does, with no refill blindness."""
    live = MeltdownDetector(config=_cfg())
    restored = MeltdownDetector(config=_cfg())
    # 220 steps of one tool: effective window 40, entropy collapsed, but
    # never fires (the episode is all one collapse from step 1, so fire the
    # alarm is pre-latched) -- force the re-armed state a live monitor would
    # have after a first alarm cleared.
    collapse = [_ev(i, signature=f"sig-{i}", tool_name="search") for i in range(220)]
    _observe_all(live, collapse)
    _observe_all(restored, collapse)
    for det in (live, restored):
        det._fired["ep"] = False  # post-alarm re-armed, condition still active

    restored.load_state(json.loads(json.dumps(restored.dump_state())))

    nxt = [_ev(500 + i, signature=f"s5{i}", tool_name="search") for i in range(3)]
    live_risks = _observe_all(live, nxt)
    restored_risks = _observe_all(restored, nxt)
    assert live_risks, "live detector must fire immediately on the active collapse"
    assert restored_risks == live_risks, (
        "restored detector must not be blinded by a base-sized window rebuild"
    )


def test_meltdown_restored_window_holds_effective_length():
    live = MeltdownDetector(config=_cfg())
    _observe_all(
        live, [_ev(i, signature=f"sig-{i}", tool_name="t") for i in range(120)]
    )
    dumped = json.loads(json.dumps(live.dump_state()))
    assert len(dumped["windows"]["ep"]) == 40, "live window must have scaled to 40"

    restored = MeltdownDetector(config=_cfg())
    restored.load_state(dumped)
    assert len(restored._eps["ep"].window) == 40, (
        "restored window must keep the effective length, not truncate to base 12"
    )


# --- loop: early occurrences inside the scaled window ------------------------


def test_loop_scaled_window_survives_restore():
    """Two early occurrences of a repeating signature -- inside the live
    40-window, outside the base 12-tail -- must survive the restore so the
    third occurrence fires, exactly like the never-restarted twin."""
    events = []
    step = 0
    # 120 unique steps grow the effective window to 40.
    for _ in range(120):
        events.append(_ev(step, signature=f"unique-{step}"))
        step += 1
    # Two loop occurrences, then 13 fillers so both fall outside the base
    # 12-tail but stay inside the scaled 40-window.
    for _ in range(2):
        events.append(_ev(step, signature="loopme"))
        step += 1
    for i in range(13):
        events.append(_ev(step, signature=f"filler-{i}"))
        step += 1

    live = LoopDetector(config=_cfg())
    restored = LoopDetector(config=_cfg())
    for det in (live, restored):
        _observe_all(det, events)
    restored.load_state(json.loads(json.dumps(restored.dump_state())))

    # The 3rd occurrence: 40/12 window holds all three for the live detector.
    nxt = [_ev(step, signature="loopme")]
    live_fired = _observe_all(live, nxt)
    restored_fired = _observe_all(restored, nxt)
    assert live_fired, "live detector must fire on the 3rd occurrence"
    assert restored_fired == live_fired, (
        "restored detector dropped the early occurrences via base-size truncation"
    )


def test_loop_restored_window_holds_effective_length():
    live = LoopDetector(config=_cfg())
    _observe_all(live, [_ev(i, signature=f"u-{i}") for i in range(120)])
    dumped = json.loads(json.dumps(live.dump_state()))
    assert len(dumped["windows"]["ep"]) == 40

    restored = LoopDetector(config=_cfg())
    restored.load_state(dumped)
    assert restored._windows["ep"].maxlen == 40, (
        "restored window maxlen must be the effective size, not the base 12"
    )
    assert len(restored._windows["ep"]) == 40


# --- error_cascade: windowed density across the restart ----------------------


def test_error_cascade_scaled_window_survives_restore():
    """Errors that fell out of the base 12-tail but live in the scaled
    40-window keep counting after restore, so the density alarm fires on
    the same event the never-restarted twin fires on."""
    events = []
    step = 0
    for _ in range(120):
        events.append(_ev(step, signature=f"u-{step}", error=False))
        step += 1
    # Two counted errors, then 13 clean steps: outside the base tail,
    # inside the scaled window.
    for _ in range(2):
        events.append(_ev(step, signature=f"e-{step}", error=True))
        step += 1
    for i in range(13):
        events.append(_ev(step, signature=f"c-{i}", error=False))
        step += 1

    live = ErrorCascadeDetector(config=_cfg())
    restored = ErrorCascadeDetector(config=_cfg())
    for det in (live, restored):
        _observe_all(det, events)
    restored.load_state(json.loads(json.dumps(restored.dump_state())))

    # 3rd error reaches cascade_error_threshold=3 inside the scaled window.
    nxt = [_ev(step, signature="boom", error=True)]
    live_fired = _observe_all(live, nxt)
    restored_fired = _observe_all(restored, nxt)
    assert live_fired, "live detector must fire on the 3rd error in window"
    assert restored_fired == live_fired, (
        "restored detector lost windowed errors via base-size truncation"
    )


def test_error_cascade_restored_window_holds_effective_length():
    live = ErrorCascadeDetector(config=_cfg())
    _observe_all(live, [_ev(i, signature=f"u-{i}") for i in range(120)])
    dumped = json.loads(json.dumps(live.dump_state()))
    assert len(dumped["windows"]["ep"]) == 40

    restored = ErrorCascadeDetector(config=_cfg())
    restored.load_state(dumped)
    assert restored._windows["ep"].maxlen == 40, (
        "restored window maxlen must be the effective size, not the base 10"
    )
    assert len(restored._windows["ep"]) == 40


# --- scaling off: restore behavior unchanged ---------------------------------


def test_no_scaling_restore_still_truncates_to_base():
    """With scaling off (default) the effective size is the base, so the
    tolerant-restore clamp behavior is preserved exactly."""
    live = LoopDetector(config=Config())
    _observe_all(live, [_ev(i, signature=f"u-{i}") for i in range(50)])
    restored = LoopDetector(config=Config())
    restored.load_state(json.loads(json.dumps(live.dump_state())))
    assert restored._windows["ep"].maxlen == live.window_size


# --- old snapshot payloads (pre-#92, no counts) still restore ----------------


def test_pre92_payload_without_counts_restores_at_effective_size():
    """Counts are absent in pre-#92 payloads: the fallback (window length)
    still sizes the restore correctly for the truncated payload."""
    live = LoopDetector(config=_cfg())
    _observe_all(live, [_ev(i, signature=f"u-{i}") for i in range(120)])
    dumped = json.loads(json.dumps(live.dump_state()))
    del dumped["counts"]  # simulate a pre-#92 payload

    restored = LoopDetector(config=_cfg())
    restored.load_state(dumped)
    assert len(restored._windows["ep"]) == 40, (
        "fallback len(sigs) must still restore the full scaled window"
    )
