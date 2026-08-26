"""Window auto-scaling and CUSUM baseline re-fit (issue #92).

Property under test: with ``window_scale_steps == 0`` (default) detector
behavior is byte-identical to the pre-#92 build; with scaling on, effective
windows stay within ``[base, max_window]``, grow monotonically, and detection
quality is preserved across episode-length fixtures (a pattern that a static
base window misses on a long episode is still caught once scaled).
"""

from __future__ import annotations

import itertools

import pytest

from snagline.config import Config
from snagline.detectors.error_cascade import ErrorCascadeDetector
from snagline.detectors.latency_anomaly import LatencyAnomalyDetector
from snagline.detectors.loop import LoopDetector
from snagline.detectors.windowing import effective_window_size, next_window
from snagline.events import StepEvent


def _event(
    step_id: str,
    timestamp: float,
    signature: str,
    *,
    error: bool = False,
    latency_ms: float | None = None,
    episode_id: str = "ep1",
) -> StepEvent:
    return StepEvent(
        step_id=step_id,
        episode_id=episode_id,
        timestamp=timestamp,
        action_type="tool_call",
        action_signature=signature,
        tool_name="t",
        error=error,
        latency_ms=latency_ms,
    )


# --- pure function properties (grid over lengths and configs) ---------------


def test_effective_window_size_properties() -> None:
    bases = [1, 5, 12, 50]
    scale_steps = [0, 1, 7, 100, 10_000]
    max_windows = [16, 512]
    lens = [0, 1, 2, 6, 99, 100, 101, 999, 100_000]
    for base, steps, cap, n in itertools.product(bases, scale_steps, max_windows, lens):
        size = effective_window_size(base, n, steps, cap)
        assert base <= size <= max(cap, base), (base, steps, cap, n, size)
        if steps <= 0 or n <= 1:
            assert size == base


def test_effective_window_size_monotonic_in_episode_length() -> None:
    prev = 0
    for n in range(1, 5001):
        size = effective_window_size(12, n, 100, 200)
        assert size >= prev
        prev = size
    assert prev == 200  # capped


def test_next_window_resizes_lazily_and_keeps_recent_items() -> None:
    windows: dict = {}
    counts: dict = {}
    w = next_window(windows, counts, "ep", base=2, scale_steps=2, max_window=4)
    w.append("a")
    w.append("b")
    # len now 2; factor ceil(2/2)=1 -> still base=2
    assert w.maxlen == 2
    w = next_window(windows, counts, "ep", 2, 2, 4)
    w.append("c")
    assert w.maxlen == 2  # event 2: ceil(2/2)=1, still base
    w = next_window(windows, counts, "ep", 2, 2, 4)
    w.append("d")
    assert w.maxlen == 4  # event 3: ceil(3/2)=2 -> grew lazily
    assert list(w) == ["b", "c", "d"]  # most recent items retained


# --- detector equivalence fixtures ------------------------------------------


def _planted_loop_episode(length: int, loop_start: int) -> list[StepEvent]:
    """A long episode of distinct steps with an exact A/B loop planted late."""
    events = []
    ts = 0.0
    for i in range(loop_start):
        events.append(_event(f"s{i}", ts, f"unique-{i}"))
        ts += 1.0
    pair = itertools.cycle(["loopA", "loopB"])
    for j in range(loop_start, length):
        events.append(_event(f"s{j}", ts, next(pair)))
        ts += 1.0
    return events


def _run_loop(events: list[StepEvent], **cfg_kwargs) -> int:
    det = LoopDetector(config=Config(**cfg_kwargs))
    fires = 0
    for e in events:
        if det.observe(e) is not None:
            fires += 1
    return fires


@pytest.mark.parametrize("length", [50, 500, 5_000])
def test_scaling_equivalent_below_scale_floor(length: int) -> None:
    """With scale_steps > length, scaling on == scaling off exactly."""
    events = _planted_loop_episode(length, length - 20)
    off = _run_loop(events)
    on = _run_loop(events, window_scale_steps=length + 1, max_window=64)
    assert on == off


def test_scaled_window_catches_loop_static_base_misses() -> None:
    """The value proposition: a sparse loop beyond the base window is caught."""
    # LoopA lands every 25 distinct steps: three repeats span 50 steps, so a
    # static 12-window holds one repeat (never fires) while the scaled window
    # (base * ceil(n/steps), capped) eventually holds three.
    length = 3_000
    events: list[StepEvent] = []
    ts = 0.0
    for i in range(length):
        sig = "loopA" if i % 25 == 0 else f"unique-{i}"
        events.append(_event(f"s{i}", ts, sig))
        ts += 1.0
    static = _run_loop(events)
    scaled = _run_loop(events, window_scale_steps=500, max_window=256)
    assert static == 0
    assert scaled > 0


def test_cascade_scaling_preserves_short_episode_behavior() -> None:
    cfg_off = Config()
    cfg_on = Config(window_scale_steps=10_000, max_window=128)
    events = [
        _event(f"s{i}", float(i), f"sig{i}", error=(i % 4 == 0)) for i in range(30)
    ]
    d_off = ErrorCascadeDetector(config=cfg_off)
    d_on = ErrorCascadeDetector(config=cfg_on)
    for e in events:
        r_off = d_off.observe(e)
        r_on = d_on.observe(e)
        assert (r_off is None) == (r_on is None)


# --- CUSUM periodic re-fit ---------------------------------------------------


def test_cusum_refit_surfaces_baseline_drift() -> None:
    cfg = Config(cusum_min_samples=5, cusum_refit_every=10)
    det = LatencyAnomalyDetector(config=cfg)
    risks = []
    ts = 0.0
    # Healthy warm-up around 100ms, then frozen.
    for i in range(8):
        e = _event(f"w{i}", ts, "s", latency_ms=100.0 + (i % 2))
        risks.append(det.observe(e))
        ts += 1.0
    # Sustained shift to ~400ms. The CUSUM alarms against the old baseline;
    # once the parallel learner adopts the new one, the accumulated shift
    # surfaces as its own risk and later steps go quiet again.
    for i in range(40):
        e = _event(f"x{i}", ts, "s", latency_ms=400.0)
        risks.append(det.observe(e))
        ts += 1.0
    triggers = [r.trigger for r in risks if r is not None]
    assert "latency_anomaly" in triggers  # CUSUM alarm fired
    details = [r.detail for r in risks if r is not None]
    assert any("baseline shifted" in d for d in details)  # drift visible
    # After adoption, 400ms IS the baseline: the detector goes quiet.
    assert all(r is None for r in risks[-5:])


def test_cusum_refit_disabled_by_default_matches_pre92_behavior() -> None:
    cfg = Config()
    det = LatencyAnomalyDetector(config=cfg)
    state = None
    ts = 0.0
    for i in range(10):
        det.observe(_event(f"w{i}", ts, "s", latency_ms=50.0))
        ts += 1.0
    key = ("ep1", "t")
    state = det._states[key]
    assert state.refit_every == 0
    assert state.learner_n == 0
    assert not state.pending_drift


def test_cusum_refit_snapshot_roundtrip_tolerates_old_payloads() -> None:
    cfg = Config(cusum_min_samples=2, cusum_refit_every=5)
    det = LatencyAnomalyDetector(config=cfg)
    ts = 0.0
    for i in range(6):
        det.observe(_event(f"w{i}", ts, "s", latency_ms=50.0 + i))
        ts += 1.0
    dumped = det.dump_state()
    det2 = LatencyAnomalyDetector(config=cfg)
    det2.load_state(dumped)
    key = ("ep1", "t")
    assert det2._states[key].refit_every == 5
    # Pre-#92 payload (no refit keys) must load cleanly into a refit-enabled
    # detector AND into a default one.
    old_raw = {
        k: v
        for k, v in dumped["states"][0][1].items()
        if not k.startswith(("refit", "learner"))
    }
    old_dump = {"states": [[dumped["states"][0][0], old_raw]]}
    det3 = LatencyAnomalyDetector(config=cfg)
    det3.load_state(old_dump)
    assert det3._states[key].learner_n == 0


def test_loop_detector_default_counts_invisible_when_scaling_off() -> None:
    det = LoopDetector(config=Config())
    events = [_planted_loop_episode(30, 25)[0]]
    for i, e in enumerate(events):
        det.observe(e)
    # Counts maintained but never consulted when scale_steps == 0.
    assert det._counts.get("ep1") == len(events)
    assert all(w.maxlen == 12 for w in det._windows.values())
