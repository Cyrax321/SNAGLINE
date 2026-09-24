"""Window auto-scaling and CUSUM baseline re-fit (issue #92).

Property under test: with ``window_scale_steps == 0`` (default) detector
behavior is byte-identical to the pre-#92 build; with scaling on, effective
windows stay within ``[base, max_window]``, grow monotonically, and detection
quality is preserved across episode-length fixtures (a pattern that a static
base window misses on a long episode is still caught once scaled).
"""

from __future__ import annotations

import itertools
import random
from collections import Counter, deque

import pytest

from snagline.config import Config
from snagline.detectors.error_cascade import ErrorCascadeDetector
from snagline.detectors.latency_anomaly import LatencyAnomalyDetector
from snagline.detectors.loop import LoopDetector
from snagline.detectors.windowing import (
    append_counted,
    effective_window_size,
    maintain_counter,
    next_window,
)
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


def test_cusum_refit_reports_sustained_shift_below_single_step_bar() -> None:
    """Regression for #482: periodic re-fit adopted the learner's baseline
    *unconditionally* but only emitted a drift risk when the move exceeded the
    single-step ``h * sigma`` alarm bar. A sustained regression in the band
    ``k*sigma < shift <= h*sigma`` -- one the CUSUM's *sustained-shift*
    sensitivity (accumulation past the dead-band ``k``) would eventually flag
    -- was therefore folded into the new baseline with NO risk, defeating the
    module's frozen-baseline promise.

    Defaults k=0.5, h=5.0. Warm-up at a constant 100ms floors sigma0 at 5ms
    (5% of the mean), so the dead-band edge is k*sigma0 = 2.5ms and the old bar
    was h*sigma0 = 25ms. A +3ms sustained shift sits between them: the CUSUM
    accumulates only 0.1/step (never reaching h before the refit adopts), so
    pre-fix the whole run is silent. The re-fit must now report it once."""
    cfg = Config(cusum_min_samples=5, cusum_refit_every=5)
    det = LatencyAnomalyDetector(config=cfg)
    risks = []
    ts = 0.0
    for i in range(5):  # constant warm-up -> mu0=100, sigma0 floored to 5ms
        risks.append(det.observe(_event(f"w{i}", ts, "s", latency_ms=100.0)))
        ts += 1.0
    for i in range(15):  # sustained +3ms: past the dead-band, below the old bar
        risks.append(det.observe(_event(f"x{i}", ts, "s", latency_ms=103.0)))
        ts += 1.0
    fired = [r for r in risks if r is not None]
    assert len(fired) == 1, f"the absorbed shift must surface exactly once: {fired}"
    assert fired[0].trigger == "latency_anomaly"
    assert fired[0].score == 0.8
    assert "baseline shifted" in fired[0].detail


def test_cusum_alarm_coincident_with_adoption_keeps_severity_and_detail() -> None:
    """Issue #244: the alarm was scored *after* the periodic re-fit advanced.
    When adoption landed on the same step as an alarm, adopt_candidate() had
    already reset cusum to 0 and replaced mu0, so the risk came out with a
    flat 0.6 score and the self-contradictory detail "300ms deviates from
    baseline (mean 300ms)". The alarm must be scored against the snapshot
    taken at alarm time, before the re-fit touches the state.
    """
    cfg = Config(cusum_min_samples=5, cusum_refit_every=3)
    det = LatencyAnomalyDetector(config=cfg)
    ts = 0.0
    for i in range(5):  # healthy warm-up at 100ms, then frozen
        det.observe(_event(f"w{i}", ts, "s", latency_ms=100.0))
        ts += 1.0
    risks = []
    for i in range(10):  # sustained regression at 300ms
        risks.append(det.observe(_event(f"x{i}", ts, "s", latency_ms=300.0)))
        ts += 1.0
    alarm_risks = [
        r for r in risks if r is not None and "deviates from baseline" in r.detail
    ]
    assert alarm_risks, "the sustained shift must alarm"
    for r in alarm_risks:
        # Score: pre-adoption cusum was far above h, so the alarm must carry
        # the severity bonus, not the 0.6 floor an adopted-reset produces.
        assert r.score > 0.6, f"coincident alarm scored flat {r.score}"
        # Detail: the mean cited must be the baseline the alarm was measured
        # against (100ms), never the just-adopted 300ms.
        assert "mean 100ms" in r.detail, f"self-contradictory detail: {r.detail}"
        assert "mean 300ms" not in r.detail


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


def test_near_duplicate_mode_counts_events_once_per_family() -> None:
    """Regression (gitar finding, PR #167): plain and near-duplicate paths
    share one observe() call per event, so each window family must keep its
    own event counter; a shared one would count every event twice and scale
    windows twice as fast.
    """
    cfg = Config(loop_near_duplicate_enabled=True, window_scale_steps=10)
    det = LoopDetector(config=cfg)
    for i in range(25):
        det.observe(_event(f"s{i}", float(i), "loopA" if i % 2 else f"u{i}"))
    assert det._counts["ep1"] == 25  # advanced once per event, not twice
    assert det._near_counts["ep1"] == 25


def test_cusum_pending_drift_survives_snapshot_restore() -> None:
    """A baseline shift detected on the same step a CUSUM alarm held the risk
    slot is deferred to the next quiet step. dump_state/load_state used to
    drop the deferral entirely, so a restart taken between detection and
    emission silently lost the "baseline shifted" risk forever.

    The fixture leaves a deferred shift held: baseline moved 9 -> 70ms while
    an alarm owned the slot.
    """
    cfg = Config(cusum_k=0.5, cusum_h=2.0, cusum_min_samples=3, cusum_refit_every=4)

    def feed(det: LatencyAnomalyDetector, seq: list[float]) -> None:
        for i, lat in enumerate(seq):
            det.observe(_event(f"s{i}", float(i), "s", latency_ms=lat))

    seq = [12.0, 8.0, 8.0, 100.0, 100.0, 100.0, 100.0, 10.0, 100.0, 200.0, 100.0]
    live = LatencyAnomalyDetector(config=cfg)
    feed(live, seq)
    assert live._states[("ep1", "t")].pending_drift, "fixture must reach the deferral"

    restored = LatencyAnomalyDetector(config=cfg)
    restored.load_state(live.dump_state())
    assert restored._states[("ep1", "t")].pending_drift, (
        "a deferred baseline shift must survive snapshot/restore"
    )

    # The deferred risk must actually be delivered on the first quiet step.
    live2 = LatencyAnomalyDetector(config=cfg)
    feed(live2, seq)
    restored2 = LatencyAnomalyDetector(config=cfg)
    restored2.load_state(live2.dump_state())
    delivered: dict[str, int] = {"live": 0, "restored": 0}
    for i in range(6):
        e = _event(f"q{i}", float(len(seq) + i), "s", latency_ms=70.0)
        for det, key in ((live2, "live"), (restored2, "restored")):
            r = det.observe(e)
            if r is not None and "baseline shifted" in r.detail:
                delivered[key] += 1
    assert delivered["live"] == 1, "the live detector must deliver the shift"
    assert delivered["restored"] == 1, (
        f"the restored detector must deliver it too, got {delivered['restored']}"
    )


def test_cusum_pending_drift_fields_stay_absent_when_refit_disabled() -> None:
    """The pending_* keys are written only with refit active, so a default
    config's snapshot stays byte-identical to a pre-#92 one."""
    det = LatencyAnomalyDetector(config=Config())
    det.observe(_event("s0", 0.0, "s", latency_ms=10.0))
    raw = det.dump_state()["states"][0][1]
    assert not any(k.startswith("pending") for k in raw)


# --- running counts vs rescanning (issue #298) ------------------------------
# The scaled path answers ``deque.count`` / ``sum(w)`` questions from a Counter
# maintained alongside the deque. If the Counter ever drifts from the window --
# on eviction, on a lazy resize, or after a snapshot restore -- the O(1) answer
# is silently *wrong*, not merely slow. These hold the invariant that the cached
# counter always equals a fresh pass over the window.


def test_append_counted_tracks_appends_evictions_and_deletes_at_zero() -> None:
    w: deque = deque(maxlen=3)
    counts: Counter = Counter()
    assert append_counted(w, counts, "a") == 1
    assert append_counted(w, counts, "a") == 2
    assert append_counted(w, counts, "b") == 1
    assert list(w) == ["a", "a", "b"]
    assert counts == Counter({"a": 2, "b": 1})
    # Window is full: appending "a" evicts the oldest "a", so the net count of
    # "a" is unchanged (2 - 1 evicted + 1 appended).
    assert append_counted(w, counts, "a") == 2
    assert list(w) == ["a", "b", "a"]
    # Evicting the last copy of a key removes it rather than leaving a stale 0,
    # so the Counter cannot accumulate one dead key per distinct signature.
    append_counted(w, counts, "c")  # evicts "a" -> a:1
    append_counted(w, counts, "d")  # evicts "b" -> b gone
    assert "b" not in counts
    assert counts == Counter({"a": 1, "c": 1, "d": 1})
    assert list(w) == ["a", "c", "d"]


def test_append_counted_on_zero_capacity_window_returns_zero() -> None:
    """A detector configured with ``window_size=0`` (Config does not reject it)
    yields ``deque(maxlen=0)``, which drops every item. The count is then always
    0 -- and the eviction branch must not read ``w[0]`` on an empty deque.
    Scaling off silently discards via the plain deque; scaling on must not turn
    that into an IndexError (CodeRabbit finding, PR #307).
    """
    w: deque = deque(maxlen=0)
    counts: Counter = Counter()
    for item in ("a", "a", "b"):
        assert append_counted(w, counts, item) == 0
    assert list(w) == []
    assert counts == Counter()


@pytest.mark.parametrize(
    "base_zero", ["loop_window_size", "cascade_window_size"], ids=["loop", "cascade"]
)
def test_zero_window_size_does_not_raise_under_scaling(base_zero: str) -> None:
    """End-to-end guard: a zero base window with scaling on must stay fail-safe
    (no risk ever fires) rather than raising out of observe(). The detector whose
    window is zeroed is the one that would raise."""
    cfg = Config(window_scale_steps=5, max_window=8, **{base_zero: 0})  # type: ignore[arg-type]
    det = (
        LoopDetector(config=cfg)
        if base_zero == "loop_window_size"
        else ErrorCascadeDetector(config=cfg)
    )
    for i in range(20):
        e = _event(f"s{i}", float(i), "loopA" if i % 2 else f"u{i}", error=(i % 3 == 0))
        assert det.observe(e) is None


def test_maintain_counter_rebuilds_when_maxlen_moves() -> None:
    counters: dict = {}
    sizes: dict = {}
    w = deque("aab", maxlen=3)
    counts = maintain_counter(counters, sizes, "ep", w)
    assert counts == Counter({"a": 2, "b": 1})
    # Cached and reused while the maxlen is unchanged.
    assert maintain_counter(counters, sizes, "ep", w) is counts
    # next_window grows by swapping in a larger deque that keeps the recent
    # items; a cached counter would then describe the *old* contents. The
    # maxlen change must trigger a rebuild.
    grown = deque(w, maxlen=6)
    grown.extend(["a", "a", "a"])
    rebuilt = maintain_counter(counters, sizes, "ep", grown)
    assert rebuilt == Counter({"a": 5, "b": 1})
    assert sizes["ep"] == 6


def test_scaled_loop_running_count_equals_the_window() -> None:
    """Every step, through evictions and lazy resizes: counter == Counter(w)."""
    det = LoopDetector(config=Config(window_scale_steps=5, max_window=8))
    # Enough steps to grow the window to the cap and churn it thoroughly.
    sigs = [f"u{i}" for i in range(30)]
    sigs += ["loopA", "loopB"] * 40  # a real repeat fills the window with 2 sigs
    for i, sig in enumerate(sigs):
        det.observe(_event(f"s{i}", float(i), sig))
        w = det._windows["ep1"]
        assert det._counts_map["ep1"] == Counter(w), f"drift at step {i}: {list(w)}"


def test_scaled_near_duplicate_running_count_equals_the_window() -> None:
    cfg = Config(loop_near_duplicate_enabled=True, window_scale_steps=5, max_window=8)
    det = LoopDetector(config=cfg)
    for i in range(60):
        det.observe(_event(f"s{i}", float(i), "loopA" if i % 2 else f"u{i}"))
        w = det._near_windows["ep1"]
        assert det._near_counts_map["ep1"] == Counter(w), f"drift at step {i}"


def test_scaled_cascade_running_total_equals_sum_of_window() -> None:
    det = ErrorCascadeDetector(config=Config(window_scale_steps=5, max_window=8))
    for i in range(60):
        det.observe(_event(f"s{i}", float(i), "sig", error=(i % 3 == 0)))
        w = det._windows["ep1"]
        # The O(1) lookup must equal what ``sum(w)`` would have returned.
        assert det._flags["ep1"][True] == sum(w), f"drift at step {i}: {list(w)}"


def test_scaled_loop_counters_rebuilt_after_snapshot_restore() -> None:
    # Near-duplicate mode on so the restore covers both counter families.
    cfg = Config(
        window_scale_steps=5,
        max_window=8,
        loop_near_duplicate_enabled=True,
    )
    det = LoopDetector(config=cfg)
    for i in range(20):
        det.observe(_event(f"s{i}", float(i), "sig-a"))
    dumped = det.dump_state()
    restored = LoopDetector(config=cfg)
    restored.load_state(dumped)
    # load_state drops the derived counters: a restored window carries its own
    # maxlen, so a cached size would be trusted against the wrong scaling
    # position. The rebuild lands on the first observe and the invariant holds.
    for i in range(20, 44):
        restored.observe(_event(f"s{i}", float(i), "sig-a" if i % 2 else "sig-b"))
        assert restored._counts_map["ep1"] == Counter(restored._windows["ep1"])
        assert restored._near_counts_map["ep1"] == Counter(
            restored._near_windows["ep1"]
        )


def test_scaled_cascade_counters_rebuilt_after_snapshot_restore() -> None:
    cfg = Config(window_scale_steps=5, max_window=8)
    det = ErrorCascadeDetector(config=cfg)
    for i in range(20):
        det.observe(_event(f"s{i}", float(i), "sig", error=(i % 4 == 0)))
    restored = ErrorCascadeDetector(config=cfg)
    restored.load_state(det.dump_state())
    for i in range(20, 44):
        restored.observe(_event(f"s{i}", float(i), "sig", error=(i % 3 == 0)))
        assert restored._flags["ep1"][True] == sum(restored._windows["ep1"])


def test_scaling_off_keeps_the_default_path_counter_free() -> None:
    """The bookkeeping is gated on scaling: with defaults it never runs, so
    the published default-path numbers cannot move (issue #298's contract)."""
    loop = LoopDetector(config=Config())
    cascade = ErrorCascadeDetector(config=Config())
    for i in range(30):
        e = _event(f"s{i}", float(i), "loopA" if i % 2 else f"u{i}", error=(i % 4 == 0))
        loop.observe(e)
        cascade.observe(e)
    assert loop._counts_map == {} and loop._window_sizes == {}
    assert loop._near_counts_map == {} and loop._near_sizes == {}
    assert cascade._flags == {} and cascade._sizes == {}


def _naive_minimal_period(w: deque) -> int | None:
    """The pre-#298 reference: full O(max_period x window) scan, no short-circuit."""
    n = len(w)
    for p in range(1, 256):
        if n < 2 * p:
            return None
        if all(w[i] == w[i + p] for i in range(n - p)):
            return p
    return None


@pytest.mark.parametrize(
    "window",
    [
        # exact periods, several lengths each
        deque((["a"] * 8), maxlen=8),
        deque((["a", "b"] * 6), maxlen=12),
        deque((["a", "b", "c"] * 5), maxlen=15),
        deque((["a", "b", "c", "d"] * 3), maxlen=12),
        deque((["x", "y", "z", "w", "v"] * 3), maxlen=15),
        # periodic prefix then a break: NOT p-periodic for any p
        deque(list("abababababc"), maxlen=11),
        deque(list("abcabcabz"), maxlen=9),
        # uniform: minimal period 1
        deque(list("qqqq"), maxlen=4),
        # too short to verify any period
        deque(list("ab"), maxlen=8),
    ],
)
def test_cycle_short_circuit_agrees_with_the_full_scan(window: deque) -> None:
    det = LoopDetector(config=Config(loop_cycle_min_period=1, loop_cycle_max_period=32))
    assert det._minimal_period(window) == _naive_minimal_period(window)


def test_cycle_short_circuit_agrees_with_the_full_scan_on_random_windows() -> None:
    """The ``w[-1] == w[-1-p]`` guard is a *necessary* condition for p-periodicity,
    so it can only skip candidates the full scan would also reject. Verified by
    brute force over many random windows rather than by trusting the algebra."""
    det = LoopDetector(config=Config(loop_cycle_min_period=1, loop_cycle_max_period=32))
    rng = random.Random(298)
    alphabet = "abcd"
    for _ in range(500):
        n = rng.randint(2, 24)
        w = deque(
            (rng.choice(alphabet) for _ in range(n)), maxlen=rng.choice([4, 8, 16, 32])
        )
        assert det._minimal_period(w) == _naive_minimal_period(w), list(w)
