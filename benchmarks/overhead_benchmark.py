"""Overhead benchmark -- the credibility artifact for "cheap enough to run on
every step" (project.md §10 / §13 step 4).

Measures the real, reproducible per-step cost of monitoring an agent in
microseconds, amortized over a large number of synthetic steps. Run directly::

    python benchmarks/overhead_benchmark.py

or via the CLI::

    snagline bench

The number is reported (median and p99) so it can be published in the README
rather than asserted. This script is stdlib-only; it imports ``snagline``.

The headline number is the **full per-step path an integrator pays**: building
the ``StepEvent`` and then ingesting it. An earlier version timed ``ingest()``
alone against events built up front, which silently excluded ``StepEvent``
construction -- a real ~1.9 us/step on a frozen dataclass, because its
generated ``__init__`` routes every field through ``object.__setattr__`` to
enforce immutability (issue #312). Construction is not optional: every adapter
builds one event per step, so a number that omits it understates the cost of
running SNAGLINE on every step by the entire construction budget. The
ingest-only split is still reported (``ingest_only``) so detector overhead
stays isolable from event construction.

Two legs are reported. The **default** leg is the full per-step path an
integrator pays -- constructing the ``StepEvent`` and ingesting it -- and is the
number the README quotes; its ingest-only split sits under ``ingest_only``. The
**scaled** legs turn on window auto-scaling (issue #92) with a large
``max_window``, because that is the path scaling was built for -- and, before
issue #298, the path where the loop and error-cascade detectors silently
degraded from O(1) to O(window) per step by rescanning the window instead of
keeping a running count. A scaled leg that climbs with ``max_window`` while the
default leg stays flat is exactly that regression class, and it is invisible to
the default leg alone. The scaled legs are ingest-only: window scaling changes
detector cost, not construction cost, so charging construction there would add a
constant to every leg and obscure exactly the gap the leg exists to measure.
"""

from __future__ import annotations

import dataclasses
import statistics
import time

from snagline import Monitor
from snagline.config import Config
from snagline.events import StepEvent, make_signature

# Scaled-leg knobs. The effective window is ``base * ceil(n / scale_steps)``
# capped at ``max_window``, and each detector has its own base (loop 12,
# cascade 10), so the slower-growing one sets the floor. scale_steps is small
# enough that every leg saturates at its cap well inside the run: with 250 and
# n=200_000, each max_window is reached by ~60k steps, so the remaining blocks
# measure *sustained* O(window) cost rather than the growth phase -- which is
# the only thing this leg exists to catch.
_SCALED_STEPS = 250
_SCALED_MAX_WINDOWS = (512, 2048)


def _make_events(n: int) -> list[StepEvent]:
    """Synthetic steps with a *unique* signature per step (so the loop detector
    does real work but never false-alarms) and a *stable* latency (a genuinely
    healthy run: the CUSUM's std==0 guard means no false alarm). This isolates
    the ingest() overhead rather than detector tuning."""
    events: list[StepEvent] = []
    for i in range(n):
        events.append(_build_event(i))
    return events


def _build_event(i: int) -> StepEvent:
    """One synthetic step, built the way an adapter builds one on the hot path.

    Factored out of :func:`_make_events` so the full-step leg (below) can build
    its events *inside* the timed region from the identical generator -- the
    two legs must see the same stream, or the construction cost they differ by
    would be confounded with a different workload.
    """
    return StepEvent(
        step_id=str(i),
        episode_id="bench",
        timestamp=time.time(),
        action_type="tool_call",
        action_signature=make_signature("tool_call", "tool", str(i)),
        tool_name="tool",
        latency_ms=100.0,
    )


def _percentiles(per_step_us: list[float]) -> tuple[float, float]:
    ordered = sorted(per_step_us)
    p99_idx = min(len(ordered) - 1, int(0.99 * len(ordered)))
    return statistics.median(per_step_us), ordered[p99_idx]


def _time_ingest(
    monitor: Monitor, events: list[StepEvent], n: int, block: int
) -> tuple[float, float]:
    """Median and p99 microseconds per ingest() over the measured blocks.

    Pre-built events are passed in, so this leg charges *only* ingest() and
    excludes StepEvent construction. It is the split, not the headline -- see
    :func:`_time_full_step`.

    The first ``block`` steps are a warm-up, swallowed before timing, so lazy
    one-time costs do not land in the numbers.
    """
    for e in events[:block]:
        monitor.ingest(e)
    per_step_us: list[float] = []
    for start in range(block, n, block):
        chunk = events[start : start + block]
        t0 = time.perf_counter()
        for e in chunk:
            monitor.ingest(e)
        t1 = time.perf_counter()
        per_step_us.append((t1 - t0) / len(chunk) * 1e6)
    return _percentiles(per_step_us)


def _time_full_step(monitor: Monitor, n: int, block: int) -> tuple[float, float]:
    """Median and p99 microseconds for the whole per-step path: build the
    StepEvent, then ingest it.

    Construction happens *inside* the timed region. This is what an integrator
    actually pays per step -- every adapter builds one event before calling
    ingest() -- and it is the number the README should quote. Timing ingest()
    alone excluded construction entirely, which on a frozen dataclass costs
    more than ingest() itself (issue #312).

    Warm-up is swallowed first, as in :func:`_time_ingest`. ``_build_event`` is
    the same generator :func:`_make_events` uses, so the only difference
    between the two legs is whether construction is charged.
    """
    for i in range(block):
        monitor.ingest(_build_event(i))
    per_step_us: list[float] = []
    for start in range(block, n, block):
        end = min(start + block, n)
        t0 = time.perf_counter()
        for i in range(start, end):
            monitor.ingest(_build_event(i))
        t1 = time.perf_counter()
        per_step_us.append((t1 - t0) / (end - start) * 1e6)
    return _percentiles(per_step_us)


def run_benchmark(
    n: int = 200_000,
    block: int = 2_000,
    *,
    scale_steps: int = _SCALED_STEPS,
    max_windows: tuple[int, ...] = _SCALED_MAX_WINDOWS,
) -> dict:
    """Time the per-step path in blocks, returning median/p99 microseconds.

    The headline ``median_us`` / ``p99_us`` keys the CLI and README quote cover
    the full path an integrator pays -- constructing the StepEvent and then
    ingesting it (issue #312). The ingest-only split lands under
    ``ingest_only`` so detector overhead stays separable from event
    construction, and the scaled legs land under ``scaled`` (issue #298).
    """
    events = _make_events(n)
    blocks = len(range(block, n, block))

    # One resolved config for every leg. ``Monitor.default()`` resolves the
    # SNAGLINE_* env layering while a bare ``Config()`` pins every knob to its
    # dataclass default, so mixing the two compares detector setups that differ
    # in more than scaling (an operator with SNAGLINE_LOOP_WINDOW_SIZE set would
    # measure a different base window per leg). Deriving each leg from one base
    # and overriding only the two scaling knobs isolates the rescanning cost.
    base_cfg = Config.resolve()

    # Headline: the full per-step path. A fresh monitor, because the warm-up
    # below has already consumed the first ``block`` events' worth of
    # detector state and the split leg must start from the same place.
    full_median_us, full_p99_us = _time_full_step(Monitor.default(base_cfg), n, block)
    # Split: ingest() alone on the same stream, for separating detector cost
    # from construction cost.
    ingest_median_us, ingest_p99_us = _time_ingest(
        Monitor.default(base_cfg), events, n, block
    )
    stats: dict = {
        "n": n,
        "blocks": blocks,
        "median_us": full_median_us,
        "p99_us": full_p99_us,
        "ingest_only": {
            "median_us": ingest_median_us,
            "p99_us": ingest_p99_us,
        },
        "scaled": [],
    }

    # Same event stream, scaling on: any gap from the default leg is the
    # rescanning cost, not detector tuning or a different workload.
    for cap in max_windows:
        monitor = Monitor.default(
            dataclasses.replace(
                base_cfg, window_scale_steps=scale_steps, max_window=cap
            )
        )
        median_us, p99_us = _time_ingest(monitor, events, n, block)
        stats["scaled"].append(
            {
                "max_window": cap,
                "median_us": median_us,
                "p99_us": p99_us,
            }
        )
    return stats


def main() -> None:
    stats = run_benchmark()
    print("snagline overhead benchmark")
    print(f"  steps measured : {stats['n']}")
    print(f"  median        : {stats['median_us']:.2f} us/step")
    print(f"  p99           : {stats['p99_us']:.2f} us/step")
    split = stats["ingest_only"]
    print(
        f"  ingest only   : median {split['median_us']:.2f} us/step, "
        f"p99 {split['p99_us']:.2f} us/step"
    )
    for leg in stats["scaled"]:
        print(
            f"  scaled max_window={leg['max_window']:<5d}: "
            f"median {leg['median_us']:.2f} us/step, "
            f"p99 {leg['p99_us']:.2f} us/step"
        )


if __name__ == "__main__":
    main()
