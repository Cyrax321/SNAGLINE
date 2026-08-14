"""Overhead benchmark -- the credibility artifact for "cheap enough to run on
every step" (project.md §10 / §13 step 4).

Measures the real, reproducible cost of ``Monitor.ingest()`` in microseconds
per call, amortized over a large number of synthetic steps. Run directly::

    python benchmarks/overhead_benchmark.py

or via the CLI::

    snagline bench

The number is reported (median and p99) so it can be published in the README
rather than asserted. This script is stdlib-only; it imports ``snagline``.
"""

from __future__ import annotations

import statistics
import time

from snagline import Monitor
from snagline.events import StepEvent, make_signature


def _make_events(n: int) -> list[StepEvent]:
    """Synthetic steps with a *unique* signature per step (so the loop detector
    does real work but never false-alarms) and a *stable* latency (a genuinely
    healthy run: the CUSUM's std==0 guard means no false alarm). This isolates
    the ingest() overhead rather than detector tuning."""
    events: list[StepEvent] = []
    for i in range(n):
        events.append(
            StepEvent(
                step_id=str(i),
                episode_id="bench",
                timestamp=time.time(),
                action_type="tool_call",
                action_signature=make_signature("tool_call", "tool", str(i)),
                tool_name="tool",
                latency_ms=100.0,
            )
        )
    return events


def run_benchmark(n: int = 200_000, block: int = 2_000) -> dict:
    """Time ``ingest()`` in blocks, returning median/p99 microseconds per step."""
    monitor = Monitor.default()
    events = _make_events(n)

    # Warm-up: let any lazy one-time costs settle before measuring.
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

    ordered = sorted(per_step_us)
    p99_idx = min(len(ordered) - 1, int(0.99 * len(ordered)))
    return {
        "n": n,
        "blocks": len(per_step_us),
        "median_us": statistics.median(per_step_us),
        "p99_us": ordered[p99_idx],
    }


def main() -> None:
    stats = run_benchmark()
    print("snagline overhead benchmark")
    print(f"  steps measured : {stats['n']}")
    print(f"  median        : {stats['median_us']:.2f} us/step")
    print(f"  p99           : {stats['p99_us']:.2f} us/step")


if __name__ == "__main__":
    main()
