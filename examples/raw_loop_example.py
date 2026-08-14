"""Runnable demo: SNAGLINE watching a plain Python agent loop (the ``raw`` adapter).

Run:
    PYTHONPATH=src python3 examples/raw_loop_example.py           # faulty run (shows detections)
    PYTHONPATH=src python3 examples/raw_loop_example.py --healthy # clean run (no detections)

Detections print as JSON lines to stderr via the default console sink.
"""

from __future__ import annotations

import argparse
import sys
import time

from snagline import Monitor
from snagline.adapters.raw import watch


def faulty_agent(step) -> None:
    # 1) healthy baseline (stable latency, unique calls) -- long enough to
    #    satisfy the latency detector's warm-up before the spike below.
    for i in range(25):
        step("tool_call", tool_name="search", args=f"q-{i}", latency_ms=80.0)
    # 2) injected retry loop (identical retries)
    for _ in range(4):
        step("tool_call", tool_name="retry", args="same", latency_ms=80.0)
    # 3) injected error cascade (3 consecutive errors, varied args)
    for j in range(3):
        step(
            "tool_call",
            tool_name="act",
            args=f"err-{j}",
            latency_ms=80.0,
            error=True,
            error_type="TimeoutError",
        )
    # 4) injected latency spike (sustained shift, varied args)
    for j in range(6):
        step("tool_call", tool_name="search", args=f"heavy-{j}", latency_ms=400.0)


def healthy_agent(step) -> None:
    for i in range(30):
        step("tool_call", tool_name="search", args=f"q-{i}", latency_ms=80.0 + (i % 5))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--healthy", action="store_true", help="Run a clean agent (no detections).")
    args = parser.parse_args()

    monitor = Monitor.default()
    with watch(monitor, "demo-episode", agent_name="demo-agent") as step:
        (healthy_agent if args.healthy else faulty_agent)(step)

    mode = "healthy (expect no detections)" if args.healthy else "faulty (expect loop + cascade + latency risks)"
    print(f"\n[demo] finished {mode}; see stderr above for any FailureRisk lines.", file=sys.stderr)


if __name__ == "__main__":
    main()
