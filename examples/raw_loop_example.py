"""Minimal end-to-end example: SNAGLINE watching a plain Python agent loop.

Run directly:
    PYTHONPATH=src python3 examples/raw_loop_example.py

Any detected loop / error-cascade is printed as a JSON line to stderr (the
default console sink). No framework, no config, no training data required.
"""

from __future__ import annotations

import time

from snagline import Monitor
from snagline.adapters.raw import watch


def agent_loop() -> None:
    monitor = Monitor.default()
    with watch(monitor, "demo-episode", agent_name="demo-agent") as step:
        for i in range(8):
            ok = i != 5  # simulate one failed tool call
            step(
                "tool_call",
                tool_name="search",
                args=f"query-{i}",
                latency_ms=80 + i * 5,
                error=not ok,
                error_type=(None if ok else "timeout"),
            )
            time.sleep(0.01)


if __name__ == "__main__":
    agent_loop()
