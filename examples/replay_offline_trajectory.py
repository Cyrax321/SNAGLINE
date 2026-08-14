"""Runnable demo: offline analysis of a trajectory file via ``snagline replay``.

Builds a small synthetic trajectory (a loop + an error cascade), writes it to a
temp file, and replays it through the detectors. Mirrors:

    snagline replay trajectory.jsonl --summary

Run:
    PYTHONPATH=src python3 examples/replay_offline_trajectory.py
"""

from __future__ import annotations

import dataclasses
import json
import os
import tempfile

from snagline.cli import replay
from snagline.events import EpisodeMeta, StepEvent, make_signature


def _build_trajectory() -> list[StepEvent]:
    ep = "demo-replay"
    events: list[StepEvent] = []

    def add(step_id, tool, args, *, error=False, latency_ms=80.0):
        events.append(
            StepEvent(
                step_id=str(step_id),
                episode_id=ep,
                timestamp=1.0 + step_id,
                action_type="tool_call",
                action_signature=make_signature("tool_call", tool, str(args)),
                tool_name=tool,
                latency_ms=latency_ms,
                error=error,
            )
        )

    # healthy baseline -- long enough to satisfy latency warm-up
    for i in range(25):
        add(i, "search", f"q-{i}")
    # injected loop: 4 identical retries
    for j in range(4):
        add(25 + j, "retry", "same")
    # injected error cascade: 3 consecutive errors
    for j in range(3):
        add(29 + j, "act", f"e-{j}", error=True)
    # injected latency spike: 6 sustained slow calls, varied args
    for j in range(6):
        add(32 + j, "search", f"heavy-{j}", latency_ms=400.0)
    return events


def main() -> None:
    events = _build_trajectory()
    path = os.path.join(tempfile.gettempdir(), "snagline_demo_trajectory.jsonl")
    with open(path, "w") as f:
        for e in events:
            f.write(json.dumps(dataclasses.asdict(e)) + "\n")

    print(f"[demo] replaying {len(events)} steps from {path}", file=__import__("sys").stderr)
    n = replay(path)
    print(f"[demo] replayed {n} steps; see stderr above for any FailureRisk lines.", file=__import__("sys").stderr)
    os.remove(path)


if __name__ == "__main__":
    main()
