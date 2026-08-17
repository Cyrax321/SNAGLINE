"""End-to-end example: baseline a healthy run, then monitor live traffic.

This walks the full next-phase pipeline without any optional dependency:

1. Write a small "known-good" trajectory (one JSON StepEvent per line).
2. Fit a healthy ``BaselineProfile`` from it via ``snagline baseline`` (or the
   ``fit_baseline_from_jsonl`` API).
3. Start a ``Monitor`` with ``goal_drift`` and ``ml_ensemble`` enabled, pointing
   at that baseline.
4. Feed a healthy episode (should stay silent) and a drifting episode (rising
   error rate) and print the risks the ensemble emits.

Run it directly::

    python examples/baseline_to_monitor.py
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from snagline import (
    Config,
    Monitor,
    fit_baseline_from_jsonl,
    load_baseline,
    save_baseline,
)
from snagline.adapters.raw import watch
from snagline.risk import FailureRisk
from snagline.sinks.base import AlertSink


class Collector(AlertSink):
    """A sink that just remembers every risk it receives."""

    def __init__(self) -> None:
        self.risks: list[FailureRisk] = []

    def emit(self, risk: FailureRisk) -> None:
        self.risks.append(risk)


def _write_healthy_trajectory(path: Path) -> None:
    # A known-good run: tool "search" runs fast and never errors.
    with path.open("a", encoding="utf-8") as fh:
        for i in range(40):
            evt = {
                "step_id": str(i),
                "episode_id": "baseline",
                "timestamp": float(i),
                "action_type": "tool_call",
                "action_signature": f"search:{i % 5}",
                "tool_name": "search",
                "latency_ms": 100.0 + (i % 3),
                "error": False,
            }
            fh.write(json.dumps(evt) + "\n")


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        traj = Path(tmp) / "healthy.jsonl"
        _write_healthy_trajectory(traj)

        # Step 1+2: fit and persist a healthy baseline.
        baseline = fit_baseline_from_jsonl(str(traj))
        baseline_path = Path(tmp) / "baseline.json"
        save_baseline(baseline, str(baseline_path))
        loaded = load_baseline(str(baseline_path))
        print(f"Fitted baseline for tools: {sorted(loaded.tools)}")

        # Step 3: monitor with goal-drift + ml ensemble enabled.
        config = Config(
            goal_drift_enabled=True,
            goal_drift_baseline=loaded,
            goal_drift_min_samples=5,
            ml_ensemble_enabled=True,
        )
        collector = Collector()
        monitor = Monitor.default(config=config, sinks=[collector])

        # Step 4a: a healthy live episode -> should stay silent.
        with watch(monitor, "live-healthy") as step:
            for i in range(10):
                step(
                    "tool_call",
                    tool_name="search",
                    args=f"query={i}",
                    latency_ms=100.0 + (i % 3),
                )
        print(f"Healthy episode risks: {len(collector.risks)} (expect 0)")

        # Step 4b: a drifting episode -> rising error rate trips goal_drift
        # (and the error-cascade detector), which the ml ensemble elevates.
        collector.risks.clear()
        with watch(monitor, "live-drift") as step:
            for i in range(10):
                step(
                    "tool_call",
                    tool_name="search",
                    args=f"query={i}",
                    latency_ms=100.0,
                    error=True,
                )
        print(f"Drifting episode risks: {len(collector.risks)} (expect >= 1)")
        for risk in collector.risks:
            print(f"  - {risk.trigger} score={risk.score:.2f}: {risk.detail}")


if __name__ == "__main__":
    main()
