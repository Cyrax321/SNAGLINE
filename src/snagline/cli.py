"""Command-line interface for SNAGLINE (project.md §9).

This build phase (v0.1) implements ``snagline replay`` and ``snagline bench``.
``watch`` and ``baseline`` are registered but intentionally not yet implemented
-- they land in later, explicitly-ordered build steps and will error clearly
rather than silently do nothing.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

from snagline.events import StepEvent
from snagline.monitor import Monitor
from snagline.risk import FailureRisk


class _CountingSink:
    """Internal sink used by the CLI to report how many risks fired."""

    def __init__(self) -> None:
        self.count = 0
        self.risks: List[FailureRisk] = []

    def emit(self, risk: FailureRisk) -> None:
        self.count += 1
        self.risks.append(risk)


def replay(path: str, monitor: Optional[Monitor] = None) -> int:
    """Replay a JSONL trajectory file through ``monitor`` (live offline analysis).

    Each line must be a JSON object with the ``StepEvent`` fields. Returns the
    number of steps replayed. With the default monitor this also prints any
    ``FailureRisk`` to stderr via the console sink.
    """
    if monitor is None:
        monitor = Monitor.default()

    steps = 0
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            event = StepEvent(**obj)
            monitor.ingest(event)
            steps += 1
    return steps


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="snagline",
        description="Real-time failure detection for AI agents (v0.1).",
    )
    sub = parser.add_subparsers(dest="command")

    p_replay = sub.add_parser(
        "replay", help="Replay a JSONL trajectory through the detectors (offline)."
    )
    p_replay.add_argument("trajectory", help="Path to a .jsonl trajectory file.")
    p_replay.add_argument(
        "--quiet", action="store_true", help="Suppress the per-risk console output."
    )
    p_replay.add_argument(
        "--summary", action="store_true", help="Print a trailing summary line."
    )

    p_bench = sub.add_parser(
        "bench", help="Run the ingest() overhead benchmark and print us/step."
    )

    # Registered but not yet implemented in this build phase.
    for name, help_text in [
        ("watch", "Live monitoring mode (dev/debug). [not in v0.1]"),
        ("baseline", "Fit a healthy-run baseline. [not in v0.1]"),
    ]:
        sp = sub.add_parser(name, help=help_text)
        sp.add_argument("__rest", nargs="*", help=argparse.SUPPRESS)

    return parser


def _cmd_bench() -> int:
    try:
        from benchmarks.overhead_benchmark import run_benchmark
    except ImportError:
        sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
        from benchmarks.overhead_benchmark import run_benchmark

    stats = run_benchmark()
    print("snagline overhead benchmark")
    print(f"  steps measured : {stats['n']}")
    print(f"  median        : {stats['median_us']:.2f} us/step")
    print(f"  p99           : {stats['p99_us']:.2f} us/step")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "replay":
        sinks = []
        counter = _CountingSink()
        if not args.quiet:
            from snagline.sinks.console import ConsoleSink

            sinks.append(ConsoleSink())
        sinks.append(counter)
        monitor = Monitor.default(sinks=sinks)
        steps = replay(args.trajectory, monitor=monitor)
        if args.summary:
            print(
                f"replayed {steps} steps; {counter.count} risk(s) emitted",
                file=sys.stderr,
            )
        return 0

    if args.command == "bench":
        return _cmd_bench()

    if args.command in {"watch", "baseline"}:
        print(
            f"snagline {args.command} is not implemented in this build phase (v0.1). "
            "It will arrive in a later, explicitly-ordered step.",
            file=sys.stderr,
        )
        return 2

    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
