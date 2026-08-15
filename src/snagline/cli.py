"""Command-line interface for SNAGLINE (project.md §9).

Implements ``snagline replay``, ``snagline bench``, ``snagline watch`` (live
stdin mode), and ``snagline serve`` (the stdlib sidecar HTTP server).
``baseline`` is registered but intentionally not implemented -- it belongs to
the ``ml`` extra, a later, explicitly-ordered build step -- and errors clearly
rather than silently doing nothing.
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

    p_watch = sub.add_parser(
        "watch",
        help="Live mode: read StepEvent JSON lines from stdin, ingest each as it arrives.",
    )
    p_watch.add_argument(
        "--episode-id",
        default="stdin",
        help="Episode id for the watched stream [default: stdin].",
    )
    p_watch.add_argument(
        "--sink",
        choices=["console", "webhook"],
        default="console",
        help="Escalation sink [default: console].",
    )
    p_watch.add_argument(
        "--webhook-url",
        default=None,
        help="Webhook endpoint (required with --sink webhook).",
    )

    p_serve = sub.add_parser(
        "serve",
        help="Run the sidecar HTTP server (POST /events, GET /health) for non-Python agents.",
    )
    p_serve.add_argument("--host", default="127.0.0.1", help="Bind address [default: 127.0.0.1].")
    p_serve.add_argument("--port", type=int, default=8787, help="Port [default: 8787].")

    # Registered but not yet implemented: belongs to the ml extra (later phase).
    sp = sub.add_parser("baseline", help="Fit a healthy-run baseline. [ml extra, not built yet]")
    sp.add_argument("__rest", nargs="*", help=argparse.SUPPRESS)

    return parser


def _cmd_watch(args: argparse.Namespace) -> int:
    sinks: list = []
    if args.sink == "webhook":
        if not args.webhook_url:
            print("--webhook-url is required with --sink webhook", file=sys.stderr)
            return 2
        from snagline.sinks.webhook import WebhookSink

        sinks.append(WebhookSink(args.webhook_url))
    monitor = Monitor.default(sinks=sinks or None)
    steps = 0
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
            event = StepEvent(**obj)
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            print(f"snagline watch: skipping malformed line: {exc}", file=sys.stderr)
            continue
        monitor.ingest(event)
        steps += 1
    monitor.end_episode(args.episode_id)
    print(f"snagline watch: ingested {steps} step(s)", file=sys.stderr)
    return 0


def _cmd_serve(args: argparse.Namespace) -> int:
    from snagline.server.http_server import serve

    print(f"snagline serve: listening on http://{args.host}:{args.port} (POST /events, GET /health)", file=sys.stderr)
    try:
        serve(Monitor.default(), host=args.host, port=args.port)
    except KeyboardInterrupt:
        pass
    return 0


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

    if args.command == "watch":
        return _cmd_watch(args)

    if args.command == "serve":
        return _cmd_serve(args)

    if args.command == "baseline":
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
