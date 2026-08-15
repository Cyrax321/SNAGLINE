"""Command-line interface for SNAGLINE (project.md §9).

Implements ``snagline replay``, ``snagline bench``, ``snagline watch`` (live
stdin or file-follow mode), ``snagline serve`` (the stdlib sidecar HTTP
server), and ``snagline hook`` (the universal command-hook bridge for
external agent processes: Claude Code, OpenClaw, Hermes, anything that can
run a shell command). ``baseline`` is registered but intentionally not
implemented - it belongs to the ``ml`` extra, a later, explicitly-ordered
build step - and errors clearly rather than silently doing nothing.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Iterator, List, Optional

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
    skipped = 0
    with open(path, "r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                event = StepEvent(**obj)
            except (json.JSONDecodeError, TypeError, ValueError) as exc:
                # Fail-soft on the input side too: a malformed trajectory line
                # must never crash the whole replay (issue #5).
                print(
                    f"snagline replay: skipping malformed line {lineno}: {exc}",
                    file=sys.stderr,
                )
                skipped += 1
                continue
            monitor.ingest(event)
            steps += 1
    if skipped:
        print(
            f"snagline replay: skipped {skipped} malformed line(s)", file=sys.stderr
        )
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
        "--file",
        default=None,
        help="Read StepEvent JSON lines from this file instead of stdin.",
    )
    p_watch.add_argument(
        "--follow",
        action="store_true",
        help="With --file: keep reading as the file grows (tail -f).",
    )
    p_watch.add_argument(
        "--episode-id",
        default=None,
        help="Episode id override [default: from the events / file name].",
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

    p_hook = sub.add_parser(
        "hook",
        help="Universal hook bridge: map a native hook payload from stdin and forward it.",
    )
    p_hook.add_argument(
        "--url",
        default=None,
        help="Forward to a snagline sidecar (POST /events with a canonical StepEvent).",
    )
    p_hook.add_argument(
        "--out",
        default=None,
        help="Append the mapped StepEvent as a JSON line to this file (for snagline watch).",
    )
    p_hook.add_argument(
        "--timeout", type=float, default=1.0, help="HTTP timeout when --url is used [s]."
    )

    # Registered but not yet implemented: belongs to the ml extra (later phase).
    sp = sub.add_parser("baseline", help="Fit a healthy-run baseline. [ml extra, not built yet]")
    sp.add_argument("__rest", nargs="*", help=argparse.SUPPRESS)

    return parser


def _iter_lines(path: Optional[str], follow: bool) -> Iterator[str]:
    """Yield lines from stdin, or from a file; with ``follow``, tail like -f."""
    if path is None:
        yield from sys.stdin
        return
    fh = open(path, "r", encoding="utf-8")
    try:
        while True:
            line = fh.readline()
            if line:
                yield line
            elif follow:
                time.sleep(0.2)
            else:
                return
    finally:
        fh.close()


def _cmd_watch(args: argparse.Namespace) -> int:
    sinks: list = []
    if args.sink == "webhook":
        if not args.webhook_url:
            print("--webhook-url is required with --sink webhook", file=sys.stderr)
            return 2
        from snagline.sinks.webhook import WebhookSink

        sinks.append(WebhookSink(args.webhook_url))
    monitor = Monitor.default(sinks=sinks or None)
    episode = args.episode_id or (args.file or "stdin")
    steps = 0
    try:
        for line in _iter_lines(args.file, args.follow):
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
    except KeyboardInterrupt:
        pass
    finally:
        monitor.end_episode(episode)
    print(f"snagline watch: ingested {steps} step(s)", file=sys.stderr)
    return 0


def _cmd_hook(args: argparse.Namespace) -> int:
    """Universal command-hook bridge (fail-open: ALWAYS exits 0).

    Reads one hook payload from stdin. Claude Code payloads (detected by
    ``hook_event_name``) are mapped to a canonical StepEvent; a payload that
    is already a StepEvent is passed through as-is. The event is then POSTed
    to ``--url`` and/or appended to ``--out``; with neither, it is ingested
    into a local default Monitor. Malformed input and network failures are
    logged to stderr and swallowed - a hook must never break the host agent.
    """
    try:
        raw = sys.stdin.read()
        payload = json.loads(raw)
        if not isinstance(payload, dict):
            raise ValueError("payload must be a JSON object")
        from snagline.adapters.claude_code import (
            HookTracker,
            is_claude_code_payload,
            payload_to_event,
        )

        if is_claude_code_payload(payload):
            event = payload_to_event(payload, tracker=None)
        else:
            event = StepEvent(**payload)  # already canonical
    except Exception as exc:
        print(f"snagline hook: ignoring malformed payload: {exc}", file=sys.stderr)
        return 0  # fail-open: never fail the host's hook invocation

    if event is None:  # unmapped lifecycle event (e.g. SessionStart)
        return 0

    if args.out:
        try:
            with open(args.out, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(_event_to_json(event)) + "\n")
        except OSError as exc:
            print(f"snagline hook: cannot append to {args.out}: {exc}", file=sys.stderr)

    if args.url:
        try:
            import urllib.request

            req = urllib.request.Request(
                args.url,
                data=json.dumps(_event_to_json(event)).encode(),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=args.timeout) as resp:
                resp.read()
        except Exception as exc:
            print(f"snagline hook: forward to {args.url} failed: {exc}", file=sys.stderr)

    if not args.url and not args.out:
        try:
            monitor = Monitor.default()
            monitor.ingest(event)
        except Exception:
            pass  # fail-open by construction; Monitor.default() is also fail-open
    return 0


def _event_to_json(event: StepEvent) -> dict:
    return {
        "step_id": event.step_id,
        "episode_id": event.episode_id,
        "timestamp": event.timestamp,
        "action_type": event.action_type,
        "action_signature": event.action_signature,
        "tool_name": event.tool_name,
        "latency_ms": event.latency_ms,
        "error": event.error,
        "error_type": event.error_type,
        "tokens_in": event.tokens_in,
        "tokens_out": event.tokens_out,
    }


def _cmd_serve(args: argparse.Namespace) -> int:
    from snagline.server.http_server import serve

    print(f"snagline serve: listening on http://{args.host}:{args.port} (POST /events, GET /health)", file=sys.stderr)
    try:
        serve(Monitor.default(), host=args.host, port=args.port)
    except KeyboardInterrupt:
        pass
    return 0


def _inline_benchmark(n: int = 200_000, block: int = 2_000) -> dict:
    """Fallback benchmark used when the ``benchmarks`` extra is not importable
    (e.g. running from an installed wheel that does not ship it). Mirrors the
    shape of ``benchmarks.overhead_benchmark.run_benchmark`` so the CLI output
    is identical (issue #6)."""
    import statistics

    from snagline import Monitor
    from snagline.events import StepEvent, make_signature

    monitor = Monitor.default()
    events = [
        StepEvent(
            step_id=str(i),
            episode_id="bench",
            timestamp=__import__("time").time(),
            action_type="tool_call",
            action_signature=make_signature("tool_call", "tool", str(i)),
            tool_name="tool",
            latency_ms=100.0,
        )
        for i in range(n)
    ]
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


def _cmd_bench() -> int:
    try:
        from benchmarks.overhead_benchmark import run_benchmark
    except ImportError:
        # Running from an installed package: the benchmarks module is not
        # shipped, so fall back to an inline measurement rather than crashing.
        run_benchmark = _inline_benchmark

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

    if args.command == "hook":
        return _cmd_hook(args)

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
