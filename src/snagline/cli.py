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
import dataclasses
import json
import os
import sys
import time
from collections.abc import Iterator
from contextlib import suppress
from pathlib import Path

from snagline.config import Config
from snagline.events import StepEvent
from snagline.monitor import Monitor
from snagline.risk import FailureRisk
from snagline.sinks.base import AlertSink


class _CountingSink:
    """Internal sink used by the CLI to report how many risks fired."""

    def __init__(self) -> None:
        self.count = 0
        self.risks: list[FailureRisk] = []

    def emit(self, risk: FailureRisk) -> None:
        self.count += 1
        self.risks.append(risk)


def replay(path: str, monitor: Monitor | None = None) -> int:
    """Replay a JSONL trajectory file through ``monitor`` (live offline analysis).

    Each line must be a JSON object with the ``StepEvent`` fields. Returns the
    number of steps replayed. With the default monitor this also prints any
    ``FailureRisk`` to stderr via the console sink.
    """
    if monitor is None:
        monitor = Monitor.default()

    steps = 0
    skipped = 0
    # Track every episode touched so we can clear per-episode detector state
    # (loop windows, cascade counters, CUSUM baselines) when replay ends.
    # Without this, reusing the same monitor across replay() calls leaks state
    # from one trajectory into the next (issue #18).
    episodes: set = set()
    with open(path, encoding="utf-8") as fh:
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
            episodes.add(event.episode_id)
            monitor.ingest(event)
            steps += 1
    # Tear down per-episode state so a later replay() on the same monitor starts
    # clean (fail-open: a teardown error must not abort the summary).
    for episode_id in episodes:
        # Fail-open: a teardown error must not abort the summary.
        with suppress(Exception):  # pragma: no cover - defensive
            monitor.end_episode(episode_id)
    if skipped:
        print(f"snagline replay: skipped {skipped} malformed line(s)", file=sys.stderr)
    return steps


def _build_config(args: argparse.Namespace) -> Config:
    """Resolve the effective Config from --config and SNAGLINE_* env vars."""
    path = getattr(args, "config", None)
    return Config.resolve(path=path)


def _maybe_dedup(sinks: list[AlertSink], cooldown_seconds: float) -> list[AlertSink]:
    """Wrap each sink in a cooldown ``DedupSink`` when ``cooldown_seconds`` > 0.

    Suppresses alert storms (issue #4) so a repeating failure pages once per
    cooldown window instead of on every step.
    """
    if not cooldown_seconds or cooldown_seconds <= 0:
        return sinks
    from snagline.sinks.dedup import DedupSink

    return [DedupSink(s, cooldown_seconds=cooldown_seconds) for s in sinks]


def _console_sinks(cfg: Config) -> list[AlertSink]:
    """Console escalation plus LoggingSink when ``log_format == "json"``.

    Issue #99 settled composition as emission "alongside console"; issue #119
    wires the knob end to end so ``SNAGLINE_LOG_FORMAT=json`` makes risks
    machine-readable with zero code changes, everywhere the console sink is
    the default choice. Explicit non-console ``--sink`` selections replace the
    console pair entirely and stay untouched.
    """
    from snagline.sinks.console import ConsoleSink

    pair: list[AlertSink] = [ConsoleSink()]
    if cfg.log_format == "json":
        from snagline.sinks.logging_sink import LoggingSink

        pair.append(LoggingSink())
    return pair


def _build_sinks(args: argparse.Namespace, cfg: Config) -> list[AlertSink]:
    """Resolve the escalation sinks from --sink / --slack-url / --pagerduty-key.

    Unknown/invalid combinations fail closed with a clear stderr message and a
    non-zero exit code (this is configuration error, not a monitoring fault).
    """
    sinks: list[AlertSink] = []
    if args.sink == "webhook":
        if not args.webhook_url:
            print("--webhook-url is required with --sink webhook", file=sys.stderr)
            raise SystemExit(2)
        from snagline.sinks.webhook import WebhookSink

        sinks.append(WebhookSink(args.webhook_url))
    elif args.sink == "slack":
        if not args.slack_url:
            print("--slack-url is required with --sink slack", file=sys.stderr)
            raise SystemExit(2)
        from snagline.sinks.slack import SlackSink

        sinks.append(SlackSink(args.slack_url, min_severity=args.min_severity))
    elif args.sink == "pagerduty":
        if not args.pagerduty_key:
            print("--pagerduty-key is required with --sink pagerduty", file=sys.stderr)
            raise SystemExit(2)
        from snagline.sinks.pagerduty import PagerDutySink

        sinks.append(
            PagerDutySink(
                args.pagerduty_key,
                source=args.pagerduty_source,
                min_severity=args.min_severity,
            )
        )
    else:
        sinks.extend(_console_sinks(cfg))
    return _maybe_dedup(sinks, args.cooldown_seconds)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="snagline",
        description="Real-time failure detection for AI agents (v0.1).",
    )
    sub = parser.add_subparsers(dest="command")

    # 12-factor configuration applies across all subcommands: a config file
    # and/or SNAGLINE_* environment variables tune the detectors (P0).
    parser.add_argument(
        "--config",
        default=None,
        help="Path to a JSON/TOML config file. Environment variables "
        "(SNAGLINE_*) override it. See docs/ATTACH_ANY_SYSTEM.md.",
    )

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

    sub.add_parser(
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
        choices=["console", "webhook", "slack", "pagerduty"],
        default="console",
        help="Escalation sink [default: console].",
    )
    p_watch.add_argument(
        "--webhook-url",
        default=None,
        help="Webhook endpoint (required with --sink webhook).",
    )
    p_watch.add_argument(
        "--slack-url",
        default=None,
        help="Slack incoming webhook URL (required with --sink slack).",
    )
    p_watch.add_argument(
        "--pagerduty-key",
        default=None,
        help="PagerDuty Events API routing key (required with --sink pagerduty).",
    )
    p_watch.add_argument(
        "--pagerduty-source",
        default="snagline",
        help="PagerDuty event source label [default: snagline].",
    )
    p_watch.add_argument(
        "--min-severity",
        default=None,
        help="Only escalate risks at or above this severity "
        "(info|warning|critical). Optional.",
    )
    p_watch.add_argument(
        "--cooldown-seconds",
        type=float,
        default=0.0,
        help="Suppress repeated alerts of the same key for this many seconds "
        "(issue #4). 0 disables (default).",
    )

    p_serve = sub.add_parser(
        "serve",
        help="Run the sidecar HTTP server (POST /events, GET /health) for non-Python agents.",
    )
    p_serve.add_argument(
        "--host", default="127.0.0.1", help="Bind address [default: 127.0.0.1]."
    )
    p_serve.add_argument("--port", type=int, default=8787, help="Port [default: 8787].")
    p_serve.add_argument(
        "--auth-token",
        default=None,
        help="Shared secret required on every request except GET /health "
        "(Authorization: Bearer <token> or X-Snagline-Token). Falls back to "
        "$SNAGLINE_SERVE_AUTH_TOKEN, which keeps the secret out of argv. "
        "Unset means all endpoints are open.",
    )
    p_serve.add_argument(
        "--certfile",
        default=None,
        help="PEM certificate enabling stdlib TLS on the sidecar listener "
        "(issue #120); requires --keyfile unless the file bundles its key. "
        "Omit for plain HTTP.",
    )
    p_serve.add_argument(
        "--keyfile",
        default=None,
        help="PEM private key matching --certfile.",
    )
    p_serve.add_argument(
        "--max-body-bytes",
        type=int,
        default=1_000_000,
        help="Reject POST bodies larger than this with 413 [default: 1000000].",
    )
    p_serve.add_argument(
        "--max-risks",
        type=int,
        default=1000,
        help="How many risks received on POST /risks to retain for GET /risks "
        "[default: 1000].",
    )
    p_serve.add_argument(
        "--sink",
        choices=["console", "webhook", "slack", "pagerduty"],
        default="console",
        help="Escalation sink [default: console].",
    )
    p_serve.add_argument(
        "--webhook-url",
        default=None,
        help="Webhook endpoint (required with --sink webhook).",
    )
    p_serve.add_argument(
        "--slack-url",
        default=None,
        help="Slack incoming webhook URL (required with --sink slack).",
    )
    p_serve.add_argument(
        "--pagerduty-key",
        default=None,
        help="PagerDuty Events API routing key (required with --sink pagerduty).",
    )
    p_serve.add_argument(
        "--pagerduty-source",
        default="snagline",
        help="PagerDuty event source label [default: snagline].",
    )
    p_serve.add_argument(
        "--min-severity",
        default=None,
        help="Only escalate risks at or above this severity "
        "(info|warning|critical). Optional.",
    )
    p_serve.add_argument(
        "--cooldown-seconds",
        type=float,
        default=0.0,
        help="Suppress repeated alerts of the same key for this many seconds "
        "(issue #4). 0 disables (default).",
    )
    p_serve.add_argument(
        "--halt-forward",
        default=None,
        help="Enable the halt_webhook enforcement policy (issue #93): risks "
        "with score >= --min-severity-for-halt are POSTed to this URL and the "
        'response {"action": "continue"|"pause", "reason": ...} is surfaced '
        "as the sidecar monitor's last_directive. Timeout or error fails open "
        "to continue. Hold this endpoint tight: it controls pause decisions.",
    )
    p_serve.add_argument(
        "--halt-timeout",
        type=float,
        default=None,
        help="Halt webhook round-trip budget in seconds [default: 0.25].",
    )
    p_serve.add_argument(
        "--min-severity-for-halt",
        type=float,
        default=None,
        help="Minimum risk score that pays the halt-webhook cost [default: 0.8].",
    )

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
        "--timeout",
        type=float,
        default=1.0,
        help="HTTP timeout when --url is used [s].",
    )

    # Fits a per-tool healthy-run profile from a trajectory and persists it as
    # JSON. Dependency-free (stdlib only); model-based baselines can build on
    # the same persisted profile later.
    sp = sub.add_parser(
        "baseline",
        help="Fit a healthy-run baseline (per-tool latency/error profile) from a trajectory.",
    )
    sp.add_argument(
        "trajectory",
        metavar="trajectory|retrain",
        help="Path to a .jsonl trajectory of a healthy run, or the literal "
        "keyword 'retrain' to refit from the newest JSONL window and bump "
        "the BaselineStore version atomically (issue #102; see docs/RETRAIN_CADENCE.md).",
    )
    sp.add_argument(
        "--jsonl",
        default=None,
        help="With 'retrain': refit from exactly this JSONL window file "
        "(overrides --windows-dir).",
    )
    sp.add_argument(
        "--windows-dir",
        default=None,
        help="With 'retrain': directory of rotated window .jsonl files; the "
        "newest by modification time is used.",
    )
    sp.add_argument(
        "--max-age",
        type=float,
        default=None,
        metavar="SECONDS",
        help="With 'retrain': warn on stderr when the currently active stored "
        "baseline is older than this many seconds (staleness guard for the "
        "host-side cadence).",
    )
    sp.add_argument(
        "--output",
        default="baseline.json",
        help="Where to write the fitted baseline JSON [default: baseline.json].",
    )
    sp.add_argument(
        "--store-dir",
        default=None,
        help="If set, store the baseline in a versioned BaselineStore at this "
        "root (per --tenant/--deployment) instead of a single --output file.",
    )
    sp.add_argument(
        "--tenant",
        default="default",
        help="Tenant scope for the BaselineStore [default: default].",
    )
    sp.add_argument(
        "--deployment",
        default="default",
        help="Deployment scope for the BaselineStore [default: default].",
    )
    sp.add_argument(
        "--list-versions",
        action="store_true",
        help="With --store-dir: list stored versions and exit (ignores fitting).",
    )
    sp.add_argument(
        "--max-versions",
        type=int,
        default=None,
        help="With --store-dir: cap retained versions (oldest pruned).",
    )

    return parser


def _iter_lines(path: str | None, follow: bool) -> Iterator[str]:
    """Yield lines from stdin, or from a file; with ``follow``, tail like -f."""
    if path is None:
        yield from sys.stdin
        return
    with open(path, encoding="utf-8") as fh:
        while True:
            line = fh.readline()
            if line:
                yield line
            elif follow:
                time.sleep(0.2)
            else:
                return


def _cmd_watch(args: argparse.Namespace) -> int:
    # Resolve the effective config once (config file -> SNAGLINE_* env) so
    # log_format and every other knob layer identically across subcommands.
    cfg = _build_config(args)
    try:
        sinks = _build_sinks(args, cfg)
    except SystemExit as exc:
        return int(exc.code or 2)
    monitor = Monitor.default(
        config=cfg,
        # _build_sinks is the single DedupSink choke point (issue #152):
        # re-wrapping here composed DedupSink(DedupSink(inner)) and made watch
        # diverge from serve, which passes _build_sinks through untouched.
        sinks=sinks,
    )
    episode = args.episode_id or (args.file or "stdin")
    steps = 0
    try:
        with suppress(KeyboardInterrupt):
            for line in _iter_lines(args.file, args.follow):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    event = StepEvent(**obj)
                except (json.JSONDecodeError, TypeError, ValueError) as exc:
                    print(
                        f"snagline watch: skipping malformed line: {exc}",
                        file=sys.stderr,
                    )
                    continue
                monitor.ingest(event)
                steps += 1
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
            print(
                f"snagline hook: forward to {args.url} failed: {exc}", file=sys.stderr
            )

    if not args.url and not args.out:
        # Fail-open by construction; Monitor.default() is also fail-open.
        with suppress(Exception):
            monitor = Monitor.default(config=_build_config(args))
            monitor.ingest(event)
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


def _newest_window(windows_dir: str) -> str | None:
    """Return the newest ``*.jsonl`` window file in ``windows_dir``, or None.

    Newest means largest modification time; ties break alphabetically so the
    pick is deterministic. Non-recursive by design: windows are rotated into
    one flat directory (issue #102).
    """
    root = Path(windows_dir)
    candidates = [p for p in root.glob("*.jsonl") if p.is_file()]
    if not candidates:
        return None
    return str(max(candidates, key=lambda p: (p.stat().st_mtime_ns, p.name)))


def _active_baseline_age(store, tenant: str, deployment: str) -> float | None:
    """Age in seconds of the newest stored version, or None if unknowable.

    Version ids written by ``save()`` are wall-clock timestamps, so the id
    doubles as the fit time. Non-numeric (custom) ids are skipped fail-open;
    staleness is a warning aid, never a hard dependency.
    """
    for version_id in reversed(store.list_versions(tenant, deployment)):
        try:
            fitted_at = float(version_id)
        except ValueError:
            continue
        return max(0.0, time.time() - fitted_at)
    return None


def _cmd_baseline_retrain(args: argparse.Namespace) -> int:
    """Refit from the newest JSONL window and atomically bump the store.

    Usage: ``snagline baseline retrain --store-dir ROOT [--tenant T]
    [--deployment D] (--jsonl FILE | --windows-dir DIR) [--max-age SECONDS]
    [--max-versions N]``. Exits 0 on success, 2 on usage errors, 3 when the
    window cannot be resolved or read.
    """
    if not args.store_dir:
        print(
            "snagline baseline retrain: --store-dir is required (versioned store root)",
            file=sys.stderr,
        )
        return 2
    if bool(args.jsonl) == bool(args.windows_dir):
        print(
            "snagline baseline retrain: give exactly one of --jsonl FILE or "
            "--windows-dir DIR",
            file=sys.stderr,
        )
        return 2

    from snagline.baseline_store import BaselineStore, retrain_from_jsonl

    store = BaselineStore(args.store_dir, max_versions=args.max_versions or 10)

    # Staleness guard on the previously active baseline, before it is
    # replaced: this is the signal that the host-side cadence slipped.
    if args.max_age is not None:
        age = _active_baseline_age(store, args.tenant, args.deployment)
        if age is not None and age > args.max_age:
            print(
                f"snagline baseline: WARNING: active baseline for "
                f"{args.tenant}/{args.deployment} is {age / 3600.0:.1f}h old "
                f"(max-age {args.max_age / 3600.0:.1f}h); consider tightening "
                f"the retrain cadence",
                file=sys.stderr,
            )

    window = args.jsonl or _newest_window(args.windows_dir)
    if window is None:
        print(
            f"snagline baseline retrain: no .jsonl window files found in "
            f"{args.windows_dir}",
            file=sys.stderr,
        )
        return 3
    try:
        version = retrain_from_jsonl(
            store,
            window,
            tenant=args.tenant,
            deployment=args.deployment,
            max_versions=args.max_versions,
        )
    except OSError as exc:
        print(
            f"snagline baseline retrain: cannot read {window}: {exc}", file=sys.stderr
        )
        return 3

    profile = store.load_version(args.tenant, args.deployment, version)
    tools = len(profile.tools) if profile is not None else 0
    steps = profile.total_steps if profile is not None else 0
    print(
        f"snagline baseline retrain: stored version {version} for "
        f"{args.tenant}/{args.deployment} from {window} "
        f"({tools} tool(s), {steps} step(s))"
    )
    return 0


def _cmd_baseline(args: argparse.Namespace) -> int:
    """Fit a healthy-run baseline from a trajectory and persist it.

    With ``--store-dir`` the baseline is stored versioned and per-tenant in a
    ``BaselineStore``; ``--list-versions`` just lists what is already stored.
    Otherwise it writes a single JSON file (``--output``).

    The literal keyword ``retrain`` instead of a trajectory path dispatches to
    the scheduled-retrain contract (issue #102).
    """
    if getattr(args, "trajectory", None) == "retrain":
        return _cmd_baseline_retrain(args)

    if args.store_dir:
        from snagline.baseline_store import (
            BaselineStore,
            capture_from_jsonl,
        )

        store = BaselineStore(args.store_dir, max_versions=args.max_versions or 10)
        if args.list_versions:
            versions = store.list_versions(args.tenant, args.deployment)
            if not versions:
                print(
                    f"snagline baseline: no stored versions for "
                    f"{args.tenant}/{args.deployment}"
                )
                return 0
            print(f"snagline baseline: versions for {args.tenant}/{args.deployment}:")
            for v in versions:
                print(f"  {v}")
            return 0
        version = capture_from_jsonl(
            store,
            args.trajectory,
            tenant=args.tenant,
            deployment=args.deployment,
            max_versions=args.max_versions,
        )
        print(
            f"snagline baseline: stored version {version} for "
            f"{args.tenant}/{args.deployment}"
        )
        return 0

    from snagline.baseline import fit_baseline_from_jsonl, save_baseline

    profile = fit_baseline_from_jsonl(args.trajectory)
    save_baseline(profile, args.output)

    tools = profile.tools
    print(
        f"snagline baseline: fitted {len(tools)} tool(s) from {profile.total_steps} step(s)"
    )
    for name, tb in sorted(tools.items()):
        print(
            f"  {name}: n={tb.count} mean={tb.mean_latency:.1f}ms "
            f"std={tb.std_latency:.1f}ms errors={tb.error_count}"
        )
    print(f"snagline baseline: wrote {args.output}")
    return 0


def _cmd_serve(args: argparse.Namespace) -> int:
    from snagline.server.http_server import serve

    cfg = _build_config(args)
    # Enforcement wiring (issue #93): --halt-forward turns the resolved config
    # into halt_webhook mode. dataclasses.replace re-runs Config.__post_init__,
    # so an invalid value fails loudly here instead of at first ingest.
    if args.halt_forward:
        cfg = dataclasses.replace(
            cfg, policy="halt_webhook", halt_url=args.halt_forward
        )
    if args.halt_timeout is not None:
        cfg = dataclasses.replace(cfg, halt_timeout_s=args.halt_timeout)
    if args.min_severity_for_halt is not None:
        cfg = dataclasses.replace(cfg, min_severity_for_halt=args.min_severity_for_halt)
    try:
        sinks = _build_sinks(args, cfg)
    except SystemExit as exc:
        return int(exc.code or 2)
    # Prefer the flag, fall back to the environment so the secret need not
    # appear in argv (visible to every other process via ps).
    auth_token = args.auth_token or os.environ.get("SNAGLINE_SERVE_AUTH_TOKEN") or None
    # Validate TLS flag pairing before printing anything: a banner that
    # advertises https:// and then dies in a traceback is worse than a clean
    # refusal up front.
    if args.keyfile and not args.certfile:
        print(
            "snagline serve: --keyfile requires --certfile",
            file=sys.stderr,
        )
        return 2
    scheme = "https" if (args.certfile or args.keyfile) else "http"
    print(
        f"snagline serve: listening on {scheme}://{args.host}:{args.port} "
        "(POST /events, GET /health)",
        file=sys.stderr,
    )
    if cfg.policy == "halt_webhook":
        print(
            f"snagline serve: halt forwarding enabled -> {cfg.halt_url} "
            f"(timeout {cfg.halt_timeout_s}s, min severity "
            f"{cfg.min_severity_for_halt}); directives land on "
            "Monitor.last_directive (issue #93)",
            file=sys.stderr,
        )
    if not auth_token:
        print(
            "snagline serve: no --auth-token / $SNAGLINE_SERVE_AUTH_TOKEN set; "
            "every endpoint is open to anyone who can reach this port",
            file=sys.stderr,
        )
    with suppress(KeyboardInterrupt):
        # TLS kwargs are forwarded only when requested: with no --certfile/
        # --keyfile the serve() call is exactly what it was before issue #120.
        tls_kwargs = (
            {"certfile": args.certfile, "keyfile": args.keyfile}
            if (args.certfile or args.keyfile)
            else {}
        )
        serve(
            Monitor.default(config=cfg, sinks=sinks),
            host=args.host,
            port=args.port,
            auth_token=auth_token,
            max_body_bytes=args.max_body_bytes,
            max_risks=args.max_risks,
            **tls_kwargs,
        )
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


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "replay":
        counter = _CountingSink()
        cfg = _build_config(args)
        sinks: list[AlertSink] = []
        if not args.quiet:
            # Same composition as Monitor.default(): console plus, for
            # log_format="json", LoggingSink alongside it (issues #99/#119).
            sinks.extend(_console_sinks(cfg))
        sinks.append(counter)
        monitor = Monitor.default(config=cfg, sinks=sinks)
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
        return _cmd_baseline(args)

    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
