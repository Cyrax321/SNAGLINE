"""Enforcement-policy latency benchmark (issue #93).

Measures the real, reproducible added wall-clock latency per ingested step
when an enforcement policy is configured, across three halt-endpoint
behaviors:

  * responding : localhost endpoint answering {"action": "continue"} at once
  * refused    : dead endpoint (connection refused immediately)
  * stalled    : endpoint accepts the connection but never answers within the
                 timeout budget -- the true worst case, where each halting
                 step pays the full halt_timeout_s on its own dispatch thread

A policy="observe" baseline leg is included for reference. Each halt leg uses
a stub detector that emits one score=0.99 risk per step, so every step pays
the full enforcement cost; sinks are empty so ONLY policy work is measured.

Run directly::

    python benchmarks/enforcement_benchmark.py

Stdlib-only; imports ``snagline``. Numbers this script prints are safe to
publish (with hardware/date noted), matching how overhead_benchmark is used.
"""

from __future__ import annotations

import json
import socket
import statistics
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from snagline.events import StepEvent
from snagline.monitor import Monitor
from snagline.risk import FailureRisk


class _RiskPerStepDetector:
    """Emits one above-threshold synthetic risk on every observe() call."""

    name = "bench_risk_per_step"

    def observe(self, event: StepEvent) -> FailureRisk | None:
        return FailureRisk(
            event.episode_id,
            event.step_id,
            0.99,
            "loop",
            "bench risk",
            event.timestamp,
        )

    def reset(self, episode_id: str) -> None:
        pass


def _make_events(n: int) -> list[StepEvent]:
    return [
        StepEvent(
            step_id=str(i),
            episode_id="bench-halt",
            timestamp=time.time(),
            action_type="tool_call",
            action_signature=f"halt-bench-{i}",
            tool_name="tool",
            latency_ms=100.0,
        )
        for i in range(n)
    ]


def _measure_us_per_step(monitor: Monitor, n: int, block: int = 200) -> dict:
    # Adaptive block size so small legs (the expensive stalled case) still
    # produce several timed blocks after a proportionally small warm-up.
    block = max(1, min(block, n // 5))
    events = _make_events(n)
    for e in events[:block]:
        monitor.ingest(e)
    per_step_us: list[float] = []
    start_at = block
    while start_at < n:
        chunk = events[start_at : start_at + block]
        t0 = time.perf_counter()
        for e in chunk:
            monitor.ingest(e)
        t1 = time.perf_counter()
        per_step_us.append((t1 - t0) / len(chunk) * 1e6)
        start_at += block
    ordered = sorted(per_step_us)
    p99_idx = min(len(ordered) - 1, int(0.99 * len(ordered)))
    return {
        "n": n,
        "median_us": statistics.median(per_step_us),
        "p99_us": ordered[p99_idx],
        # Fail-open fallbacks that fired during the leg; published numbers
        # should always state this count (it means some steps paid the error
        # path instead of a real round trip).
        "policy_errors": monitor.metrics()["policy_errors"],
    }


class _RespondingEndpoint:
    def __init__(self) -> None:
        class _Handler(BaseHTTPRequestHandler):
            def log_message(self, format: str, *args) -> None:  # noqa: A002
                pass

            def do_POST(self) -> None:  # noqa: N802
                length = int(self.headers.get("Content-Length") or 0)
                self.rfile.read(length)
                data = json.dumps({"action": "continue", "reason": ""}).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        self._server.daemon_threads = True
        self._server.request_queue_size = 128
        self.url = f"http://127.0.0.1:{self._server.server_port}/halt"
        self._thread = threading.Thread(target=self._server.serve_forever)
        self._thread.start()

    def stop(self) -> None:
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=2)


class _StalledEndpoint:
    """Accepts requests, then sleeps far past any sane timeout budget."""

    def __init__(self, stall_seconds: float = 5.0) -> None:
        class _Handler(BaseHTTPRequestHandler):
            def log_message(self, format: str, *args) -> None:  # noqa: A002
                pass

            def do_POST(self) -> None:  # noqa: N802
                length = int(self.headers.get("Content-Length") or 0)
                self.rfile.read(length)
                time.sleep(stall_seconds)
                self.send_response(200)
                self.send_header("Content-Length", "2")
                self.end_headers()
                self.wfile.write(b"{}")

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        self._server.daemon_threads = True
        self._server.request_queue_size = 128
        self.url = f"http://127.0.0.1:{self._server.server_port}/halt"
        self._thread = threading.Thread(target=self._server.serve_forever)
        self._thread.start()

    def stop(self) -> None:
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=2)


def _dead_url() -> str:
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    return f"http://127.0.0.1:{port}/halt"


def run_enforcement_benchmark(
    n_fast: int = 2_000, n_stalled: int = 12, halt_timeout_s: float = 0.25
) -> dict:
    """Return per-leg median/p99 microseconds-per-step for the four setups."""
    results: dict[str, dict] = {}

    baseline_monitor = Monitor([_RiskPerStepDetector()], [], policy="observe")
    results["observe"] = _measure_us_per_step(baseline_monitor, n_fast)

    responding = _RespondingEndpoint()
    try:
        monitor = Monitor(
            [_RiskPerStepDetector()],
            [],
            policy="halt_webhook",
            halt_url=responding.url,
            halt_timeout_s=halt_timeout_s,
        )
        results["responding"] = _measure_us_per_step(monitor, n_fast)
    finally:
        responding.stop()

    monitor = Monitor(
        [_RiskPerStepDetector()],
        [],
        policy="halt_webhook",
        halt_url=_dead_url(),
        halt_timeout_s=halt_timeout_s,
    )
    results["refused"] = _measure_us_per_step(monitor, n_fast)

    stalled = _StalledEndpoint(stall_seconds=halt_timeout_s * 20)
    try:
        monitor = Monitor(
            [_RiskPerStepDetector()],
            [],
            policy="halt_webhook",
            halt_url=stalled.url,
            halt_timeout_s=halt_timeout_s,
        )
        results["stalled"] = _measure_us_per_step(monitor, n_stalled)
    finally:
        stalled.stop()

    return results


def main() -> None:
    halt_timeout_s = 0.25
    stats = run_enforcement_benchmark(halt_timeout_s=halt_timeout_s)
    print("snagline enforcement-policy latency benchmark (issue #93)")
    print(f"  halt timeout budget : {halt_timeout_s * 1000:.0f} ms")
    for leg in ("observe", "responding", "refused", "stalled"):
        s = stats[leg]
        print(
            f"  {leg:<11}: median {s['median_us']:>12.2f} us/step, "
            f"p99 {s['p99_us']:>12.2f} us/step (n={s['n']}, "
            f"fail-open fallbacks: {s['policy_errors']})"
        )


if __name__ == "__main__":
    main()
