"""Enforcement policy layer (issue #93): callback + halt_webhook, fail-open.

Parity with tests/test_monitor_fail_open.py is the core contract: the
enforcement tail must be exactly as safe as the sink loop it mirrors.
Documented ordering: detectors -> sinks -> policy. The halt webhook runs
outside every episode lock and never raises into the host loop under
fail_open=True: timeout, error, dead endpoint, malformed body, or unknown
action all leave ``monitor.last_directive`` at continue.
"""

from __future__ import annotations

import json
import logging
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from snagline.config import Config
from snagline.events import StepEvent
from snagline.monitor import HaltDirective, Monitor
from snagline.risk import FailureRisk, TriggerType


def _event(step_id: str = "s1", episode_id: str = "ep1") -> StepEvent:
    return StepEvent(
        step_id=step_id,
        episode_id=episode_id,
        timestamp=1.0,
        action_type="tool_call",
        action_signature=f"sig-{step_id}",
    )


class _FixedRiskDetector:
    """Emits one synthetic risk at a fixed score on every observe() call."""

    name = "fixed_risk"

    def __init__(self, score: float = 0.9, trigger: TriggerType = "loop") -> None:
        self.score = score
        self.trigger: TriggerType = trigger

    def observe(self, event: StepEvent) -> FailureRisk | None:
        return FailureRisk(
            event.episode_id,
            event.step_id,
            self.score,
            self.trigger,
            "synthetic risk",
            event.timestamp,
        )

    def reset(self, episode_id: str) -> None:
        pass


class _FinalizeRiskDetector:
    """Silent during observe(); emits one risk from finalize()."""

    name = "finalize_risk"

    def observe(self, event: StepEvent) -> FailureRisk | None:
        return None

    def finalize(self, episode_id: str) -> FailureRisk | None:
        return FailureRisk(
            episode_id,
            "final",
            0.95,
            "silent_abort",
            "synthetic finalize risk",
            2.0,
        )

    def reset(self, episode_id: str) -> None:
        pass


class _BlockStepRiskDetector:
    """Emits a high-score risk ONLY for the step named 'block'.

    Lets the outside-the-lock test give one thread a halting risk while a
    concurrent ingest of the SAME episode stays risk-free, isolating the
    episode-lock behavior from the webhook itself.
    """

    name = "block_step_risk"

    def observe(self, event: StepEvent) -> FailureRisk | None:
        if event.step_id != "block":
            return None
        return FailureRisk(
            event.episode_id,
            event.step_id,
            0.95,
            "loop",
            "synthetic blocking risk",
            event.timestamp,
        )

    def reset(self, episode_id: str) -> None:
        pass


class _RecordingSink:
    def __init__(self) -> None:
        self.received: list[FailureRisk] = []

    def emit(self, risk: FailureRisk) -> None:
        self.received.append(risk)


class _HaltEndpoint:
    """Configurable local halt endpoint used by the webhook tests.

    responder(parsed_body) -> (status, response_obj); when raw_body is set it
    is sent verbatim instead. delay > 0 stalls every response until the
    release event fires or the delay elapses, whichever comes first.
    """

    def __init__(self, responder=None, raw_body=None, delay: float = 0.0) -> None:
        self.requests: list[dict] = []
        self.lock = threading.Lock()
        self.entered = threading.Event()
        self.release = threading.Event()
        self._responder = responder
        self._raw_body = raw_body
        self._delay = delay
        outer = self

        class _Handler(BaseHTTPRequestHandler):
            def log_message(self, format: str, *args) -> None:  # noqa: A002
                pass

            def do_POST(self) -> None:  # noqa: N802
                length = int(self.headers.get("Content-Length") or 0)
                body = self.rfile.read(length)
                with outer.lock:
                    outer.requests.append(json.loads(body.decode("utf-8")))
                outer.entered.set()
                if outer._delay > 0:
                    outer.release.wait(outer._delay)
                if outer._raw_body is not None:
                    payload, status = outer._raw_body, 200
                elif outer._responder is not None:
                    status, payload = outer._responder(outer.requests[-1])
                else:
                    status, payload = 200, {"action": "continue", "reason": ""}
                data = (
                    payload
                    if isinstance(payload, bytes)
                    else json.dumps(payload).encode("utf-8")
                )
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        self._server.daemon_threads = True
        self.url = f"http://127.0.0.1:{self._server.server_port}/halt"
        self._thread = threading.Thread(target=self._server.serve_forever)
        self._thread.start()

    def wait_entered(self, timeout: float = 2.0) -> bool:
        return self.entered.wait(timeout)

    def stop(self) -> None:
        self.release.set()
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=2)


@pytest.fixture
def halt_endpoint():
    endpoint = _HaltEndpoint()
    yield endpoint
    endpoint.stop()


def _dead_url() -> str:
    """An address where nothing listens anymore (bound once, then closed)."""
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    return f"http://127.0.0.1:{port}/halt"


# --- Default construction unchanged (acceptance criterion 1) -----------------


def test_default_construction_unchanged_observe_policy():
    sink = _RecordingSink()
    monitor = Monitor([_FixedRiskDetector()], [sink])
    assert monitor.policy == "observe"
    assert monitor.halt_url is None
    assert monitor.last_directive.action == "continue"
    monitor.ingest(_event())
    # Identical pre-#93 dispatch behavior plus a zero-count policy counter.
    assert len(sink.received) == 1
    assert monitor.metrics()["policy_errors"] == 0


def test_observe_policy_ignores_enforcement_arguments():
    # Passing halt-only arguments without opting into a policy changes nothing.
    sink = _RecordingSink()
    monitor = Monitor(
        [_FixedRiskDetector(score=0.99)],
        [sink],
        halt_url=_dead_url(),
        halt_timeout_s=0.01,
    )
    assert monitor.policy == "observe"
    monitor.ingest(_event())
    assert len(sink.received) == 1
    assert monitor.last_directive.action == "continue"


# --- Callback mode (parity with test_monitor_fail_open.py) --------------------


def test_callback_invoked_after_sinks_in_documented_order():
    order: list[str] = []

    class OrderSink:
        def emit(self, risk: FailureRisk) -> None:
            order.append("sink")

    def on_risk(risk: FailureRisk) -> None:
        order.append("callback")

    monitor = Monitor(
        [_FixedRiskDetector()],
        [OrderSink()],
        policy="callback",
        on_risk=on_risk,
    )
    monitor.ingest(_event())
    assert order == ["sink", "callback"]


def test_callback_receives_the_risk_object():
    seen: list[FailureRisk] = []
    monitor = Monitor(
        [_FixedRiskDetector(trigger="error_cascade")],
        [],
        policy="callback",
        on_risk=seen.append,
    )
    monitor.ingest(_event("s9"))
    assert len(seen) == 1
    assert seen[0].trigger == "error_cascade"
    assert seen[0].step_id == "s9"


class _RaisingCallback:
    def __call__(self, risk: FailureRisk) -> None:
        raise RuntimeError("boom in callback")


def test_callback_exception_swallowed_fail_open():
    sink = _RecordingSink()
    monitor = Monitor(
        [_FixedRiskDetector()],
        [sink],
        policy="callback",
        on_risk=_RaisingCallback(),
    )
    # Must NOT raise, mirroring test_fail_open_default_swallows_sink_exception.
    monitor.ingest(_event("a"))
    monitor.ingest(_event("b"))
    # Both risks still reached the sink; the callback never broke ingest.
    assert len(sink.received) == 2


def test_callback_exception_logged_and_counted(caplog):
    monitor = Monitor(
        [_FixedRiskDetector()],
        [],
        policy="callback",
        on_risk=_RaisingCallback(),
    )
    with caplog.at_level(logging.ERROR, logger="snagline"):
        monitor.ingest(_event("a"))
        monitor.ingest(_event("b"))
    assert any("callback" in r.message for r in caplog.records)
    assert any("fail-open" in r.message for r in caplog.records)
    # Counted on every occurrence even though logged only once (issue #14).
    assert monitor.metrics()["policy_errors"] == 2


def test_callback_exception_propagates_when_fail_open_false():
    monitor = Monitor(
        [_FixedRiskDetector()],
        [],
        fail_open=False,
        policy="callback",
        on_risk=_RaisingCallback(),
    )
    with pytest.raises(RuntimeError):
        monitor.ingest(_event())


def test_callback_not_filtered_by_min_severity():
    seen: list[FailureRisk] = []
    monitor = Monitor(
        [_FixedRiskDetector(score=0.2)],
        [],
        policy="callback",
        on_risk=seen.append,
        min_severity_for_halt=0.99,
    )
    monitor.ingest(_event())
    # The threshold gates only the webhook cost; callbacks see every risk.
    assert len(seen) == 1


def test_callback_policy_without_callable_warns_but_never_crashes(caplog):
    with caplog.at_level(logging.WARNING, logger="snagline"):
        monitor = Monitor([], [], policy="callback")
    assert monitor.policy == "callback"
    assert any("on_risk" in r.message for r in caplog.records)


# --- Halt webhook mode --------------------------------------------------------


def test_directive_starts_at_continue_before_any_webhook(halt_endpoint):
    monitor = Monitor(
        [_FixedRiskDetector()],
        [],
        policy="halt_webhook",
        halt_url=halt_endpoint.url,
    )
    assert monitor.last_directive == HaltDirective()


def test_halt_webhook_pause_response_surfaced_as_last_directive(halt_endpoint):
    halt_endpoint._responder = lambda req: (
        200,
        {"action": "pause", "reason": "budget breach"},
    )
    monitor = Monitor(
        [_FixedRiskDetector()],
        [],
        policy="halt_webhook",
        halt_url=halt_endpoint.url,
    )
    monitor.ingest(_event())
    directive = monitor.last_directive
    assert directive.action == "pause"
    assert directive.reason == "budget breach"
    assert directive.timestamp > 0


def test_halt_webhook_continue_response(halt_endpoint):
    halt_endpoint._responder = lambda req: (200, {"action": "continue"})
    monitor = Monitor(
        [_FixedRiskDetector()],
        [],
        policy="halt_webhook",
        halt_url=halt_endpoint.url,
    )
    monitor.ingest(_event())
    assert monitor.last_directive.action == "continue"


def test_halt_webhook_payload_carries_only_risk_fields(halt_endpoint):
    monitor = Monitor(
        [_FixedRiskDetector()],
        [],
        policy="halt_webhook",
        halt_url=halt_endpoint.url,
    )
    monitor.ingest(_event("sp"))
    assert halt_endpoint.wait_entered()
    body = halt_endpoint.requests[0]
    assert set(body) == {
        "episode_id",
        "step_id",
        "score",
        "trigger",
        "detail",
        "timestamp",
        "severity",
    }
    assert body["step_id"] == "sp"
    assert "metadata" not in body


def test_halt_webhook_below_threshold_skips_post(halt_endpoint):
    monitor = Monitor(
        [_FixedRiskDetector(score=0.5)],
        [],
        policy="halt_webhook",
        halt_url=halt_endpoint.url,
    )
    monitor.ingest(_event())
    time.sleep(0.05)
    assert halt_endpoint.requests == []
    assert monitor.last_directive.action == "continue"


def test_halt_webhook_at_threshold_posts(halt_endpoint):
    monitor = Monitor(
        [_FixedRiskDetector(score=0.8)],
        [],
        policy="halt_webhook",
        halt_url=halt_endpoint.url,
    )
    monitor.ingest(_event())
    assert halt_endpoint.wait_entered()
    assert len(halt_endpoint.requests) == 1


def test_halt_webhook_dead_endpoint_defaults_to_continue():
    sink = _RecordingSink()
    monitor = Monitor(
        [_FixedRiskDetector()],
        [sink],
        policy="halt_webhook",
        halt_url=_dead_url(),
    )
    start = time.perf_counter()
    monitor.ingest(_event())  # must NOT raise
    elapsed = time.perf_counter() - start
    assert monitor.last_directive.action == "continue"
    assert monitor.metrics()["policy_errors"] == 1
    # Sink dispatch was unaffected by the failed consultation.
    assert len(sink.received) == 1
    # Connection-refused fails fast locally; guard against surprise stalls.
    assert elapsed < 1.0


def test_halt_webhook_stalled_endpoint_times_out_to_continue():
    endpoint = _HaltEndpoint(delay=5.0)
    try:
        monitor = Monitor(
            [_FixedRiskDetector()],
            [],
            policy="halt_webhook",
            halt_url=endpoint.url,
            halt_timeout_s=0.15,
        )
        start = time.perf_counter()
        monitor.ingest(_event())  # must NOT raise
        elapsed = time.perf_counter() - start
        # Paid roughly the timeout budget, not the endpoint's full stall.
        # Allow small epsilon for timer granularity on Windows (issue #156 era).
        assert 0.12 <= elapsed < 2.0
        assert monitor.last_directive.action == "continue"
        assert monitor.metrics()["policy_errors"] == 1
    finally:
        endpoint.stop()


def test_halt_webhook_malformed_body_defaults_to_continue():
    endpoint = _HaltEndpoint(raw_body=b"this is not json")
    try:
        monitor = Monitor(
            [_FixedRiskDetector()],
            [],
            policy="halt_webhook",
            halt_url=endpoint.url,
        )
        monitor.ingest(_event())
        assert monitor.last_directive.action == "continue"
        assert monitor.metrics()["policy_errors"] == 1
    finally:
        endpoint.stop()


def test_halt_webhook_unknown_action_defaults_to_continue():
    endpoint = _HaltEndpoint(responder=lambda req: (200, {"action": "explode"}))
    try:
        monitor = Monitor(
            [_FixedRiskDetector()],
            [],
            policy="halt_webhook",
            halt_url=endpoint.url,
        )
        monitor.ingest(_event())
        assert monitor.last_directive.action == "continue"
        assert monitor.metrics()["policy_errors"] == 1
    finally:
        endpoint.stop()


def test_halt_webhook_error_status_defaults_to_continue():
    endpoint = _HaltEndpoint(responder=lambda req: (500, {"action": "pause"}))
    try:
        monitor = Monitor(
            [_FixedRiskDetector()],
            [],
            policy="halt_webhook",
            halt_url=endpoint.url,
        )
        monitor.ingest(_event())
        assert monitor.last_directive.action == "continue"
        assert monitor.metrics()["policy_errors"] == 1
    finally:
        endpoint.stop()


def test_halt_webhook_failure_propagates_when_fail_open_false():
    monitor = Monitor(
        [_FixedRiskDetector()],
        [],
        fail_open=False,
        policy="halt_webhook",
        halt_url=_dead_url(),
    )
    with pytest.raises(Exception):
        monitor.ingest(_event())


def test_halt_webhook_runs_outside_ingest_lock():
    """A stalled halt consultation must not hold the episode lock (issue #93).

    Thread A ingests a high-score event for ep1 and parks inside the webhook;
    while it is parked, a low-score ingest for the SAME episode must still
    complete its locked detection phase. If the webhook ever moved inside the
    episode lock, the second ingest would hang until the release.
    """
    endpoint = _HaltEndpoint(delay=30.0)
    try:
        monitor = Monitor(
            [_BlockStepRiskDetector()],
            [],
            policy="halt_webhook",
            halt_url=endpoint.url,
            halt_timeout_s=10.0,
        )
        blocker = threading.Thread(
            target=monitor.ingest, args=(_event("block", "ep1"),)
        )
        blocker.start()
        assert endpoint.wait_entered(2.0), "webhook was never consulted"

        done = threading.Event()

        def second_ingest() -> None:
            monitor.ingest(_event("after", "ep1"))
            done.set()

        worker = threading.Thread(target=second_ingest)
        worker.start()
        assert done.wait(2.0), (
            "second ingest blocked: halt webhook appears to hold the "
            "episode lock (ordering detectors -> sinks -> policy violated)"
        )
        # Low-score risk skipped the webhook, so exactly one POST so far.
        with endpoint.lock:
            assert len(endpoint.requests) == 1
        endpoint.release.set()
        blocker.join(timeout=15)
        assert not blocker.is_alive()
        worker.join(timeout=2)
    finally:
        endpoint.stop()


def test_last_directive_thread_safe_under_concurrent_ingest(halt_endpoint):
    halt_endpoint._responder = lambda req: (
        200,
        {"action": "pause" if req["step_id"].endswith("0") else "continue"},
    )
    failures: list[str] = []

    def check(directive: HaltDirective) -> None:
        if directive.action not in ("continue", "pause"):
            failures.append(f"torn directive: {directive!r}")

    monitor = Monitor(
        [_FixedRiskDetector()],
        [],
        policy="halt_webhook",
        halt_url=halt_endpoint.url,
    )
    threads = [
        threading.Thread(
            target=lambda k=k: [
                (check(monitor.last_directive), monitor.ingest(_event(f"t{k}-{i}")))[1]  # type: ignore[func-returns-value]
                for i in range(10)
            ]
        )
        for k in range(8)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)
    assert failures == []
    assert monitor.last_directive.action in ("continue", "pause")


def test_finalize_risks_also_reach_policy_tail(halt_endpoint):
    """end_episode() dispatches finalized risks through the same tail."""
    halt_endpoint._responder = lambda req: (
        200,
        {"action": "pause", "reason": "silent abort"},
    )
    monitor = Monitor(
        [_FinalizeRiskDetector()],
        [],
        policy="halt_webhook",
        halt_url=halt_endpoint.url,
    )
    monitor.ingest(_event())
    assert monitor.last_directive.action == "continue"
    monitor.end_episode("ep1")
    assert monitor.last_directive.action == "pause"


# --- Configuration and validation ---------------------------------------------


def test_invalid_policy_raises_valueerror():
    with pytest.raises(ValueError):
        Monitor([], [], policy="nope")
    with pytest.raises(ValueError):
        Config(policy="nope")


def test_halt_webhook_without_url_raises():
    with pytest.raises(ValueError):
        Monitor([], [], policy="halt_webhook")
    with pytest.raises(ValueError):
        Monitor([], [], policy="halt_webhook", halt_url="")


def test_halt_webhook_nonpositive_timeout_raises():
    with pytest.raises(ValueError):
        Monitor([], [], policy="halt_webhook", halt_url="http://x/", halt_timeout_s=0)


def test_halt_webhook_rejects_non_http_scheme():
    # Control-plane endpoint: urllib would happily try file://, ftp://, etc.;
    # only http(s) is a legitimate halt transport.
    for bad in ("file:///etc/passwd", "ftp://host/halt", "/etc/passwd"):
        with pytest.raises(ValueError):
            Monitor([], [], policy="halt_webhook", halt_url=bad)


def test_min_severity_for_halt_range_validated():
    for bad in (-0.1, 1.1, 42):
        with pytest.raises(ValueError):
            Monitor([], [], min_severity_for_halt=bad)
    # The domain edges themselves are legal.
    for edge in (0.0, 1.0):
        monitor = Monitor([], [], min_severity_for_halt=edge)
        assert monitor._min_severity_for_halt == edge


def test_monitor_default_wires_policy_from_config(halt_endpoint):
    monitor = Monitor.default(
        config=Config(policy="halt_webhook", halt_url=halt_endpoint.url),
        sinks=[],
    )
    assert monitor.policy == "halt_webhook"
    assert monitor.halt_url == halt_endpoint.url
    assert monitor.last_directive.action == "continue"
    # And the stock default stays observe.
    stock = Monitor.default(config=Config(), sinks=[])
    assert stock.policy == "observe"


def test_config_env_layering_for_policy_fields():
    cfg = Config.resolve(
        environ={
            "SNAGLINE_POLICY": "Halt_Webhook ",
            "SNAGLINE_HALT_URL": "http://127.0.0.1:9/halt",
            "SNAGLINE_HALT_TIMEOUT_S": "0.4",
            "SNAGLINE_MIN_SEVERITY_FOR_HALT": "0.6",
        }
    )
    assert cfg.policy == "halt_webhook"
    assert cfg.halt_url == "http://127.0.0.1:9/halt"
    assert cfg.halt_timeout_s == 0.4
    assert cfg.min_severity_for_halt == 0.6


def test_no_extras_import_smoke():
    """Bare import works stdlib-only; enforcement adds no third-party deps."""
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, "-c", "import snagline; print(snagline.__version__)"],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip()


# --- CLI: snagline serve --halt-forward ----------------------------------------


def test_serve_halt_forward_enables_policy(monkeypatch):
    from snagline import cli

    captured: dict = {}

    def _fake_serve(monitor, host="127.0.0.1", port=8787, **kwargs):
        captured["monitor"] = monitor

    monkeypatch.setattr("snagline.server.http_server.serve", _fake_serve)
    url = "http://127.0.0.1:9/halt"
    assert cli.main(["serve", "--halt-forward", url]) == 0
    monitor = captured["monitor"]
    assert monitor.policy == "halt_webhook"
    assert monitor.halt_url == url
    assert monitor.last_directive.action == "continue"


def test_serve_without_halt_forward_stays_observe(monkeypatch):
    from snagline import cli

    captured: dict = {}

    def _fake_serve(monitor, host="127.0.0.1", port=8787, **kwargs):
        captured["monitor"] = monitor

    monkeypatch.delenv("SNAGLINE_SERVE_AUTH_TOKEN", raising=False)
    monkeypatch.setattr("snagline.server.http_server.serve", _fake_serve)
    assert cli.main(["serve"]) == 0
    assert captured["monitor"].policy == "observe"


def test_serve_halt_flags_map_to_config(monkeypatch):
    from snagline import cli

    captured: dict = {}

    def _fake_serve(monitor, host="127.0.0.1", port=8787, **kwargs):
        captured["monitor"] = monitor

    monkeypatch.setattr("snagline.server.http_server.serve", _fake_serve)
    assert (
        cli.main(
            [
                "serve",
                "--halt-forward",
                "http://127.0.0.1:9/halt",
                "--halt-timeout",
                "0.5",
                "--min-severity-for-halt",
                "0.6",
            ]
        )
        == 0
    )
    monitor = captured["monitor"]
    assert monitor._halt_timeout_s == 0.5
    assert monitor._min_severity_for_halt == 0.6


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
