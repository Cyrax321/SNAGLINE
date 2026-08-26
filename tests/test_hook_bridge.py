"""End-to-end tests for the external-process bridges:

* sidecar ``POST /hooks/claude-code`` (Claude Code native ``http`` hooks)
* ``snagline hook`` (command-hook bridge: stdin payload -> file/HTTP/local)
* ``snagline watch --file`` (file bridge)

Teardown contract (issue #69): every spawned child stops through
``_graceful_stop``, a cross-platform ladder (graceful signal on POSIX,
terminate() on Windows, then wait -> terminate -> kill escalation), so no
platform orphans the watch subprocess and no ProcessLookupError escapes.
"""

from __future__ import annotations

import json
import os
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.request
from contextlib import suppress

import pytest

from snagline.monitor import Monitor
from snagline.risk import FailureRisk
from snagline.server.http_server import make_server


class _RecordingSink:
    def __init__(self):
        self.risks: list[FailureRisk] = []

    def emit(self, risk: FailureRisk) -> None:
        self.risks.append(risk)


def _tool_payload(i: int, event: str = "PreToolUse") -> dict:
    # tool_input stays IDENTICAL across calls (only the volatile tool_use_id
    # differs) so the signature is stable and the loop detector can see the
    # repetition.
    return {
        "session_id": "sess-e2e",
        "hook_event_name": event,
        "tool_name": "Bash",
        "tool_input": {"command": "npm test"},
        "tool_use_id": f"toolu_{i}",
    }


def _post(url: str, obj: dict) -> dict:
    req = urllib.request.Request(
        url, data=json.dumps(obj).encode(), headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=5) as resp:
        return json.loads(resp.read())


_STOP_TIMEOUT_S = 10.0


class _HookTargetServer:
    """Minimal localhost POST target for webhook-sink teardown tests.

    Raw stdlib socket accept loop on an ephemeral 127.0.0.1 port. With
    ``respond_ok`` it answers 200 like a healthy sink endpoint; in drop mode
    it accepts and immediately closes, forcing the WebhookSink failure path
    while still giving the test a deterministic readiness signal (the first
    accepted connection). No sleeps needed anywhere.
    """

    def __init__(self, respond_ok: bool) -> None:
        self.respond_ok = respond_ok
        self.bodies: list[bytes] = []
        self.accepts = 0
        self._connected = threading.Event()
        self._sock = socket.socket()
        self._sock.bind(("127.0.0.1", 0))
        self._sock.listen(8)
        self.port: int = self._sock.getsockname()[1]
        self._thread = threading.Thread(target=self._serve, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def close(self) -> None:
        with suppress(OSError):
            self._sock.close()
        self._thread.join(timeout=5)

    def wait_for_connection(self, timeout: float) -> bool:
        return self._connected.wait(timeout)

    def _serve(self) -> None:
        while True:
            try:
                conn, _ = self._sock.accept()
            except OSError:
                return  # listening socket closed by close()
            self.accepts += 1
            self._connected.set()
            try:
                data = conn.recv(65536)
                if self.respond_ok and data:
                    self.bodies.append(data)
                    conn.sendall(b"HTTP/1.0 200 OK\r\nContent-Length: 0\r\n\r\n")
                # Drop mode: closing with no response IS the failure the
                # sink must survive.
            except OSError:
                pass
            finally:
                with suppress(OSError):
                    conn.close()


def _await_exit(proc: subprocess.Popen, timeout: float) -> None:
    """Wait for exit, escalating terminate() -> kill() until reaped."""
    try:
        proc.wait(timeout=timeout)
        return
    except subprocess.TimeoutExpired:
        pass
    # Rung 1: terminate() (SIGTERM on POSIX, TerminateProcess on Windows).
    with suppress(Exception):
        proc.terminate()
    try:
        proc.wait(timeout=timeout)
        return
    except subprocess.TimeoutExpired:
        pass
    # Rung 2: hard kill, then reap unconditionally.
    with suppress(Exception):
        proc.kill()
    with suppress(Exception):
        proc.wait(timeout=timeout)


def _graceful_stop(proc: subprocess.Popen, timeout: float = _STOP_TIMEOUT_S) -> None:
    """Cross-platform child teardown for issue #69.

    ``Popen.send_signal(SIGINT)`` raises ValueError on Windows (only SIGTERM
    and the CTRL_* events are supported there), which used to abort the
    finally block in the follow-mode test and orphan the watch subprocess.
    This ladder always ends with the child reaped, on every platform:

    - graceful step: SIGINT on POSIX (identical to Ctrl-C on the CLI, so its
      KeyboardInterrupt-suppressed summary still prints); terminate() first
      on Windows where SIGINT does not exist;
    - then wait(timeout) -> terminate() -> wait(timeout) -> kill().
    """
    if proc.poll() is None:
        if os.name == "nt":
            proc.terminate()
        else:
            proc.send_signal(signal.SIGINT)
    _await_exit(proc, timeout)


@pytest.fixture()
def children():
    """Track every Popen a test spawns; teardown asserts zero survivors."""
    handles: list[subprocess.Popen] = []
    yield handles
    for proc in handles:
        if proc.poll() is None:
            _graceful_stop(proc, timeout=_STOP_TIMEOUT_S)
    for proc in handles:
        assert proc.poll() is not None, f"lingering child not reaped: {proc.args}"


def test_sidecar_accepts_native_claude_code_payloads_and_fires_loop():
    sink = _RecordingSink()
    server = make_server(Monitor.default(sinks=[sink]), host="127.0.0.1", port=0)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        # A pre/start + post/end pair, repeated with IDENTICAL tool_input
        # (tool_use_id differs; the signature ignores it on purpose).
        for i in range(4):
            assert (
                _post(base + "/hooks/claude-code", _tool_payload(i))["status"]
                == "ingested"
            )
            assert (
                _post(base + "/hooks/claude-code", _tool_payload(i, "PostToolUse"))[
                    "status"
                ]
                == "ingested"
            )
        # Unmapped lifecycle events are acknowledged and ignored.
        assert (
            _post(base + "/hooks/claude-code", {"hook_event_name": "SessionStart"})[
                "status"
            ]
            == "ignored"
        )
        assert any(r.trigger == "loop" for r in sink.risks)
        assert all(r.episode_id == "sess-e2e" for r in sink.risks)
    finally:
        server.shutdown()
        server.server_close()


def _run(args: list, stdin_text: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "snagline.cli", *args],
        input=stdin_text,
        capture_output=True,
        text=True,
        timeout=30,
    )


def test_hook_cli_appends_canonical_event_to_file(tmp_path):
    out = tmp_path / "events.jsonl"
    for i in range(4):
        r = _run(["hook", "--out", str(out)], json.dumps(_tool_payload(i)))
        assert r.returncode == 0  # fail-open: hook ALWAYS exits 0
    lines = [json.loads(line) for line in out.read_text().splitlines()]
    assert len(lines) == 4
    assert {line["episode_id"] for line in lines} == {"sess-e2e"}
    assert {line["action_type"] for line in lines} == {"tool_call"}
    # The appended file is valid snagline watch input and trips the detector.
    r = _run(["watch", "--file", str(out), "--episode-id", "sess-e2e"], "")
    assert r.returncode == 0
    assert '"trigger": "loop"' in r.stderr


def test_hook_cli_forwards_to_sidecar(tmp_path):
    sink = _RecordingSink()
    server = make_server(Monitor.default(sinks=[sink]), host="127.0.0.1", port=0)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{server.server_address[1]}/events"
    try:
        for i in range(4):
            r = _run(["hook", "--url", url], json.dumps(_tool_payload(i)))
            assert r.returncode == 0
        assert any(x.trigger == "loop" for x in sink.risks)
    finally:
        server.shutdown()
        server.server_close()


def test_hook_cli_never_fails_the_host():
    r = _run(["hook", "--out", "/nonexistent/dir/x.jsonl"], "not json at all")
    assert r.returncode == 0
    r2 = _run(
        ["hook", "--url", "http://127.0.0.1:1/nope"], json.dumps(_tool_payload(1))
    )
    assert r2.returncode == 0
    r3 = _run(["hook"], json.dumps({"hook_event_name": "SessionStart"}))
    assert r3.returncode == 0  # unmapped events are silently dropped


def test_watch_file_follow_sees_appended_lines(tmp_path, children):
    path = tmp_path / "live.jsonl"
    path.write_text(json.dumps(_base_event(0)) + "\n")
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "snagline.cli",
            "watch",
            "--file",
            str(path),
            "--follow",
            "--episode-id",
            "ep-follow",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    children.append(proc)
    try:
        time.sleep(1.0)  # let it consume the first line
        with open(path, "a") as fh:
            for i in range(1, 4):
                fh.write(json.dumps(_base_event(i)) + "\n")
                fh.flush()
                time.sleep(0.3)
        time.sleep(0.8)
    finally:
        _graceful_stop(proc, timeout=_STOP_TIMEOUT_S)
        out, err = proc.communicate(timeout=15)
    assert proc.poll() is not None
    if os.name == "nt":
        # Windows teardown is terminate()-based: SIGINT does not exist for
        # arbitrary children (issue #69), so the CLI never reaches its
        # KeyboardInterrupt-path summary there. Reaping is the contract and
        # the regression tests below assert it on every platform.
        return
    assert "ingested 4 step(s)" in err
    assert '"trigger": "loop"' in err


def _base_event(i: int) -> dict:
    from snagline.events import make_signature

    return {
        "step_id": str(i),
        "episode_id": "ep-follow",
        "timestamp": 1718300000.0 + i,
        "action_type": "tool_call",
        "action_signature": make_signature("tool_call", "Bash", "npm test"),
        "tool_name": "Bash",
    }


_MALFORMED_MARKER = "malformed line"


class _StderrCatcher:
    """Drain a child's stderr on a thread without racing communicate().

    Exposes the joined text plus a one-shot event that fires once a line
    containing ``marker`` appears: a deterministic readiness signal with no
    sleeps, so tests stop the child only after it provably finished the
    work under test.
    """

    def __init__(self, proc: subprocess.Popen, marker: str) -> None:
        self.lines: list[str] = []
        self.marker_seen = threading.Event()
        self._marker = marker
        assert proc.stderr is not None
        self._stream = proc.stderr
        self._thread = threading.Thread(target=self._drain, daemon=True)
        self._thread.start()

    def _drain(self) -> None:
        for line in self._stream:
            self.lines.append(line)
            if self._marker in line:
                self.marker_seen.set()

    @property
    def text(self) -> str:
        return "".join(self.lines)

    def join(self, timeout: float) -> None:
        self._thread.join(timeout=timeout)


def _spawn_follow_watcher(
    path, webhook_url: str, children: list[subprocess.Popen]
) -> subprocess.Popen:
    """Spawn ``snagline watch --follow`` against a webhook sink endpoint."""
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "snagline.cli",
            "watch",
            "--file",
            str(path),
            "--follow",
            "--episode-id",
            "ep-follow",
            "--sink",
            "webhook",
            "--webhook-url",
            webhook_url,
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )
    children.append(proc)
    return proc


def test_watch_follow_teardown_reaps_the_child_on_a_healthy_run(tmp_path, children):
    """Issue #69 regression, healthy path, runs on every platform.

    The follow-mode watcher dispatches its loop risk to a live localhost
    sink endpoint. Delivery is proven deadline-based: the test polls
    ``server.bodies`` for the expected loop-trigger payload and proceeds
    only once the POST has landed (issue #156 -- the former readiness
    signal, the first accepted connection, raced the actual body landing,
    which lost intermittently on windows-latest). A malformed sentinel line
    then proves the child finished counting all four events before
    ``_graceful_stop`` ever runs. The child must end fully reaped: poll()
    non-None, no ProcessLookupError, and the module-level children fixture
    leaves zero lingering processes.
    """
    path = tmp_path / "healthy.jsonl"
    path.write_text("\n".join(json.dumps(_base_event(i)) for i in range(4)) + "\n")
    server = _HookTargetServer(respond_ok=True)
    server.start()
    try:
        proc = _spawn_follow_watcher(
            path, f"http://127.0.0.1:{server.port}/sink", children
        )
        catcher = _StderrCatcher(proc, _MALFORMED_MARKER)
        # Contract under test: the loop risk is delivered END-TO-END through
        # watch --follow to an HTTP sink. Poll for the delivered payload with
        # a generous deadline instead of trusting connection-level readiness;
        # teardown must not begin until delivery is observed (issue #156).
        deadline = time.monotonic() + 30.0
        while not any(b'"trigger": "loop"' in body for body in server.bodies):
            if time.monotonic() >= deadline:
                pytest.fail(
                    "watcher never delivered the loop risk to the sink "
                    f"endpoint within 30s; bodies={server.bodies!r}"
                )
            time.sleep(0.05)
        # Sentinel: once the CLI logs this line it has already counted every
        # seeded event and is back in the follow poll loop, so stopping it
        # now cannot truncate the run into an exit-code race.
        with open(path, "a", encoding="utf-8") as fh:
            fh.write("{not json at all\n")
        assert catcher.marker_seen.wait(30), "watcher never reached the sentinel line"
        _graceful_stop(proc, timeout=_STOP_TIMEOUT_S)
        catcher.join(10)
    finally:
        server.close()
    assert proc.poll() is not None
    assert any(b'"trigger": "loop"' in body for body in server.bodies), server.bodies
    if os.name != "nt":
        # SIGINT path: clean CLI shutdown with the full summary.
        assert proc.returncode == 0
        assert "ingested 4 step(s)" in catcher.text


def test_watch_follow_cleanup_reaps_the_child_when_the_sink_endpoint_is_dead(
    tmp_path, children
):
    """Issue #69 regression, failure path, runs on every platform.

    The endpoint accepts connections and drops them without responding, so
    the watcher's webhook POST fails mid-run. Even then the cleanup ladder
    must reap the child cleanly after it finishes the whole seeded file.
    """
    path = tmp_path / "dead-endpoint.jsonl"
    path.write_text("\n".join(json.dumps(_base_event(i)) for i in range(4)) + "\n")
    server = _HookTargetServer(respond_ok=False)
    server.start()
    try:
        proc = _spawn_follow_watcher(
            path, f"http://127.0.0.1:{server.port}/sink", children
        )
        catcher = _StderrCatcher(proc, _MALFORMED_MARKER)
        assert server.wait_for_connection(30), (
            "watcher never attempted a webhook POST against the dead endpoint"
        )
        with open(path, "a", encoding="utf-8") as fh:
            fh.write("{not json at all\n")
        assert catcher.marker_seen.wait(30), "watcher never reached the sentinel line"
        _graceful_stop(proc, timeout=_STOP_TIMEOUT_S)
        catcher.join(10)
    finally:
        server.close()
    assert proc.poll() is not None
    assert server.accepts >= 1  # the failing POST genuinely happened
    if os.name != "nt":
        # Fail-open: the sink failure never crashes the watch loop...
        assert proc.returncode == 0
        assert "ingested 4 step(s)" in catcher.text
        # ...it is logged and swallowed instead (guaranteed to be in the
        # captured stream by now: logging precedes the sentinel line).
        assert "webhook sink POST" in catcher.text


def test_graceful_stop_tolerates_an_already_exited_child():
    """No ProcessLookupError escapes when the child died before the stop."""
    proc = subprocess.Popen([sys.executable, "-c", "pass"])
    assert proc.wait(timeout=30) == 0
    _graceful_stop(proc, timeout=_STOP_TIMEOUT_S)
    assert proc.poll() is not None
    assert proc.returncode == 0


_IGNORES_SIGNALS_CHILD = (
    "import signal, time\n"
    "signal.signal(signal.SIGINT, signal.SIG_IGN)\n"
    "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
    "print('ready', flush=True)\n"
    "time.sleep(120)\n"
)


def test_graceful_stop_escalates_to_kill_when_the_child_ignores_signals(children):
    """A child ignoring SIGINT and SIGTERM is still torn down by kill()."""
    proc = subprocess.Popen(
        [sys.executable, "-c", _IGNORES_SIGNALS_CHILD],
        stdout=subprocess.PIPE,
        text=True,
    )
    children.append(proc)
    ready = threading.Event()

    def _wait_ready() -> None:
        assert proc.stdout is not None
        if proc.stdout.readline():
            ready.set()

    pump = threading.Thread(target=_wait_ready, daemon=True)
    pump.start()
    assert ready.wait(30), "stubborn child never signalled readiness"
    _graceful_stop(proc, timeout=5.0)
    assert proc.poll() is not None
    pump.join(timeout=5)
    with suppress(Exception):
        proc.stdout.close()
    if os.name != "nt":
        # POSIX proof of escalation: only SIGKILL gets through the ignores.
        assert proc.returncode == -signal.SIGKILL
