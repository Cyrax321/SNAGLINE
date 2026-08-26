"""A stalled sender must not pin a sidecar handler thread (#130).

``BaseHTTPRequestHandler`` inherits ``timeout = None``, so before this every
``rfile.read(n)`` blocked until exactly n bytes arrived or the peer closed. A
client could declare ``Content-Length: 500000``, write seven bytes and go
quiet, holding one thread of the ``ThreadingHTTPServer`` for as long as it
liked -- slowloris-style retention, repeatable from many connections. These
tests pin the remedy: the connection is dropped and its thread released
within the budget, senders that keep making progress are still served, and
the knob resolves from argument, environment and default in that order.
"""

from __future__ import annotations

import json
import socket
import threading
import time

from snagline.config import Config
from snagline.monitor import Monitor
from snagline.server.http_server import _resolve_read_timeout, make_handler, make_server

_EVENT = {
    "step_id": "0",
    "episode_id": "ep-timeout",
    "timestamp": 1718300000.0,
    "action_type": "tool_call",
    "action_signature": "aaaa1111bbbb2222",
}


def _start(**kwargs):
    server = make_server(Monitor([], []), host="127.0.0.1", port=0, **kwargs)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, int(server.server_address[1])


def _head(length: int) -> bytes:
    return (
        "POST /events HTTP/1.0\r\n"
        "Host: localhost\r\n"
        "Content-Type: application/json\r\n"
        f"Content-Length: {length}\r\n"
        "\r\n"
    ).encode()


def test_stalled_sender_is_dropped_and_its_thread_released():
    """The core regression: declare a big body, send a little, go silent."""
    server, port = _start(read_timeout=0.5)
    baseline = threading.active_count()
    try:
        sock = socket.create_connection(("127.0.0.1", port), timeout=10)
        try:
            sock.sendall(_head(500_000))
            sock.sendall(b'{"a":1}')  # 7 bytes of a 500 KB promise, then quiet
            started = time.perf_counter()
            # The server closing its side is what ends this recv; before the
            # fix it blocked until the client's own timeout instead.
            leftover = sock.recv(65536)
            elapsed = time.perf_counter() - started
        finally:
            sock.close()
        assert leftover == b"", leftover  # closed, not answered
        assert elapsed < 10.0  # the client timeout never had to fire
        # Poll rather than sleep a fixed amount: the handler is torn down just
        # after the socket closes, and CI schedulers are not prompt.
        deadline = time.perf_counter() + 10.0
        while threading.active_count() > baseline and time.perf_counter() < deadline:
            time.sleep(0.02)
        assert threading.active_count() <= baseline, "handler thread was not released"
    finally:
        server.shutdown()
        server.server_close()


def test_a_slow_but_progressing_sender_is_still_served():
    """The budget is per read, so trickling a body over longer is fine.

    Total transfer time here exceeds the timeout several times over; what
    matters is that no single read waits that long.
    """
    raw = json.dumps(_EVENT).encode()
    server, port = _start(read_timeout=0.3)
    try:
        sock = socket.create_connection(("127.0.0.1", port), timeout=10)
        try:
            sock.sendall(_head(len(raw)))
            for index in range(0, len(raw), 16):
                sock.sendall(raw[index : index + 16])
                time.sleep(0.01)  # steady progress, well inside the budget
            chunks: list[bytes] = []
            while True:
                data = sock.recv(65536)
                if not data:
                    break
                chunks.append(data)
            response = b"".join(chunks)
        finally:
            sock.close()
    finally:
        server.shutdown()
        server.server_close()
    assert response.startswith(b"HTTP/1.0 202"), response[:120]
    assert b'"status": "ingested"' in response


def test_well_formed_requests_are_unaffected_by_the_default_timeout():
    raw = json.dumps(_EVENT).encode()
    server, port = _start()  # default budget, nothing passed
    try:
        sock = socket.create_connection(("127.0.0.1", port), timeout=10)
        try:
            sock.sendall(_head(len(raw)))
            sock.sendall(raw)
            chunks: list[bytes] = []
            while True:
                data = sock.recv(65536)
                if not data:
                    break
                chunks.append(data)
            response = b"".join(chunks)
        finally:
            sock.close()
    finally:
        server.shutdown()
        server.server_close()
    assert response.startswith(b"HTTP/1.0 202"), response[:120]


def test_handler_class_carries_the_resolved_timeout():
    assert make_handler(Monitor([], [])).timeout == Config().serve_read_timeout_s
    assert make_handler(Monitor([], []), read_timeout=2.5).timeout == 2.5
    # At or below zero is the documented opt-out: back to blocking forever.
    assert make_handler(Monitor([], []), read_timeout=0).timeout is None
    assert make_handler(Monitor([], []), read_timeout=-1).timeout is None


def test_resolve_read_timeout_precedence(monkeypatch):
    monkeypatch.delenv("SNAGLINE_SERVE_READ_TIMEOUT_S", raising=False)
    assert _resolve_read_timeout(None) == Config().serve_read_timeout_s
    assert _resolve_read_timeout(12.0) == 12.0

    monkeypatch.setenv("SNAGLINE_SERVE_READ_TIMEOUT_S", "1.5")
    assert _resolve_read_timeout(None) == 1.5
    assert _resolve_read_timeout(12.0) == 12.0  # explicit still wins
    monkeypatch.setenv("SNAGLINE_SERVE_READ_TIMEOUT_S", "0")
    assert _resolve_read_timeout(None) is None

    # Unusable env values are ignored by Config.from_env_overrides, so the
    # default stands: configuration must never take the sidecar down.
    monkeypatch.setenv("SNAGLINE_SERVE_READ_TIMEOUT_S", "soon")
    assert _resolve_read_timeout(None) == Config().serve_read_timeout_s


def test_resolve_read_timeout_survives_unusable_arguments():
    assert _resolve_read_timeout(float("nan")) is None
    assert _resolve_read_timeout("also-not-a-number") == Config().serve_read_timeout_s


def test_resolve_read_timeout_survives_a_broken_environment(monkeypatch):
    """A failing env lookup must fall back, not take the sidecar down."""

    def _explode(*args, **kwargs):
        raise RuntimeError("environment unavailable")

    monkeypatch.setattr(Config, "from_env_overrides", classmethod(_explode))
    assert _resolve_read_timeout(None) == Config().serve_read_timeout_s
    assert _resolve_read_timeout(4.0) == 4.0  # explicit never consults the env
