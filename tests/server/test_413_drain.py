"""Over-cap POST handling: the sender must be able to read the 413 (#121).

Closing the connection with megabytes of unread request body makes the
kernel reset the peer, so direct clients saw EPIPE/ECONNRESET instead of the
status line. These tests pin the remedy: within a small window above the
body cap the body is drained and discarded before replying, exactly-at-cap
bodies stay accepted, and the drain itself is bounded no matter what
Content-Length claims.
"""

from __future__ import annotations

import json
import socket
import threading
import types

from snagline.monitor import Monitor
from snagline.server.http_server import (
    _DRAIN_CHUNK_BYTES,
    _MAX_OVERCAP_DRAIN_EXCESS,
    make_handler,
    make_server,
)

_CAP = 1_000_000


def _start(max_body_bytes: int):
    server = make_server(
        Monitor([], []),
        host="127.0.0.1",
        port=0,
        max_body_bytes=max_body_bytes,
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, int(server.server_address[1])


def test_overcap_streamed_body_reads_413_status_line():
    """A client streaming an over-cap body must see the 413, not EPIPE."""
    body_len = _CAP + _MAX_OVERCAP_DRAIN_EXCESS  # top of the drain window
    server, port = _start(_CAP)
    try:
        sock = socket.create_connection(("127.0.0.1", port), timeout=10)
        try:
            head = (
                "POST /events HTTP/1.0\r\n"
                "Host: localhost\r\n"
                "Content-Type: application/json\r\n"
                f"Content-Length: {body_len}\r\n"
                "\r\n"
            ).encode()
            sock.sendall(head)
            sock.sendall(b"x" * body_len)
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
    assert response.startswith(b"HTTP/1.0 413"), response[:120]
    assert b"payload too large" in response


def test_exactly_at_cap_body_still_accepted():
    event = {
        "step_id": "0",
        "episode_id": "ep-cap",
        "timestamp": 1718300000.0,
        "action_type": "tool_call",
        "action_signature": "aaaa1111bbbb2222",
        "metadata": {"pad": ""},
    }
    raw = json.dumps(event).encode()
    filler = _CAP - len(raw)
    assert filler > 0
    event["metadata"]["pad"] = "a" * filler
    raw = json.dumps(event).encode()
    assert len(raw) == _CAP  # byte-exact, mirroring the verified cap behavior
    server, port = _start(_CAP)
    try:
        sock = socket.create_connection(("127.0.0.1", port), timeout=10)
        try:
            head = (
                "POST /events HTTP/1.0\r\n"
                "Host: localhost\r\n"
                "Content-Type: application/json\r\n"
                f"Content-Length: {len(raw)}\r\n"
                "\r\n"
            ).encode()
            sock.sendall(head)
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
    assert b'"status": "ingested"' in response


def test_drain_is_bounded_regardless_of_claimed_content_length():
    """A huge claimed length must not make the sidecar read forever."""
    handler_cls = make_handler(Monitor([], []), max_body_bytes=1_000)

    reads: list[int] = []

    class _EndlessRFile:
        def __init__(self) -> None:
            self.consumed = 0

        def read(self, n: int) -> bytes:
            reads.append(n)
            self.consumed += n
            return b"x" * n

    fake = types.SimpleNamespace(rfile=_EndlessRFile(), snagline_max_body=1_000)
    handler_cls._discard_overcap_body(fake, 10**9)
    assert fake.rfile.consumed == 1_000 + _MAX_OVERCAP_DRAIN_EXCESS
    assert reads
    assert max(reads) <= _DRAIN_CHUNK_BYTES


def test_drain_survives_client_hangup_and_socket_errors():
    handler_cls = make_handler(Monitor([], []), max_body_bytes=1_000)

    class _HungUpRFile:
        def read(self, n: int) -> bytes:
            return b""  # immediate EOF: client vanished mid-body

    fake = types.SimpleNamespace(rfile=_HungUpRFile(), snagline_max_body=1_000)
    handler_cls._discard_overcap_body(fake, 5_000)  # must not raise

    class _ResetRFile:
        def read(self, n: int) -> bytes:
            raise ConnectionResetError("peer reset mid-drain")

    fake = types.SimpleNamespace(rfile=_ResetRFile(), snagline_max_body=1_000)
    handler_cls._discard_overcap_body(fake, 5_000)  # must not raise
