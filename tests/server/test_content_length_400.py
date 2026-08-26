"""Malformed ``Content-Length`` must become a 400, not a traceback (#129).

``int(self.headers.get("Content-Length") or 0)`` raised inside ``do_POST``,
so ``socketserver`` printed a full traceback per bad request and dropped the
connection: the client saw an empty reply (or a reset, for negative values
that reached ``rfile.read(-n)``) instead of a status line, and a repeating
sender produced unbounded server-side log noise. These tests pin the
remedy -- a clean 400 on the wire, no exception escaping the handler, and
the well-formed and absent-header paths left untouched.
"""

from __future__ import annotations

import json
import socket
import threading
import time
import types

from snagline.monitor import Monitor
from snagline.server import http_server
from snagline.server.http_server import (
    _BAD_FRAMING_DRAIN_SECONDS,
    _DRAIN_CHUNK_BYTES,
    _MAX_OVERCAP_DRAIN_EXCESS,
    make_handler,
    make_server,
)


def _start(**kwargs):
    server = make_server(Monitor([], []), host="127.0.0.1", port=0, **kwargs)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, int(server.server_address[1])


def _post_raw(port: int, header_value: str | None, body: bytes, path: str = "/events"):
    """Send one POST with a verbatim Content-Length header, return the reply.

    The header is written by hand rather than through a client library
    precisely because no client library will emit an invalid one.
    """
    sock = socket.create_connection(("127.0.0.1", port), timeout=10)
    try:
        head = f"POST {path} HTTP/1.0\r\nHost: localhost\r\n"
        if header_value is not None:
            head += f"Content-Length: {header_value}\r\n"
        head += "\r\n"
        sock.sendall(head.encode())
        if body:
            sock.sendall(body)
        chunks: list[bytes] = []
        while True:
            data = sock.recv(65536)
            if not data:
                break
            chunks.append(data)
        return b"".join(chunks)
    finally:
        sock.close()


def test_non_numeric_content_length_gets_400():
    server, port = _start()
    try:
        response = _post_raw(port, "abc", b'{"step_id": "1"}')
    finally:
        server.shutdown()
        server.server_close()
    assert response.startswith(b"HTTP/1.0 400"), response[:120]
    assert b"invalid Content-Length" in response


def test_negative_content_length_gets_400():
    """Negatives never reach ``rfile.read(-n)``, which raises its own error."""
    server, port = _start()
    try:
        response = _post_raw(port, "-5", b'{"step_id": "1"}')
    finally:
        server.shutdown()
        server.server_close()
    assert response.startswith(b"HTTP/1.0 400"), response[:120]
    assert b"invalid Content-Length" in response


def test_bad_content_length_is_rejected_before_auth_and_routing():
    """Framing is settled first: no drain, no route dispatch, no 401/404.

    Every later branch reads or drains the body, and none of them can know
    where an unparseable body ends -- so an authenticated sidecar and an
    unknown path both still answer 400 here.
    """
    server, port = _start(auth_token="secret")
    try:
        unauthorized = _post_raw(port, "abc", b"")
        unknown_path = _post_raw(port, "abc", b"", path="/nope")
    finally:
        server.shutdown()
        server.server_close()
    assert unauthorized.startswith(b"HTTP/1.0 400"), unauthorized[:120]
    assert b"unauthorized" not in unauthorized
    assert unknown_path.startswith(b"HTTP/1.0 400"), unknown_path[:120]


def test_no_traceback_escapes_the_handler(capfd):
    """socketserver prints to stderr when a handler raises; it must not."""
    server, port = _start()
    try:
        for value in ("abc", "0x10", "-5", "1e3", "1 2"):
            response = _post_raw(port, value, b"x")
            assert response.startswith(b"HTTP/1.0 400"), (value, response[:120])
    finally:
        server.shutdown()
        server.server_close()
    captured = capfd.readouterr()
    assert "Traceback" not in captured.err, captured.err
    assert "ValueError" not in captured.err, captured.err


def test_absent_content_length_still_reads_as_empty_body():
    """No header means length 0, so /events answers on the JSON parse."""
    server, port = _start()
    try:
        response = _post_raw(port, None, b"")
    finally:
        server.shutdown()
        server.server_close()
    assert response.startswith(b"HTTP/1.0 400"), response[:120]
    assert b"invalid StepEvent JSON" in response


def test_well_formed_content_length_is_unaffected():
    event = {
        "step_id": "0",
        "episode_id": "ep-cl",
        "timestamp": 1718300000.0,
        "action_type": "tool_call",
        "action_signature": "aaaa1111bbbb2222",
    }
    raw = json.dumps(event).encode()
    server, port = _start()
    try:
        response = _post_raw(port, str(len(raw)), raw)
    finally:
        server.shutdown()
        server.server_close()
    assert response.startswith(b"HTTP/1.0 202"), response[:120]
    assert b'"status": "ingested"' in response


def test_content_length_helper_classifies_values():
    """Unit-level table for the shared parse helper."""
    handler_cls = make_handler(Monitor([], []))

    def parse(value):
        headers = {} if value is None else {"Content-Length": value}
        fake = types.SimpleNamespace(headers=headers)
        return handler_cls._content_length(fake)

    assert parse(None) == 0  # absent header: no body
    assert parse("") == 0
    assert parse("   ") == 0
    assert parse("0") == 0
    assert parse("42") == 42
    assert parse(" 42 ") == 42  # int() tolerates surrounding whitespace
    assert parse("abc") is None
    assert parse("0x10") is None
    assert parse("1e3") is None
    assert parse("-1") is None  # would otherwise become read(-1): read to EOF


def test_read_body_returns_empty_on_unusable_length():
    """The shared helper keeps _read_body from raising if reached directly."""
    handler_cls = make_handler(Monitor([], []))

    class _Unused:
        def read(self, n: int) -> bytes:  # pragma: no cover - must not be called
            raise AssertionError("body must not be read for bad framing")

    fake = types.SimpleNamespace(headers={"Content-Length": "abc"}, rfile=_Unused())
    fake._content_length = lambda: handler_cls._content_length(fake)
    assert handler_cls._read_body(fake) == b""


def test_bad_framing_drain_releases_a_stalled_sender():
    """The drain is time-bounded: a silent sender cannot pin the handler.

    Nothing is written after the headers, so there is no body for the drain
    to find; the handler must still finish within the deadline (plus slack
    for scheduling) rather than blocking on a read that never completes.
    """
    server, port = _start()
    try:
        sock = socket.create_connection(("127.0.0.1", port), timeout=10)
        try:
            sock.sendall(
                b"POST /events HTTP/1.0\r\n"
                b"Host: localhost\r\n"
                b"Content-Length: abc\r\n"
                b"\r\n"
            )
            started = time.perf_counter()
            chunks: list[bytes] = []
            while True:
                data = sock.recv(65536)
                if not data:
                    break
                chunks.append(data)
            elapsed = time.perf_counter() - started
            response = b"".join(chunks)
        finally:
            sock.close()
    finally:
        server.shutdown()
        server.server_close()
    assert response.startswith(b"HTTP/1.0 400"), response[:120]
    # Generous ceiling: the point is that it is bounded at all, not the exact
    # figure, which is why the assertion is not a tight one (CI runners are
    # noisy).
    assert elapsed < _BAD_FRAMING_DRAIN_SECONDS + 5.0, elapsed


def test_bad_framing_drain_is_byte_bounded_and_error_tolerant():
    """Unit level: the drain stops at the cap and swallows socket errors."""
    handler_cls = make_handler(Monitor([], []), max_body_bytes=1_000)
    budget = 1_000 + _MAX_OVERCAP_DRAIN_EXCESS

    class _EndlessConnection:
        def __init__(self) -> None:
            self.consumed = 0
            self.reads: list[int] = []
            self.timeouts: list[float | None] = []

        def gettimeout(self) -> float | None:
            return None

        def settimeout(self, value: float | None) -> None:
            self.timeouts.append(value)

        def recv(self, n: int) -> bytes:
            self.reads.append(n)
            self.consumed += n
            return b"x" * n

    connection = _EndlessConnection()
    fake = types.SimpleNamespace(
        connection=connection,
        wfile=types.SimpleNamespace(flush=lambda: None),
        snagline_max_body=1_000,
    )
    handler_cls._drain_unknown_body(fake)
    assert connection.consumed == budget
    assert max(connection.reads) <= _DRAIN_CHUNK_BYTES
    assert connection.timeouts[-1] is None  # original timeout restored

    class _ResetConnection(_EndlessConnection):
        def recv(self, n: int) -> bytes:
            raise ConnectionResetError("peer reset before the 400 was read")

    connection = _ResetConnection()
    fake = types.SimpleNamespace(
        connection=connection,
        wfile=types.SimpleNamespace(flush=lambda: None),
        snagline_max_body=1_000,
    )
    handler_cls._drain_unknown_body(fake)  # must not raise
    assert connection.timeouts[-1] is None


def test_bad_framing_drain_stops_on_hangup_and_on_a_spent_deadline(monkeypatch):
    """The two quiet exits: peer EOF, and the wall-clock budget running out."""
    handler_cls = make_handler(Monitor([], []), max_body_bytes=1_000)

    class _Connection:
        def __init__(self, chunks):
            self.chunks = list(chunks)
            self.timeouts: list[float | None] = []

        def gettimeout(self) -> float | None:
            return 7.5  # a distinctive prior value, to prove it is restored

        def settimeout(self, value: float | None) -> None:
            self.timeouts.append(value)

        def recv(self, n: int) -> bytes:
            return self.chunks.pop(0) if self.chunks else b""

    def drain(connection):
        fake = types.SimpleNamespace(
            connection=connection,
            wfile=types.SimpleNamespace(flush=lambda: None),
            snagline_max_body=1_000,
        )
        handler_cls._drain_unknown_body(fake)

    hung_up = _Connection([b"partial", b""])
    drain(hung_up)  # stops at EOF well inside both budgets
    assert hung_up.timeouts[-1] == 7.5

    # A zero-length deadline: the loop must give up before its first recv
    # rather than reading with a non-positive timeout.
    monkeypatch.setattr(http_server, "_BAD_FRAMING_DRAIN_SECONDS", 0.0)
    spent = _Connection([b"x" * _DRAIN_CHUNK_BYTES])
    drain(spent)
    assert spent.timeouts == [7.5]  # restore only: no read timeout was ever set
