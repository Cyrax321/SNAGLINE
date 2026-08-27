"""Malformed Content-Length must get 400, not a dropped connection (#129).

Before the fix, ``int(self.headers.get(\"Content-Length\") or 0)`` raised
``ValueError`` on ``Content-Length: abc``, the handler printed a traceback
and the client saw an empty reply (connection dropped). The helper
``_parse_content_length`` now catches the error and answers 400 without
reading any body bytes, shared by ``do_POST`` and ``_read_body``.
"""

from __future__ import annotations

import socket
import threading
import types

from snagline.monitor import Monitor
from snagline.server.http_server import make_handler, make_server


def _start(auth_token: str | None = None):
    """Start a sidecar on an ephemeral port; returns (server, port)."""
    server = make_server(
        Monitor([], []),
        host="127.0.0.1",
        port=0,
        auth_token=auth_token,
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, int(server.server_address[1])


def _raw_request(port: int, raw: bytes) -> bytes:
    """Send one raw request and return the full wire response."""
    sock = socket.create_connection(("127.0.0.1", port), timeout=10)
    try:
        sock.sendall(raw)
        chunks: list[bytes] = []
        sock.settimeout(5)
        while True:
            try:
                data = sock.recv(65536)
            except TimeoutError:
                break
            if not data:
                break
            chunks.append(data)
        return b"".join(chunks)
    finally:
        sock.close()


def test_malformed_content_length_gets_400_not_dropped_connection():
    """Client must receive a 400 status line, not an empty reply."""
    server, port = _start()
    try:
        for bad in [b"abc", b"-5", b"12.34", b""]:
            raw = (
                b"POST /events HTTP/1.0\r\n"
                b"Host: localhost\r\n"
                b"Content-Type: application/json\r\n"
                b"Content-Length: " + bad + b"\r\n"
                b"\r\n"
                b"{}"
            )
            response = _raw_request(port, raw)
            assert response.startswith(b"HTTP/1.0 400"), (bad, response[:200])
            assert b"invalid Content-Length" in response, (bad, response[:300])
        # Server still alive afterwards.
        ok = _raw_request(
            port,
            b"GET /health HTTP/1.0\r\nHost: localhost\r\n\r\n",
        )
        assert ok.startswith(b"HTTP/1.0 200"), ok[:120]
    finally:
        server.shutdown()
        server.server_close()


def test_malformed_length_on_any_post_path_is_400():
    """The helper is shared, so every POST path sees the same 400."""
    server, port = _start()
    try:
        for path in [
            b"/events",
            b"/hooks/claude-code",
            b"/risks",
            b"/episodes/end",
            b"/unknown",
        ]:
            raw = (
                b"POST " + path + b" HTTP/1.0\r\n"
                b"Host: localhost\r\n"
                b"Content-Length: bad\r\n"
                b"\r\n"
            )
            response = _raw_request(port, raw)
            assert response.startswith(b"HTTP/1.0 400"), (path, response[:200])
            assert b"invalid Content-Length" in response
    finally:
        server.shutdown()
        server.server_close()


def test_absent_content_length_is_treated_as_zero():
    """Absent header means length 0; body is silently empty (docstring)."""
    server, port = _start()
    try:
        # No Content-Length at all, empty body -> _read_body reads 0 bytes,
        # then the endpoint sees empty JSON and answers 400 invalid JSON,
        # not 400 invalid Content-Length, and never crashes.
        raw = b"POST /events HTTP/1.0\r\nHost: localhost\r\n\r\n"
        response = _raw_request(port, raw)
        assert response.startswith(b"HTTP/1.0 400"), response[:200]
        assert b"invalid StepEvent JSON" in response
        # Also via the helper directly: absent header returns 0.
        handler_cls = make_handler(Monitor([], []))

        class _FakeHeaders(dict):
            def get(self, key, default=None):  # type: ignore[override]
                return None if key == "Content-Length" else super().get(key, default)

        fake = types.SimpleNamespace(headers=_FakeHeaders())
        # bind helper as method
        length = handler_cls._parse_content_length(fake)  # type: ignore[arg-type]
        assert length == 0
    finally:
        server.shutdown()
        server.server_close()


def test_malformed_length_does_not_block_subsequent_requests():
    """A bad request must not pin the handler thread."""
    server, port = _start()
    try:
        raw_bad = (
            b"POST /events HTTP/1.0\r\n"
            b"Host: localhost\r\n"
            b"Content-Length: not-a-number\r\n"
            b"\r\n"
            b"{}"
        )
        r1 = _raw_request(port, raw_bad)
        assert r1.startswith(b"HTTP/1.0 400")

        raw_ok = (
            b"POST /events HTTP/1.0\r\n"
            b"Host: localhost\r\n"
            b"Content-Type: application/json\r\n"
            b"Content-Length: 2\r\n"
            b"\r\n"
            b"{}"
        )
        r2 = _raw_request(port, raw_ok)
        # Empty dict {} is not a valid StepEvent (missing fields) -> 400 invalid JSON,
        # but the point is the server still responds cleanly, not dropped.
        assert r2.startswith(b"HTTP/1.0 400")
        assert b"invalid StepEvent" in r2
    finally:
        server.shutdown()
        server.server_close()


def test_parse_helper_docstring_mentions_absent_is_zero():
    """Guard the docstring contract required by the issue."""
    handler_cls = make_handler(Monitor([], []))
    doc = handler_cls._parse_content_length.__doc__ or ""  # type: ignore[attr-defined]
    assert "absent" in doc.lower()
    assert "0" in doc
