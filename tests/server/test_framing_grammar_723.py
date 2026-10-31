"""Regression tests for RFC 9110 / RFC 9112 framing grammar (#723).

Content-Length and chunk-size parsers previously accepted Python-style + and _
numerals via bare int(), causing invalid framing to be accepted or hanging
handler threads instead of answering 400 immediately.
"""

from __future__ import annotations

import json
import socket
import threading
from typing import Any

import pytest

from snagline.monitor import Monitor
from snagline.server.http_server import make_server

CRLF = b"\r\n"


def _start_server(monitor: Monitor | None = None) -> tuple[Any, int, Monitor]:
    """Start sidecar on ephemeral port; returns (server, port, monitor)."""
    mon = monitor or Monitor([], [])
    server = make_server(mon, host="127.0.0.1", port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, int(server.server_address[1]), mon


def _raw_request(port: int, raw: bytes, timeout: float = 3.0) -> bytes:
    """Send one raw request and return the full response bytes."""
    sock = socket.create_connection(("127.0.0.1", port), timeout=5)
    try:
        sock.sendall(raw)
        chunks: list[bytes] = []
        sock.settimeout(timeout)
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


def _valid_body() -> bytes:
    return json.dumps(
        {
            "step_id": "s1",
            "episode_id": "ep",
            "timestamp": 1.0,
            "action_type": "tool_call",
            "action_signature": "a" * 64,
        }
    ).encode("utf-8")


@pytest.mark.parametrize(
    "bad_cl",
    [
        b"+171",
        b"+0",
        b"-1",
        b"-171",
        b"1_170",
        b"1_0",
        b"_171",
        b"12.34",
        b"abc",
        b"",
        "\u0661\u0667\u0661".encode("utf-8"),  # Unicode Arabic-Indic digits
    ],
)
def test_content_length_rejects_signs_underscores_and_non_digits(bad_cl: bytes) -> None:
    """RFC 9110 §8.6 pins Content-Length = 1*DIGIT."""
    server, port, mon = _start_server()
    try:
        body = _valid_body()
        req = (
            b"POST /events HTTP/1.1\r\n"
            b"Host: 127.0.0.1\r\n"
            b"Content-Type: application/json\r\n"
            b"Connection: close\r\n"
            b"Content-Length: " + bad_cl + b"\r\n\r\n" + body
        )
        resp = _raw_request(port, req)
        assert resp.startswith(b"HTTP/1.0 400") or resp.startswith(b"HTTP/1.1 400"), (
            bad_cl,
            resp,
        )
        assert b"invalid Content-Length" in resp, (bad_cl, resp)
        assert mon.metrics()["events_ingested"] == 0
    finally:
        server.shutdown()
        server.server_close()


@pytest.mark.parametrize(
    "bad_chunk_size",
    [
        b"+ab",
        b"+10",
        b"-1a",
        b"1_ab",
        b"1_0",
        b"_ab",
        b"zz",
        b"",
    ],
)
def test_chunk_size_rejects_signs_underscores_and_non_hexdig(
    bad_chunk_size: bytes,
) -> None:
    """RFC 9112 §7.1 pins chunk-size = 1*HEXDIG."""
    server, port, mon = _start_server()
    try:
        body = _valid_body()
        req = (
            b"POST /events HTTP/1.1\r\n"
            b"Host: 127.0.0.1\r\n"
            b"Content-Type: application/json\r\n"
            b"Transfer-Encoding: chunked\r\n"
            b"Connection: close\r\n\r\n"
            + bad_chunk_size
            + CRLF
            + body
            + CRLF
            + b"0\r\n\r\n"
        )
        resp = _raw_request(port, req)
        assert resp.startswith(b"HTTP/1.0 400") or resp.startswith(b"HTTP/1.1 400"), (
            bad_chunk_size,
            resp,
        )
        assert b"invalid chunk size" in resp, (bad_chunk_size, resp)
        assert mon.metrics()["events_ingested"] == 0
    finally:
        server.shutdown()
        server.server_close()


def test_chunk_size_with_extensions_and_casing_accepted() -> None:
    """Valid hex chunk sizes with extensions and uppercase letters are accepted."""
    server, port, mon = _start_server()
    try:
        body = _valid_body()
        hex_len = f"{len(body):X}".encode("ascii")  # uppercase hex
        req = (
            b"POST /events HTTP/1.1\r\n"
            b"Host: 127.0.0.1\r\n"
            b"Content-Type: application/json\r\n"
            b"Transfer-Encoding: chunked\r\n"
            b"Connection: close\r\n\r\n"
            + hex_len
            + b";ext=val\r\n"
            + body
            + CRLF
            + b"0\r\n\r\n"
        )
        resp = _raw_request(port, req)
        assert b" 202" in resp.split(CRLF)[0]
        assert mon.metrics()["events_ingested"] == 1
    finally:
        server.shutdown()
        server.server_close()


def test_issue_723_repro_does_not_hang_or_ingest_short_reads() -> None:
    """Direct reproduction of issue #723.

    Non-conforming numerals must answer 400 immediately, never hang, and
    never ingest phantom bodies when the client disconnects.
    """
    server, port, mon = _start_server()
    try:
        body = _valid_body()
        body_len = str(len(body)).encode("ascii")
        hex_size = format(len(body), "x").encode("ascii")

        # 1. Content-Length control
        req = (
            b"POST /events HTTP/1.1\r\nHost: h\r\n"
            b"Content-Type: application/json\r\n"
            b"Content-Length: " + body_len + b"\r\n\r\n" + body
        )
        r = _raw_request(port, req, timeout=1.0)
        assert b" 202" in r.split(CRLF)[0]

        # 2. Content-Length: +171
        req = (
            b"POST /events HTTP/1.1\r\nHost: h\r\n"
            b"Content-Type: application/json\r\n"
            b"Content-Length: +" + body_len + b"\r\n\r\n" + body
        )
        r = _raw_request(port, req, timeout=1.0)
        assert b" 400" in r.split(CRLF)[0]
        assert b"invalid Content-Length" in r

        # 3. Content-Length: 1_170
        req = (
            b"POST /events HTTP/1.1\r\nHost: h\r\n"
            b"Content-Type: application/json\r\n"
            b"Content-Length: 1_170\r\n\r\n" + body
        )
        r = _raw_request(port, req, timeout=1.0)
        assert b" 400" in r.split(CRLF)[0]
        assert b"invalid Content-Length" in r

        # 4. Chunk size control
        chead = (
            b"POST /events HTTP/1.1\r\nHost: h\r\n"
            b"Content-Type: application/json\r\n"
            b"Transfer-Encoding: chunked\r\n\r\n"
        )
        req = chead + hex_size + CRLF + body + CRLF + b"0" + CRLF + CRLF
        r = _raw_request(port, req, timeout=1.0)
        assert b" 202" in r.split(CRLF)[0]

        # 5. Chunk size 1_ab
        req = chead + b"1_ab" + CRLF + body + CRLF + b"0" + CRLF + CRLF
        r = _raw_request(port, req, timeout=1.0)
        assert b" 400" in r.split(CRLF)[0]
        assert b"invalid chunk size" in r

        # 6. Chunk size zz control
        req = chead + b"zz" + CRLF + body
        r = _raw_request(port, req, timeout=1.0)
        assert b" 400" in r.split(CRLF)[0]
        assert b"invalid chunk size" in r

        # Exactly 2 ingested (the two controls), not 4
        assert mon.metrics()["events_ingested"] == 2
    finally:
        server.shutdown()
        server.server_close()


def test_drain_chunked_rest_aborts_on_invalid_size_framing() -> None:
    """_drain_chunked_rest must stop immediately on non-HEXDIG chunk sizes."""
    import io
    import types

    from snagline.server.http_server import make_handler

    handler_cls = make_handler(Monitor([], []))
    stream = io.BytesIO(b"12345\r\n1_ab\r\npayload\r\n0\r\n\r\n")
    fake = types.SimpleNamespace(rfile=stream)
    handler_cls._drain_chunked_rest(fake, pending=5)  # type: ignore[attr-defined]
    # 5 pending bytes + CRLF (7) + readline 1_ab\r\n (6) = 13 bytes
    assert stream.tell() == len(b"12345\r\n1_ab\r\n")
