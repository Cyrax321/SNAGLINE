"""In-process TLS termination for the sidecar listener (issue #120).

Every HTTPS test uses a self-signed pair generated at test time with
subprocess ``openssl`` (stdlib has no certificate generator) against an
ephemeral loopback port. The client skips verification, which is exactly
what ``curl -k`` does in the documented acceptance check.
"""

from __future__ import annotations

import json
import shutil
import socket
import ssl
import subprocess
import threading
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from snagline.monitor import Monitor
from snagline.server.http_server import make_server

requires_openssl = pytest.mark.skipif(
    shutil.which("openssl") is None, reason="openssl binary not available"
)

# Minimal valid StepEvent JSON as accepted by POST /events.
_EVENT = {
    "step_id": "1",
    "episode_id": "run",
    "timestamp": 1735689600.0,
    "action_type": "tool_call",
    "action_signature": "deadbeefdeadbeef",
}


def _generate_self_signed_cert(directory: Path) -> tuple[str, str]:
    """Create a throwaway self-signed cert/key pair; return both paths."""
    cert_path = directory / "cert.pem"
    key_path = directory / "key.pem"
    subprocess.run(
        [
            "openssl",
            "req",
            "-x509",
            "-newkey",
            "rsa:2048",
            "-keyout",
            str(key_path),
            "-out",
            str(cert_path),
            "-days",
            "2",
            "-nodes",
            "-subj",
            "/CN=127.0.0.1",
        ],
        check=True,
        capture_output=True,
        timeout=60,
    )
    return str(cert_path), str(key_path)


def _insecure_client_context() -> ssl.SSLContext:
    """Client context equivalent to ``curl -k``: TLS on, verification off."""
    context = ssl.create_default_context()
    context.check_hostname = False
    context.verify_mode = ssl.CERT_NONE
    return context


def _start_tls_server(
    tmp_path: Path, auth_token: str | None = None, **kwargs
) -> tuple[object, str]:
    certfile, keyfile = _generate_self_signed_cert(tmp_path)
    server = make_server(
        Monitor.default(),
        host="127.0.0.1",
        port=0,
        auth_token=auth_token,
        certfile=certfile,
        keyfile=keyfile,
        **kwargs,
    )
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, f"https://127.0.0.1:{server.server_address[1]}"


def _request_tls(method: str, base: str, path: str, **kw):
    req = urllib.request.Request(base + path, method=method, **kw)
    try:
        with urllib.request.urlopen(
            req, timeout=5, context=_insecure_client_context()
        ) as resp:
            return int(resp.status), json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        return int(exc.code), json.loads(exc.read())


@requires_openssl
def test_tls_handshake_serves_health_over_https(tmp_path):
    server, base = _start_tls_server(tmp_path)
    try:
        # Listener stays raw; wrapping happens per connection in the worker
        # thread so one stalled handshake cannot block other senders.
        assert not isinstance(server.socket, ssl.SSLSocket)
        assert isinstance(server.snagline_ssl_context, ssl.SSLContext)
        status, body = _request_tls("GET", base, "/health")
        assert status == 200
        assert body == {"status": "ok"}
    finally:
        server.shutdown()
        server.server_close()


@requires_openssl
def test_stalled_handshake_does_not_block_other_connections(tmp_path):
    # One client opens a TCP connection and never advances the TLS handshake;
    # a second client must still be served because the handshake runs on the
    # stalled connection's own worker thread, not on the accept loop.
    server, base = _start_tls_server(tmp_path)
    host, port = server.server_address[0], server.server_address[1]
    stalled = socket.create_connection((host, port), timeout=5)
    try:
        status, body = _request_tls("GET", base, "/health")
        assert status == 200
        assert body == {"status": "ok"}
    finally:
        stalled.close()
        server.shutdown()
        server.server_close()


@requires_openssl
def test_plaintext_probe_is_not_served_and_server_survives(tmp_path):
    # Both sides: plaintext bytes to the TLS listener must never get an HTTP
    # answer, and the sidecar must keep serving proper TLS clients after.
    server, base = _start_tls_server(tmp_path)
    host, port = server.server_address[0], server.server_address[1]
    probe = socket.create_connection((host, port), timeout=5)
    try:
        probe.sendall(b"GET /health HTTP/1.0\r\n\r\n")
        probe.settimeout(5)
        reply = b""
        try:
            reply = probe.recv(1024)  # alert bytes or abrupt close: both fine
        except (ssl.SSLError, OSError):
            pass
        assert b"HTTP/1." not in reply  # no plaintext service on the TLS port
        probe.close()
        status, body = _request_tls("GET", base, "/health")
        assert status == 200
        assert body == {"status": "ok"}
    finally:
        probe.close()
        server.shutdown()
        server.server_close()


@requires_openssl
def test_malformed_content_length_gets_400_over_tls(tmp_path):
    """The #129 400 path also works when ``self.connection`` is an SSLSocket.

    Worth pinning separately: the bad-framing drain reads and re-times the
    raw connection object, which is a wrapped socket on this listener, and
    no client library will emit an invalid Content-Length for us.
    """
    server, _ = _start_tls_server(tmp_path)
    host, port = server.server_address[0], server.server_address[1]
    try:
        raw = socket.create_connection((host, port), timeout=10)
        client = _insecure_client_context().wrap_socket(raw)
        try:
            client.sendall(
                b"POST /events HTTP/1.0\r\n"
                b"Host: localhost\r\n"
                b"Content-Length: abc\r\n"
                b"\r\n"
            )
            client.sendall(b"x")  # unread body: the reset race the drain covers
            chunks = []
            while True:
                data = client.recv(65536)
                if not data:
                    break
                chunks.append(data)
            reply = b"".join(chunks)
        finally:
            client.close()
    finally:
        server.shutdown()
        server.server_close()
    assert reply.startswith(b"HTTP/1.0 400"), reply[:120]
    assert b"invalid Content-Length" in reply


@requires_openssl
def test_make_server_accepts_ready_ssl_context(tmp_path):
    certfile, keyfile = _generate_self_signed_cert(tmp_path)
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(certfile, keyfile)
    server = make_server(
        Monitor.default(),
        host="127.0.0.1",
        port=0,
        ssl_context=context,
    )
    # Start serving before asserting anything so a failed assertion still
    # tears the server down cleanly instead of deadlocking on shutdown().
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        assert not isinstance(server.socket, ssl.SSLSocket)
        assert isinstance(context, ssl.SSLContext)
        base = f"https://127.0.0.1:{server.server_address[1]}"
        status, body = _request_tls("GET", base, "/health")
        assert status == 200
        assert body == {"status": "ok"}
    finally:
        server.shutdown()
        server.server_close()


@requires_openssl
def test_auth_is_enforced_over_the_tls_listener(tmp_path):
    # Both sides: without a token every protected endpoint returns 401, with
    # the correct bearer token ingestion works, and GET /health stays open.
    server, base = _start_tls_server(tmp_path, auth_token="s3cret")
    try:
        no_token_status, _ = _request_tls("GET", base, "/risks")
        assert no_token_status == 401

        event_body = json.dumps(_EVENT).encode("utf-8")
        rejected = _request_tls(
            "POST",
            base,
            "/events",
            data=event_body,
            headers={"Content-Type": "application/json"},
        )
        assert rejected[0] == 401

        accepted = _request_tls(
            "POST",
            base,
            "/events",
            data=event_body,
            headers={
                "Content-Type": "application/json",
                "Authorization": "Bearer s3cret",
            },
        )
        assert accepted[0] == 202
        assert accepted[1]["status"] == "ingested"

        health_status, health_body = _request_tls("GET", base, "/health")
        assert health_status == 200
        assert health_body == {"status": "ok"}
    finally:
        server.shutdown()
        server.server_close()


def test_plain_mode_is_untouched_when_no_tls_arguments_are_given():
    server = make_server(Monitor.default(), host="127.0.0.1", port=0)
    # Start serving before asserting anything so a failed assertion still
    # tears the server down cleanly instead of deadlocking on shutdown().
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        assert not isinstance(server.socket, ssl.SSLSocket)
        base = f"http://127.0.0.1:{server.server_address[1]}"
        with urllib.request.urlopen(base + "/health", timeout=5) as resp:
            assert resp.status == 200
            assert json.loads(resp.read()) == {"status": "ok"}
    finally:
        server.shutdown()
        server.server_close()


def test_keyfile_without_certfile_is_rejected_before_binding():
    with pytest.raises(ValueError, match="keyfile requires certfile"):
        make_server(Monitor.default(), host="127.0.0.1", port=0, keyfile="/x/k.pem")


@requires_openssl
def test_ssl_context_and_certfile_together_are_rejected(tmp_path):
    certfile, keyfile = _generate_self_signed_cert(tmp_path)
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(certfile, keyfile)
    with pytest.raises(ValueError, match="not both"):
        make_server(
            Monitor.default(),
            host="127.0.0.1",
            port=0,
            ssl_context=context,
            certfile=certfile,
        )


@requires_openssl
def test_unloadable_certfile_raises_at_startup(tmp_path):
    bad = tmp_path / "not-a-cert.pem"
    bad.write_text("this is not a certificate\n")
    with pytest.raises((ssl.SSLError, OSError)):
        make_server(Monitor.default(), host="127.0.0.1", port=0, certfile=str(bad))


def test_cli_serve_forwards_tls_flags(monkeypatch):
    from snagline.cli import main

    captured: dict = {}

    def _fake_serve(monitor, **kwargs):
        captured.update(kwargs)

    monkeypatch.setattr("snagline.server.http_server.serve", _fake_serve)
    assert (
        main(
            ["serve", "--port", "0", "--certfile", "/t/c.pem", "--keyfile", "/t/k.pem"]
        )
        == 0
    )
    assert captured["certfile"] == "/t/c.pem"
    assert captured["keyfile"] == "/t/k.pem"


def test_cli_serve_without_tls_flags_passes_none(monkeypatch):
    # Plain-mode regression on the CLI path: with absent flags serve() is
    # called without any TLS kwargs at all, byte-for-byte the old invocation.
    from snagline.cli import main

    captured: dict = {}

    def _fake_serve(monitor, **kwargs):
        captured.update(kwargs)

    monkeypatch.setattr("snagline.server.http_server.serve", _fake_serve)
    assert main(["serve", "--port", "0"]) == 0
    assert "certfile" not in captured
    assert "keyfile" not in captured


def test_cli_serve_rejects_bare_keyfile_before_starting(monkeypatch):
    # A bare --keyfile must fail fast with a clean refusal (exit 2), never
    # print the https banner and then die inside serve().
    from snagline.cli import main

    def _fake_serve(monitor, **kwargs):
        raise AssertionError("serve() must not be called for a bad TLS config")

    monkeypatch.setattr("snagline.server.http_server.serve", _fake_serve)
    assert main(["serve", "--port", "0", "--keyfile", "/t/k.pem"]) == 2
