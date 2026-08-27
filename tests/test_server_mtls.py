"""Optional mTLS knob (--client-ca) for the TLS sidecar listener (issue #145).

All TLS/mTLS handshakes run against an ephemeral loopback port with keys and
certs generated at test time via the ``openssl`` binary. The plain and
plain+server-TLS modes are regression-checked byte-for-byte when --client-ca
is absent.
"""

from __future__ import annotations

import json
import shutil
import ssl
import subprocess
import threading
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from snagline.monitor import Monitor
from snagline.server.http_server import _resolve_ssl_context, make_server

requires_openssl = pytest.mark.skipif(
    shutil.which("openssl") is None, reason="openssl binary not available"
)

_EVENT = {
    "step_id": "1",
    "episode_id": "run",
    "timestamp": 1735689600.0,
    "action_type": "tool_call",
    "action_signature": "deadbeefdeadbeef",
}


def _generate_ca(directory: Path, name: str = "ca") -> tuple[str, str]:
    """Create a self-signed CA; return (ca_cert, ca_key)."""
    ca_key = str(directory / f"{name}.key")
    ca_cert = str(directory / f"{name}.pem")
    subprocess.run(
        [
            "openssl",
            "req",
            "-x509",
            "-newkey",
            "rsa:2048",
            "-keyout",
            ca_key,
            "-out",
            ca_cert,
            "-days",
            "2",
            "-nodes",
            "-subj",
            f"/CN={name}",
        ],
        check=True,
        capture_output=True,
        timeout=60,
    )
    return ca_cert, ca_key


def _generate_signed_cert(
    directory: Path,
    ca_cert: str,
    ca_key: str,
    cn: str,
    name: str,
) -> tuple[str, str]:
    """Create a key and cert signed by the given CA; return (cert, key)."""
    key_path = str(directory / f"{name}.key")
    csr_path = str(directory / f"{name}.csr")
    cert_path = str(directory / f"{name}.crt")
    subprocess.run(
        [
            "openssl",
            "req",
            "-newkey",
            "rsa:2048",
            "-nodes",
            "-keyout",
            key_path,
            "-out",
            csr_path,
            "-subj",
            f"/CN={cn}",
        ],
        check=True,
        capture_output=True,
        timeout=60,
    )
    subprocess.run(
        [
            "openssl",
            "x509",
            "-req",
            "-in",
            csr_path,
            "-CA",
            ca_cert,
            "-CAkey",
            ca_key,
            "-CAcreateserial",
            "-out",
            cert_path,
            "-days",
            "2",
        ],
        check=True,
        capture_output=True,
        timeout=60,
    )
    return cert_path, key_path


def _insecure_client_context() -> ssl.SSLContext:
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return ctx


def _mtls_client_context(
    ca_cert: str | None = None,
    client_cert: str | None = None,
    client_key: str | None = None,
) -> ssl.SSLContext:
    """Client context that optionally verifies server and presents a client cert."""
    ctx = (
        ssl.create_default_context(cafile=ca_cert)
        if ca_cert
        else ssl.create_default_context()
    )
    # For test simplicity skip server verification unless a CA is given explicitly
    # to verify the server. Most mTLS tests only care about server verifying
    # the client, so we keep server verification off when ca_cert is None.
    if ca_cert is None:
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
    else:
        ctx.check_hostname = False
        # keep default verification against ca_cert
        ctx.verify_mode = ssl.CERT_REQUIRED
    if client_cert and client_key:
        ctx.load_cert_chain(client_cert, client_key)
    elif client_cert:
        ctx.load_cert_chain(client_cert)
    return ctx


def _request_with_context(base: str, path: str, context: ssl.SSLContext, **kw):
    req = urllib.request.Request(base + path, **kw)
    try:
        with urllib.request.urlopen(req, timeout=5, context=context) as resp:
            return int(resp.status), resp.read()
    except urllib.error.HTTPError as exc:
        return int(exc.code), exc.read()
    except (urllib.error.URLError, ssl.SSLError, OSError):
        # Handshake failures surface here when client cert is missing or untrusted.
        raise


# ---------------------------------------------------------------------------
# _resolve_ssl_context unit tests (no network)
# ---------------------------------------------------------------------------


def test_client_ca_requires_certfile():
    with pytest.raises(ValueError, match="client-ca requires certfile"):
        _resolve_ssl_context(None, None, None, client_ca="/tmp/ca.pem")


def test_client_ca_with_ssl_context_is_rejected(tmp_path):
    cert, key = (
        _generate_ca(tmp_path) if shutil.which("openssl") else ("/tmp/x", "/tmp/y")
    )
    # Use a dummy context to test rejection logic without needing real certs
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    with pytest.raises(ValueError, match="not both"):
        _resolve_ssl_context(ctx, None, None, client_ca="/tmp/ca.pem")


def test_resolve_sets_verify_mode_only_when_client_ca_given(tmp_path):
    ca_cert, ca_key = _generate_ca(tmp_path)
    server_cert, server_key = _generate_signed_cert(
        tmp_path, ca_cert, ca_key, "127.0.0.1", "server"
    )
    # Without client_ca: verify_mode stays CERT_NONE (default)
    ctx_plain = _resolve_ssl_context(None, server_cert, server_key, None)
    assert ctx_plain is not None
    assert ctx_plain.verify_mode == ssl.CERT_NONE
    # With client_ca: verify_mode becomes CERT_REQUIRED and CA is loaded
    ctx_mtls = _resolve_ssl_context(None, server_cert, server_key, ca_cert)
    assert ctx_mtls is not None
    assert ctx_mtls.verify_mode == ssl.CERT_REQUIRED


def test_plain_and_server_tls_modes_unchanged_without_client_ca(tmp_path):
    # Byte-for-byte unchanged: plain mode still returns None, server-TLS still
    # returns a context with CERT_NONE and no client verification.
    assert _resolve_ssl_context(None, None, None) is None
    assert _resolve_ssl_context(None, None, None, None) is None
    ca_cert, ca_key = _generate_ca(tmp_path)
    server_cert, server_key = _generate_signed_cert(
        tmp_path, ca_cert, ca_key, "127.0.0.1", "server"
    )
    ctx = _resolve_ssl_context(None, server_cert, server_key)
    assert ctx is not None
    assert ctx.verify_mode == ssl.CERT_NONE
    # Explicit None client_ca must behave identically
    ctx2 = _resolve_ssl_context(None, server_cert, server_key, None)
    assert ctx2 is not None
    assert ctx2.verify_mode == ssl.CERT_NONE


# ---------------------------------------------------------------------------
# CLI forwarding tests
# ---------------------------------------------------------------------------


def test_cli_serve_forwards_client_ca(monkeypatch):
    from snagline.cli import main

    captured: dict = {}

    def _fake_serve(monitor, **kwargs):
        captured.update(kwargs)

    monkeypatch.setattr("snagline.server.http_server.serve", _fake_serve)
    assert (
        main(
            [
                "serve",
                "--port",
                "0",
                "--certfile",
                "/t/c.pem",
                "--keyfile",
                "/t/k.pem",
                "--client-ca",
                "/t/ca.pem",
            ]
        )
        == 0
    )
    assert captured["certfile"] == "/t/c.pem"
    assert captured["keyfile"] == "/t/k.pem"
    assert captured["client_ca"] == "/t/ca.pem"


def test_cli_serve_without_client_ca_passes_none_or_absent(monkeypatch):
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
    # When not given, client_ca is None (explicit) or absent; both are plain server-TLS
    assert captured.get("client_ca") is None


def test_cli_serve_rejects_bare_client_ca(monkeypatch):
    from snagline.cli import main

    def _fake_serve(monitor, **kwargs):
        raise AssertionError("serve() must not be called for bare --client-ca")

    monkeypatch.setattr("snagline.server.http_server.serve", _fake_serve)
    assert main(["serve", "--port", "0", "--client-ca", "/t/ca.pem"]) == 2


def test_cli_plain_mode_still_passes_no_tls_kwargs(monkeypatch):
    from snagline.cli import main

    captured: dict = {}

    def _fake_serve(monitor, **kwargs):
        captured.update(kwargs)

    monkeypatch.setattr("snagline.server.http_server.serve", _fake_serve)
    assert main(["serve", "--port", "0"]) == 0
    assert "certfile" not in captured
    assert "keyfile" not in captured
    # client_ca must not appear or be None in plain mode
    assert captured.get("client_ca") is None or "client_ca" not in captured


# ---------------------------------------------------------------------------
# mTLS handshake tests against ephemeral loopback port
# ---------------------------------------------------------------------------


@requires_openssl
def test_mtls_handshake_succeeds_with_trusted_client_cert(tmp_path):
    ca_cert, ca_key = _generate_ca(tmp_path, "ca1")
    server_cert, server_key = _generate_signed_cert(
        tmp_path, ca_cert, ca_key, "127.0.0.1", "server"
    )
    client_cert, client_key = _generate_signed_cert(
        tmp_path, ca_cert, ca_key, "client", "client"
    )

    server = make_server(
        Monitor.default(),
        host="127.0.0.1",
        port=0,
        certfile=server_cert,
        keyfile=server_key,
        client_ca=ca_cert,
    )
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        base = f"https://127.0.0.1:{server.server_address[1]}"
        ctx = _mtls_client_context(client_cert=client_cert, client_key=client_key)
        status, body = _request_with_context(base, "/health", ctx, method="GET")
        assert status == 200
        assert json.loads(body) == {"status": "ok"}

        # Also verify POST /events still works with mTLS
        event_body = json.dumps(_EVENT).encode("utf-8")
        status2, body2 = _request_with_context(
            base,
            "/events",
            ctx,
            method="POST",
            data=event_body,
            headers={"Content-Type": "application/json"},
        )
        assert status2 == 202
        assert json.loads(body2)["status"] == "ingested"
    finally:
        server.shutdown()
        server.server_close()


@requires_openssl
def test_mtls_handshake_fails_without_client_cert(tmp_path):
    ca_cert, ca_key = _generate_ca(tmp_path, "ca1")
    server_cert, server_key = _generate_signed_cert(
        tmp_path, ca_cert, ca_key, "127.0.0.1", "server"
    )

    server = make_server(
        Monitor.default(),
        host="127.0.0.1",
        port=0,
        certfile=server_cert,
        keyfile=server_key,
        client_ca=ca_cert,
    )
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        base = f"https://127.0.0.1:{server.server_address[1]}"
        # No client cert presented; server requires one
        ctx = _insecure_client_context()
        with pytest.raises((urllib.error.URLError, ssl.SSLError, OSError)):
            _request_with_context(base, "/health", ctx, method="GET")

        # Server must still be alive for a trusted client after the failed handshake
        client_cert, client_key = _generate_signed_cert(
            tmp_path, ca_cert, ca_key, "client2", "client2"
        )
        ctx2 = _mtls_client_context(client_cert=client_cert, client_key=client_key)
        status, body = _request_with_context(base, "/health", ctx2, method="GET")
        assert status == 200
        assert json.loads(body) == {"status": "ok"}
    finally:
        server.shutdown()
        server.server_close()


@requires_openssl
def test_mtls_handshake_fails_with_untrusted_client_cert(tmp_path):
    ca_cert, ca_key = _generate_ca(tmp_path, "ca1")
    server_cert, server_key = _generate_signed_cert(
        tmp_path, ca_cert, ca_key, "127.0.0.1", "server"
    )
    # Untrusted CA and client cert signed by it
    ca2_cert, ca2_key = _generate_ca(tmp_path, "ca2")
    bad_client_cert, bad_client_key = _generate_signed_cert(
        tmp_path, ca2_cert, ca2_key, "badclient", "badclient"
    )

    server = make_server(
        Monitor.default(),
        host="127.0.0.1",
        port=0,
        certfile=server_cert,
        keyfile=server_key,
        client_ca=ca_cert,
    )
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        base = f"https://127.0.0.1:{server.server_address[1]}"
        ctx_bad = _mtls_client_context(
            client_cert=bad_client_cert, client_key=bad_client_key
        )
        with pytest.raises((urllib.error.URLError, ssl.SSLError, OSError)):
            _request_with_context(base, "/health", ctx_bad, method="GET")

        # Trusted client still succeeds on same server
        good_client_cert, good_client_key = _generate_signed_cert(
            tmp_path, ca_cert, ca_key, "good", "good"
        )
        ctx_good = _mtls_client_context(
            client_cert=good_client_cert, client_key=good_client_key
        )
        status, body = _request_with_context(base, "/health", ctx_good, method="GET")
        assert status == 200
        assert json.loads(body) == {"status": "ok"}
    finally:
        server.shutdown()
        server.server_close()


@requires_openssl
def test_plain_and_server_tls_modes_regression(tmp_path):
    # Plain mode must stay plain HTTP
    plain = make_server(Monitor.default(), host="127.0.0.1", port=0)
    threading.Thread(target=plain.serve_forever, daemon=True).start()
    try:
        assert not hasattr(plain, "snagline_ssl_context")
        base = f"http://127.0.0.1:{plain.server_address[1]}"
        with urllib.request.urlopen(base + "/health", timeout=5) as resp:
            assert resp.status == 200
    finally:
        plain.shutdown()
        plain.server_close()

    # Server-TLS without client_ca must not require a client cert
    ca_cert, ca_key = _generate_ca(tmp_path, "ca1")
    server_cert, server_key = _generate_signed_cert(
        tmp_path, ca_cert, ca_key, "127.0.0.1", "server"
    )
    tls_plain = make_server(
        Monitor.default(),
        host="127.0.0.1",
        port=0,
        certfile=server_cert,
        keyfile=server_key,
    )
    threading.Thread(target=tls_plain.serve_forever, daemon=True).start()
    try:
        # No client cert required; insecure context succeeds
        base2 = f"https://127.0.0.1:{tls_plain.server_address[1]}"
        ctx = _insecure_client_context()
        status, body = _request_with_context(base2, "/health", ctx, method="GET")
        assert status == 200
        assert json.loads(body) == {"status": "ok"}
        # Verify server context is not in mTLS mode
        assert tls_plain.snagline_ssl_context.verify_mode == ssl.CERT_NONE  # type: ignore[attr-defined]
    finally:
        tls_plain.shutdown()
        tls_plain.server_close()
