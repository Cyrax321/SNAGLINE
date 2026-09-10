"""Server routing, batch atomicity, and serve-config layering.

Three related defects in the sidecar:

- Batched ``POST /events`` validated and ingested items in the same loop, so
  a bad item at position k answered 400 with items 0..k-1 already inside the
  Monitor -- a retrying client re-fed the accepted prefix on every attempt
  (issue #239).
- ``do_POST`` and the open ``/health`` check compared ``self.path`` exactly,
  query string included, so ``/events?batch=1`` 404'd and ``/health?probe=1``
  fell through to the auth gate and answered 401 with a token set (issue
  #241). The GET side already stripped the query via ``urlsplit``.
- ``snagline serve --config`` resolved the full layered config but forwarded
  only the CLI flags for ``metrics_format`` / ``server_read_timeout`` /
  ``episode_ttl_seconds``; the ``_resolve_*`` helpers in http_server read env
  only, so a config-file value for those three knobs was silently dropped
  (issue #240).
"""

from __future__ import annotations

import argparse
import json
import socket
import threading

from snagline.monitor import Monitor
from snagline.server.http_server import make_server


def _start():
    server = make_server(Monitor([], []), host="127.0.0.1", port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, int(server.server_address[1])


def _request(sock: socket.socket, request: bytes) -> str:
    sock.sendall(request)
    chunks: list[bytes] = []
    while True:
        data = sock.recv(65536)
        if not data:
            break
        chunks.append(data)
    return b"".join(chunks).decode("utf-8", "replace")


def _post(port: int, path: str, body: bytes) -> tuple[int, str]:
    sock = socket.create_connection(("127.0.0.1", port), timeout=10)
    try:
        head = (
            f"POST {path} HTTP/1.0\r\n"
            "Host: localhost\r\n"
            "Content-Type: application/json\r\n"
            f"Content-Length: {len(body)}\r\n"
            "\r\n"
        ).encode()
        response = _request(sock, head + body)
    finally:
        sock.close()
    status_line, _, rest = response.partition("\r\n")
    return int(status_line.split()[1]), rest


def _get(port: int, path: str) -> tuple[int, str]:
    sock = socket.create_connection(("127.0.0.1", port), timeout=10)
    try:
        response = _request(
            sock, f"GET {path} HTTP/1.0\r\nHost: localhost\r\n\r\n".encode()
        )
    finally:
        sock.close()
    status_line, _, rest = response.partition("\r\n")
    return int(status_line.split()[1]), rest


def _event(step: str) -> dict:
    return {
        "step_id": step,
        "episode_id": "ep",
        "timestamp": 0.0,
        "action_type": "tool_call",
        "action_signature": f"sig-{step}",
    }


# --- Batched POST atomicity (issue #239) -----------------------------------


def test_rejected_batch_ingests_nothing() -> None:
    """A 400 on item k must not leave items 0..k-1 inside the monitor.

    A well-behaved client retries the batch after fixing it; pre-fix, every
    retry re-ingested the accepted prefix.
    """
    server, port = _start()
    try:
        collector = server.RequestHandlerClass.snagline_collector
        before = collector._ingest_total if hasattr(collector, "_ingest_total") else 0
        good = _event("s1")
        bad = {**_event("s2"), "unknown_field": True}
        status, _ = _post(
            port,
            "/events",
            json.dumps([good, bad]).encode(),
        )
        assert status == 400
        # The good event must NOT have been ingested: query the monitor's
        # own metrics counter, which ingest increments.
        assert (
            server.RequestHandlerClass.snagline_monitor.metrics()["events_ingested"]
            == before
        ), "a rejected batch must ingest zero events (issue #239)"
    finally:
        server.shutdown()
        server.server_close()


def test_valid_batch_still_ingests_every_item() -> None:
    """The two-pass fix must not change the happy path."""
    server, port = _start()
    try:
        events = [_event(f"s{i}") for i in range(3)]
        status, body = _post(port, "/events", json.dumps(events).encode())
        assert status == 202
        assert json.loads(body[body.index("{") :])["count"] == 3
        assert (
            server.RequestHandlerClass.snagline_monitor.metrics()["events_ingested"]
            == 3
        )
    finally:
        server.shutdown()
        server.server_close()


def test_rejected_batch_retry_ingests_exactly_once() -> None:
    """End to end: fix the bad item, retry, and no step is duplicated."""
    server, port = _start()
    try:
        monitor = server.RequestHandlerClass.snagline_monitor
        bad = {**_event("s2"), "unknown_field": True}
        status, _ = _post(port, "/events", json.dumps([_event("s1"), bad]).encode())
        assert status == 400
        status, _ = _post(
            port, "/events", json.dumps([_event("s1"), _event("s2")]).encode()
        )
        assert status == 202
        assert monitor.metrics()["events_ingested"] == 2
    finally:
        server.shutdown()
        server.server_close()


# --- Query-string routing (issue #241) --------------------------------------


def test_post_routes_accept_query_strings() -> None:
    server, port = _start()
    try:
        status, _ = _post(port, "/events?batch=1", json.dumps(_event("q1")).encode())
        assert status == 202, "POST /events?batch=1 used to 404 (issue #241)"
        assert (
            server.RequestHandlerClass.snagline_monitor.metrics()["events_ingested"]
            == 1
        )
    finally:
        server.shutdown()
        server.server_close()


def test_health_answers_200_with_query_string() -> None:
    server, port = _start()
    try:
        status, _ = _get(port, "/health?probe=1")
        assert status == 200, "GET /health?probe=1 must stay the open liveness probe"
    finally:
        server.shutdown()
        server.server_close()


def test_unknown_post_path_with_query_still_404() -> None:
    server, port = _start()
    try:
        status, _ = _post(port, "/nope?x=1", json.dumps(_event("n1")).encode())
        assert status == 404
    finally:
        server.shutdown()
        server.server_close()


# --- serve --config layering (issue #240) -----------------------------------


def test_cmd_serve_passes_layered_config_values(monkeypatch, tmp_path) -> None:
    """_cmd_serve must forward the resolved cfg values for the three knobs
    the _resolve_* helpers re-resolve from env only."""
    import snagline.cli as cli_mod
    from snagline.cli import _cmd_serve

    cfg_file = tmp_path / "cfg.json"
    cfg_file.write_text(
        json.dumps(
            {
                "metrics_format": "classic",
                "server_read_timeout": 12.5,
                "episode_ttl_seconds": 90.0,
            }
        )
    )

    captured: dict = {}

    def fake_serve(monitor, **kwargs):
        captured.update(kwargs)

    args = argparse.Namespace(
        config=str(cfg_file),
        halt_forward=None,
        halt_timeout=None,
        min_severity_for_halt=None,
        host="127.0.0.1",
        port=0,
        auth_token=None,
        keyfile=None,
        certfile=None,
        client_ca=None,
        max_body_bytes=1000,
        max_risks=10,
        read_timeout=None,
        episode_ttl_seconds=None,
        sink="console",
    )

    monkeypatch.setattr(cli_mod, "_build_sinks", lambda args, cfg: [])
    monkeypatch.setattr("snagline.server.http_server.serve", fake_serve)
    monkeypatch.setattr(cli_mod, "Monitor", Monitor)
    # Serve runs in the foreground; patch suppress so the with-block exits
    # immediately after the (fake) serve call.
    import contextlib

    real_suppress = contextlib.suppress
    monkeypatch.setattr(
        "snagline.cli.suppress",
        lambda *exc: real_suppress(KeyboardInterrupt),
    )

    rc = _cmd_serve(args)
    assert rc == 0
    assert captured["metrics_format"] == "classic"
    assert captured["read_timeout"] == 12.5
    assert captured["episode_ttl_seconds"] == 90.0


def test_cmd_serve_flag_beats_config_file(monkeypatch, tmp_path) -> None:
    """CLI flags keep priority over the file layer (the resolve contract)."""
    import snagline.cli as cli_mod
    from snagline.cli import _cmd_serve

    cfg_file = tmp_path / "cfg.json"
    cfg_file.write_text(json.dumps({"server_read_timeout": 12.5}))

    captured: dict = {}

    def fake_serve(monitor, **kwargs):
        captured.update(kwargs)

    args = argparse.Namespace(
        config=str(cfg_file),
        halt_forward=None,
        halt_timeout=None,
        min_severity_for_halt=None,
        host="127.0.0.1",
        port=0,
        auth_token=None,
        keyfile=None,
        certfile=None,
        client_ca=None,
        max_body_bytes=1000,
        max_risks=10,
        read_timeout=30.0,
        episode_ttl_seconds=45.0,
        sink="console",
    )

    monkeypatch.setattr(cli_mod, "_build_sinks", lambda args, cfg: [])
    monkeypatch.setattr("snagline.server.http_server.serve", fake_serve)
    monkeypatch.setattr(cli_mod, "Monitor", Monitor)
    import contextlib

    real_suppress = contextlib.suppress
    monkeypatch.setattr(
        "snagline.cli.suppress",
        lambda *exc: real_suppress(KeyboardInterrupt),
    )

    rc = _cmd_serve(args)
    assert rc == 0
    assert captured["read_timeout"] == 30.0
    assert captured["episode_ttl_seconds"] == 45.0
