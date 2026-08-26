"""End-of-episode signal: POST /episodes/end expires the gauge entry (#123).

The wire format previously had no way to say "this episode is over", so
``snagline_episodes_active`` could only shrink via least-recently-seen cap
eviction. These tests pin the remedy: the signal forwards to
``Monitor.end_episode``, removes the id from the metrics table immediately,
applies auth like every POST, and treats malformed signals as 400 -- never
a crash, never retained content (ids only).
"""

from __future__ import annotations

import json
import socket
import threading

from snagline.monitor import Monitor
from snagline.server.http_server import (
    SidecarMetricsCollector,
    make_server,
)

_EVENT_FIELDS = {
    "timestamp": 1718300000.0,
    "action_type": "tool_call",
    "action_signature": "aaaa1111bbbb2222",
    "metadata": {},
}


class _RecordingMonitor:
    """Duck-typed Monitor stand-in that records end_episode calls."""

    def __init__(self) -> None:
        self.ended: list[str] = []

    def add_sink(self, sink: object) -> None:
        pass

    def ingest(self, event: object) -> None:
        pass

    def end_episode(self, episode_id: str) -> None:
        self.ended.append(str(episode_id))

    def metrics(self) -> dict:
        return {}


class _ExplodingMonitor(_RecordingMonitor):
    def end_episode(self, episode_id: str) -> None:
        raise RuntimeError("boom: monitor misconfigured")


def _start(monitor: object | None = None, auth_token: str | None = None):
    server = make_server(
        monitor or Monitor([], []),  # type: ignore[arg-type]
        host="127.0.0.1",
        port=0,
        auth_token=auth_token,
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, int(server.server_address[1])


def _request(
    port: int,
    method: str,
    path: str,
    body: bytes = b"",
    token: str | None = None,
) -> bytes:
    """One raw-socket HTTP/1.0 request; returns the full wire response."""
    sock = socket.create_connection(("127.0.0.1", port), timeout=10)
    try:
        lines = [f"{method} {path} HTTP/1.0", "Host: localhost"]
        if token:
            lines.append(f"Authorization: Bearer {token}")
        if body:
            lines.append(f"Content-Length: {len(body)}")
        lines.append("")
        sock.sendall(("\r\n".join(lines) + "\r\n").encode() + body)
        chunks: list[bytes] = []
        while True:
            data = sock.recv(65536)
            if not data:
                break
            chunks.append(data)
        return b"".join(chunks)
    finally:
        sock.close()


def _post_event(port: int, episode_id: str) -> bytes:
    event = {"step_id": "0", "episode_id": episode_id, **_EVENT_FIELDS}
    return _request(port, "POST", "/events", json.dumps(event).encode())


def test_end_signal_drops_episode_from_gauge_immediately():
    """Full raw-socket flow: ingest -> gauge 1, end -> gauge 0, no eviction."""
    server, port = _start()
    try:
        response = _post_event(port, "ep-gauge")
        assert response.startswith(b"HTTP/1.0 202"), response[:120]
        scrape = _request(port, "GET", "/metrics")
        assert b"snagline_episodes_active 1\n" in scrape

        end = _request(port, "POST", "/episodes/end", b'{"episode_id": "ep-gauge"}')
        assert end.startswith(b"HTTP/1.0 200"), end[:120]
        assert b'"status": "ended"' in end
        assert b'"episode_id": "ep-gauge"' in end

        scrape_after = _request(port, "GET", "/metrics")
        assert b"snagline_episodes_active 0\n" in scrape_after
    finally:
        server.shutdown()
        server.server_close()


def test_end_forwards_to_monitor_end_episode():
    monitor = _RecordingMonitor()
    server, port = _start(monitor)
    try:
        response = _request(port, "POST", "/episodes/end", b'{"episode_id": "run-42"}')
        assert response.startswith(b"HTTP/1.0 200"), response[:120]
        assert monitor.ended == ["run-42"]
    finally:
        server.shutdown()
        server.server_close()


def test_auth_applies_to_end_endpoint_like_every_post():
    monitor = _RecordingMonitor()
    server, port = _start(monitor, auth_token="s3cr3t")
    try:
        denied = _request(port, "POST", "/episodes/end", b'{"episode_id": "run-42"}')
        assert denied.startswith(b"HTTP/1.0 401"), denied[:120]
        assert monitor.ended == []

        allowed = _request(
            port,
            "POST",
            "/episodes/end",
            b'{"episode_id": "run-42"}',
            token="s3cr3t",
        )
        assert allowed.startswith(b"HTTP/1.0 200"), allowed[:120]
        assert monitor.ended == ["run-42"]
    finally:
        server.shutdown()
        server.server_close()


def test_malformed_signals_get_400_and_sidecar_stays_up():
    """Bad bodies are rejected with 400; the sidecar keeps serving."""
    bad_bodies = [
        b"not json at all",
        b"[1, 2, 3]",
        b'"just a string"',
        b'{"step_id": "no-episode-id"}',
        b'{"episode_id": ""}',
        b'{"episode_id": 42}',
    ]
    server, port = _start()
    try:
        for bad in bad_bodies:
            response = _request(port, "POST", "/episodes/end", bad)
            assert response.startswith(b"HTTP/1.0 400"), (bad, response[:120])
        # No Content-Length at all: body silently empty -> invalid JSON -> 400.
        response = _request(port, "POST", "/episodes/end")
        assert response.startswith(b"HTTP/1.0 400"), response[:120]
        # Still alive and serving afterwards.
        health = _request(port, "GET", "/health")
        assert health.startswith(b"HTTP/1.0 200"), health[:120]
    finally:
        server.shutdown()
        server.server_close()


def test_monitor_failure_never_breaks_the_endpoint():
    """A raising Monitor is swallowed; the client still gets a clean reply."""
    monitor = _ExplodingMonitor()
    server, port = _start(monitor)
    try:
        response = _request(port, "POST", "/episodes/end", b'{"episode_id": "run-7"}')
        assert response.startswith(b"HTTP/1.0 200"), response[:120]
        assert b'"status": "ended"' in response
        health = _request(port, "GET", "/health")
        assert health.startswith(b"HTTP/1.0 200"), health[:120]
    finally:
        server.shutdown()
        server.server_close()


def test_end_is_idempotent_and_ids_only():
    """Ending twice is a no-op the second time; ids never become content."""
    collector = SidecarMetricsCollector()
    collector.record_ingest("ep-a", 0.001)
    assert collector.snapshot()["episodes_active"] == 1
    collector.end_episode("ep-a")
    assert collector.snapshot()["episodes_active"] == 0
    collector.end_episode("ep-a")  # second end: still zero, never raises
    collector.end_episode("never-seen")  # unknown id: no-op, never raises
    assert collector.snapshot()["episodes_active"] == 0


def test_collector_expiry_does_not_wait_for_cap_eviction():
    """The freed slot is reusable: the quietest old id survives because the
    explicit end made room, where cap eviction would have forgotten it."""
    collector = SidecarMetricsCollector(max_episodes=2)
    collector.record_ingest("ep-a", 0.001)
    collector.record_ingest("ep-b", 0.001)
    assert collector.snapshot()["episodes_active"] == 2
    collector.end_episode("ep-a")
    assert collector.snapshot()["episodes_active"] == 1
    collector.record_ingest("ep-c", 0.001)
    # Had the end signal not freed a slot, adding c at cap would have evicted
    # b (the least-recently-seen); instead b and c are both still tracked.
    assert list(collector._episodes) == ["ep-b", "ep-c"]
    collector.record_ingest("ep-a", 0.001)  # ended ids can be recounted later
    assert list(collector._episodes) == ["ep-c", "ep-a"]
