"""The sidecar must answer HEAD on its probe routes and 405 (not 501) for
other methods on routes that exist, draining the body first (issue #433)."""

from __future__ import annotations

import http.client
import threading

import pytest

from snagline import Monitor
from snagline.server.http_server import make_server


@pytest.fixture
def server():
    mon = Monitor.default()
    httpd = make_server(mon, host="127.0.0.1", port=0, auth_token=None)
    port = httpd.server_address[1]
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield port
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=5)


def _request(port, method, path, body=None):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    conn.request(method, path, body=body)
    resp = conn.getresponse()
    data = resp.read()
    conn.close()
    return resp.status, resp.headers, data


# ------------------------------------------------------------ HEAD ----


@pytest.mark.parametrize("path", ["/health", "/metrics"])
def test_head_on_probe_routes_returns_200_not_501(server, path):
    status, headers, body = _request(server, "HEAD", path)
    assert status == 200, (status, body)
    # The whole point of HEAD: no body, same status.
    assert body == b""
    assert int(headers.get("Content-Length", 0)) >= 0


def test_head_health_advertises_the_same_length_as_get(server):
    get_status, get_headers, get_body = _request(server, "GET", "/health")
    head_status, head_headers, head_body = _request(server, "HEAD", "/health")
    assert get_status == head_status == 200
    assert head_body == b""
    # HEAD promises the same headers as GET, including the content length.
    assert head_headers.get("Content-Length") == get_headers.get("Content-Length")
    assert get_body != b""


def test_head_metrics_advertises_the_same_length_as_get(server):
    get_status, get_headers, get_body = _request(server, "GET", "/metrics")
    head_status, head_headers, head_body = _request(server, "HEAD", "/metrics")
    assert get_status == head_status == 200
    assert head_body == b""
    assert head_headers.get("Content-Length") == get_headers.get("Content-Length")
    assert get_body != b""


def test_head_unknown_route_is_a_404(server):
    status, _, body = _request(server, "HEAD", "/nope")
    assert status == 404, (status, body)


# ------------------------------------------------- other methods ----


@pytest.mark.parametrize("method", ["PUT", "DELETE", "PATCH", "OPTIONS", "FOO"])
@pytest.mark.parametrize("path", ["/health", "/metrics", "/events"])
def test_other_method_on_a_known_route_is_405_not_501(server, method, path):
    status, headers, body = _request(server, method, path)
    assert status == 405, (status, body)
    allow = headers.get("Allow")
    assert allow is not None
    assert "GET" in allow and "POST" in allow and "HEAD" in allow


def test_other_method_on_an_unknown_route_is_still_404(server):
    status, _, body = _request(server, "PUT", "/nope")
    assert status == 404, (status, body)


def test_other_method_drains_its_body_before_answering(server):
    """A body-carrying method on a known route must not be RST'd before the
    status line is read."""
    body = b"x" * (1024 * 1024)  # 1 MiB: an unread body reliably trips a reset
    status, _, resp_body = _request(server, "PUT", "/events", body=body)
    assert status == 405, (status, resp_body)


def test_get_and_post_are_unchanged(server):
    assert _request(server, "GET", "/health")[0] == 200
    assert _request(server, "GET", "/nope")[0] == 404
    # POST to an unknown route stays 404, not 405.
    assert _request(server, "POST", "/nope")[0] == 404
