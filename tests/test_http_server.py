"""End-to-end tests for the stdlib sidecar HTTP server (project.md §7)."""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from typing import Any

from snagline.monitor import Monitor
from snagline.risk import FailureRisk
from snagline.server.http_server import make_server


class _RecordingSink:
    def __init__(self):
        self.risks: list[FailureRisk] = []

    def emit(self, risk: FailureRisk) -> None:
        self.risks.append(risk)


def _start_server(
    sink: _RecordingSink,
    auth_token: str | None = None,
    max_body_bytes: int | None = None,
    max_risks: int | None = None,
) -> tuple[Any, str]:
    kwargs: dict[str, Any] = {}
    if max_body_bytes is not None:
        kwargs["max_body_bytes"] = max_body_bytes
    if max_risks is not None:
        kwargs["max_risks"] = max_risks
    server = make_server(
        Monitor.default(sinks=[sink]),
        host="127.0.0.1",
        port=0,
        auth_token=auth_token,
        **kwargs,
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, f"http://127.0.0.1:{server.server_address[1]}"


def _get(base: str, path: str, headers: dict[str, str] | None = None) -> int:
    """GET ``path`` and return the status code, HTTPError codes included."""
    req = urllib.request.Request(base + path, headers=headers or {}, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            return int(resp.status)
    except urllib.error.HTTPError as exc:
        return int(exc.code)


def test_health_endpoint() -> None:
    server, base = _start_server(_RecordingSink())
    try:
        with urllib.request.urlopen(base + "/health", timeout=5) as resp:
            assert resp.status == 200
            assert json.loads(resp.read()) == {"status": "ok"}
    finally:
        server.shutdown()
        server.server_close()


def test_events_endpoint_ingests_and_fires_loop_detector() -> None:
    sink = _RecordingSink()
    server, base = _start_server(sink)
    try:
        event = {
            "step_id": "0",
            "episode_id": "ep-http",
            "timestamp": 1718300000.0,
            "action_type": "tool_call",
            "action_signature": "aaaa1111bbbb2222",
            "tool_name": "search",
        }
        for _ in range(4):  # loop detector default: 3 repeats in window of 4
            req = urllib.request.Request(
                base + "/events",
                data=json.dumps(event).encode(),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=5) as resp:
                assert resp.status == 202
                assert json.loads(resp.read())["status"] == "ingested"
        assert any(r.trigger == "loop" for r in sink.risks)
    finally:
        server.shutdown()
        server.server_close()


def test_events_endpoint_rejects_malformed_body() -> None:
    server, base = _start_server(_RecordingSink())
    try:
        req = urllib.request.Request(
            base + "/events",
            data=b"not json",
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            urllib.request.urlopen(req, timeout=5)
            raise AssertionError("expected HTTP 400")
        except urllib.error.HTTPError as exc:
            assert exc.code == 400
    finally:
        server.shutdown()
        server.server_close()


def test_post_requires_token_when_configured() -> None:
    sink = _RecordingSink()
    server, base = _start_server(sink, auth_token="secret")
    try:
        event = {
            "step_id": "0",
            "episode_id": "ep-auth",
            "timestamp": 1718300000.0,
            "action_type": "tool_call",
            "action_signature": "aaaa1111bbbb2222",
            "tool_name": "search",
        }
        # No token -> 401.
        req = urllib.request.Request(
            base + "/events", data=json.dumps(event).encode(), method="POST"
        )
        try:
            urllib.request.urlopen(req, timeout=5)
            raise AssertionError("expected HTTP 401")
        except urllib.error.HTTPError as exc:
            assert exc.code == 401

        # Wrong token -> 401.
        req = urllib.request.Request(
            base + "/events",
            data=json.dumps(event).encode(),
            headers={"Authorization": "Bearer wrong"},
            method="POST",
        )
        try:
            urllib.request.urlopen(req, timeout=5)
            raise AssertionError("expected HTTP 401")
        except urllib.error.HTTPError as exc:
            assert exc.code == 401

        # Correct bearer token -> 202. (A request that clears the gate still
        # has to send the JSON content type; the token and the content type are
        # independent requirements, issue #388.)
        _JSON = {"Content-Type": "application/json"}
        req = urllib.request.Request(
            base + "/events",
            data=json.dumps(event).encode(),
            headers={**_JSON, "Authorization": "Bearer secret"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            assert resp.status == 202

        # X-Snagline-Token header also accepted.
        req = urllib.request.Request(
            base + "/events",
            data=json.dumps(event).encode(),
            headers={**_JSON, "X-Snagline-Token": "secret"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            assert resp.status == 202
    finally:
        server.shutdown()
        server.server_close()


def test_events_endpoint_accepts_batch() -> None:
    sink = _RecordingSink()
    server, base = _start_server(sink)
    try:
        events = [
            {
                "step_id": str(i),
                "episode_id": "ep-batch",
                "timestamp": 1718300000.0 + i,
                "action_type": "tool_call",
                "action_signature": f"sig-{i}",
                "tool_name": "search",
            }
            for i in range(3)
        ]
        req = urllib.request.Request(
            base + "/events",
            data=json.dumps(events).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            assert resp.status == 202
            assert json.loads(resp.read())["count"] == 3
        assert len(sink.risks) == 0  # batch of unique signatures => no loop
    finally:
        server.shutdown()
        server.server_close()


def test_post_payload_too_large_returns_413() -> None:
    sink = _RecordingSink()
    server, base = _start_server(sink, max_body_bytes=10)
    try:
        event = {
            "step_id": "0",
            "episode_id": "ep-big",
            "timestamp": 1718300000.0,
            "action_type": "tool_call",
            "action_signature": "aaaa1111bbbb2222",
            "tool_name": "search",
        }
        req = urllib.request.Request(
            base + "/events",
            data=json.dumps(event).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            urllib.request.urlopen(req, timeout=5)
            raise AssertionError("expected HTTP 413")
        except urllib.error.HTTPError as exc:
            assert exc.code == 413
    finally:
        server.shutdown()
        server.server_close()


def test_health_open_without_token() -> None:
    server, base = _start_server(_RecordingSink(), auth_token="secret")
    try:
        with urllib.request.urlopen(base + "/health", timeout=5) as resp:
            assert resp.status == 200
    finally:
        server.shutdown()
        server.server_close()


def test_metrics_endpoint_reports_ingested_counts() -> None:
    server, base = _start_server(_RecordingSink())
    try:
        event = {
            "step_id": "0",
            "episode_id": "ep-metrics",
            "timestamp": 1718300000.0,
            "action_type": "tool_call",
            "action_signature": "aaaa1111bbbb2222",
            "tool_name": "search",
        }
        req = urllib.request.Request(
            base + "/events",
            data=json.dumps(event).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            assert resp.status == 202
        # Issue #98 made prometheus the default /metrics body; the JSON
        # counters this test exercises remain available via ?format=classic.
        with urllib.request.urlopen(
            base + "/metrics?format=classic", timeout=5
        ) as resp:
            assert resp.status == 200
            body = json.loads(resp.read())
        assert body["events_ingested"] >= 1
        assert "risks_emitted" in body
        assert "detector_errors" in body
    finally:
        server.shutdown()
        server.server_close()


def test_unknown_paths_are_404() -> None:
    server, base = _start_server(_RecordingSink())
    try:
        try:
            urllib.request.urlopen(base + "/nope", timeout=5)
            raise AssertionError("expected HTTP 404")
        except urllib.error.HTTPError as exc:
            assert exc.code == 404
    finally:
        server.shutdown()
        server.server_close()


def test_claude_code_hook_endpoint_ingests() -> None:
    # Issue #22 path: a native Claude Code hook payload is mapped and ingested.
    # Three identical PostToolUse events must map to repeated tool_call
    # signatures and trip the loop detector end to end.
    sink = _RecordingSink()
    server, base = _start_server(sink)
    try:
        payload = {
            "hook_event_name": "PostToolUse",
            "session_id": "sess-http",
            "tool_use_id": "tu-1",
            "tool_name": "search",
            "tool_input": {"q": "cat"},
        }
        for _ in range(3):
            req = urllib.request.Request(
                base + "/hooks/claude-code",
                data=json.dumps(payload).encode(),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=5) as resp:
                assert resp.status == 202
        assert any(r.trigger == "loop" for r in sink.risks)
    finally:
        server.shutdown()
        server.server_close()


def test_risks_endpoint_records_received_risk() -> None:
    sink = _RecordingSink()
    server, base = _start_server(sink)
    try:
        risk = {
            "episode_id": "ep-x",
            "step_id": "s1",
            "score": 0.9,
            "trigger": "loop",
            "detail": "repeated",
            "timestamp": 1.0,
        }
        req = urllib.request.Request(
            base + "/risks",
            data=json.dumps(risk).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            assert resp.status == 202
        with urllib.request.urlopen(base + "/risks", timeout=5) as resp:
            assert resp.status == 200
            assert json.loads(resp.read())["risks"]
    finally:
        server.shutdown()
        server.server_close()


def test_get_metrics_requires_token_when_configured() -> None:
    """Regression: do_GET never consulted _authorized(), so /metrics leaked."""
    server, base = _start_server(_RecordingSink(), auth_token="secret")
    try:
        assert _get(base, "/metrics") == 401
        assert _get(base, "/metrics", {"Authorization": "Bearer wrong"}) == 401
        assert _get(base, "/metrics", {"Authorization": "Bearer secret"}) == 200
        assert _get(base, "/metrics", {"X-Snagline-Token": "secret"}) == 200
    finally:
        server.shutdown()
        server.server_close()


def test_get_risks_requires_token_when_configured() -> None:
    """Regression: /risks returned every risk the sidecar had ever received --
    episode ids, triggers, and detail strings -- to an unauthenticated caller."""
    sink = _RecordingSink()
    server, base = _start_server(sink, auth_token="secret")
    try:
        risk = {
            "episode_id": "ep-secret",
            "step_id": "s1",
            "score": 0.9,
            "trigger": "loop",
            "detail": "repeated",
            "timestamp": 1.0,
        }
        req = urllib.request.Request(
            base + "/risks",
            data=json.dumps(risk).encode(),
            headers={
                "Content-Type": "application/json",
                "Authorization": "Bearer secret",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            assert resp.status == 202

        assert _get(base, "/risks") == 401
        req = urllib.request.Request(
            base + "/risks", headers={"Authorization": "Bearer secret"}, method="GET"
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            assert resp.status == 200
            assert json.loads(resp.read())["risks"][0]["episode_id"] == "ep-secret"
    finally:
        server.shutdown()
        server.server_close()


def test_unknown_get_path_is_401_not_404_when_token_configured() -> None:
    """An unauthenticated caller must not be able to enumerate which paths
    exist, so the 404 fallthrough sits behind the token too."""
    server, base = _start_server(_RecordingSink(), auth_token="secret")
    try:
        assert _get(base, "/nope") == 401
        assert _get(base, "/nope", {"Authorization": "Bearer secret"}) == 404
    finally:
        server.shutdown()
        server.server_close()


def test_get_endpoints_stay_open_when_no_token_is_configured() -> None:
    """Without auth_token the sidecar is unchanged: GETs are open."""
    server, base = _start_server(_RecordingSink())
    try:
        assert _get(base, "/health") == 200
        assert _get(base, "/metrics") == 200
        assert _get(base, "/risks") == 200
    finally:
        server.shutdown()
        server.server_close()


def test_received_risks_are_bounded() -> None:
    """POST /risks is an open-ended ingest point; retention must be capped."""
    server, base = _start_server(_RecordingSink(), max_risks=3)
    try:
        for i in range(5):
            req = urllib.request.Request(
                base + "/risks",
                data=json.dumps(
                    {
                        "episode_id": "ep",
                        "step_id": str(i),
                        "score": 0.9,
                        "trigger": "loop",
                        "detail": "d",
                        "timestamp": float(i),
                    }
                ).encode(),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=5) as resp:
                assert resp.status == 202
        with urllib.request.urlopen(base + "/risks", timeout=5) as resp:
            risks = json.loads(resp.read())["risks"]
        # Oldest dropped, newest kept, and still JSON-serializable.
        assert [r["step_id"] for r in risks] == ["2", "3", "4"]
    finally:
        server.shutdown()
        server.server_close()


# --- request origin (issue #388) --------------------------------------------
# Authenticating the token does not authenticate the sender: ``snagline serve``
# defaults to no token, and a page the operator is visiting can POST to the
# loopback sidecar as a CORS "simple request" -- no preflight, so the browser
# sends it and the write lands. These cover the two gates that close that.

_JSON_HEADERS = {"Content-Type": "application/json"}


def _post(base: str, path: str, body: Any, headers: dict[str, str]) -> int:
    """POST ``body`` and return the status code, HTTPError codes included."""
    req = urllib.request.Request(
        base + path,
        data=json.dumps(body).encode(),
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            return int(resp.status)
    except urllib.error.HTTPError as exc:
        return int(exc.code)


def test_cross_site_post_is_refused_even_without_a_token() -> None:
    """The reproduction from the issue. Stock config, no token, and a page at
    another origin issues the ``text/plain`` simple request that needs no
    preflight. Refusing it is the difference between an episode's detection
    state surviving and being silently discarded."""
    sink = _RecordingSink()
    server, base = _start_server(sink)
    try:
        assert (
            _post(
                base,
                "/episodes/end",
                {"episode_id": "ep-forged"},
                {"Content-Type": "text/plain", "Origin": "https://evil.example"},
            )
            == 403
        )
    finally:
        server.shutdown()
        server.server_close()


def _active_episodes(base: str) -> int:
    """Read ``snagline_episodes_active`` off the Prometheus exposition.

    The gauge drops an id the moment ``end_episode`` lands, so it is the
    observable difference between a forgery that took effect and one refused.
    """
    with urllib.request.urlopen(base + "/metrics", timeout=5) as resp:
        for line in resp.read().decode().splitlines():
            if line.startswith("snagline_episodes_active "):
                return int(line.split()[-1])
    raise AssertionError("snagline_episodes_active not exported")


def test_cross_site_post_does_not_end_the_episode() -> None:
    """The refused POST must have no effect. ``end_episode`` clears the
    episode from the active gauge and resets every detector for it -- the
    loop detector's window, the CUSUM baselines -- so a forgery that landed
    destroys in-flight detection state the operator never sees lost."""
    sink = _RecordingSink()
    server, base = _start_server(sink)
    try:
        event = {
            "step_id": "0",
            "episode_id": "ep-forged",
            "timestamp": 1718300000.0,
            "action_type": "tool_call",
            "action_signature": "aaaa1111bbbb2222",
            "tool_name": "search",
        }
        assert _post(base, "/events", event, _JSON_HEADERS) == 202
        assert _active_episodes(base) == 1, "the episode is in flight"

        # Had this landed, the gauge would read 0 and every detector for the
        # episode would have been reset.
        assert (
            _post(
                base,
                "/episodes/end",
                {"episode_id": "ep-forged"},
                {"Content-Type": "text/plain", "Origin": "https://evil.example"},
            )
            == 403
        )
        assert _active_episodes(base) == 1, (
            "a refused forgery must leave the episode's state intact"
        )

        # A same-site, JSON end still works -- the fix refuses the origin, not
        # the endpoint.
        assert (
            _post(
                base,
                "/episodes/end",
                {"episode_id": "ep-forged"},
                {**_JSON_HEADERS, "Sec-Fetch-Site": "same-origin"},
            )
            == 200
        )
        assert _active_episodes(base) == 0
    finally:
        server.shutdown()
        server.server_close()


def test_cross_site_post_is_refused_even_with_a_json_content_type() -> None:
    """Defense in depth: the origin gate does not lean on the content-type
    gate. A cross-site request that somehow carried ``application/json``
    (or a browser extension that defeats the preflight) is still refused."""
    server, base = _start_server(_RecordingSink())
    try:
        assert (
            _post(
                base,
                "/episodes/end",
                {"episode_id": "ep"},
                {**_JSON_HEADERS, "Origin": "https://evil.example"},
            )
            == 403
        )
    finally:
        server.shutdown()
        server.server_close()


def test_null_origin_is_treated_as_cross_site() -> None:
    """``Origin: null`` is a sandboxed iframe or a ``file://`` page -- a page
    the sidecar never served, so it is not same-site."""
    server, base = _start_server(_RecordingSink())
    try:
        assert (
            _post(
                base,
                "/episodes/end",
                {"episode_id": "ep"},
                {**_JSON_HEADERS, "Origin": "null"},
            )
            == 403
        )
    finally:
        server.shutdown()
        server.server_close()


def test_sec_fetch_site_cross_site_is_refused() -> None:
    """``Sec-Fetch-Site`` is authoritative when present: it is set on every
    browser fetch and names the relationship without needing a Host
    comparison."""
    server, base = _start_server(_RecordingSink())
    try:
        assert (
            _post(
                base,
                "/episodes/end",
                {"episode_id": "ep"},
                {**_JSON_HEADERS, "Sec-Fetch-Site": "cross-site"},
            )
            == 403
        )
    finally:
        server.shutdown()
        server.server_close()


def test_sec_fetch_site_same_origin_is_accepted() -> None:
    """A page the operator loads from the sidecar itself is legitimate: the
    gate must not lock out a future browser dashboard served on the same
    origin."""
    sink = _RecordingSink()
    server, base = _start_server(sink)
    try:
        assert (
            _post(
                base,
                "/events",
                {
                    "step_id": "0",
                    "episode_id": "ep-same-origin",
                    "timestamp": 1718300000.0,
                    "action_type": "tool_call",
                    "action_signature": "aaaa1111bbbb2222",
                    "tool_name": "search",
                },
                {**_JSON_HEADERS, "Sec-Fetch-Site": "same-origin"},
            )
            == 202
        )
    finally:
        server.shutdown()
        server.server_close()


def test_matching_origin_header_is_accepted() -> None:
    """The ``Origin`` fallback compares against the ``Host`` the request
    arrived on, so a same-origin browser POST is not mistaken for a forgery
    on browsers that predate ``Sec-Fetch-Site``."""
    sink = _RecordingSink()
    server, base = _start_server(sink)
    try:
        assert (
            _post(
                base,
                "/events",
                {
                    "step_id": "0",
                    "episode_id": "ep-same-origin",
                    "timestamp": 1718300000.0,
                    "action_type": "tool_call",
                    "action_signature": "aaaa1111bbbb2222",
                    "tool_name": "search",
                },
                # No Sec-Fetch-Site, so the Origin/Host comparison decides.
                {**_JSON_HEADERS, "Origin": base},
            )
            == 202
        )
    finally:
        server.shutdown()
        server.server_close()


def test_post_without_origin_headers_is_accepted() -> None:
    """The gate is opt-in by header: curl, urllib and the shipped sinks send
    neither ``Origin`` nor ``Sec-Fetch-Site``, and a missing declaration is
    not a forgery. This is the guard against the fix locking out every
    non-browser client."""
    sink = _RecordingSink()
    server, base = _start_server(sink)
    try:
        assert (
            _post(
                base,
                "/events",
                {
                    "step_id": "0",
                    "episode_id": "ep-no-origin",
                    "timestamp": 1718300000.0,
                    "action_type": "tool_call",
                    "action_signature": "aaaa1111bbbb2222",
                    "tool_name": "search",
                },
                _JSON_HEADERS,
            )
            == 202
        )
    finally:
        server.shutdown()
        server.server_close()


def test_post_requires_a_json_content_type() -> None:
    """The CORS-safelisted types are exactly what a cross-site ``fetch`` can
    send without a preflight, so they are refused -- including from a
    non-browser client, which the shipped ones never do."""
    server, base = _start_server(_RecordingSink())
    try:
        for content_type in (
            "text/plain",
            "application/x-www-form-urlencoded",
            "multipart/form-data; boundary=x",
            "",
        ):
            assert (
                _post(
                    base,
                    "/episodes/end",
                    {"episode_id": "ep"},
                    {"Content-Type": content_type},
                )
                == 415
            ), f"{content_type!r} must not reach the router"
    finally:
        server.shutdown()
        server.server_close()


def test_json_content_type_with_charset_is_accepted() -> None:
    """Only the media type is compared, so a client that appends a charset
    parameter is not broken."""
    sink = _RecordingSink()
    server, base = _start_server(sink)
    try:
        assert (
            _post(
                base,
                "/events",
                {
                    "step_id": "0",
                    "episode_id": "ep-charset",
                    "timestamp": 1718300000.0,
                    "action_type": "tool_call",
                    "action_signature": "aaaa1111bbbb2222",
                    "tool_name": "search",
                },
                {"Content-Type": "application/json; charset=utf-8"},
            )
            == 202
        )
    finally:
        server.shutdown()
        server.server_close()


def test_cross_site_attempt_at_the_auth_gate_is_logged(caplog) -> None:
    """With a token set, a cross-site request carries no ``Authorization``
    and 401s -- the correct outcome, but the failure used to be
    indistinguishable from a mistyped token. The origin is now named in a
    warning so an attempt is visible."""
    sink = _RecordingSink()
    server, base = _start_server(sink, auth_token="secret")
    try:
        with caplog.at_level("WARNING", logger="snagline"):
            assert (
                _post(
                    base,
                    "/episodes/end",
                    {"episode_id": "ep"},
                    {"Content-Type": "text/plain", "Origin": "https://evil.example"},
                )
                == 401
            )
        assert any(
            "cross-site" in r.message and "evil.example" in r.message
            for r in caplog.records
        ), "a cross-site 401 must be distinguishable from a mistyped token"
    finally:
        server.shutdown()
        server.server_close()


def test_refused_cross_site_post_is_logged(caplog) -> None:
    """The token-less case -- the one the auth gate cannot cover -- must
    leave a trace rather than refusing silently."""
    sink = _RecordingSink()
    server, base = _start_server(sink)
    try:
        with caplog.at_level("WARNING", logger="snagline"):
            _post(
                base,
                "/episodes/end",
                {"episode_id": "ep"},
                {**_JSON_HEADERS, "Sec-Fetch-Site": "cross-site"},
            )
        assert any("cross-site" in r.message for r in caplog.records), (
            "a refused forgery must be logged"
        )
    finally:
        server.shutdown()
        server.server_close()


def test_get_endpoints_are_not_gated_on_origin() -> None:
    """Only the mutating POSTs are gated: a cross-site GET can be issued but
    cannot read the response without CORS headers, so the origin check would
    add nothing and would break a status page embed."""
    server, base = _start_server(_RecordingSink(), auth_token=None)
    try:
        assert (
            _get(
                base,
                "/health",
                {"Origin": "https://evil.example", "Sec-Fetch-Site": "cross-site"},
            )
            == 200
        )
    finally:
        server.shutdown()
        server.server_close()
