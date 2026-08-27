"""Tests for the stdlib WebhookSink (project.md §8)."""

from __future__ import annotations

import json
import urllib.request
from unittest import mock

from snagline.risk import FailureRisk
from snagline.sinks.webhook import WebhookSink


def _risk() -> FailureRisk:
    return FailureRisk(
        episode_id="ep-1",
        step_id="3",
        score=0.5,
        trigger="loop",
        detail="action repeated 3x in last 4 steps",
        timestamp=1718300000.0,
    )


def test_emit_posts_failure_risk_fields_only() -> None:
    captured = {}

    class _Resp:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def read(self):
            return b"{}"

    def fake_urlopen(req, timeout=None):
        captured["req"] = req
        captured["timeout"] = timeout
        return _Resp()

    with mock.patch.object(urllib.request, "urlopen", side_effect=fake_urlopen):
        WebhookSink("http://hooks.example/alerts", timeout=1.5).emit(_risk())

    body = json.loads(captured["req"].data.decode())
    assert body == {
        "episode_id": "ep-1",
        "step_id": "3",
        "score": 0.5,
        "trigger": "loop",
        "detail": "action repeated 3x in last 4 steps",
        "timestamp": 1718300000.0,
    }
    # The payload deliberately has no metadata field -- no exfiltration path.
    assert "metadata" not in body
    assert captured["req"].get_header("Content-type") == "application/json"
    assert captured["timeout"] == 1.5


def test_emit_never_raises_on_network_failure() -> None:
    with mock.patch.object(
        urllib.request, "urlopen", side_effect=OSError("connection refused")
    ):
        sink = WebhookSink("http://dead.invalid/hook")
        sink.emit(_risk())  # must be a silent no-op, not a raise


def test_emit_never_raises_on_bad_status() -> None:
    import urllib.error

    with mock.patch.object(
        urllib.request,
        "urlopen",
        side_effect=urllib.error.HTTPError("url", 500, "boom", hdrs=None, fp=None),  # type: ignore[arg-type]
    ):
        WebhookSink("http://hooks.example/alerts").emit(_risk())
