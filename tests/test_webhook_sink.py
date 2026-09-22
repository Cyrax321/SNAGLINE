"""Tests for the stdlib WebhookSink (project.md §8)."""

from __future__ import annotations

import json
import logging
import urllib.error
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

    with mock.patch("snagline.sinks.base._opener.open", side_effect=fake_urlopen):
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
    with mock.patch(
        "snagline.sinks.base._opener.open", side_effect=OSError("connection refused")
    ):
        sink = WebhookSink("http://dead.invalid/hook")
        sink.emit(_risk())  # must be a silent no-op, not a raise


def test_emit_never_raises_on_bad_status() -> None:
    import urllib.error

    with mock.patch(
        "snagline.sinks.base._opener.open",
        side_effect=urllib.error.HTTPError("url", 500, "boom", hdrs=None, fp=None),  # type: ignore[arg-type]
    ):
        WebhookSink("http://hooks.example/alerts").emit(_risk())


# --- min_severity filtering (issue #248) --------------------------------------
class _Resp:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self):
        return b"{}"


def _posting_urlopen(posted: list) -> object:
    def fake_urlopen(req, timeout=None):
        posted.append(json.loads(req.data.decode()))
        return _Resp()

    return fake_urlopen


def _risk_with_severity(severity: str) -> FailureRisk:
    return FailureRisk(
        episode_id="ep-1",
        step_id="3",
        score=0.5,
        trigger="loop",
        detail="action repeated 3x in last 4 steps",
        timestamp=1718300000.0,
        severity=severity,
    )


def test_min_severity_info_passes_everything() -> None:
    posted: list = []
    with mock.patch(
        "snagline.sinks.base._opener.open", side_effect=_posting_urlopen(posted)
    ):
        sink = WebhookSink("http://x", min_severity="info")
        sink.emit(_risk_with_severity("info"))
        sink.emit(_risk_with_severity("warning"))
        sink.emit(_risk_with_severity("critical"))
    assert len(posted) == 3


def test_min_severity_critical_suppresses_lower() -> None:
    posted: list = []
    with mock.patch(
        "snagline.sinks.base._opener.open", side_effect=_posting_urlopen(posted)
    ):
        sink = WebhookSink("http://x", min_severity="critical")
        sink.emit(_risk_with_severity("info"))
        sink.emit(_risk_with_severity("warning"))
        sink.emit(_risk_with_severity("critical"))
    assert len(posted) == 1, (
        "critical-only endpoint must not receive info/warning risks"
    )
    assert posted[0]["trigger"] == "loop"


def test_min_severity_unset_is_unfiltered() -> None:
    posted: list = []
    with mock.patch(
        "snagline.sinks.base._opener.open", side_effect=_posting_urlopen(posted)
    ):
        sink = WebhookSink("http://x")
        sink.emit(_risk_with_severity("info"))
    assert len(posted) == 1


# --- a failed POST must not log the destination URL (issue #390) --------------
# The URL is the credential: it can carry basic auth (``user:pass@host``) and,
# for some providers, a secret path segment. A dead endpoint is exactly when an
# operator reads the log.

_HOOK_URL = "https://alice:hunter2@hooks.example/alerts"


def test_failure_log_omits_basic_auth_credentials(caplog) -> None:
    sink = WebhookSink(_HOOK_URL)
    with caplog.at_level(logging.ERROR, logger="snagline"):
        with mock.patch(
            "snagline.sinks.base._opener.open",
            side_effect=OSError("connection refused"),
        ):
            sink.emit(_risk())
    assert caplog.records, "the failed POST must be logged"
    for record in caplog.records:
        text = record.getMessage()
        assert "hunter2" not in text, "basic-auth password reached the log record"
        assert "alice" not in text, "basic-auth username reached the log record"
        assert "hooks.example" in text, "the log must still name the host"


def test_repr_omits_basic_auth_credentials() -> None:
    # repr lands in diagnostic dumps and unhandled-exception reports, so it
    # must not carry the credential either.
    assert "hunter2" not in repr(WebhookSink(_HOOK_URL))
    assert "hooks.example" in repr(WebhookSink(_HOOK_URL))


def test_failure_log_omits_url_from_exception_traceback(caplog) -> None:
    # ``URLError`` embeds the destination in its reason for some failures
    # (``no host given: <url>``), so a ``logger.exception`` call implies
    # ``exc_info=True`` and the credential walks out attached to the
    # traceback -- from inside the fail-open ``except`` block, where a second
    # failure would be the last thing the operator is told (issue #390).
    #
    # ``caplog.text`` is the formatted output an operator reads; a record's
    # ``getMessage()`` excludes the traceback, which is why the leak survives
    # a test that only checks the message.
    def boom(req, timeout=None):
        raise urllib.error.URLError(f"no host given: {_HOOK_URL}")

    sink = WebhookSink(_HOOK_URL)
    with caplog.at_level(logging.ERROR, logger="snagline"):
        with mock.patch.object(urllib.request, "urlopen", side_effect=boom):
            sink.emit(_risk())
    assert caplog.records, "the failed POST must be logged"
    assert "hunter2" not in caplog.text, "basic-auth password reached the log"
    assert "alice" not in caplog.text, "basic-auth username reached the log"
    assert "hooks.example" in caplog.text, "the log must still name the host"
