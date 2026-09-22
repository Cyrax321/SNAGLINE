"""Tests for the Slack escalation sink (P1 alerting)."""

from __future__ import annotations

import json
import logging
from unittest import mock

from snagline.risk import SEVERITY_CRITICAL, SEVERITY_INFO, FailureRisk
from snagline.sinks.slack import SlackSink


def _risk(severity: str = SEVERITY_INFO, **kw) -> FailureRisk:
    return FailureRisk(
        episode_id="ep",
        step_id="s1",
        score=0.9,
        trigger="loop",
        detail="repeating loop detected",
        timestamp=1.0,
        severity=severity,
        **kw,
    )


def test_slack_posts_formatted_text():
    sink = SlackSink("https://hooks.slack.com/xyz")
    with mock.patch("urllib.request.urlopen") as urlopen:
        sink.emit(_risk(SEVERITY_CRITICAL))
    assert urlopen.called
    req = urlopen.call_args[0][0]
    body = json.loads(req.data.decode())
    assert "loop" in body["text"]
    assert "CRITICAL" in body["text"]
    assert dict(req.headers).get("Content-type") == "application/json"


def test_slack_min_severity_filters_lower():
    sink = SlackSink("https://hooks.slack.com/xyz", min_severity=SEVERITY_CRITICAL)
    with mock.patch("urllib.request.urlopen") as urlopen:
        sink.emit(_risk(SEVERITY_INFO))  # below threshold -> dropped
    assert not urlopen.called


def test_slack_swallows_post_errors():
    sink = SlackSink("https://hooks.slack.com/xyz")
    with mock.patch("urllib.request.urlopen", side_effect=OSError("down")):
        # Must not raise; fail-open.
        sink.emit(_risk(SEVERITY_CRITICAL))


# --- a failed POST must not log the webhook URL (issue #390) ------------------
# The URL is the credential: a Slack incoming webhook embeds its secret as the
# final path segment. A dead endpoint is exactly when an operator reads the log.

_WEBHOOK_URL = "https://hooks.slack.com/services/T000/B000/SECRET_TOKEN_DO_NOT_LOG"


def test_slack_failure_log_omits_the_secret(caplog):
    sink = SlackSink(_WEBHOOK_URL)
    with caplog.at_level(logging.ERROR, logger="snagline"):
        with mock.patch("urllib.request.urlopen", side_effect=OSError("down")):
            sink.emit(_risk(SEVERITY_CRITICAL))
    assert caplog.records, "the failed POST must be logged"
    for record in caplog.records:
        text = record.getMessage()
        assert "SECRET_TOKEN_DO_NOT_LOG" not in text, (
            "webhook secret reached the log record"
        )
        assert "hooks.slack.com" in text, "the log must still name the host"


def test_slack_repr_omits_the_secret():
    # repr lands in diagnostic dumps and unhandled-exception reports, so it
    # must not carry the credential either.
    assert "SECRET_TOKEN_DO_NOT_LOG" not in repr(SlackSink(_WEBHOOK_URL))
    assert "hooks.slack.com" in repr(SlackSink(_WEBHOOK_URL))
