"""Tests for the Slack escalation sink (P1 alerting)."""

from __future__ import annotations

import json
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
