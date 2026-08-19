"""Tests for the PagerDuty escalation sink (P1 alerting)."""

from __future__ import annotations

import json
from unittest import mock

from snagline.risk import SEVERITY_CRITICAL, SEVERITY_INFO, FailureRisk
from snagline.sinks.pagerduty import _EVENTS_API, PagerDutySink


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


def test_pagerduty_posts_trigger_event():
    sink = PagerDutySink("RKEY")
    with mock.patch("urllib.request.urlopen") as urlopen:
        sink.emit(_risk(SEVERITY_CRITICAL))
    assert urlopen.called
    req = urlopen.call_args[0][0]
    assert req.full_url == _EVENTS_API
    body = json.loads(req.data.decode())
    assert body["routing_key"] == "RKEY"
    assert body["event_action"] == "trigger"
    assert body["payload"]["severity"] == "critical"
    assert "loop" in body["payload"]["summary"]
    assert body["custom_details"]["episode_id"] == "ep"


def test_pagerduty_maps_info_severity():
    sink = PagerDutySink("RKEY")
    with mock.patch("urllib.request.urlopen") as urlopen:
        sink.emit(_risk(SEVERITY_INFO))
    body = json.loads(urlopen.call_args[0][0].data.decode())
    assert body["payload"]["severity"] == "info"


def test_pagerduty_min_severity_filters_lower():
    sink = PagerDutySink("RKEY", min_severity=SEVERITY_CRITICAL)
    with mock.patch("urllib.request.urlopen") as urlopen:
        sink.emit(_risk(SEVERITY_INFO))
    assert not urlopen.called


def test_pagerduty_swallows_post_errors():
    sink = PagerDutySink("RKEY")
    with mock.patch("urllib.request.urlopen", side_effect=OSError("down")):
        sink.emit(_risk(SEVERITY_CRITICAL))  # must not raise
