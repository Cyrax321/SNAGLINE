"""Tests for the shared sink plumbing in ``sinks/base.py``."""

from __future__ import annotations

from snagline.sinks.base import redacted_destination


def test_redacted_strips_basic_auth_credentials():
    # ``https://user:pass@host/`` -- the userinfo is the credential.
    assert redacted_destination("https://user:pass@hooks.example/hook") == (
        "https://hooks.example/"
    )


def test_redacted_strips_slack_path_secret():
    # A Slack incoming webhook embeds its secret as the final path segment.
    url = "https://hooks.slack.com/services/T0/B0/SECRET_TOKEN_HERE"
    assert redacted_destination(url) == "https://hooks.slack.com/"
    assert "SECRET_TOKEN_HERE" not in redacted_destination(url)


def test_redacted_strips_query_and_path():
    # Everything but scheme + host is dropped, including query strings that
    # may carry a token.
    out = redacted_destination("https://hooks.example/a/b?token=[REDACTED]")
    assert out == "https://hooks.example/"
    assert "SECRET" not in out


def test_redacted_keeps_port():
    # A non-default port is needed to tell two internal endpoints apart and
    # is not a credential.
    assert redacted_destination("http://localhost:8080/hook") == (
        "http://localhost:8080/"
    )


def test_redacted_handles_url_without_scheme():
    # An unparseable destination must be reported, not echoed back.
    out = redacted_destination("not a url at all")
    assert out == "<invalid destination url>"
    assert "not a url" not in out


def test_redacted_handles_url_without_host():
    out = redacted_destination("https://")
    assert out == "<invalid destination url>"


def test_redacted_handles_ipv6_loopback():
    out = redacted_destination("https://[::1]:8443/hook")
    assert out == "https://[::1]:8443/"


def test_redacted_handles_userinfo_without_host():
    # Degenerate input must not echo anything back.
    out = redacted_destination("https://user:pass@")
    assert out == "<invalid destination url>"
