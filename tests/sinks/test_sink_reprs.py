"""Every user-facing sink has a helpful, secret-free ``__repr__`` (issue #463).

Without these, a sink prints as ``<snagline.sinks.console.ConsoleSink object at
0x...>`` -- useless when inspecting a Monitor's sink list or logging a config.
The reprs must also never surface a credential (the #390/#404 rule): PagerDuty's
routing key in particular must not appear.

Slack/WebhookSink reprs are intentionally NOT covered here -- they carry a
destination URL treated as a credential and are handled (redacted) by the
#390/#404 work; this file owns only the sinks that PR does not touch.
"""

from __future__ import annotations

import logging

from snagline.sinks.batching import BatchingSink
from snagline.sinks.console import ConsoleSink
from snagline.sinks.continuum_sink import ContinuumSink
from snagline.sinks.dedup import DedupSink
from snagline.sinks.heartbeat import HeartbeatSink
from snagline.sinks.logging_sink import LoggingSink
from snagline.sinks.pagerduty import PagerDutySink


def test_console_repr_names_class_and_level() -> None:
    r = repr(ConsoleSink(level=logging.WARNING))
    assert r.startswith("ConsoleSink(")
    assert "WARNING" in r
    assert "0x" not in r  # not the default object repr


def test_logging_sink_repr_names_logger_and_level() -> None:
    r = repr(LoggingSink(logger=logging.getLogger("snagline.test"), level=logging.INFO))
    assert r.startswith("LoggingSink(")
    assert "snagline.test" in r
    assert "INFO" in r


def test_heartbeat_repr_shows_path() -> None:
    r = repr(HeartbeatSink(path="/tmp/hb.json"))
    assert r == "HeartbeatSink(path='/tmp/hb.json')"


def test_continuum_repr_shows_run_id_only() -> None:
    sink = ContinuumSink(object(), "run-42", ledger_factory=lambda s, r: object())
    assert repr(sink) == "ContinuumSink(run_id='run-42')"


def test_dedup_repr_nests_inner_sink_and_cooldown() -> None:
    inner = ConsoleSink(level=logging.WARNING)
    r = repr(DedupSink(inner, cooldown_seconds=30.0))
    assert r.startswith("DedupSink(inner=ConsoleSink(")
    assert "cooldown_seconds=30.0" in r


def test_batching_repr_nests_inner_sink_and_knobs() -> None:
    inner = ConsoleSink(level=logging.WARNING)
    r = repr(BatchingSink(inner, max_batch=50, flush_interval=2.5))
    assert r.startswith("BatchingSink(inner=ConsoleSink(")
    assert "max_batch=50" in r
    assert "flush_interval=2.5" in r


def test_pagerduty_repr_omits_the_routing_key() -> None:
    secret = "R0UT1NG-K3Y-SECRET"
    r = repr(PagerDutySink(routing_key=secret, source="svc", min_severity="critical"))
    assert secret not in r, "PagerDutySink repr must never surface the routing key"
    assert r.startswith("PagerDutySink(")
    assert "source='svc'" in r
    assert "min_severity='critical'" in r
