"""Prometheus exposition tests for GET /metrics (issue #98).

Every detector-side assertion here has both sides: an injected-failure
sequence that must fire (exact trigger label) and a healthy sequence that
must stay silent. Expected counts are derived from the documented detector
thresholds, never by calling the code under test.
"""

from __future__ import annotations

import json
import re
import threading
import urllib.request

from snagline.monitor import Monitor
from snagline.server.http_server import (
    PROMETHEUS_CONTENT_TYPE,
    SidecarMetricsCollector,
    _escape_label,
    make_server,
)

# Sanity shape every sample line must satisfy (spec of issue #98).
SAMPLE_LINE_RE = re.compile(r"^[a-zA-Z_:][a-zA-Z0-9_:]*(\{.*\})? [0-9.eE+-]+$")


def _start_server(**kwargs):
    server = make_server(Monitor.default(sinks=[]), host="127.0.0.1", port=0, **kwargs)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, f"http://127.0.0.1:{server.server_address[1]}"


def _stop(server) -> None:
    server.shutdown()
    server.server_close()


def _get(base: str, path: str, headers: dict | None = None):
    req = urllib.request.Request(base + path, headers=headers or {}, method="GET")
    with urllib.request.urlopen(req, timeout=5) as resp:
        return resp.status, dict(resp.headers), resp.read().decode("utf-8")


def _post(base: str, payload) -> None:
    req = urllib.request.Request(
        base + "/events",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=5) as resp:
        assert resp.status == 202


def _step(index, sig, episode="ep-metrics", error=False):
    return {
        "step_id": str(index),
        "episode_id": episode,
        "timestamp": 1718300000.0 + index,
        "action_type": "tool_call",
        "action_signature": sig,
        "tool_name": "search",
        "error": error,
    }


def _parse_samples(body: str) -> dict:
    """Parse exposition lines into {(name, labels): value}, asserting shape."""
    samples: dict = {}
    for line in body.splitlines():
        if not line or line.startswith("#"):
            continue
        assert SAMPLE_LINE_RE.match(line), f"bad exposition line: {line!r}"
        head, _, value = line.rpartition(" ")
        name, _, labels = head.partition("{")
        samples[(name, labels.rstrip("}"))] = float(value)
    return samples


def test_empty_monitor_serves_valid_prometheus():
    server, base = _start_server()
    try:
        status, headers, body = _get(base, "/metrics")
        assert status == 200
        assert headers["Content-Type"] == PROMETHEUS_CONTENT_TYPE
        # HELP and TYPE lines exist for every family even with zero traffic.
        for family in (
            "snagline_events_total",
            "snagline_risks_total",
            "snagline_episodes_active",
            "snagline_ingest_seconds",
            "snagline_monitor_events_ingested_total",
            "snagline_monitor_risks_emitted_total",
            "snagline_monitor_detector_errors_total",
            "snagline_monitor_sink_errors_total",
        ):
            assert f"# HELP {family} " in body, family
            assert f"# TYPE {family} " in body, family
        samples = _parse_samples(body)  # also asserts per-line sanity regex
        assert samples[("snagline_events_total", "")] == 0.0
        assert samples[("snagline_episodes_active", "")] == 0.0
        assert samples[("snagline_ingest_seconds_count", "")] == 0.0
        assert samples[("snagline_ingest_seconds_sum", "")] == 0.0
        assert samples[("snagline_monitor_detector_errors_total", "")] == 0.0
        # No risk samples at all on an empty monitor.
        assert not any(name == "snagline_risks_total" for name, _ in samples)
    finally:
        _stop(server)


def test_loop_fires_with_exact_labels_and_counters_grow_monotonically():
    server, base = _start_server()
    try:
        # Loop detector default repeat_threshold=3: the third identical
        # signature fires once; the fourth is suppressed by dedup (issue #4).
        loop_sig = "aaaa1111bbbb2222"
        for i in range(4):
            _post(base, _step(i, loop_sig, episode="ep-loop"))
        _, _, first_body = _get(base, "/metrics")
        first = _parse_samples(first_body)
        assert (
            first[("snagline_risks_total", 'trigger="loop",severity="warning"')] == 1.0
        )
        assert first[("snagline_events_total", "")] == 4.0
        assert first[("snagline_ingest_seconds_count", "")] == 4.0
        assert first[("snagline_monitor_risks_emitted_total", "")] == 1.0
        # Score at count=3 is min(1, 3/3*0.5)=0.5, so severity is "warning"
        # exactly; no other trigger may have fired from this sequence.
        assert (
            "snagline_risks_total",
            'trigger="loop",severity="critical"',
        ) not in first

        # Error-cascade default consecutive_threshold=3: three consecutive
        # failing tool calls with distinct signatures fire error_cascade once
        # with score 1.0, i.e. severity "critical", and never trip loop.
        cascade_sigs = ["bbb11111cccc2222", "bbb22222cccc2222", "bbb33333cccc2222"]
        _post(
            base,
            [
                _step(i + 10, s, episode="ep-cascade", error=True)
                for i, s in enumerate(cascade_sigs)
            ],
        )
        _, _, second_body = _get(base, "/metrics")
        second = _parse_samples(second_body)

        assert (
            second[
                ("snagline_risks_total", 'trigger="error_cascade",severity="critical"')
            ]
            == 1.0
        )
        # The earlier loop finding is unchanged (dedup keeps it at one).
        assert (
            second[("snagline_risks_total", 'trigger="loop",severity="warning"')] == 1.0
        )
        assert second[("snagline_monitor_risks_emitted_total", "")] == 2.0
        # Monotonic growth between the two scrapes, exact deltas where known.
        for key, value in first.items():
            if key[0].endswith("_total") or key[0].startswith("snagline_ingest"):
                assert second[key] >= value, key
        assert (
            second[("snagline_events_total", "")]
            == first[("snagline_events_total", "")] + 3.0
        )
        assert second[("snagline_episodes_active", "")] >= 2.0
    finally:
        _stop(server)


def test_healthy_sequence_stays_completely_silent():
    server, base = _start_server()
    try:
        # Six distinct successful steps: below every threshold, so nothing
        # may fire and no risk sample line may appear at all.
        healthy_sigs = [f"dead{i:04x}beef5678" for i in range(6)]
        for i, s in enumerate(healthy_sigs):
            _post(base, _step(i, s))
        _, _, body = _get(base, "/metrics")
        assert "snagline_risks_total{" not in body
        samples = _parse_samples(body)
        assert samples[("snagline_monitor_risks_emitted_total", "")] == 0.0
        assert samples[("snagline_episodes_active", "")] == 1.0
        assert samples[("snagline_events_total", "")] == 6.0
        # Ingest latency was observed count-wise, sum stays tiny but positive.
        assert samples[("snagline_ingest_seconds_count", "")] == 6.0
        assert samples[("snagline_ingest_seconds_sum", "")] > 0.0
    finally:
        _stop(server)


def test_classic_format_stays_available_for_old_clients():
    server, base = _start_server()
    try:
        status, headers, body = _get(base, "/metrics?format=classic")
        assert status == 200
        assert headers["Content-Type"] == "application/json"
        parsed = json.loads(body)
        for key in (
            "events_ingested",
            "risks_emitted",
            "detector_errors",
            "sink_errors",
        ):
            assert key in parsed

        # Old JSON clients are also honored via Accept negotiation.
        status, headers, body = _get(
            base, "/metrics", headers={"Accept": "application/json"}
        )
        assert status == 200
        assert headers["Content-Type"] == "application/json"
        json.loads(body)

        # Explicit ?format=prometheus wins over a JSON Accept header.
        status, headers, body = _get(
            base, "/metrics?format=prometheus", headers={"Accept": "application/json"}
        )
        assert status == 200
        assert headers["Content-Type"] == PROMETHEUS_CONTENT_TYPE
    finally:
        _stop(server)


def test_env_toggle_switches_default_format(monkeypatch):
    monkeypatch.setenv("SNAGLINE_METRICS_FORMAT", "classic")
    server, base = _start_server()
    try:
        _, headers, body = _get(base, "/metrics")
        assert headers["Content-Type"] == "application/json"
        json.loads(body)
    finally:
        _stop(server)

    # Unknown values fall back to prometheus instead of breaking startup.
    monkeypatch.setenv("SNAGLINE_METRICS_FORMAT", "nonsense")
    server, base = _start_server()
    try:
        _, headers, _body = _get(base, "/metrics")
        assert headers["Content-Type"] == PROMETHEUS_CONTENT_TYPE
    finally:
        _stop(server)


def test_escape_label_handles_backslash_quote_newline():
    # Hand-derived expectation: backslash doubles, quote gets a backslash,
    # newline becomes the two-character escape.
    assert _escape_label('a"b\\c\nd') == 'a\\"b\\\\c\\nd'
    assert _escape_label("plain") == "plain"


def test_episode_table_evicts_oldest_at_cap():
    collector = SidecarMetricsCollector(max_episodes=2)
    collector.record_ingest("ep-a", 0.001)
    collector.record_ingest("ep-b", 0.001)
    assert collector.snapshot()["episodes_active"] == 2
    collector.record_ingest("ep-c", 0.001)
    snap = collector.snapshot()
    assert snap["episodes_active"] == 2
    assert snap["events_total"] == 3
    # Rendering still yields a valid single gauge line after eviction.
    body = collector.render_prometheus(
        {
            "events_ingested": 0,
            "risks_emitted": 0,
            "detector_errors": 0,
            "sink_errors": 0,
        }
    )
    samples = _parse_samples(body)
    assert samples[("snagline_episodes_active", "")] == 2.0


def test_collector_counts_risks_per_label_pair():
    from snagline.risk import FailureRisk

    collector = SidecarMetricsCollector()
    ts = 1718300000.0
    collector.emit(FailureRisk("ep", "s1", 0.9, "loop", "detail", ts))
    collector.emit(FailureRisk("ep", "s2", 1.0, "error_cascade", "detail", ts))
    collector.emit(FailureRisk("ep", "s3", 0.6, "loop", "detail", ts))
    body = collector.render_prometheus(
        {
            "events_ingested": 0,
            "risks_emitted": 0,
            "detector_errors": 0,
            "sink_errors": 0,
        }
    )
    samples = _parse_samples(body)
    # Hand-derived from severity_from_score: >=0.8 critical, >=0.5 warning.
    # 0.9 -> critical, 1.0 -> critical, 0.6 -> warning.
    assert samples[("snagline_risks_total", 'trigger="loop",severity="warning"')] == 1.0
    assert (
        samples[("snagline_risks_total", 'trigger="loop",severity="critical"')] == 1.0
    )
    assert (
        samples[("snagline_risks_total", 'trigger="error_cascade",severity="critical"')]
        == 1.0
    )
