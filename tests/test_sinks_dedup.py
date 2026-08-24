"""Tests for FailureRisk severity and the DedupSink cooldown (P1, issue #4)."""

from __future__ import annotations

import time

from snagline.risk import (
    SEVERITY_CRITICAL,
    SEVERITY_INFO,
    SEVERITY_WARNING,
    FailureRisk,
    severity_from_score,
)
from snagline.sinks.dedup import DedupSink


def _risk(score: float, episode_id: str = "ep", **kw) -> FailureRisk:
    return FailureRisk(
        episode_id=episode_id,
        step_id="s1",
        score=score,
        trigger="loop",
        detail="repeating loop",
        timestamp=0.0,
        **kw,
    )


def test_severity_from_score_bands():
    assert severity_from_score(0.9) == SEVERITY_CRITICAL
    assert severity_from_score(0.6) == SEVERITY_WARNING
    assert severity_from_score(0.2) == SEVERITY_INFO


def test_risk_default_severity_derived_from_score():
    assert _risk(0.9).severity == SEVERITY_CRITICAL
    assert _risk(0.6).severity == SEVERITY_WARNING
    assert _risk(0.2).severity == SEVERITY_INFO


def test_risk_explicit_severity_kept():
    # When the caller sets severity explicitly, it is preserved exactly.
    assert _risk(0.9, severity=SEVERITY_INFO).severity == SEVERITY_INFO


def test_risk_explicit_warning_severity_is_not_overwritten():
    # "warning" used to double as the "unset" sentinel, so an explicit
    # severity=SEVERITY_WARNING was indistinguishable from an omitted one and
    # got silently replaced by the score-derived severity. Both directions
    # regressed: a high score escalated it to critical and a low score
    # downgraded it to info, against the documented "kept exactly" contract.
    assert _risk(0.9, severity=SEVERITY_WARNING).severity == SEVERITY_WARNING
    assert _risk(0.2, severity=SEVERITY_WARNING).severity == SEVERITY_WARNING
    # The band where derivation happens to agree must stay correct too.
    assert _risk(0.6, severity=SEVERITY_WARNING).severity == SEVERITY_WARNING


def test_risk_explicit_severity_survives_the_dedup_key():
    # severity is part of DedupSink's default key, so an overwritten severity
    # silently re-buckets an alert. Two risks the caller pinned to the same
    # severity must dedupe together even when their scores land in different
    # derivation bands.
    inner = _RecordingSink()
    sink = DedupSink(inner, cooldown_seconds=300.0)
    sink.emit(_risk(0.9, severity=SEVERITY_WARNING))
    sink.emit(_risk(0.2, severity=SEVERITY_WARNING))
    assert len(inner.emitted) == 1


class _RecordingSink:
    def __init__(self):
        self.emitted: list[FailureRisk] = []

    def emit(self, risk: FailureRisk) -> None:
        self.emitted.append(risk)


def test_dedup_suppresses_within_cooldown():
    inner = _RecordingSink()
    sink = DedupSink(inner, cooldown_seconds=10.0)
    sink.emit(_risk(0.9))
    sink.emit(_risk(0.9))  # same key, within cooldown -> suppressed
    assert len(inner.emitted) == 1


def test_dedup_emits_after_cooldown():
    inner = _RecordingSink()
    sink = DedupSink(inner, cooldown_seconds=0.05)
    sink.emit(_risk(0.9))
    time.sleep(0.08)
    sink.emit(_risk(0.9))  # cooldown elapsed -> re-emit
    assert len(inner.emitted) == 2


def test_dedup_distinct_keys_not_suppressed():
    inner = _RecordingSink()
    sink = DedupSink(inner, cooldown_seconds=100.0)
    sink.emit(_risk(0.9, episode_id="ep1"))
    sink.emit(_risk(0.9, episode_id="ep2"))
    assert len(inner.emitted) == 2


def test_dedup_custom_key_fn():
    inner = _RecordingSink()
    # Dedupe only by trigger, ignoring episode/severity.
    sink = DedupSink(inner, cooldown_seconds=100.0, key_fn=lambda r: r.trigger)
    sink.emit(_risk(0.9, episode_id="ep1"))
    sink.emit(_risk(0.2, episode_id="ep2"))  # different episode/severity, same trigger
    assert len(inner.emitted) == 1  # suppressed by custom key
