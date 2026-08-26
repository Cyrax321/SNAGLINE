"""Tests for the CONTINUUM sink (issue #79).

Unit tests run against a fake ledger mirroring the verified
``ActionLedger.flag_for_review(key, reason)`` surface. One optional test
exercises the REAL ``continuum.actions.ActionLedger`` over a minimal fake
storage when CONTINUUM is importable; it skips cleanly otherwise (zero-dep CI).
"""

from __future__ import annotations

import logging
from typing import Any

import pytest

from snagline.risk import FailureRisk
from snagline.sinks.continuum_sink import ContinuumSink


class FakeLedger:
    """Mirrors the verified flag_for_review(key, reason) -> Action surface."""

    def __init__(self) -> None:
        self.flagged: list[tuple[str, str]] = []
        self.fail = False

    def flag_for_review(self, key: str, reason: str) -> Any:
        if self.fail:
            raise RuntimeError("ledger down")
        self.flagged.append((key, reason))
        return {"key": key, "status": "requires_review"}


def make_risk(**overrides: Any) -> FailureRisk:
    fields: dict[str, Any] = {
        "episode_id": "run-1",
        "step_id": "42",
        "score": 0.91,
        "trigger": "loop",
        "detail": "same signature 5 times",
        "timestamp": 1000.0,
    }
    fields.update(overrides)
    return FailureRisk(**fields)


def make_sink(ledger: FakeLedger, **kwargs: Any) -> tuple[ContinuumSink, Any]:
    storage = object()  # the sink never touches storage directly, only the ledger
    sink = ContinuumSink(storage, "run-1", ledger_factory=lambda s, r: ledger, **kwargs)
    return sink, ledger


def test_emit_flags_action_for_review() -> None:
    sink, ledger = make_sink(FakeLedger(), key_from_risk=lambda r: "k1")
    risk = make_risk()
    sink.emit(risk)
    assert len(ledger.flagged) == 1
    key, reason = ledger.flagged[0]
    assert key == "k1"
    # structural reason string: trigger/score/severity/ids/detail, no metadata
    assert "[snagline:loop]" in reason
    assert "score=0.91" in reason
    assert "step=42" in reason
    assert "episode=run-1" in reason
    assert "same signature 5 times" in reason


def test_emit_without_key_mapping_drops_cleanly(
    caplog: pytest.LogCaptureFixture,
) -> None:
    sink, ledger = make_sink(FakeLedger())
    with caplog.at_level(logging.WARNING, logger="snagline"):
        sink.emit(make_risk())  # must not raise despite no channel
    assert ledger.flagged == []
    assert any("no action key" in rec.message for rec in caplog.records)


def test_key_from_risk_returning_none_drops(caplog: pytest.LogCaptureFixture) -> None:
    sink, ledger = make_sink(FakeLedger(), key_from_risk=lambda r: None)
    with caplog.at_level(logging.WARNING, logger="snagline"):
        sink.emit(make_risk())
    assert ledger.flagged == []


def test_ledger_failure_is_swallowed_fail_open(
    caplog: pytest.LogCaptureFixture,
) -> None:
    ledger = FakeLedger()
    ledger.fail = True
    sink, _ = make_sink(ledger, key_from_risk=lambda r: "k1")
    with caplog.at_level(logging.WARNING, logger="snagline"):
        sink.emit(make_risk())  # monitoring failure never reaches the host agent
    assert any("flag_for_review failed" in rec.message for rec in caplog.records)


def test_escalate_action_direct() -> None:
    sink, ledger = make_sink(FakeLedger())
    sink.escalate_action("k9", "human needed")
    assert ledger.flagged == [("k9", "human needed")]


def test_missing_continuum_raises_helpful_importerror(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(__import__("sys").modules, "continuum", None)
    monkeypatch.setitem(__import__("sys").modules, "continuum.actions", None)
    with pytest.raises(ImportError, match=r"snagline-agent\[continuum\]"):
        ContinuumSink(object(), "run-1")


def test_real_actionledger_flags_review() -> None:
    """Live check of the real request-human path; skipped without CONTINUUM."""
    continuum = pytest.importorskip("continuum")
    events_mod = pytest.importorskip("continuum.events")

    class MiniStorage:
        """Just enough Storage for ActionLedger's fold + append."""

        def __init__(self) -> None:
            self.events: list[Any] = []

        def read_events(self, run_id: str) -> list[Any]:
            return list(self.events)

        def read_archived_events(self, run_id: str) -> list[Any]:
            return []  # fake holds no compacted prefix

        def last_sequence(self, run_id: str) -> int:
            return len(self.events)

        def append_event(
            self,
            run_id: str,
            type: Any = None,
            payload: Any = None,
        ) -> Any:
            event = events_mod.Event(
                run_id=run_id,
                sequence=len(self.events) + 1,
                type=type,
                payload=dict(payload or {}),
            )
            self.events.append(event)
            return event

    storage = MiniStorage()
    ledger = continuum.ActionLedger(storage, "run-1")  # type: ignore[attr-defined]
    outcome = ledger.claim("send_email", {"to": "x"})
    # key_from_risk must hand the sink the *resolved* idempotency key (or
    # action_id); a bare action name does not resolve, as the real ledger
    # refuses it with LedgerError.
    action = ledger.flag_for_review(outcome.key, "[snagline] review me")
    assert str(action.status.value) == "requires_review"
    statuses = [str(e.payload.get("status")) for e in storage.events]
    assert "requires_review" in statuses
    # and the real EventType vocabulary matches what the adapter translates
    assert events_mod.EventType.PERCEPTION_OBSERVED.value == "PERCEPTION_OBSERVED"
    assert events_mod.EventType.BRANCH_RESOLVED.value == "BRANCH_RESOLVED"
