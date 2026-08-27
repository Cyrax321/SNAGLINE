"""Tests for the CONTINUUM ledger adapter (issue #79).

The ``FakeStorage`` here implements ONLY the API surface verified against
current CONTINUUM source: ``read_events(run_id, *, after_sequence=0,
upto=None)`` and ``last_sequence(run_id)``. Any other attribute access raises,
so if the adapter starts depending on unverified Storage internals the tests
go red instead of silently coupling to them.

CONTINUUM itself is never imported; entries are hand-built duck-typed objects
with the verified ``Event`` fields (sequence, type, timestamp, payload).
"""

from __future__ import annotations

import importlib
import logging
import sys
from datetime import datetime, timezone
from typing import Any

import pytest

from snagline.adapters.continuum_adapter import ContinuumAdapter


class FakeEntry:
    """Duck-typed stand-in for ``continuum.events.Event``."""

    def __init__(
        self,
        sequence: int,
        type_: str,
        payload: dict[str, Any],
        timestamp: Any | None = None,
    ) -> None:
        self.sequence = sequence
        self.type = type_
        self.payload = payload
        self.timestamp = timestamp or datetime.now(timezone.utc)


class FakeStorage:
    """Exactly the verified read-by-sequence surface -- nothing more."""

    def __init__(self, entries: list[FakeEntry] | None = None) -> None:
        self.entries = entries or []
        self.read_calls: list[dict[str, Any]] = []
        self.fail_reads = False

    def read_events(
        self,
        run_id: str,
        *,
        after_sequence: int = 0,
        upto: int | None = None,
    ) -> list[FakeEntry]:
        self.read_calls.append(
            {"run_id": run_id, "after": after_sequence, "upto": upto}
        )
        if self.fail_reads:
            raise RuntimeError("storage unavailable")
        return [e for e in self.entries if e.sequence > after_sequence]

    def last_sequence(self, run_id: str) -> int:
        return self.entries[-1].sequence if self.entries else 0

    def __getattr__(self, name: str) -> Any:
        raise AssertionError(f"unexpected Storage API access: {name!r}")


class RecordingMonitor:
    def __init__(self) -> None:
        self.events: list[Any] = []

    def ingest(self, event: Any) -> None:
        self.events.append(event)


def make_perception(seq: int, **overrides: Any) -> FakeEntry:
    payload = {
        "observation_id": "obs_1",
        "source": "environment_observed",
        "trust_level": "verified",
        "verifier": "dom+consensus",
        "content_hash": "a" * 64,
        "q_vlm_model": "q-vlm-1",
        "raw_claim": "TOP SECRET SCREENSHOT TEXT",
    }
    payload.update(overrides)
    return FakeEntry(seq, "PERCEPTION_OBSERVED", payload)


def make_branch(seq: int, requires_review: bool = False) -> FakeEntry:
    return FakeEntry(
        seq,
        "BRANCH_RESOLVED",
        {
            "branch": {"branch_id": "br-9", "name": "deploy-prod"},
            "observation": {"observation_id": "obs_1"},
            "requires_review": requires_review,
        },
    )


def make_action(
    seq: int,
    status: str,
    key: str = "k1",
    action_type: str = "send_email",
    name: str = "ACTION_RECORDED",
    timestamp: Any | None = None,
) -> FakeEntry:
    return FakeEntry(
        seq,
        name,
        {
            "key": key,
            "action_id": f"action_{seq}",
            "action_type": action_type,
            "status": status,
            "external_id": "ext-1",
            "action": {"status": status},
        },
        timestamp=timestamp,
    )


def test_poll_translates_full_lifecycle() -> None:
    storage = FakeStorage(
        [
            make_perception(1),
            make_branch(2),
            make_action(3, "started"),
            make_action(4, "completed"),
            make_action(5, "failed", key="k2"),
            FakeEntry(
                6,
                "ACTION_RECONCILED",
                {"key": "k2", "status": "unknown", "action_type": "send_email"},
            ),
            FakeEntry(
                7,
                "ACTION_COMPENSATED",
                {"key": "k1", "status": "compensated", "action_type": "send_email"},
            ),
        ]
    )
    monitor = RecordingMonitor()
    adapter = ContinuumAdapter(monitor, storage, "run-1", start_at_tail=False)

    assert adapter.poll() == 7
    kinds = [e.action_type for e in monitor.events]
    assert kinds == [
        "observation",
        "plan_step",
        "tool_call",
        "tool_call",
        "tool_call",
        "action_reconciled",
        "action_compensated",
    ]
    assert [e.step_id for e in monitor.events] == ["1", "2", "3", "4", "5", "6", "7"]
    assert all(e.episode_id == "run-1" for e in monitor.events)
    # error flag rides the failed status only
    assert [e.error for e in monitor.events] == [
        False,
        False,
        False,
        False,
        True,
        False,
        False,
    ]
    # tool names come from the CONTINUUM action type / observation source
    assert monitor.events[0].tool_name == "environment_observed"
    assert {e.tool_name for e in monitor.events[2:]} == {"send_email"}
    # every event keeps structural provenance in metadata
    assert monitor.events[0].metadata["event_type"] == "PERCEPTION_OBSERVED"
    assert monitor.events[1].metadata["requires_review"] is False
    assert monitor.events[3].metadata["status"] == "completed"
    # reads advanced by sequence
    assert storage.read_calls[0]["after"] == 0
    adapter.poll()  # second poll sees nothing new
    assert storage.read_calls[1]["after"] == 7


def dt(epoch: float) -> Any:
    return datetime.fromtimestamp(epoch, tz=timezone.utc)


def test_latency_paired_from_ledger_times() -> None:
    """Claim and completion carry real ledger timestamps 1.5s apart."""
    storage = FakeStorage(
        [
            make_action(1, "started", timestamp=dt(1000.0)),
            make_action(2, "completed", timestamp=dt(1001.5)),
        ]
    )
    monitor = RecordingMonitor()
    adapter = ContinuumAdapter(monitor, storage, "run-1", start_at_tail=False)
    adapter.poll()
    assert adapter._pending_claims == {}  # terminal status closes the pairing
    assert monitor.events[1].latency_ms == pytest.approx(1500.0)
    assert monitor.events[0].latency_ms is None


def test_step_timestamps_come_from_the_ledger_not_the_wall_clock() -> None:
    """Replay must reproduce when steps actually happened (review fix)."""
    storage = FakeStorage([make_perception(1), make_branch(2)])
    for e in storage.entries:
        e.timestamp = dt(1700000000.0 + e.sequence)
    monitor = RecordingMonitor()
    ContinuumAdapter(monitor, storage, "run-1", start_at_tail=False).poll()
    assert monitor.events[0].timestamp == pytest.approx(1700000001.0)
    assert monitor.events[1].timestamp == pytest.approx(1700000002.0)


def test_unusable_entry_timestamp_falls_back_to_clock() -> None:
    storage = FakeStorage([make_perception(1)])
    storage.entries[0].timestamp = "not-a-time"
    ticks = iter([555.0])
    monitor = RecordingMonitor()
    adapter = ContinuumAdapter(
        monitor, storage, "run-1", start_at_tail=False, clock=lambda: next(ticks)
    )
    adapter.poll()
    assert monitor.events[0].timestamp == 555.0


def test_pending_claims_bounded_by_eviction(monkeypatch: pytest.MonkeyPatch) -> None:
    """Never-terminating actions are evicted oldest-first (review fix)."""
    import snagline.adapters.continuum_adapter as mod

    monkeypatch.setattr(mod, "_MAX_PENDING_CLAIMS", 2)
    entries = [make_action(i, "started", key=f"k{i}") for i in range(1, 5)]
    storage = FakeStorage(entries)
    monitor = RecordingMonitor()
    adapter = ContinuumAdapter(monitor, storage, "run-1", start_at_tail=False)
    adapter.poll()
    assert set(adapter._pending_claims) == {"k3", "k4"}  # oldest two evicted
    # an evicted key can no longer pair; a late terminal yields latency None
    storage.entries.append(make_action(5, "completed", key="k1"))
    adapter.poll()
    assert monitor.events[-1].latency_ms is None


def test_repeat_claim_keeps_original_start() -> None:
    storage = FakeStorage(
        [
            make_action(1, "started", timestamp=dt(10.0)),
            make_action(2, "started", timestamp=dt(11.0)),
            make_action(3, "completed", timestamp=dt(12.0)),
        ]
    )
    monitor = RecordingMonitor()
    adapter = ContinuumAdapter(monitor, storage, "run-1", start_at_tail=False)
    adapter.poll()
    # retry at t=11 must NOT reset the t=10 start: total 2000ms, not 1000ms
    assert monitor.events[-1].latency_ms == pytest.approx(2000.0)


def test_poll_defaults_to_live_tail() -> None:
    storage = FakeStorage([make_perception(i) for i in range(1, 6)] + [make_branch(6)])
    monitor = RecordingMonitor()
    adapter = ContinuumAdapter(monitor, storage, "run-1")
    # attaching skips all history, including the entry that was the tail
    assert adapter.poll() == 0
    assert monitor.events == []
    # entries appended after attaching are picked up
    storage.entries.append(make_perception(7))
    assert adapter.poll() == 1
    assert [e.step_id for e in monitor.events] == ["7"]


def test_unknown_entry_types_skipped_but_cursor_advances() -> None:
    storage = FakeStorage(
        [
            FakeEntry(1, "RUN_STARTED", {"goal": "x"}),
            FakeEntry(2, "TASK_UPDATED", {"progress": 1}),
            make_perception(3),
        ]
    )
    monitor = RecordingMonitor()
    adapter = ContinuumAdapter(monitor, storage, "run-1", start_at_tail=False)
    assert adapter.poll() == 1
    assert monitor.events[0].step_id == "3"
    adapter.poll()
    assert storage.read_calls[-1]["after"] == 3


def test_raw_claim_never_reaches_step_event() -> None:
    storage = FakeStorage([make_perception(1)])
    monitor = RecordingMonitor()
    ContinuumAdapter(monitor, storage, "run-1", start_at_tail=False).poll()

    def _walk(obj: Any) -> None:
        if isinstance(obj, dict):
            for k, v in obj.items():
                assert "raw_claim" != k
                _walk(v)
        elif isinstance(obj, (list, tuple)):
            for item in obj:
                _walk(item)

    for event in monitor.events:
        blob = repr(event.metadata) + repr(event.action_signature)
        assert "TOP SECRET" not in blob
        _walk(event.metadata)


def test_storage_error_is_swallowed_fail_open() -> None:
    storage = FakeStorage([make_perception(1)])
    storage.fail_reads = True
    monitor = RecordingMonitor()
    adapter = ContinuumAdapter(monitor, storage, "run-1", start_at_tail=False)
    assert adapter.poll() == 0  # no raise, no events
    storage.fail_reads = False
    assert adapter.poll() == 1  # recovers on the next poll


def test_poisoned_entry_does_not_stop_the_rest(
    caplog: pytest.LogCaptureFixture,
) -> None:
    class PoisonedEntry:
        def __init__(self) -> None:
            self.sequence = 1
            self.type = "PERCEPTION_OBSERVED"

        @property
        def payload(self) -> dict[str, Any]:
            raise RuntimeError("boom")

    bad = PoisonedEntry()
    good = make_perception(2)
    storage = FakeStorage([bad, good])
    monitor = RecordingMonitor()
    adapter = ContinuumAdapter(monitor, storage, "run-1", start_at_tail=False)
    with caplog.at_level(logging.WARNING, logger="snagline"):
        assert adapter.poll() == 1
    assert monitor.events[0].step_id == "2"


def test_monitor_failure_is_contained_and_poll_continues() -> None:
    class PickyMonitor(RecordingMonitor):
        def ingest(self, event: Any) -> None:
            if event.step_id == "1":
                raise RuntimeError("monitor exploded")
            super().ingest(event)

    storage = FakeStorage([make_perception(1), make_branch(2)])
    monitor = PickyMonitor()
    adapter = ContinuumAdapter(monitor, storage, "run-1", start_at_tail=False)
    assert adapter.poll() == 1  # branch still made it through
    assert [e.step_id for e in monitor.events] == ["2"]
    adapter.poll()
    assert storage.read_calls[-1]["after"] == 2


def test_push_mode_for_live_tails() -> None:
    storage = FakeStorage()
    monitor = RecordingMonitor()
    adapter = ContinuumAdapter(monitor, storage, "run-1")
    assert adapter.push(make_action(41, "started")) is True
    assert adapter.push(FakeEntry(42, "STATE_CHECKPOINTED", {})) is False
    assert [e.step_id for e in monitor.events] == ["41"]


def test_datetime_timestamps_accepted() -> None:
    entry = make_perception(1)
    entry.timestamp = datetime(2026, 8, 26, tzinfo=timezone.utc)
    storage = FakeStorage([entry])
    monitor = RecordingMonitor()
    assert ContinuumAdapter(monitor, storage, "run-1", start_at_tail=False).poll() == 1
    assert monitor.events[0].timestamp == entry.timestamp.timestamp()


def test_modules_import_without_continuum(monkeypatch: pytest.MonkeyPatch) -> None:
    """The extra is truly optional: poisoning 'continuum' changes nothing."""
    monkeypatch.setitem(sys.modules, "continuum", None)
    monkeypatch.setitem(sys.modules, "continuum.actions", None)
    adapter_mod = importlib.import_module("snagline.adapters.continuum_adapter")
    sink_mod = importlib.import_module("snagline.sinks.continuum_sink")
    assert adapter_mod.ContinuumAdapter is not None
    assert sink_mod.ContinuumSink is not None


def test_poll_rate_limits_storage_warnings(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Persistent outage logs once then suppresses repeats (issue #171)."""
    storage = FakeStorage([make_perception(1)])
    storage.fail_reads = True
    monitor = RecordingMonitor()
    adapter = ContinuumAdapter(monitor, storage, "run-1", start_at_tail=False)
    with caplog.at_level(logging.WARNING, logger="snagline"):
        for _ in range(5):
            assert adapter.poll() == 0
        warnings = [
            r for r in caplog.records if "read failed for run run-1" in r.message
        ]
        assert len(warnings) == 1
        assert warnings[0].levelname == "WARNING"
        assert adapter.consecutive_failures == 5
    # Recovery logs exactly once and resets the counter.
    storage.fail_reads = False
    caplog.clear()
    with caplog.at_level(logging.INFO, logger="snagline"):
        assert adapter.poll() == 1
        recoveries = [
            r for r in caplog.records if "read recovered for run run-1" in r.message
        ]
        assert len(recoveries) == 1
        assert "after 5 failure" in recoveries[0].message
        assert adapter.consecutive_failures == 0
    # Next poll with no new data must not log another recovery.
    caplog.clear()
    with caplog.at_level(logging.INFO, logger="snagline"):
        adapter.poll()
        assert not [r for r in caplog.records if "read recovered" in r.message]


def test_poll_logs_new_warning_after_recovery(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Intermittent failure after recovery surfaces a new warning (issue #171)."""
    storage = FakeStorage([make_perception(1)])
    storage.fail_reads = True
    monitor = RecordingMonitor()
    adapter = ContinuumAdapter(monitor, storage, "run-1", start_at_tail=False)
    with caplog.at_level(logging.WARNING, logger="snagline"):
        adapter.poll()
        assert len([r for r in caplog.records if "read failed" in r.message]) == 1
    storage.fail_reads = False
    with caplog.at_level(logging.INFO, logger="snagline"):
        adapter.poll()
        assert len([r for r in caplog.records if "read recovered" in r.message]) == 1
    # New outage after recovery must log a fresh warning.
    storage.fail_reads = True
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="snagline"):
        adapter.poll()
        warnings = [r for r in caplog.records if "read failed" in r.message]
        assert len(warnings) == 1
        assert adapter.consecutive_failures == 1
    # Fail-open still holds throughout.
    storage.fail_reads = False
    assert adapter.poll() == 0  # no new entries, but read succeeded
    assert adapter.consecutive_failures == 0
