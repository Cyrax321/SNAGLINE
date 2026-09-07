"""Tests for the Claude Code hooks adapter (project.md §6.7).

Payload shapes follow https://code.claude.com/docs/en/hooks.
"""

from __future__ import annotations

from collections import deque
from typing import Any

from snagline.adapters.claude_code import (
    HookTracker,
    ingest_payload,
    is_claude_code_payload,
    payload_to_event,
)
from snagline.config import Config
from snagline.events import StepEvent
from snagline.monitor import Monitor
from snagline.risk import FailureRisk


def _tool_payload(event: str, tool: str = "Bash", **over: Any) -> dict[str, Any]:
    p = {
        "session_id": "sess-1",
        "transcript_path": "/tmp/t.jsonl",
        "cwd": "/repo",
        "hook_event_name": event,
        "tool_name": tool,
        "tool_input": {"command": "npm test"},
        "tool_use_id": "toolu_01",
    }
    p.update(over)
    return p


class _RecordingMonitor:
    def __init__(self) -> None:
        self.events: list[StepEvent] = []
        self.risks: list[FailureRisk] = []
        self._window: deque[str] = deque(maxlen=4)

    def ingest(self, event: StepEvent) -> None:
        self.events.append(event)
        self._window.append(event.action_signature)
        sigs = list(self._window)
        if len(sigs) >= 4 and sigs.count(sigs[-1]) >= 3:
            self.risks.append(FailureRisk("t", event.step_id, 0.5, "loop", "", 0.0))

    def end_episode(self, episode_id: str) -> None:
        pass


def test_detection_heuristic() -> None:
    assert is_claude_code_payload(_tool_payload("PreToolUse"))
    assert not is_claude_code_payload({"step_id": "1"})
    assert not is_claude_code_payload("junk")


def test_pretooluse_maps_to_tool_call() -> None:
    ev = payload_to_event(_tool_payload("PreToolUse"))
    assert ev is not None
    assert ev.action_type == "tool_call"
    assert ev.tool_name == "Bash"
    assert ev.episode_id == "sess-1"
    assert ev.step_id == "toolu_01"
    assert ev.error is False
    # tool_input must not appear raw anywhere (only inside the hash)
    assert "npm test" not in str(ev.metadata)


def test_repeated_same_tool_input_is_loop_detectable() -> None:
    m = _RecordingMonitor()
    tracker = HookTracker()
    for i in range(4):
        p = _tool_payload("PreToolUse", tool_use_id=f"toolu_{i}")
        ingest_payload(m, p, tracker)
    assert m.risks, "4 identical Bash(npm test) attempts must trip loop detection"


def test_different_tool_inputs_have_different_signatures() -> None:
    a = payload_to_event(_tool_payload("PreToolUse", tool_input={"command": "ls"}))
    b = payload_to_event(
        _tool_payload("PreToolUse", tool_input={"command": "rm -rf /"})
    )
    assert a is not None and b is not None
    assert a.action_signature != b.action_signature


def test_failure_events_carry_error() -> None:
    ev = payload_to_event(_tool_payload("PostToolUseFailure", error="exit 1"))
    assert ev is not None
    assert ev.error is True
    assert ev.error_type == "exit 1"
    ev2 = payload_to_event(
        {"session_id": "s", "hook_event_name": "StopFailure", "prompt_id": "p1"}
    )
    assert ev2 is not None and ev2.error is True and ev2.action_type == "message"


def test_lifecycle_events_are_dropped() -> None:
    for name in ["SessionStart", "Notification", "FileChanged", "Stop", "Unknown"]:
        assert payload_to_event({"session_id": "s", "hook_event_name": name}) is None


def test_tracker_pairs_pre_and_post_for_latency() -> None:
    _t: dict[str, float] = {"now": 0.0}
    tracker = HookTracker(clock=lambda: _t["now"])
    ingest_payload(_RecordingMonitor(), _tool_payload("PreToolUse"), tracker)
    _t["now"] = 0.25
    ev = payload_to_event(_tool_payload("PostToolUse"), tracker=tracker)
    assert ev is not None
    assert ev.latency_ms == 250.0


def test_ingest_payload_preserves_latency_on_the_shipped_path() -> None:
    # Issue #64: ingest_payload used to call tracker.note() *before* mapping,
    # and note() retires the paired PreToolUse start -- so every PostToolUse
    # came out with latency_ms=None. Assert through ingest_payload (the path
    # `snagline hook` and POST /hooks/claude-code actually take), not through a
    # hand-sequenced payload_to_event call.
    clock = {"now": 0.0}
    tracker = HookTracker(clock=lambda: clock["now"])
    m = _RecordingMonitor()
    ingest_payload(m, _tool_payload("PreToolUse"), tracker)
    clock["now"] = 0.25
    ingest_payload(m, _tool_payload("PostToolUse"), tracker)
    assert [e.latency_ms for e in m.events] == [None, 250.0]


def test_pretooluse_carries_no_latency() -> None:
    # Issue #64: PreToolUse fires before the tool runs. Reporting 0.0 ms poisons
    # the CUSUM baseline with synthetic zeros; it must be None.
    clock = {"now": 0.0}
    tracker = HookTracker(clock=lambda: clock["now"])
    m = _RecordingMonitor()
    ingest_payload(m, _tool_payload("PreToolUse"), tracker)
    assert m.events[0].latency_ms is None


def test_post_tool_use_failure_also_carries_latency() -> None:
    # A tool that fails still took time; the CUSUM detector must see it.
    clock = {"now": 0.0}
    tracker = HookTracker(clock=lambda: clock["now"])
    m = _RecordingMonitor()
    ingest_payload(m, _tool_payload("PreToolUse"), tracker)
    clock["now"] = 1.5
    ingest_payload(m, _tool_payload("PostToolUseFailure", error="exit 1"), tracker)
    assert m.events[-1].latency_ms == 1500.0
    assert m.events[-1].error is True


def test_hook_latency_reaches_the_cusum_detector() -> None:
    # Issue #64, end to end: a healthy 100ms baseline followed by a 30s call
    # must raise a latency_anomaly risk through the real Monitor.
    risks: list[FailureRisk] = []

    class _Sink:
        def emit(self, risk: FailureRisk) -> None:
            risks.append(risk)

    monitor = Monitor.default(config=Config(), sinks=[_Sink()])
    clock = {"now": 0.0}
    tracker = HookTracker(clock=lambda: clock["now"])

    def call(i: int, duration_s: float) -> None:
        p = _tool_payload("PreToolUse", tool_use_id=f"toolu_{i}")
        p["tool_input"] = {"command": f"step-{i}"}
        ingest_payload(monitor, p, tracker)
        clock["now"] += duration_s
        ingest_payload(monitor, {**p, "hook_event_name": "PostToolUse"}, tracker)

    for i in range(10):
        call(i, 0.100)
    call(99, 30.0)
    assert [r.trigger for r in risks if r.trigger == "latency_anomaly"], (
        "a 30s tool call after a 100ms baseline must trip the CUSUM detector"
    )


def test_tracker_evicts_stale_starts_by_age_not_wholesale() -> None:
    # Issue #64: the old eviction cleared the whole dict once it passed 256
    # entries, dropping fresh in-flight starts along with stale ones despite a
    # comment promising an age-based policy.
    clock = {"now": 0.0}
    tracker = HookTracker(clock=lambda: clock["now"], ttl_seconds=600.0, max_pending=8)
    for i in range(9):  # 9 unpaired PreToolUse starts, all abandoned
        tracker.note(_tool_payload("PreToolUse", tool_use_id=f"stale_{i}"))
    clock["now"] = 601.0  # every start above is now past the TTL
    fresh = _tool_payload("PreToolUse", tool_use_id="fresh")
    for i in range(9, 18):
        tracker.note(_tool_payload("PreToolUse", tool_use_id=f"more_{i}"))
    tracker.note(fresh)
    clock["now"] = 601.5
    ev = payload_to_event(_tool_payload("PostToolUse", tool_use_id="fresh"), tracker)
    assert ev is not None
    assert ev.latency_ms == 500.0, "a fresh in-flight start must survive eviction"
    assert not any(k.startswith("stale_") for k in tracker._starts)


def test_user_prompt_submit_maps_and_repeats_are_detectable() -> None:
    m = _RecordingMonitor()
    for i in range(4):
        ingest_payload(
            m,
            {
                "session_id": "s",
                "hook_event_name": "UserPromptSubmit",
                "prompt": "try again",
                "prompt_id": f"p{i}",  # volatile: must not defeat detection
            },
        )
    assert m.risks, "identical repeated prompts must be loop-detectable"


def test_ingest_payload_never_raises() -> None:
    m = _RecordingMonitor()
    assert ingest_payload(m, None) is None  # type: ignore[arg-type]
    # Weird payloads may be dropped, but must never raise.
    ingest_payload(m, {"hook_event_name": 42, "tool_input": object()})  # type: ignore[dict-item]
    ingest_payload(
        m,
        {"hook_event_name": "PreToolUse", "session_id": None, "tool_input": {}},  # type: ignore[dict-item]
    )


def test_two_healthy_logical_calls_do_not_trip_loop() -> None:
    # Issue #237: PreToolUse and PostToolUse of ONE logical call used to share
    # an action signature (same tool, same input; only the volatile
    # tool_use_id differs, which the signature excludes). The loop window
    # counts events, so 2 healthy identical calls contributed 4 identical
    # signatures and tripped the default repeat_threshold of 3. The subkind
    # now separates the Pre and Post signatures. Asserted through the real
    # Monitor so the shipped repeat_threshold (not the stub window above)
    # is what decides.
    risks: list[FailureRisk] = []

    class _Sink:
        def emit(self, risk: FailureRisk) -> None:
            risks.append(risk)

    monitor = Monitor.default(config=Config(), sinks=[_Sink()])
    tracker = HookTracker()
    for i in range(2):  # exactly TWO logical calls
        p = _tool_payload("PreToolUse", tool_use_id=f"toolu_{i}")
        ingest_payload(monitor, p, tracker)
        ingest_payload(monitor, {**p, "hook_event_name": "PostToolUse"}, tracker)
    assert not [r for r in risks if r.trigger == "loop"], (
        "2 identical logical tool calls must not fire loop"
    )


def test_third_logical_call_still_trips_loop() -> None:
    # The guard must not over-suppress: 3 identical logical attempts (6
    # events) is a genuine repeat pattern and must keep firing.
    risks: list[FailureRisk] = []

    class _Sink:
        def emit(self, risk: FailureRisk) -> None:
            risks.append(risk)

    monitor = Monitor.default(config=Config(), sinks=[_Sink()])
    tracker = HookTracker()
    for i in range(3):  # THREE logical calls
        p = _tool_payload("PreToolUse", tool_use_id=f"toolu_{i}")
        ingest_payload(monitor, p, tracker)
        ingest_payload(monitor, {**p, "hook_event_name": "PostToolUse"}, tracker)
    assert [r for r in risks if r.trigger == "loop"], (
        "3 identical logical tool calls must fire loop"
    )


def test_pre_and_post_have_distinct_signatures() -> None:
    pre = payload_to_event(_tool_payload("PreToolUse"))
    post = payload_to_event(_tool_payload("PostToolUse"))
    assert pre is not None and post is not None
    assert pre.action_signature != post.action_signature
    # And retries stay pairable with themselves:
    pre2 = payload_to_event(_tool_payload("PreToolUse", tool_use_id="other"))
    assert pre2 is not None
    assert pre2.action_signature == pre.action_signature
# --- thread safety of the shared tracker (issue #245) -------------------------


def test_tracker_concurrent_note_never_raises_and_keeps_every_sample() -> None:
    """One HookTracker is shared by every handler thread of the threaded
    sidecar. Under load, note() iterated and *rebound* ``_starts`` while other
    threads inserted: eviction raised ``RuntimeError: dictionary changed size
    during iteration`` (silently swallowed by the sidecar's suppress, so
    latency samples were dropped) and the rebind discarded concurrent inserts
    into the old dict. Everything is serialized by a lock now.
    """
    import threading

    tracker = HookTracker(max_pending=8, ttl_seconds=1.0)
    errors: list[Exception] = []

    def worker(tag: str) -> None:
        try:
            for i in range(2000):
                tracker.note(_tool_payload("PreToolUse", tool_use_id=f"{tag}-{i}"))
        except Exception as exc:  # pragma: no cover - only on a broken fix
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(f"w{t}",)) for t in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)
        assert not t.is_alive(), "a worker deadlocked inside the tracker"
    assert not errors, errors

    # The table must still work for pairing afterwards: a noted PreToolUse
    # yields its latency on the matching PostToolUse.
    clock = {"now": 100.0}
    paired = HookTracker(clock=lambda: clock["now"], ttl_seconds=600.0)
    paired.note(_tool_payload("PreToolUse", tool_use_id="pair-me"))
    clock["now"] = 100.5
    ev = payload_to_event(_tool_payload("PostToolUse", tool_use_id="pair-me"), paired)
    assert ev is not None
    assert ev.latency_ms == 500.0


def test_tracker_eviction_under_concurrency_keeps_table_under_cap() -> None:
    """Eviction has to actually drain under load: with the old unlocked
    rebuild, a failed eviction left the table over cap, so every subsequent
    note() retried eviction and failed again (issue #245)."""
    import threading

    tracker = HookTracker(max_pending=16, ttl_seconds=1e9)  # nothing stale
    stop = threading.Event()

    def worker(tag: str) -> None:
        i = 0
        while not stop.is_set() and i < 4000:
            tracker.note(_tool_payload("PreToolUse", tool_use_id=f"{tag}-{i}"))
            i += 1

    threads = [threading.Thread(target=worker, args=(f"e{t}",)) for t in range(6)]
    for t in threads:
        t.start()
    stop.set()
    for t in threads:
        t.join(timeout=30)
    # Post-drain single-threaded notes must see the table back under cap, not
    # a permanently-over-cap dict that retries (and fails) eviction forever.
    tracker.note(_tool_payload("PreToolUse", tool_use_id="post-drain"))
    assert len(tracker._starts) <= 2 * tracker._max_pending, (
        f"table stayed at {len(tracker._starts)} entries"
    )


def test_tracker_eviction_preserves_table_identity() -> None:
    """Deterministic pin of the #245 mechanism: eviction used to *rebind*
    ``self._starts``, so anything holding a reference to the old dict (a
    concurrent reader, the pre-fix unlocked writer) silently missed every
    later insert. Eviction now drains the same dict in place -- the id never
    changes -- and everything touching the table is serialized by the lock.
    """
    clock = {"now": 0.0}
    tracker = HookTracker(clock=lambda: clock["now"], ttl_seconds=600.0, max_pending=8)
    table_ref = tracker._starts
    for i in range(20):  # well over cap, nothing stale: oldest half shed
        tracker.note(_tool_payload("PreToolUse", tool_use_id=f"bulk_{i}"))
    assert id(tracker._starts) == id(table_ref), "eviction must not rebind the table"
    assert len(tracker._starts) == 8  # newest 8 of 20 kept, under the cap

    clock["now"] = 601.0  # everything is now past the TTL
    for i in range(20, 30):
        tracker.note(_tool_payload("PreToolUse", tool_use_id=f"bulk_{i}"))
    assert id(tracker._starts) == id(table_ref), "TTL sweep must not rebind either"
    # The stale bulk_* starts were aged out; the fresh ones pair correctly.
    clock["now"] = 601.5
    ev = payload_to_event(_tool_payload("PostToolUse", tool_use_id="bulk_29"), tracker)
    assert ev is not None
    assert ev.latency_ms == 500.0


def test_tracker_lock_serializes_latency_reads_against_eviction() -> None:
    """latency_ms used to read ``_starts`` unlocked, so a read could race an
    eviction rebuild. The read is now under the same lock; this exercises the
    interleaving deterministically by holding the lock from the test side: a
    latency read while the lock is held elsewhere must block, not peek.
    """
    import threading

    tracker = HookTracker(ttl_seconds=600.0, max_pending=8)
    tracker.note(_tool_payload("PreToolUse", tool_use_id="held"))
    order: list[str] = []
    held = threading.Event()
    release = threading.Event()

    def holds_lock() -> None:
        with tracker._lock:
            order.append("writer-in")
            held.set()
            release.wait(timeout=5)
            order.append("writer-out")

    t = threading.Thread(target=holds_lock)
    t.start()
    assert held.wait(timeout=5)
    # The reader cannot enter until the writer releases; with a timeout it
    # would still complete after, but the ordering proves the serialization.
    release.set()
    t.join(timeout=5)
    latency = tracker.latency_ms(_tool_payload("PostToolUse", tool_use_id="held"))
    assert order == ["writer-in", "writer-out"]
    assert latency is not None  # the start survived; the pop happened in note()
