"""Tests for the Claude Code hooks adapter (project.md §6.7).

Payload shapes follow https://code.claude.com/docs/en/hooks.
"""

from __future__ import annotations

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


def _tool_payload(event: str, tool: str = "Bash", **over) -> dict:
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
    def __init__(self):
        self.events: list[StepEvent] = []
        self.risks: list[FailureRisk] = []

    def ingest(self, event: StepEvent) -> None:
        self.events.append(event)
        from collections import deque

        self._window = getattr(self, "_window", deque(maxlen=4))
        self._window.append(event.action_signature)
        sigs = list(self._window)
        if len(sigs) >= 4 and sigs.count(sigs[-1]) >= 3:
            self.risks.append(FailureRisk("t", event.step_id, 0.5, "loop", "", 0.0))

    def end_episode(self, episode_id: str) -> None:
        pass


def test_detection_heuristic():
    assert is_claude_code_payload(_tool_payload("PreToolUse"))
    assert not is_claude_code_payload({"step_id": "1"})
    assert not is_claude_code_payload("junk")


def test_pretooluse_maps_to_tool_call():
    ev = payload_to_event(_tool_payload("PreToolUse"))
    assert ev is not None
    assert ev.action_type == "tool_call"
    assert ev.tool_name == "Bash"
    assert ev.episode_id == "sess-1"
    assert ev.step_id == "toolu_01"
    assert ev.error is False
    # tool_input must not appear raw anywhere (only inside the hash)
    assert "npm test" not in str(ev.metadata)


def test_repeated_same_tool_input_is_loop_detectable():
    m = _RecordingMonitor()
    tracker = HookTracker()
    for i in range(4):
        p = _tool_payload("PreToolUse", tool_use_id=f"toolu_{i}")
        ingest_payload(m, p, tracker)
    assert m.risks, "4 identical Bash(npm test) attempts must trip loop detection"


def test_different_tool_inputs_have_different_signatures():
    a = payload_to_event(_tool_payload("PreToolUse", tool_input={"command": "ls"}))
    b = payload_to_event(
        _tool_payload("PreToolUse", tool_input={"command": "rm -rf /"})
    )
    assert a.action_signature != b.action_signature


def test_failure_events_carry_error():
    ev = payload_to_event(_tool_payload("PostToolUseFailure", error="exit 1"))
    assert ev.error is True
    assert ev.error_type == "exit 1"
    ev2 = payload_to_event(
        {"session_id": "s", "hook_event_name": "StopFailure", "prompt_id": "p1"}
    )
    assert ev2 is not None and ev2.error is True and ev2.action_type == "message"


def test_lifecycle_events_are_dropped():
    for name in ["SessionStart", "Notification", "FileChanged", "Stop", "Unknown"]:
        assert payload_to_event({"session_id": "s", "hook_event_name": name}) is None


def test_tracker_pairs_pre_and_post_for_latency():
    tracker = HookTracker(clock=lambda: _t["now"])
    _t = {"now": 0.0}
    ingest_payload(_RecordingMonitor(), _tool_payload("PreToolUse"), tracker)
    _t["now"] = 0.25
    ev = payload_to_event(_tool_payload("PostToolUse"), tracker=tracker)
    assert ev is not None
    assert ev.latency_ms == 250.0


def test_ingest_payload_preserves_latency_on_the_shipped_path():
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


def test_pretooluse_carries_no_latency():
    # Issue #64: PreToolUse fires before the tool runs. Reporting 0.0 ms poisons
    # the CUSUM baseline with synthetic zeros; it must be None.
    clock = {"now": 0.0}
    tracker = HookTracker(clock=lambda: clock["now"])
    m = _RecordingMonitor()
    ingest_payload(m, _tool_payload("PreToolUse"), tracker)
    assert m.events[0].latency_ms is None


def test_post_tool_use_failure_also_carries_latency():
    # A tool that fails still took time; the CUSUM detector must see it.
    clock = {"now": 0.0}
    tracker = HookTracker(clock=lambda: clock["now"])
    m = _RecordingMonitor()
    ingest_payload(m, _tool_payload("PreToolUse"), tracker)
    clock["now"] = 1.5
    ingest_payload(m, _tool_payload("PostToolUseFailure", error="exit 1"), tracker)
    assert m.events[-1].latency_ms == 1500.0
    assert m.events[-1].error is True


def test_hook_latency_reaches_the_cusum_detector():
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


def test_tracker_evicts_stale_starts_by_age_not_wholesale():
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


def test_user_prompt_submit_maps_and_repeats_are_detectable():
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


def test_ingest_payload_never_raises():
    m = _RecordingMonitor()
    assert ingest_payload(m, None) is None  # type: ignore[arg-type]
    # Weird payloads may be dropped, but must never raise.
    ingest_payload(m, {"hook_event_name": 42, "tool_input": object()})
    ingest_payload(
        m, {"hook_event_name": "PreToolUse", "session_id": None, "tool_input": {}}
    )
