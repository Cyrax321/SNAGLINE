"""Time-axis awareness at the top of Monitor.ingest (issue #92).

Property under test: every idle-gap and wall-clock-budget decision is derived
ONLY from StepEvent timestamps, computed at the top of ingest(), fires at most
once per episode per threshold, and disappears entirely when the new config
options are unset -- replaying an old trajectory must produce byte-identical
results to the pre-#92 build.
"""

from __future__ import annotations

import glob
import json
import logging
import os

import pytest

from snagline.config import Config
from snagline.detectors.token_runaway import TokenRunawayDetector
from snagline.events import StepEvent
from snagline.monitor import Monitor
from snagline.risk import FailureRisk


class CapturingSink:
    """Records every dispatched risk; nothing else."""

    def __init__(self) -> None:
        self.risks: list[FailureRisk] = []

    def emit(self, risk: FailureRisk) -> None:
        self.risks.append(risk)


def _event(
    step_id: str,
    timestamp: float,
    episode_id: str = "ep1",
    latency_ms: float | None = None,
) -> StepEvent:
    return StepEvent(
        step_id=step_id,
        episode_id=episode_id,
        timestamp=timestamp,
        action_type="tool_call",
        action_signature=f"sig-{step_id}",
        tool_name="t",
        latency_ms=latency_ms,
    )


def _monitor(sink: CapturingSink, **cfg_kwargs) -> Monitor:
    cfg = Config(**cfg_kwargs)
    return Monitor([], [sink], config=cfg)


def _feed(monitor: Monitor, *events: StepEvent, end: bool = False) -> None:
    for e in events:
        monitor.ingest(e)
    if end:
        monitor.end_episode(events[-1].episode_id if events else "ep1")


# --- idle_gap ---------------------------------------------------------------


def test_idle_gap_fires_once_per_episode() -> None:
    sink = CapturingSink()
    m = _monitor(sink, idle_warn_seconds=10.0)
    _feed(
        m,
        _event("s1", 0.0),
        _event("s2", 5.0),  # gap 5s: quiet
        _event("s3", 20.0),  # gap 15s: fire once
        _event("s4", 40.0),  # gap 20s: already fired, stays quiet
    )
    idle = [r for r in sink.risks if r.trigger == "idle_gap"]
    assert len(idle) == 1
    assert idle[0].step_id == "s3"
    assert idle[0].score == 0.8


def test_idle_gap_disabled_by_default_and_small_gaps_quiet() -> None:
    sink = CapturingSink()
    m = _monitor(sink)
    _feed(m, _event("s1", 0.0), _event("s2", 99999.0))
    assert sink.risks == []
    sink2 = CapturingSink()
    m2 = _monitor(sink2, idle_warn_seconds=100.0)
    _feed(m2, _event("s1", 0.0), _event("s2", 99.9))
    assert [r for r in sink2.risks if r.trigger == "idle_gap"] == []


def test_idle_gap_resets_with_episode() -> None:
    sink = CapturingSink()
    m = _monitor(sink, idle_warn_seconds=10.0)
    _feed(m, _event("s1", 0.0), _event("s2", 50.0), end=True)
    assert len([r for r in sink.risks if r.trigger == "idle_gap"]) == 1
    # Same episode id reused after end_episode: a fresh clock, so a later
    # silence can fire again.
    _feed(m, _event("s3", 100.0), _event("s4", 200.0))
    assert len([r for r in sink.risks if r.trigger == "idle_gap"]) == 2


def test_first_event_never_fires_idle_or_budget() -> None:
    sink = CapturingSink()
    m = _monitor(sink, idle_warn_seconds=0.001, max_episode_wall_seconds=0.001)
    _feed(m, _event("s1", 0.0))
    assert sink.risks == []


# --- wall_clock_budget ------------------------------------------------------


def test_budget_warn_then_breach_each_fire_once() -> None:
    sink = CapturingSink()
    m = _monitor(sink, max_episode_wall_seconds=100.0, warn_fraction=0.8)
    _feed(
        m,
        _event("s1", 0.0),
        _event("s2", 30.0),  # elapsed 30 < 80: quiet
        _event("s3", 85.0),  # elapsed 85 >= 80: warn
        _event("s4", 90.0),  # elapsed 90: still warned, no re-fire
        _event("s5", 120.0),  # elapsed 120 >= 100: breach
        _event("s6", 500.0),  # already breached, no re-fire
    )
    budget = [r for r in sink.risks if r.trigger == "wall_clock_budget"]
    assert [(r.step_id, r.score) for r in budget] == [("s3", 0.7), ("s5", 1.0)]
    assert budget[0].severity == "warning"
    assert budget[1].severity == "critical"


def test_single_jump_past_budget_fires_only_breach() -> None:
    sink = CapturingSink()
    m = _monitor(sink, max_episode_wall_seconds=10.0, warn_fraction=0.8)
    _feed(m, _event("s1", 0.0), _event("s2", 1000.0))
    budget = [r for r in sink.risks if r.trigger == "wall_clock_budget"]
    assert [(r.step_id, r.score) for r in budget] == [("s2", 1.0)]


def test_negative_delta_does_not_refund_budget() -> None:
    sink = CapturingSink()
    m = _monitor(sink, max_episode_wall_seconds=10.0)
    _feed(
        m,
        _event("s1", 0.0),
        _event("s2", 12.0),  # breach
        _event("s3", 1.0),  # skewed backwards: must not un-breach
        _event("s4", 30.0),
    )
    breaches = [
        r for r in sink.risks if r.trigger == "wall_clock_budget" and r.score == 1.0
    ]
    assert len(breaches) == 1


def test_out_of_order_events_do_not_double_count_the_same_span() -> None:
    """Issue #249: a negative delta was excluded from ``elapsed`` but still
    rewound ``last_ts``, so the next event measured its delta from the moved-
    back reference and counted an already-counted span a second time. Events
    at ts 0, 60, 0, 60 -- a 60-second true span, the second half a skewed
    replay of the same range -- used to yield ``elapsed == 120`` and breach a
    100 s budget. ``last_ts`` must only move forward.
    """
    sink = CapturingSink()
    m = _monitor(sink, max_episode_wall_seconds=100.0)
    _feed(
        m,
        _event("s1", 0.0),
        _event("s2", 60.0),
        _event("s3", 0.0),  # skewed backwards: rewinds nothing now
        _event("s4", 60.0),  # delta measured from 60, not from 0
    )
    clock = m._clocks["ep1"]
    assert clock.elapsed == 60.0, "the same wall-clock span was counted twice"
    assert clock.last_ts == 60.0, "last_ts must not move backwards"
    assert sink.risks == [], "60s of real time must not breach a 100s budget"


def test_out_of_order_replay_still_counts_genuinely_new_time() -> None:
    """The fix must not under-count either: after a backwards event, a delta
    past the previous high-water mark is real new time and still spends
    budget."""
    sink = CapturingSink()
    m = _monitor(sink, max_episode_wall_seconds=100.0)
    _feed(
        m,
        _event("s1", 0.0),
        _event("s2", 60.0),
        _event("s3", 0.0),
        _event("s4", 150.0),  # 90s past the high-water mark of 60
    )
    breaches = [
        r for r in sink.risks if r.trigger == "wall_clock_budget" and r.score == 1.0
    ]
    assert len(breaches) == 1, "genuinely new time must still breach"
    assert m._clocks["ep1"].elapsed == 150.0


# --- determinism / fail-open / region contract ------------------------------


def test_no_wall_clock_reads_in_ingest(monkeypatch: object) -> None:
    """ZERO time.time() reads during ingest: replay stays deterministic."""
    import time as time_module

    def _explode() -> float:
        raise AssertionError("wall clock read during ingest")

    monkeypatch.setattr(time_module, "time", _explode)  # type: ignore[attr-defined]
    sink = CapturingSink()
    m = _monitor(sink, idle_warn_seconds=1.0, max_episode_wall_seconds=2.0)
    # Timestamps come from the events only; the frozen wall clock is never read.
    _feed(m, _event("s1", 0.0), _event("s2", 5.0))


def test_replay_fingerprints_unchanged_when_options_unset() -> None:
    """Old trajectories produce identical results with new options unset.

    The expected fingerprints were captured on the pre-#92 build; any change
    in default-config behavior fails here.
    """
    expected = {
        "healthy_run.jsonl": [],
        # Graded scoring (issue #538): the cascade first crosses the default
        # threshold at step 22 with 3 consecutive errors, so the score is the
        # first-crossing 0.5 -- not the flat 1.0 it was before -- matching the
        # loop fixture's first crossing.
        "injected_error_cascade.jsonl": [("error_cascade", "22", 0.5)],
        # The governance fixture's trailing lookup/write/search calls are three
        # distinct tools, so under the default config (the compaction tripwire
        # is opt-in and off) it emits nothing. It used to emit a spurious loop:
        # the lookup and write rows carried search's action_signature, and
        # LoopDetector keys on the signature, not the tool name (issue #574).
        "injected_governance_decay.jsonl": [],
        "injected_latency_spike.jsonl": [
            ("latency_anomaly", str(i), 1.0) for i in range(40, 52)
        ],
        "injected_loop.jsonl": [("loop", "22", 0.5)],
    }
    for path in sorted(glob.glob("tests/fixtures/trajectories/*.jsonl")):
        name = os.path.basename(path)
        sink = CapturingSink()
        monitor = Monitor.default(config=Config(), sinks=[sink])
        episode = None
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                event = StepEvent(**json.loads(line))
                episode = event.episode_id
                monitor.ingest(event)
        monitor.end_episode(episode)  # type: ignore[arg-type]
        got = [(r.trigger, r.step_id, round(r.score, 4)) for r in sink.risks]
        assert got == expected[name], f"fingerprint changed for {name}"


def test_fixture_signatures_match_their_tool_names() -> None:
    """A fixture's tool_call rows must not share a signature unintentionally.

    LoopDetector keys on ``action_signature``, not ``tool_name``, so two rows
    that name different tools but carry one digest manufacture a loop the
    fixture never intended. The governance fixture's lookup and write rows
    both carried search's signature, so its replay reported a spurious loop
    instead of the governance decay it was built to show (issue #574).

    The other fixtures are hand-written with a *unique* digest per step
    (deliberately, so no accidental loop can fire), so the invariant to pin is
    pairwise-distinctness among a fixture's tool_call rows, not equality with
    ``make_signature``: hashing the tool name would itself defeat the loop
    fixture, whose four ``retry`` rows are meant to collide.
    """
    for path in sorted(glob.glob("tests/fixtures/trajectories/*.jsonl")):
        name = os.path.basename(path)
        seen: dict[str, str] = {}
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                raw = json.loads(line)
                if raw["action_type"] != "tool_call":
                    continue
                sig = raw["action_signature"]
                tool = raw.get("tool_name")
                if sig in seen and seen[sig] != tool:
                    pytest.fail(
                        f"{name} step {raw['step_id']}: action_signature "
                        f"{sig[:16]}… is shared with a row named "
                        f"{seen[sig]!r} but this row names {tool!r}; "
                        "LoopDetector keys on the signature, so this is a "
                        "loop the fixture did not intend"
                    )
                seen.setdefault(sig, tool)


def test_time_axis_fail_open_on_pathological_timestamps(caplog) -> None:
    """NaN timestamps must not crash ingest; fault logged once, never raised."""
    sink = CapturingSink()
    m = _monitor(sink, idle_warn_seconds=1.0)
    nan = float("nan")
    with caplog.at_level(logging.DEBUG, logger="snagline"):
        _feed(m, _event("s1", 0.0), _event("s2", nan), _event("s3", nan))
    assert True  # reaching here IS the assertion: nothing propagated


def test_horizon_risks_counted_in_metrics() -> None:
    sink = CapturingSink()
    m = _monitor(sink, idle_warn_seconds=1.0)
    _feed(m, _event("s1", 0.0), _event("s2", 9.0))
    metrics = m.metrics()
    assert metrics["risks_emitted"] == 1
    assert metrics["events_ingested"] == 2


def test_horizon_knobs_reach_monitor_via_default(monkeypatch) -> None:
    """Monitor.default wires SNAGLINE_* env through to the time axis."""
    env = {"SNAGLINE_IDLE_WARN_SECONDS": "5"}
    monkeypatch.setenv("SNAGLINE_IDLE_WARN_SECONDS", "5")
    cfg = Config.resolve(environ=env)
    assert cfg.idle_warn_seconds == 5.0


def test_invalid_horizon_config_fails_loudly() -> None:
    import pytest

    with pytest.raises(ValueError):
        Config(warn_fraction=0.0)
    with pytest.raises(ValueError):
        Config(warn_fraction=1.5)
    with pytest.raises(ValueError):
        Config(max_episode_wall_seconds=-1.0)
    with pytest.raises(ValueError):
        Config(idle_warn_seconds=0.0)
    with pytest.raises(ValueError):
        Config(window_scale_steps=-1)
    with pytest.raises(ValueError):
        Config(max_window=0)
    with pytest.raises(ValueError):
        Config(cusum_refit_every=-1)


def test_jump_past_budget_emits_no_stale_warning_afterward() -> None:
    """Issue #224: the breach fired, but ``clock.warned`` stayed False, so
    the *next* step emitted the 0.7 pre-breach warning for an episode already
    reported critical -- severity running backwards, with the self-contradicting
    text "at 201% of its 100s wall-clock budget". A jump straight past the
    budget now marks the warning threshold as passed, for good.
    """
    sink = CapturingSink()
    m = _monitor(sink, max_episode_wall_seconds=100.0, warn_fraction=0.8)
    _feed(
        m,
        _event("s1", 0.0),
        _event("s2", 200.0),  # jumps straight past the budget: breach only
        _event("s3", 201.0),  # already breached -- no stale warning may fire
        _event("s4", 202.0),
    )
    budget = [r for r in sink.risks if r.trigger == "wall_clock_budget"]
    assert [(r.step_id, r.score) for r in budget] == [("s2", 1.0)]
    assert budget[0].severity == "critical"


def test_wall_clock_budget_and_token_runaway_grade_their_warning_identically():
    """The two budget envelopes are the same signal at different scales, and
    both the code and the trigger table in ``risk.py`` claim they grade their
    pre-breach warning alike (``token_runaway`` carries the comment "the two
    must grade their pre-breach signal identically"). The claim is asserted
    here rather than left as prose so a change to one envelope's score cannot
    leave the other -- and the documented severity -- behind."""
    sink = CapturingSink()
    cfg = Config(
        max_episode_wall_seconds=100.0,
        warn_fraction=0.5,
        token_runaway_enabled=True,
        episode_token_budget=1000,
        token_budget_warn_fraction=0.5,
    )
    m = Monitor([TokenRunawayDetector(config=cfg)], [sink], config=cfg)
    for i in range(6):
        m.ingest(  # 20s per step, 100 tokens per step
            StepEvent(
                step_id=f"s{i}",
                episode_id="ep1",
                timestamp=float(i * 20),
                action_type="tool_call",
                action_signature=f"sig-{i}",
                tool_name="t",
                tokens_in=100,
            )
        )

    wall_warn = next(
        (r for r in sink.risks if r.trigger == "wall_clock_budget" and r.score < 1.0),
        None,
    )
    token_warn = next((r for r in sink.risks if r.trigger == "token_runaway"), None)
    assert wall_warn is not None and token_warn is not None, (
        "both envelopes must fire their pre-breach warning"
    )
    assert wall_warn.score == token_warn.score, (
        "the twin envelopes must grade their pre-breach warning identically: "
        f"wall_clock_budget={wall_warn.score} token_runaway={token_warn.score}"
    )
    assert wall_warn.severity == token_warn.severity
    # The shared grade must stay inside the warning band: ``severity_from_score``
    # promotes >= 0.8 to critical, which would page on a merely-at-budget signal
    # and, under policy="halt_webhook", consult the halt endpoint (default
    # ``min_severity_for_halt`` is 0.8) for something that is not a breach.
    assert wall_warn.score < 0.8
    assert wall_warn.severity == "warning"


# --- restore across a process restart ---------------------------------------


def test_restore_does_not_fabricate_horizon_risks_from_a_dead_clock(tmp_path) -> None:
    """A restored clock's ``last_ts`` is a raw event timestamp from the process
    that wrote the snapshot. The shipped auto-instrumentation stamps events
    with ``perf_counter``, whose epoch is process-local, so the first
    post-restore event must not measure its delta from the old epoch: it
    invented both a 40000s idle_gap and a 40000s budget breach from a single
    healthy event. The clock re-anchors on that event instead, as a first
    event does.
    """
    path = str(tmp_path / "snap.json")
    src = _monitor(
        CapturingSink(),
        max_episode_wall_seconds=100.0,
        warn_fraction=0.8,
        idle_warn_seconds=30.0,
    )
    _feed(src, _event("s1", 0.4))  # short-lived source process
    src.snapshot(path)

    sink = CapturingSink()
    dst = _monitor(
        sink, max_episode_wall_seconds=100.0, warn_fraction=0.8, idle_warn_seconds=30.0
    )
    dst.restore(path)
    # A new process whose perf_counter is tens of thousands of seconds on.
    _feed(dst, _event("s2", 40_000.0))

    spurious = [r for r in sink.risks if r.trigger in ("idle_gap", "wall_clock_budget")]
    assert not spurious, [(r.trigger, r.score, r.detail) for r in spurious]


def test_restore_does_not_freeze_the_budget_below_the_old_epoch(tmp_path) -> None:
    """The mirror case: the source process ran long (its perf_counter is far
    above the new one's), so every post-restore delta is negative, ``elapsed``
    never advanced, and 200s of real new time produced no breach at all. The
    budget was silently extended.
    """
    path = str(tmp_path / "snap.json")
    src = _monitor(CapturingSink(), max_episode_wall_seconds=100.0)
    _feed(src, _event("s1", 3000.0), _event("s2", 3095.0))  # 95s spent, warned
    src.snapshot(path)

    sink = CapturingSink()
    dst = _monitor(sink, max_episode_wall_seconds=100.0)
    dst.restore(path)
    for ts in range(0, 200):  # 200s of genuinely new time
        _feed(dst, _event(f"r{ts}", float(ts)))

    clock = dst._clocks["ep1"]
    assert clock.elapsed >= 100.0, f"budget frozen at {clock.elapsed}s"
    assert clock.breached, "the real breach never fired"


def test_restore_preserves_budget_already_spent(tmp_path) -> None:
    """Re-anchoring must not forgive budget: the 95s consumed before the
    restart still counts, so a further 10s breaches exactly as it should."""
    path = str(tmp_path / "snap.json")
    src = _monitor(CapturingSink(), max_episode_wall_seconds=100.0)
    _feed(src, _event("s1", 0.0), _event("s2", 95.0))
    src.snapshot(path)

    sink = CapturingSink()
    dst = _monitor(sink, max_episode_wall_seconds=100.0)
    dst.restore(path)
    _feed(dst, _event("s3", 0.0), _event("s4", 10.0))  # fresh clock, +10s

    assert dst._clocks["ep1"].elapsed == 105.0
    assert any(
        r.trigger == "wall_clock_budget" and r.score == 1.0 for r in sink.risks
    ), "the carried-over 95s must still count"


def test_restore_replaces_the_time_axis_like_everything_else() -> None:
    """restore_dict rebuilds every detector wholesale and clears the LRU, but
    used to *merge* _clocks. A live monitor restored onto a snapshot that does
    not carry episode A left A's clock behind: A resumed with 95s of budget
    already spent and its warning latch set while every detector treated it
    as brand new.
    """
    sink = CapturingSink()
    m = _monitor(sink, max_episode_wall_seconds=100.0, warn_fraction=0.8)
    _feed(m, _event("a1", 0.0, "A"), _event("a2", 95.0, "A"))
    m.restore_dict(
        {
            "format_version": 1,
            "detectors": {},
            "sinks": {},
            "time_axis": {
                "B": {
                    "last_ts": 5.0,
                    "elapsed": 0.0,
                    "idle_fired": False,
                    "warned": False,
                    "breached": False,
                }
            },
            "live_episodes": ["B"],
        }
    )
    assert "A" not in m._clocks, (
        f"orphan clock survives restore: {m._clocks['A'].elapsed}s spent"
    )
    assert "B" in m._clocks


def test_mixed_clock_domains_reanchor_instead_of_fabricating_a_breach() -> None:
    """Issue #532: two adapters in one episode stamping different clocks.

    The shipped adapters share one process-local monotonic clock, but a stray
    epoch stamp (~1.79e9) meeting a monotonic one (~1e3) used to produce a
    ~1.79-billion-second delta on a single step. That fired BOTH time-axis
    risks as critical on one event and -- because ``elapsed`` can never be
    un-spent -- latched ``breached`` forever, so the episode could not recover
    even after every later step was genuine.
    """
    sink = CapturingSink()
    m = _monitor(sink, max_episode_wall_seconds=60.0, idle_warn_seconds=5.0)

    # A monotonic-clock step, then an epoch step into the same episode.
    _feed(m, _event("a1", 100.0), _event("a2", 1_790_000_000.0))

    assert sink.risks == [], (
        "a clock-domain jump must not score an idle gap or a budget breach: "
        f"{[(r.trigger, r.score) for r in sink.risks]}"
    )

    # The span was dropped, not spent: the episode is not wedged, and steps
    # after the jump still score against a sane, moving reference.
    _feed(
        m,
        _event("a3", 1_790_000_000.0 + 50.0),  # 50s in -> warning at 0.8 of 60s
        _event("a4", 1_790_000_000.0 + 70.0),  # 70s in -> breach
    )
    # The warning precedes the breach (a jump straight past the budget must
    # not leave a stale warning behind it -- issue #224).
    horizon = [r for r in sink.risks if r.trigger == "wall_clock_budget"]
    assert len(horizon) == 2
    assert horizon[0].score < 1.0
    assert horizon[1].score == 1.0


def test_mixed_clock_domains_warn_once_and_stay_healthy(caplog) -> None:
    """The re-anchor is logged fault-once per episode and never raises."""
    sink = CapturingSink()
    m = _monitor(sink, max_episode_wall_seconds=60.0, idle_warn_seconds=5.0)

    _feed(m, _event("b0", 100.0))  # monotonic reference point
    with caplog.at_level(logging.ERROR):
        for i in range(1, 6):  # a foreign epoch clock, step after step
            _feed(m, _event(f"b{i}", 1_790_000_000.0 + i))

    assert len(sink.risks) == 0
    assert sum("clock domains" in rec.message for rec in caplog.records) == 1, (
        "a persistently mismatched source must warn once, not per step"
    )


def test_continuum_epoch_source_mixed_with_a_monotonic_source_stays_healthy() -> None:
    """The CONTINUUM adapter legitimately stamps unix epoch, not step_clock.

    It pairs a claim's observed time with its terminal record to report a real
    claim-to-terminal duration, so its timestamps cannot move to a monotonic
    clock without breaking that measurement. This pins the documented
    contract instead: mixing it with a monotonic adapter degrades to a dropped
    interval, not a fabricated breach (#532).
    """
    sink = CapturingSink()
    m = _monitor(sink, max_episode_wall_seconds=60.0, idle_warn_seconds=5.0)

    # A CONTINUUM ledger event (epoch), then live instrumentation (perf_counter).
    _feed(m, _event("cc0", 1_790_000_000.0))
    _feed(m, _event("cc1", 12_345.0))
    # And the episode keeps working normally afterwards.
    _feed(m, _event("cc2", 12_346.0), _event("cc3", 12_347.0))

    assert sink.risks == [], f"a mixed-domain episode must not alarm: {sink.risks}"


def test_a_plausibly_slow_step_still_fires_genuinely() -> None:
    """The re-anchor bound is a day: a real multi-minute step must still fire.

    A tool call that genuinely hangs past the budget is a real breach and must
    not be swallowed by the clock-domain guard.
    """
    sink = CapturingSink()
    m = _monitor(sink, max_episode_wall_seconds=60.0, idle_warn_seconds=5.0)

    _feed(m, _event("c1", 100.0), _event("c2", 100.0 + 3600.0))  # 1 hour

    assert [r.trigger for r in sink.risks] == ["idle_gap", "wall_clock_budget"]
    assert sink.risks[1].score == 1.0
