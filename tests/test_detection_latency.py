"""Stage 2 tests: detection-step recording, latency, and confusion metrics.

These were written BEFORE the harness gained the behavior they check (issue
#82 stage 2). Three families:

* :ref:`first-detection` -- replaying a real toy trajectory and asserting the
  FIRST step a trigger fires is the one remembered, even when the trigger
  re-arms and fires again later.
* latency arithmetic -- the pure ``detection_latency`` helper: same-step is 0,
  a later step is the difference, and a miss is ``None`` (never a sentinel
  number).
* one-vs-rest confusion -- TP/FP/FN/TN for one trigger across labeled and
  healthy episodes, and the guarantee that adding TN left Stage 1's
  precision/recall/F1 arithmetic untouched.

Step-index convention used by every assertion: ZERO-BASED position in the
episode's ``events`` list, which is also the ``s{i}`` suffix the fixture
generator writes into each ``step_id``.
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import sys
from pathlib import Path
from typing import Any

import pytest

from snagline.events import make_signature

_REPO_ROOT = Path(__file__).resolve().parents[1]
_HARNESS_PATH = _REPO_ROOT / "benchmarks" / "detection_accuracy.py"

_BASE_TS = 1_700_000_000.0
_GAP_S = 1.5


def _load_harness():
    spec = importlib.util.spec_from_file_location(
        "detection_latency_harness", _HARNESS_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def harness():
    return _load_harness()


# --------------------------------------------------------------------------
# Toy-event helpers (mirror test_detection_accuracy.py's conventions)
# --------------------------------------------------------------------------


def _ev(ep: str, i: int, sig: str, *, tool: str | None = "fetch") -> dict:
    return {
        "step_id": f"{ep}-s{i}",
        "episode_id": ep,
        "timestamp": _BASE_TS + i * _GAP_S,
        "action_type": "tool_call",
        "action_signature": sig,
        "tool_name": tool,
        "latency_ms": None,
        "error": False,
        "error_type": None,
        "tokens_in": None,
        "tokens_out": None,
    }


def _out_step(ep: str, i: int) -> dict:
    return {
        "step_id": f"{ep}-s{i}",
        "episode_id": ep,
        "timestamp": _BASE_TS + i * _GAP_S,
        "action_type": "message",
        "action_signature": make_signature("message", None, "done"),
        "tool_name": None,
        "latency_ms": None,
        "error": False,
        "error_type": None,
        "tokens_in": None,
        "tokens_out": None,
    }


_SIG_A = make_signature("tool_call", "fetch", "id=1")
_SIG_B = make_signature("tool_call", "parse", "doc=9")
_SIG_C = make_signature("tool_call", "store", "key=7")


def _write(tmp_path: Path, name: str, records: list[dict]) -> Path:
    out = tmp_path / "fixtures"
    out.mkdir(exist_ok=True)
    (out / name).write_text(
        "\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8"
    )
    return out


def _replay_first_burst_episode(tmp_path: Path, harness) -> Any:
    """Loop fires at step 2 (third identical signature), re-arms, fires again.

    Steps 0-2 are the same signature: the window count reaches 3 at step 2, so
    ``loop`` first fires there. Steps 3-4 differ, dropping the count below the
    threshold and re-arming the latch. Steps 5-7 repeat the original signature
    again, so ``loop`` fires a SECOND time at step 7. The remembered first
    detection must stay 2.
    """
    ep = "two-bursts"
    events = [
        _ev(ep, 0, _SIG_A),
        _ev(ep, 1, _SIG_A),
        _ev(ep, 2, _SIG_A),  # first firing: window count 3 >= threshold
        _ev(ep, 3, _SIG_B),
        _ev(ep, 4, _SIG_C),  # re-arm
        _ev(ep, 5, _SIG_A),
        _ev(ep, 6, _SIG_A),
        _ev(ep, 7, _SIG_A),  # second, independent firing
        _out_step(ep, 8),
    ]
    record = {
        "episode_id": ep,
        "label": {"trigger": "loop", "onset_step": 0},
        "events": events,
    }
    fixtures = _write(tmp_path, "bursts.jsonl", [record])
    episodes = list(harness.iter_fixtures(fixtures))
    return harness.replay_episode(episodes[0])


# --------------------------------------------------------------------------
# Test A + I: first detection step, and repeated alerts not replacing it
# --------------------------------------------------------------------------


def test_first_detection_step_is_the_earliest_firing(harness, tmp_path: Path) -> None:
    """Test A: firings at 3, 4, 5 must remember 3. The toy episode above fires
    at 2 and 7; the first detection is 2, not the later re-fire."""
    outcome = _replay_first_burst_episode(tmp_path, harness)
    assert outcome.predicted == {"loop"}
    assert outcome.first_detection["loop"] == 2


def test_repeated_alerts_do_not_move_first_detection(harness, tmp_path: Path) -> None:
    """Test I: a second firing burst at step 7 must not overwrite step 2."""
    outcome = _replay_first_burst_episode(tmp_path, harness)
    assert outcome.first_detection["loop"] == 2
    # The trigger genuinely fired twice (two separate bursts) but only one
    # first-detection step exists, and it is the earliest.
    assert outcome.first_detection == {"loop": 2}


def test_first_detection_is_per_trigger(harness, tmp_path: Path) -> None:
    """Two triggers firing on the same episode each keep their own earliest
    step. The loop repeat lands before the latency CUSUM's spike, so the two
    recorded steps must differ.

    All six events use one tool so the latency CUSUM (per-tool, five-sample
    warmup) freezes on the first five and alarms on the 2500ms spike at index
    5; the loop detector fires earlier, at the third identical signature
    (index 2).
    """
    ep = "two-triggers"
    events = [
        _ev(ep, 0, _SIG_A, tool="search_web"),
        _ev(ep, 1, _SIG_A, tool="search_web"),
        _ev(ep, 2, _SIG_A, tool="search_web"),  # loop fires here
        _ev(ep, 3, _SIG_B, tool="search_web"),
        _ev(ep, 4, _SIG_C, tool="search_web"),
        _ev(ep, 5, _SIG_A, tool="search_web"),  # latency spike here
    ]
    for i, ms in enumerate((400.0, 395.0, 405.0, 398.0, 402.0, 2500.0)):
        events[i]["latency_ms"] = ms
    events.append(_out_step(ep, 6))
    record = {"episode_id": ep, "label": {"trigger": "loop"}, "events": events}
    fixtures = _write(tmp_path, "two.jsonl", [record])
    outcome = harness.replay_episode(list(harness.iter_fixtures(fixtures))[0])
    assert "loop" in outcome.predicted
    assert outcome.first_detection["loop"] == 2
    # latency_anomaly's spike is at index 5, strictly after the loop repeat:
    # per-trigger first detection keeps the two distinct.
    assert outcome.first_detection["latency_anomaly"] == 5


# --------------------------------------------------------------------------
# Test B, C, D: latency arithmetic
# --------------------------------------------------------------------------


def test_latency_is_detection_minus_onset(harness) -> None:
    """Test B: onset 3, detection 5 -> latency 2."""
    assert harness.detection_latency(3, 5) == 2


def test_same_step_detection_is_zero_latency(harness) -> None:
    """Test C: onset 3, detection 3 -> latency 0."""
    assert harness.detection_latency(3, 3) == 0


def test_missed_detection_is_null_not_a_big_number(harness) -> None:
    """Test D: no detection yields None, never a sentinel like -1 or 9999."""
    assert harness.detection_latency(3, None) is None
    # An onset is meaningless without a detection to subtract it from.
    assert harness.detection_latency(None, 5) is None
    assert harness.detection_latency(None, None) is None


def test_latency_is_negative_only_when_detection_precedes_the_onset(harness) -> None:
    """A detector firing before the recorded onset reports negative latency.
    That is a real signal (the onset derivation is wrong, or the detector is
    reacting to a precursor), so the harness reports it rather than clamping
    to zero."""
    assert harness.detection_latency(5, 2) == -3


# --------------------------------------------------------------------------
# Test E-H: one-vs-rest confusion with TN
# --------------------------------------------------------------------------


def test_tn_increments_for_non_target_silent_episode(harness) -> None:
    """Test E: a labeled episode expecting a different trigger, on which this
    trigger stays silent, contributes TN."""
    outcomes = [
        harness.EpisodeOutcome("e1", frozenset({"error_cascade"}), predicted=set()),
    ]
    report = harness.score(outcomes)
    assert report.per_trigger["loop"].tn == 1


def test_fp_increments_for_non_target_firing_episode(harness) -> None:
    """Test F: a labeled episode expecting a different trigger, on which this
    trigger fires, contributes FP (and no TN)."""
    outcomes = [
        harness.EpisodeOutcome("e1", frozenset({"error_cascade"}), predicted={"loop"}),
    ]
    report = harness.score(outcomes)
    s = report.per_trigger["loop"]
    assert s.fp == 1
    assert s.tn == 0


def test_tp_increments_for_target_firing_episode(harness) -> None:
    """Test G."""
    outcomes = [
        harness.EpisodeOutcome("e1", frozenset({"loop"}), predicted={"loop"}),
    ]
    report = harness.score(outcomes)
    s = report.per_trigger["loop"]
    assert s.tp == 1
    assert s.fp == 0
    assert s.fn == 0


def test_fn_increments_for_target_silent_episode(harness) -> None:
    """Test H: the expected trigger never firing is an FN. One-vs-rest, that
    episode is NOT also a TN for this trigger -- TN requires the trigger to be
    unexpected (see the two tests below)."""
    outcomes = [harness.EpisodeOutcome("e1", frozenset({"loop"}), predicted=set())]
    report = harness.score(outcomes)
    s = report.per_trigger["loop"]
    assert s.fn == 1
    assert s.tn == 0
    assert s.tp == 0 and s.fp == 0
    # The silent episode IS a true negative for every trigger it did not
    # expect: that is the asymmetry between FN and TN.
    assert report.per_trigger["error_cascade"].tn == 1


def test_tn_and_fn_are_mutually_exclusive_per_trigger(harness) -> None:
    """For one trigger an episode lands in exactly one cell: expecting-and-
    silent is FN, unexpected-and-silent is TN -- never both."""
    outcomes = [
        harness.EpisodeOutcome("miss", frozenset({"loop"}), predicted=set()),
        harness.EpisodeOutcome("other", frozenset({"error_cascade"}), predicted=set()),
    ]
    report = harness.score(outcomes)
    loop = report.per_trigger["loop"]
    assert (loop.tp, loop.fp, loop.tn, loop.fn) == (0, 0, 1, 1)


def test_confusion_matrix_sums_to_the_episode_count(harness) -> None:
    """One-vs-rest: every episode lands in exactly one of TP/FP/TN/FN for a
    given trigger, so the four cells sum to the number of episodes."""
    outcomes = [
        harness.EpisodeOutcome("tp", frozenset({"loop"}), predicted={"loop"}),
        harness.EpisodeOutcome("fn", frozenset({"loop"}), predicted=set()),
        harness.EpisodeOutcome("fp", frozenset({"error_cascade"}), predicted={"loop"}),
        harness.EpisodeOutcome("tn", frozenset({"error_cascade"}), predicted=set()),
        harness.EpisodeOutcome("healthy-silent", None, predicted=set()),
        harness.EpisodeOutcome("healthy-noisy", None, predicted={"loop"}),
    ]
    report = harness.score(outcomes)
    s = report.per_trigger["loop"]
    assert (s.tp, s.fp, s.tn, s.fn) == (1, 2, 2, 1)
    assert s.tp + s.fp + s.tn + s.fn == len(outcomes)


def test_healthy_controls_contribute_tn_when_silent(harness) -> None:
    """A silent healthy control is a true negative for every trigger."""
    outcomes = [harness.EpisodeOutcome("h1", None, predicted=set())]
    report = harness.score(outcomes)
    for trig in harness.SHIPPED_TRIGGERS:
        assert report.per_trigger[trig].tn == 1, trig


def test_specificity_is_tn_over_tn_plus_fp(harness) -> None:
    assert harness.TriggerStats(tn=9, fp=1).specificity == pytest.approx(0.9)
    # No negatives at all: define 0.0 rather than dividing by zero.
    assert harness.TriggerStats().specificity == 0.0


# --------------------------------------------------------------------------
# Test J: Stage 1 baseline compatibility
# --------------------------------------------------------------------------


def test_tn_addition_leaves_stage1_metric_arithmetic_untouched(harness) -> None:
    """precision/recall/F1 and macro-F1 must be computed from TP/FP/FN only,
    exactly as Stage 1 did; adding a TN cell cannot move them."""
    stats = harness.TriggerStats(tp=3, fp=1, fn=1, tn=95)
    assert stats.precision == pytest.approx(0.75)
    assert stats.recall == pytest.approx(0.75)
    assert stats.f1 == pytest.approx(0.75)
    # A trigger with no support at all still reports 0, not NaN.
    empty = harness.TriggerStats()
    assert empty.precision == 0.0 and empty.recall == 0.0 and empty.f1 == 0.0


@pytest.fixture(scope="module")
def committed_outcomes(harness) -> list:
    fixtures = _REPO_ROOT / "benchmarks" / "fixtures"
    return [
        harness.replay_episode(ep, fixtures) for ep in harness.iter_fixtures(fixtures)
    ]


def test_committed_corpus_reproduces_stage3_aggregates(
    harness, committed_outcomes
) -> None:
    """The committed corpus is pinned here so a detector or scoring change
    cannot silently move the published numbers.

    Stage 3 expanded the corpus from 109 to 230 episodes (17 triggers x 10
    variants, plus 60 healthy controls), so Stage 1's TP=68/FP=12/macro-F1
    0.951 baseline no longer applies: the new variants make the documented
    loop/meltdown/stall overlaps fire far more often (see
    ``test_committed_corpus_separates_cleanly`` and reports/EVAL_FINDINGS.md
    section 6). Every labeled episode still fires its own trigger (FN=0) and
    every healthy control stays silent.
    """
    report = harness.score(committed_outcomes)
    tp = sum(s.tp for s in report.per_trigger.values())
    fp = sum(s.fp for s in report.per_trigger.values())
    fn = sum(s.fn for s in report.per_trigger.values())
    tn = sum(s.tn for s in report.per_trigger.values())
    assert (tp, fp, fn) == (170, 34, 0)
    assert tn > 0  # TN is the new cell; it was simply not counted before.
    assert report.healthy_fired == 0
    assert report.macro_f1 == pytest.approx(0.9461, abs=1e-3)
    assert report.n_labeled == 170 and report.n_healthy == 60


def test_committed_corpus_onset_is_present_and_consistent(harness) -> None:
    """Every labeled fixture carries a generator-derived onset and every
    healthy control carries none, and each onset is a valid in-range event
    index."""
    fixtures = _REPO_ROOT / "benchmarks" / "fixtures"
    labeled = 0
    healthy = 0
    for ep in harness.iter_fixtures(fixtures):
        if ep.label_triggers is None:
            assert ep.onset_step is None, ep.episode_id
            healthy += 1
            continue
        assert ep.onset_step is not None, ep.episode_id
        assert 0 <= ep.onset_step < len(ep.events), ep.episode_id
        # The onset is an index into this episode's events: the step_id suffix
        # of that event must match the index.
        assert ep.events[ep.onset_step].step_id == f"{ep.episode_id}-s{ep.onset_step}"
        labeled += 1
    assert labeled == 170 and healthy == 60


def test_every_emitted_step_resolves_to_a_fixture_event_index(harness) -> None:
    """first_detection is only meaningful when every emitted risk can be
    pinned to a fixture event. The committed corpus must satisfy this: the
    finalize pass (silent_abort) attributes its risk to the last ingested
    event, which is in the fixture."""
    fixtures = _REPO_ROOT / "benchmarks" / "fixtures"
    for ep in harness.iter_fixtures(fixtures):
        outcome = harness.replay_episode(ep, fixtures)
        assert set(outcome.first_detection) == outcome.predicted, ep.episode_id
        for step in outcome.first_detection.values():
            assert 0 <= step < len(ep.events), ep.episode_id


def test_committed_corpus_latencies_are_non_negative(
    harness, committed_outcomes
) -> None:
    """Every expected trigger fired at or after its recorded onset. A negative
    latency would mean the onset derivation disagrees with the detector."""
    fixtures = _REPO_ROOT / "benchmarks" / "fixtures"
    onsets = {ep.episode_id: ep.onset_step for ep in harness.iter_fixtures(fixtures)}
    for outcome in committed_outcomes:
        if outcome.label_triggers is None:
            continue
        onset = onsets[outcome.episode_id]
        for trig in outcome.label_triggers:
            assert outcome.first_detection[trig] >= onset, (
                outcome.episode_id,
                trig,
                outcome.first_detection[trig],
                onset,
            )


# --------------------------------------------------------------------------
# Latency aggregation
# --------------------------------------------------------------------------


def test_latency_aggregation_over_outcomes(harness) -> None:
    """Latency is collected only for expected-trigger hits with a known onset
    (TP with onset). Mean/median/p95 come from those values alone."""
    outcomes = [
        harness.EpisodeOutcome(
            "a", frozenset({"loop"}), predicted={"loop"}, onset_step=2
        ),
        harness.EpisodeOutcome(
            "b", frozenset({"loop"}), predicted={"loop"}, onset_step=1
        ),
        harness.EpisodeOutcome(
            "c", frozenset({"loop"}), predicted={"loop"}, onset_step=5
        ),
        # No onset recorded: contributes a detection but no latency sample.
        harness.EpisodeOutcome("d", frozenset({"loop"}), predicted={"loop"}),
        # Expected trigger never fired: FN, no latency sample.
        harness.EpisodeOutcome("e", frozenset({"loop"}), predicted=set(), onset_step=0),
    ]
    for o, step in zip(outcomes, (4, 3, 5, 9, None)):
        if step is not None:
            o.first_detection["loop"] = step
    report = harness.score(outcomes)
    s = report.per_trigger["loop"]
    assert s.tp == 4
    assert s.fn == 1
    # Latencies 2, 2, 0 from the three onsets 2/1/5 against detections 4/3/5.
    assert s.latencies == [2, 2, 0]
    assert s.mean_latency == pytest.approx(4 / 3)
    assert s.median_latency == pytest.approx(2.0)
    # Nearest-rank p95 of [0, 2, 2]: ceil(0.95 * 3) = 3 -> the 3rd (largest).
    assert s.p95_latency == 2


def test_latency_stats_are_null_without_samples(harness) -> None:
    s = harness.TriggerStats()
    assert s.latencies == []
    assert s.mean_latency is None
    assert s.median_latency is None
    assert s.p95_latency is None


def test_percentile_nearest_rank_matches_the_documented_method(harness) -> None:
    """p95 uses nearest rank: ceil(p/100 * n), clipped to n, 1-based."""
    assert harness.percentile([0, 2, 2], 95) == 2
    assert harness.percentile([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 95) == 10
    assert harness.percentile([4.0, 6.0, 8.0], 50) == pytest.approx(6.0)
    assert harness.percentile([], 95) is None


def _latency_rows(harness, report):
    """(header, trigger rows) of the latency table, summary lines stripped."""
    lines = harness.format_latency_table(report).splitlines()
    header = next(ln for ln in lines if ln.startswith("trigger"))
    rule = "-" * len(header)
    rows = [
        ln
        for ln in lines
        if ln
        and ln != rule
        and not ln.startswith(("detection latency", "trigger", "-", "p95 is"))
    ]
    return header, rows


def test_latency_table_reports_four_distinct_populations(harness) -> None:
    """The latency table must not present the corpus total as a per-trigger
    sample size (issue #82 stage 3). ``labeled``, ``detected`` and ``samples``
    are three different populations and the corpus total belongs in the table
    header, not a column.

    Built by hand: 3 episodes expect ``loop``; 2 more fire ``loop`` without
    expecting it; 2 of the 3 labeled episodes record an onset and were
    detected, so latency has 2 samples while the corpus has 5 episodes.
    """
    stats = harness.TriggerStats(tp=2, fp=2, fn=1, tn=0)
    stats.latencies.extend([3, 7])
    stats.n_onset = 2
    report = harness.ScoreReport(
        per_trigger={"loop": stats},
        confusion={},
        healthy_fired=0,
        n_labeled=3,
        n_healthy=2,
    )
    header, rows = _latency_rows(harness, report)
    (row,) = [r for r in rows if r.split()[0] == "loop"]

    # The corpus total (5) appears once, in the header -- never in a row.
    assert "corpus = 5 episodes (3 labeled, 2 healthy controls)" in "\n".join(
        harness.format_latency_table(report).splitlines()[:1]
    )
    assert "5" not in row.split()[1:4], row

    # labeled == TP + FN == 3; detected == TP + FP == 4; samples == 2.
    cols = header.split()
    idx = {name: cols.index(name) for name in ("labeled", "detected", "samples")}
    vals = row.split()
    assert int(vals[idx["labeled"]]) == 3
    assert int(vals[idx["detected"]]) == 4
    assert int(vals[idx["samples"]]) == 2
    # detected > labeled is the visible signature of cross-firing.
    assert int(vals[idx["detected"]]) > int(vals[idx["labeled"]])
    assert int(vals[idx["samples"]]) < int(vals[idx["labeled"]])


def test_latency_table_hides_percentiles_below_the_meaningful_sample_count(
    harness,
) -> None:
    """p95 of a 4-sample column would just be the maximum, so the table prints
    n/a and keeps the raw value in JSON only."""
    stats = harness.TriggerStats(tp=4, fp=0, fn=0)
    stats.latencies.extend([1, 2, 3, 40])
    report = harness.ScoreReport(
        per_trigger={"loop": stats},
        confusion={},
        healthy_fired=0,
        n_labeled=4,
        n_healthy=0,
    )
    header, rows = _latency_rows(harness, report)
    (row,) = [r for r in rows if r.split()[0] == "loop"]
    p95 = row.split()[header.split().index("p95")]
    assert p95 == "n/a", "4 < MEANINGFUL_PERCENTILE_N (5)"
    assert harness.TriggerStats(latencies=[1, 2, 3, 40]).p95_latency == 40
    # median/min/max are still reported on the same row.
    vals = row.split()
    assert int(vals[header.split().index("min")]) == 1
    assert int(vals[header.split().index("max")]) == 40


# --------------------------------------------------------------------------
# Output formats
# --------------------------------------------------------------------------


def test_json_output_carries_metadata_and_per_episode_records(
    harness, tmp_path: Path
) -> None:
    ep = "json-hit"
    record = {
        "episode_id": ep,
        "label": {"trigger": "loop", "onset_step": 2},
        "events": [
            _ev(ep, 0, _SIG_A),
            _ev(ep, 1, _SIG_A),
            _ev(ep, 2, _SIG_A),  # onset and detection on the same step
            _out_step(ep, 3),
        ],
    }
    fixtures = _write(tmp_path, "json.jsonl", [record])
    stdout = io.StringIO()
    with contextlib.redirect_stdout(stdout):
        assert harness.main(["--fixtures", str(fixtures), "--format", "json"]) == 0
    payload = json.loads(stdout.getvalue())
    meta = payload["metadata"]
    assert meta["fixture_count"] == 1
    assert meta["labeled_count"] == 1
    assert meta["healthy_count"] == 0
    assert meta["step_index_convention"] == "zero-based"
    assert meta["evaluation_schema"]
    # Per-episode record: onset, first detection, and a zero latency.
    (episode,) = payload["episode_records"]
    assert episode["episode_id"] == ep
    assert episode["expected_trigger"] == "loop"
    assert episode["onset_step"] == 2
    assert episode["first_detection_step"] == 2
    assert episode["detection_latency"] == 0
    assert episode["classification"] == "TP"
    trig = payload["per_trigger"]["loop"]
    assert trig["tp"] == 1 and trig["fp"] == 0 and trig["tn"] == 0 and trig["fn"] == 0
    assert trig["mean_latency"] == 0.0
