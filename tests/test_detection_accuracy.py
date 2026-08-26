"""Tests for the detection-accuracy harness (issue #82).

Two layers of honesty live here:

* ``test_toy_corpus_*`` replays toy corpora whose expected
  TP/FP/FN/precision/recall/F1 were derived BY HAND on paper before running
  anything (the arithmetic is spelled out in the test bodies). This is the
  anti-tautology anchor: expectations come from independent arithmetic, never
  from calling the scorer twice. Since issue #118 there is one toy corpus per
  newly covered opt-in trigger: ``goal_drift`` (against its own hand-built
  baseline fixture) and ``ml_ensemble`` (under the noisy-OR variant).
* The remaining tests pin scorer semantics that the fixture corpus depends on
  (label-null episodes can never produce false negatives; duplicate firings
  count once) and verify the committed fixture corpus still separates cleanly
  against the real detector stack.
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import sys
from pathlib import Path

import pytest

from snagline.events import make_signature

_REPO_ROOT = Path(__file__).resolve().parents[1]
_HARNESS_PATH = _REPO_ROOT / "benchmarks" / "detection_accuracy.py"


def _load_harness():
    spec = importlib.util.spec_from_file_location("detection_accuracy", _HARNESS_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # Register before exec: slots dataclasses resolve their module while the
    # module body runs.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def harness():
    return _load_harness()


# --------------------------------------------------------------------------
# Toy corpus helpers
# --------------------------------------------------------------------------

_SIG_X = make_signature("tool_call", "fetch", "id=1")
_SIG_Y = make_signature("tool_call", "fetch", "id=2")
_SIG_OTHER_A = make_signature("tool_call", "parse", "doc=9")
_SIG_OTHER_B = make_signature("tool_call", "store", "key=7")


def _toy_event(ep: str, i: int, sig: str, *, tool: str | None = "fetch") -> dict:
    return {
        "step_id": f"{ep}-s{i}",
        "episode_id": ep,
        "timestamp": 1_700_000_000.0 + i * 1.5,
        "action_type": "tool_call",
        "action_signature": sig,
        "tool_name": tool,
        "latency_ms": None,
        "error": False,
        "error_type": None,
        "tokens_in": None,
        "tokens_out": None,
    }


def _output_step(ep: str, i: int) -> dict:
    return {
        "step_id": f"{ep}-s{i}",
        "episode_id": ep,
        "timestamp": 1_700_000_000.0 + i * 1.5,
        "action_type": "message",
        "action_signature": make_signature("message", None, "done"),
        "tool_name": None,
        "latency_ms": None,
        "error": False,
        "error_type": None,
        "tokens_in": None,
        "tokens_out": None,
    }


def _finish(ep: str, events: list[dict]) -> list[dict]:
    """Close on a message step so silent_abort has nothing to judge."""
    events.append(_output_step(ep, len(events)))
    return events


@pytest.fixture()
def toy_dir(tmp_path: Path) -> Path:
    """Three-episode toy corpus.

    Paper derivation of expected numbers (done BEFORE any code ran):

    * ``toy-hit`` (label loop): signatures X, X, X then a message. The loop
      window holds 12; the third X raises its window count to 3 >= threshold
      3, so ``loop`` fires exactly once. Predicted = {loop}. Contributes
      TP(loop) += 1.
    * ``toy-miss`` (label loop): X, X, other, other, message. The count for X
      peaks at 2 < 3, so nothing fires. Predicted = {}. Contributes
      FN(loop) += 1.
    * ``toy-fp`` (label null): X, Y, X, Y, X. The count for X reaches 3 at the
      fifth step (window 12), so ``loop`` fires on an UNLABELED episode.
      Predicted = {loop} against no label. Contributes FP(loop) += 1 and
      healthy_fired += 1.

    Corpus aggregates for loop, by hand:
      TP=1, FP=1, FN=1
      precision = 1 / (1 + 1) = 0.5
      recall    = 1 / (1 + 1) = 0.5
      f1        = 2 * 0.5 * 0.5 / (0.5 + 0.5) = 0.5
      macro-F1 over supported triggers = 0.5 (loop is the only one)
      healthy controls fired = 1
    Every other trigger has TP=FP=FN=0.

    Each episode stays under 20 tool calls (meltdown window never fills),
    carries no latency/token/error fields (those detectors are inert), and
    ends on a message step (silent abort silent), so loop is the ONLY trigger
    that can fire anywhere in this corpus.
    """
    hit = [
        _toy_event("toy-hit", 0, _SIG_X),
        _toy_event("toy-hit", 1, _SIG_X),
        _toy_event("toy-hit", 2, _SIG_X),
    ]
    miss = [
        _toy_event("toy-miss", 0, _SIG_X),
        _toy_event("toy-miss", 1, _SIG_X),
        _toy_event("toy-miss", 2, _SIG_OTHER_A),
        _toy_event("toy-miss", 3, _SIG_OTHER_B),
    ]
    fp = [
        _toy_event("toy-fp", 0, _SIG_X),
        _toy_event("toy-fp", 1, _SIG_Y),
        _toy_event("toy-fp", 2, _SIG_X),
        _toy_event("toy-fp", 3, _SIG_Y),
        _toy_event("toy-fp", 4, _SIG_X),
    ]
    lines = [
        json.dumps(
            {
                "episode_id": "toy-hit",
                "label": {"trigger": "loop"},
                "events": _finish("toy-hit", hit),
            }
        ),
        json.dumps(
            {
                "episode_id": "toy-miss",
                "label": {"trigger": "loop"},
                "events": _finish("toy-miss", miss),
            }
        ),
        json.dumps(
            {"episode_id": "toy-fp", "label": None, "events": _finish("toy-fp", fp)}
        ),
    ]
    out = tmp_path / "fixtures"
    out.mkdir()
    (out / "toy.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


# --------------------------------------------------------------------------
# Hand-computed toy corpus (anti-tautology anchor)
# --------------------------------------------------------------------------


def test_toy_corpus_matches_hand_computed_scores(harness, toy_dir: Path) -> None:
    outcomes = [harness.replay_episode(ep) for ep in harness.iter_fixtures(toy_dir)]
    report = harness.score(outcomes)

    # Hand values: TP=1 FP=1 FN=1 -> P=R=F1=0.5 for loop, and only loop has
    # support in this corpus.
    s = report.per_trigger["loop"]
    assert (s.tp, s.fp, s.fn) == (1, 1, 1)
    assert s.precision == pytest.approx(0.5)
    assert s.recall == pytest.approx(0.5)
    assert s.f1 == pytest.approx(0.5)
    assert report.macro_f1 == pytest.approx(0.5)

    # Every other shipped trigger saw no support at all.
    for trig in harness.SHIPPED_TRIGGERS:
        if trig == "loop":
            continue
        other = report.per_trigger[trig]
        assert (other.tp, other.fp, other.fn) == (0, 0, 0)

    assert report.n_labeled == 2
    assert report.n_healthy == 1
    assert report.healthy_fired == 1


def test_toy_corpus_cli_exit_code_and_json_output(harness, toy_dir: Path) -> None:
    stdout = io.StringIO()
    with contextlib.redirect_stdout(stdout):
        code = harness.main(["--fixtures", str(toy_dir), "--format", "json"])
    # The toy corpus contains a firing healthy control (toy-fp), so the
    # false-positive gate must fail the run.
    assert code == 1
    payload = json.loads(stdout.getvalue())
    assert payload["healthy_controls_fired"] == 1
    assert payload["episodes"] == {"labeled": 2, "healthy": 1}
    loop = payload["per_trigger"]["loop"]
    assert loop["tp"] == 1 and loop["fp"] == 1 and loop["fn"] == 1
    assert loop["precision"] == pytest.approx(0.5)
    assert loop["recall"] == pytest.approx(0.5)
    assert loop["f1"] == pytest.approx(0.5)
    assert payload["macro_f1"] == pytest.approx(0.5)


# --------------------------------------------------------------------------
# Toy corpora for the issue-#118 triggers (anti-tautology anchors)
# --------------------------------------------------------------------------


def _write_goal_drift_baseline(out: Path) -> None:
    """Hand-built healthy reference identical in shape to the committed one:
    search_web frozen at 100.0ms, read_file at 50.0ms, run_sql at 80.0ms,
    all with std 0 and error rate 0. Constant values make the detector's
    floored spreads exact: max(0, 1ms, 5% of mean)."""
    from snagline.baseline import BaselineProfile, save_baseline
    from snagline.events import StepEvent

    profile = BaselineProfile()
    i = 0
    for tool, ms in (("search_web", 100.0), ("read_file", 50.0), ("run_sql", 80.0)):
        for j in range(40):
            profile.add_event(
                StepEvent(
                    step_id=f"base-s{i}",
                    episode_id="baseline-fit",
                    timestamp=1_700_000_000.0 + i,
                    action_type="tool_call",
                    action_signature=make_signature("tool_call", tool, f"q={j}"),
                    tool_name=tool,
                    latency_ms=ms,
                )
            )
            i += 1
    save_baseline(profile, str(out / "goal_drift_baseline.json"))


@pytest.fixture()
def goal_drift_toy_dir(tmp_path: Path) -> Path:
    """Three-episode goal_drift toy corpus, replayed under the goal_drift
    variant against the baseline written above.

    Paper derivation of expected numbers (done BEFORE any code ran):

    * ``gd-hit`` (label goal_drift): ten search_web calls at exactly 200ms,
      then a message. After goal_drift_min_samples = 10 the live mean is
      200 against a reference mean of 100 whose floored spread is
      max(0, 1, 5% of 100) = 5ms: z = (200 - 100) / 5 = 20 > k = 3, so the
      contribution is min(1, (20 - 3) / 10) = 1.0 >= threshold 0.5.
      Predicted = {goal_drift}. Contributes TP(goal_drift) += 1.
    * ``gd-miss`` (label goal_drift): ten search_web calls at 120ms. Same
      spread: z = (120 - 100) / 5 = 4 > k = 3, but the contribution is only
      (4 - 3) / 10 = 0.1 < 0.5, so nothing fires. Predicted = {}.
      Contributes FN(goal_drift) += 1.
    * ``gd-fp`` (label null): nine calls cycling the three baseline tools at
      their exact reference latencies (z = 0 everywhere), then one call to
      mystery_tool, absent from the baseline: absent tools contribute a flat
      0.6 >= 0.5 at that tenth live sample. Predicted = {goal_drift} against
      no label. Contributes FP(goal_drift) += 1 and healthy_fired += 1.

    Corpus aggregates for goal_drift, by hand:
      TP=1, FP=1, FN=1
      precision = 1 / (1 + 1) = 0.5
      recall    = 1 / (1 + 1) = 0.5
      f1        = 2 * 0.5 * 0.5 / (0.5 + 0.5) = 0.5
      macro-F1 over supported triggers = 0.5 (goal_drift is the only one)
      healthy controls fired = 1
    Every other trigger has TP=FP=FN=0.

    No other detector can fire in this corpus: all latencies are constant so
    the CUSUM increments never exceed 0 post-warmup, args are unique so loop
    counts peak at 1, there are no errors, no tokens, fewer than 20 tool
    calls (meltdown window unfilled), and every episode ends on a message.
    """
    hit = [
        _toy_event(
            "gd-hit",
            i,
            make_signature("tool_call", "search_web", f"q={i}"),
            tool="search_web",
        )
        | {"latency_ms": 200.0}
        for i in range(10)
    ]
    miss = [
        _toy_event(
            "gd-miss",
            i,
            make_signature("tool_call", "search_web", f"r={i}"),
            tool="search_web",
        )
        | {"latency_ms": 120.0}
        for i in range(10)
    ]
    fp = [
        _toy_event("gd-fp", 0, _SIG_X, tool="search_web") | {"latency_ms": 100.0},
        _toy_event("gd-fp", 1, _SIG_Y, tool="read_file") | {"latency_ms": 50.0},
        _toy_event(
            "gd-fp", 2, make_signature("tool_call", "run_sql", "r=1"), tool="run_sql"
        )
        | {"latency_ms": 80.0},
        _toy_event("gd-fp", 3, _SIG_X, tool="search_web") | {"latency_ms": 100.0},
        _toy_event("gd-fp", 4, _SIG_Y, tool="read_file") | {"latency_ms": 50.0},
        _toy_event(
            "gd-fp", 5, make_signature("tool_call", "run_sql", "r=2"), tool="run_sql"
        )
        | {"latency_ms": 80.0},
        _toy_event(
            "gd-fp",
            6,
            make_signature("tool_call", "search_web", "s=1"),
            tool="search_web",
        )
        | {"latency_ms": 100.0},
        _toy_event(
            "gd-fp",
            7,
            make_signature("tool_call", "read_file", "f=1"),
            tool="read_file",
        )
        | {"latency_ms": 50.0},
        _toy_event(
            "gd-fp", 8, make_signature("tool_call", "run_sql", "r=3"), tool="run_sql"
        )
        | {"latency_ms": 80.0},
        _toy_event(
            "gd-fp",
            9,
            make_signature("tool_call", "mystery_tool", "m=1"),
            tool="mystery_tool",
        )
        | {"latency_ms": 70.0},
    ]
    lines = [
        json.dumps(
            {
                "episode_id": "gd-hit",
                "label": {"trigger": "goal_drift"},
                "config": "goal_drift",
                "events": _finish("gd-hit", hit),
            },
            sort_keys=True,
        ),
        json.dumps(
            {
                "episode_id": "gd-miss",
                "label": {"trigger": "goal_drift"},
                "config": "goal_drift",
                "events": _finish("gd-miss", miss),
            },
            sort_keys=True,
        ),
        json.dumps(
            {
                "episode_id": "gd-fp",
                "label": None,
                "config": "goal_drift",
                "events": _finish("gd-fp", fp),
            },
            sort_keys=True,
        ),
    ]
    out = tmp_path / "gd-fixtures"
    out.mkdir()
    _write_goal_drift_baseline(out)
    (out / "toy_goal_drift.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


@pytest.fixture()
def ml_ensemble_toy_dir(tmp_path: Path) -> Path:
    """Three-episode ml_ensemble toy corpus, replayed under the ml_ensemble
    variant where every base detector is wrapped in the noisy-OR orchestrator.

    Paper derivation of expected numbers (done BEFORE any code ran):

    * ``ml-hit`` (label ml_ensemble): signatures X, X, X then a message. At
      the third X the loop detector emits score count/threshold*0.5 =
      3/3*0.5 = 0.5. The noisy-OR over that lone score is
      1 - (1 - 0.5) = 0.5, which is NOT below the 0.5 ensemble threshold
      (the guard is strict '<'), so exactly one ml_ensemble risk emits.
      Predicted = {ml_ensemble}. Contributes TP(ml_ensemble) += 1.
    * ``ml-miss`` (label ml_ensemble): X, X, other then a message. The count
      for X peaks at 2 < 3, no base detector emits anything, the orchestrator
      sees no scores and stays silent. Predicted = {}.
      Contributes FN(ml_ensemble) += 1.
    * ``ml-fp`` (label null): X, Y, X, Y, X. The count for X reaches 3 at the
      fifth step, loop score 0.5 crosses the ensemble threshold, and the
      COMBINED trigger fires on an UNLABELED episode -- note the individual
      loop trigger itself never reaches sinks under this variant.
      Predicted = {ml_ensemble}. Contributes FP(ml_ensemble) += 1 and
      healthy_fired += 1.

    Corpus aggregates for ml_ensemble, by hand:
      TP=1, FP=1, FN=1 -> precision = recall = f1 = 0.5
      macro-F1 over supported triggers = 0.5 (ml_ensemble is the only one)
      healthy controls fired = 1
    Every other trigger has TP=FP=FN=0.

    All episodes stay under 20 tool calls, carry no latency/token/error
    fields, and end on message steps, so no other base signal exists that the
    orchestrator could combine.
    """
    hit = [
        _toy_event("ml-hit", 0, _SIG_X),
        _toy_event("ml-hit", 1, _SIG_X),
        _toy_event("ml-hit", 2, _SIG_X),
    ]
    miss = [
        _toy_event("ml-miss", 0, _SIG_X),
        _toy_event("ml-miss", 1, _SIG_X),
        _toy_event("ml-miss", 2, _SIG_OTHER_A),
    ]
    fp = [
        _toy_event("ml-fp", 0, _SIG_X),
        _toy_event("ml-fp", 1, _SIG_Y),
        _toy_event("ml-fp", 2, _SIG_X),
        _toy_event("ml-fp", 3, _SIG_Y),
        _toy_event("ml-fp", 4, _SIG_X),
    ]
    lines = []
    for ep_id, label, events in (
        ("ml-hit", {"trigger": "ml_ensemble"}, hit),
        ("ml-miss", {"trigger": "ml_ensemble"}, miss),
        ("ml-fp", None, fp),
    ):
        record: dict = {"episode_id": ep_id, "label": label, "config": "ml_ensemble"}
        record["events"] = _finish(ep_id, list(events))
        lines.append(json.dumps(record, sort_keys=True))
    out = tmp_path / "ml-fixtures"
    out.mkdir()
    (out / "toy_ml_ensemble.jsonl").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    return out


def test_toy_corpus_goal_drift_matches_hand_computed_scores(
    harness, goal_drift_toy_dir: Path
) -> None:
    outcomes = [
        harness.replay_episode(ep, goal_drift_toy_dir)
        for ep in harness.iter_fixtures(goal_drift_toy_dir)
    ]
    report = harness.score(outcomes)

    s = report.per_trigger["goal_drift"]
    assert (s.tp, s.fp, s.fn) == (1, 1, 1)
    assert s.precision == pytest.approx(0.5)
    assert s.recall == pytest.approx(0.5)
    assert s.f1 == pytest.approx(0.5)
    assert report.macro_f1 == pytest.approx(0.5)

    # Every other shipped trigger saw no support at all.
    for trig in harness.SHIPPED_TRIGGERS:
        if trig == "goal_drift":
            continue
        other = report.per_trigger[trig]
        assert (other.tp, other.fp, other.fn) == (0, 0, 0)

    assert report.n_labeled == 2
    assert report.n_healthy == 1
    assert report.healthy_fired == 1


def test_toy_corpus_ml_ensemble_matches_hand_computed_scores(
    harness, ml_ensemble_toy_dir: Path
) -> None:
    outcomes = [
        harness.replay_episode(ep, ml_ensemble_toy_dir)
        for ep in harness.iter_fixtures(ml_ensemble_toy_dir)
    ]
    report = harness.score(outcomes)

    s = report.per_trigger["ml_ensemble"]
    assert (s.tp, s.fp, s.fn) == (1, 1, 1)
    assert s.precision == pytest.approx(0.5)
    assert s.recall == pytest.approx(0.5)
    assert s.f1 == pytest.approx(0.5)
    assert report.macro_f1 == pytest.approx(0.5)

    for trig in harness.SHIPPED_TRIGGERS:
        if trig == "ml_ensemble":
            continue
        other = report.per_trigger[trig]
        assert (other.tp, other.fp, other.fn) == (0, 0, 0)

    # The combined trigger fully replaces the individual loop trigger under
    # this variant: the firing episodes must show ONLY ml_ensemble.
    by_id = {o.episode_id: o.predicted for o in outcomes}
    assert by_id["ml-hit"] == {"ml_ensemble"}
    assert by_id["ml-fp"] == {"ml_ensemble"}

    assert report.n_labeled == 2
    assert report.n_healthy == 1
    assert report.healthy_fired == 1


def test_new_toy_corpora_cli_exit_codes(
    harness, goal_drift_toy_dir: Path, ml_ensemble_toy_dir: Path
) -> None:
    """Both toy corpora contain one firing healthy control each, so the CI
    false-positive gate must fail (exit 1) on both."""
    for toy_dir in (goal_drift_toy_dir, ml_ensemble_toy_dir):
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            code = harness.main(["--fixtures", str(toy_dir), "--format", "json"])
        assert code == 1
        payload = json.loads(stdout.getvalue())
        assert payload["healthy_controls_fired"] == 1
        assert payload["episodes"] == {"labeled": 2, "healthy": 1}


def test_loader_rejects_unknown_config_variant(harness, tmp_path: Path) -> None:
    bad = {
        "episode_id": "bad-config",
        "label": None,
        "config": "nope",
        "events": [_toy_event("bad-config", 0, _SIG_X)],
    }
    target = tmp_path / "cfg.jsonl"
    target.write_text(json.dumps(bad) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unknown config variant"):
        list(harness.iter_fixtures(tmp_path))


def test_harness_config_rejects_unknown_variant(harness) -> None:
    with pytest.raises(ValueError, match="unknown harness config variant"):
        harness.harness_config("bogus")


# --------------------------------------------------------------------------
# Scorer semantics
# --------------------------------------------------------------------------


def test_label_null_episode_never_produces_false_negative(harness) -> None:
    """A healthy control has no label to miss: whatever it emits can only be
    an FP, and silence contributes nothing at all."""
    silent = harness.EpisodeOutcome("h1", None)
    noisy = harness.EpisodeOutcome("h2", None, predicted={"loop"})
    report = harness.score([silent, noisy])
    assert report.n_labeled == 0
    assert report.n_healthy == 2
    for trig, stats in report.per_trigger.items():
        assert stats.fn == 0, f"unlabeled episodes produced FN for {trig}"
    assert report.per_trigger["loop"].fp == 1
    assert all(s.tp == 0 for s in report.per_trigger.values())


def test_duplicate_firings_count_once_per_episode(harness, tmp_path: Path) -> None:
    """One episode where the loop detector legitimately fires TWICE (fires,
    clears below threshold, fires again after re-arm) must score a single TP:
    'a detector fires' means it emitted the trigger at least once."""
    lines = []
    events = [
        # First firing burst: third repeat crosses the threshold.
        _toy_event("dedupe", 0, _SIG_X),
        _toy_event("dedupe", 1, _SIG_X),
        _toy_event("dedupe", 2, _SIG_X),
        # Two distinct steps drop the count below 3 and re-arm the latch.
        _toy_event("dedupe", 3, _SIG_OTHER_A),
        _toy_event("dedupe", 4, _SIG_OTHER_B),
        # Second independent burst: fires again.
        _toy_event("dedupe", 5, _SIG_X),
        _toy_event("dedupe", 6, _SIG_X),
        _toy_event("dedupe", 7, _SIG_X),
    ]
    lines.append(
        json.dumps(
            {
                "episode_id": "dedupe",
                "label": {"trigger": "loop"},
                "events": _finish("dedupe", events),
            }
        )
    )
    out = tmp_path / "fixtures"
    out.mkdir()
    (out / "dedupe.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")

    episodes = list(harness.iter_fixtures(out))
    outcome = harness.replay_episode(episodes[0])
    assert outcome.predicted == {"loop"}

    report = harness.score([outcome])
    assert report.per_trigger["loop"].tp == 1
    assert report.per_trigger["loop"].fp == 0
    assert report.per_trigger["loop"].fn == 0


def test_meltdown_risk_maps_to_label_space_by_shape(harness) -> None:
    from snagline.risk import FailureRisk

    low = FailureRisk(
        "e",
        "s1",
        0.7,
        "meltdown",
        "tool-choice entropy collapsed to 0.00 bits (1 distinct in last 20 steps)",
        0.0,
    )
    high = FailureRisk(
        "e",
        "s2",
        0.6,
        "meltdown",
        "tool-choice entropy spiked to 3.52 bits (12 distinct in last 20 steps)",
        0.0,
    )
    plain = FailureRisk("e", "s3", 0.9, "loop", "action repeated 3x", 0.0)
    assert harness.risk_to_label_trigger(low) == "meltdown_low"
    assert harness.risk_to_label_trigger(high) == "meltdown_high"
    assert harness.risk_to_label_trigger(plain) == "loop"


def test_loader_rejects_metadata_and_unknown_labels(harness, tmp_path: Path) -> None:
    bad_meta = {
        "episode_id": "bad-meta",
        "label": None,
        "events": [_toy_event("bad-meta", 0, _SIG_X) | {"metadata": {"p": "x"}}],
    }
    bad_label = {
        "episode_id": "bad-label",
        "label": {"trigger": "nope"},
        "events": [_toy_event("bad-label", 0, _SIG_X)],
    }
    for name, record, fragment in (
        ("meta.jsonl", bad_meta, "unknown fields"),
        ("label.jsonl", bad_label, "unknown labeled trigger"),
    ):
        target = tmp_path / name
        target.write_text(json.dumps(record) + "\n", encoding="utf-8")
        with pytest.raises(ValueError, match=fragment):
            list(harness.iter_fixtures(tmp_path))


# --------------------------------------------------------------------------
# Committed fixture corpus integrity
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def committed_outcomes(harness) -> list:
    fixtures = _REPO_ROOT / "benchmarks" / "fixtures"
    return [harness.replay_episode(ep) for ep in harness.iter_fixtures(fixtures)]


def test_committed_corpus_shape(harness) -> None:
    fixtures = _REPO_ROOT / "benchmarks" / "fixtures"
    episodes = list(harness.iter_fixtures(fixtures))
    assert len(episodes) >= 60
    labeled = [e for e in episodes if e.label_triggers is not None]
    healthy = [e for e in episodes if e.label_triggers is None]
    assert len(labeled) >= 28 and len(healthy) >= 28
    covered = set().union(*(e.label_triggers for e in labeled))
    assert covered == set(harness.SHIPPED_TRIGGERS)


def test_committed_corpus_separates_cleanly(harness, committed_outcomes) -> None:
    """Every labeled episode fires exactly its intended trigger and every
    healthy control stays silent. Expectations come from the generator's
    documented per-builder design (each builder's docstring derives the
    arithmetic), not from calling the scorer."""
    for outcome in committed_outcomes:
        intended = outcome.label_triggers or set()
        assert outcome.predicted == intended, (
            f"{outcome.episode_id}: expected {sorted(intended)}, "
            f"got {sorted(outcome.predicted)}"
        )
    report = harness.score(committed_outcomes)
    for trig in harness.SHIPPED_TRIGGERS:
        stats = report.per_trigger[trig]
        assert stats.tp == 4 and stats.fp == 0 and stats.fn == 0, trig
    assert report.macro_f1 == pytest.approx(1.0)
    assert report.confusion == {}
    assert report.healthy_fired == 0


def test_cli_exit_zero_on_committed_fixtures(harness) -> None:
    stdout = io.StringIO()
    with contextlib.redirect_stdout(stdout):
        code = harness.main(
            [
                "--fixtures",
                str(_REPO_ROOT / "benchmarks" / "fixtures"),
                "--format",
                "table",
            ]
        )
    assert code == 0
    table = stdout.getvalue()
    assert "macro-F1" in table
    assert "silent_abort" in table
