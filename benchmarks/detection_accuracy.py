"""Detection-accuracy harness -- the honesty gate for every accuracy claim
(issue #82, project.md section 14).

Replays each labeled trajectory episode under ``--fixtures`` through a real
``Monitor`` (built by ``Monitor.default()`` plus exactly the opt-in flags the
labels require) and scores which triggers fired against the ground-truth
labels:

* A trigger "fires" on an episode when at least one ``FailureRisk`` carrying
  it is emitted at any point during the episode (including the
  ``end_episode`` finalize pass). Repeat emissions of one trigger on one
  episode count once.
* Per trigger the harness reports TP / FP / FN, precision, recall, and F1,
  plus macro-F1 over the shipped triggers with any support and a confusion summary
  listing which trigger fired on another trigger's data.
* Episodes with ``label: null`` are healthy controls: anything they emit is a
  false positive, and they can never contribute a false negative.
* An episode envelope may carry an optional ``config`` field naming a harness
  config variant (issue #118): ``"standard"`` (the default, used when the key
  is absent), ``"goal_drift"``, or ``"ml_ensemble"``. Variants are additive on
  the standard flags: ``goal_drift`` additionally enables the goal-drift
  detector against the corpus's committed baseline fixture
  (``goal_drift_baseline.json`` inside the scanned fixtures directory), and
  ``ml_ensemble`` additionally wraps every base detector in the noisy-OR
  ``MLOrchestrator``, so ensemble episodes can only ever emit the single
  combined ``ml_ensemble`` trigger.

The meltdown detector emits a single ``meltdown`` trigger for both failure
shapes; the harness maps an emission to label-space ``meltdown_low`` or
``meltdown_high`` using the documented detail wording ("collapsed" for the
rote shape, "spiked" for churn). See ``risk_to_label_trigger``.

Exit code: 0 when every healthy control stays silent, 1 when any control
fires, so CI can consume this as a false-positive gate. Accuracy numbers are
reported honestly either way; they do not affect the exit code.

Pure stdlib; imports only ``snagline``. Run::

    python benchmarks/detection_accuracy.py --fixtures benchmarks/fixtures \
        --format table
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import statistics
import subprocess
import sys
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from snagline import Monitor
from snagline.config import Config
from snagline.events import StepEvent
from snagline.risk import FailureRisk
from snagline.sinks.base import AlertSink

# Label-space vocabulary: the shipped detector triggers, with the two
# meltdown collapse shapes split apart. Order defines table row order.
SHIPPED_TRIGGERS: tuple[str, ...] = (
    "loop",
    "error_cascade",
    "latency_anomaly",
    "token_runaway",
    "budget_breach",
    "meltdown_low",
    "meltdown_high",
    "silent_abort",
    # Opt-in detectors given labeled coverage by issue #118. Both need a
    # harness config variant (see the ``config`` envelope field below).
    "goal_drift",
    "ml_ensemble",
    # Shipped detectors that were missing from the allowlist before #181.
    # cycle/stall are LoopDetector hardening modes, stagnation/side_effect/
    # governance are opt-in detectors, idle_gap/wall_clock_budget are
    # Monitor horizon time-axis signals. All are stdlib-only and have been
    # driven end-to-end on clean master.
    "stagnation",
    "side_effect_duplicate",
    "governance_decay",
    "cycle",
    "stall",
    "idle_gap",
    "wall_clock_budget",
)
# Triggers that exist in code but are intentionally not gated here.
# - near_duplicate_loop: LoopDetector hardening mode behind its own flag; not
#   yet given labeled fixtures, so it would be counted as uncovered if listed.
# - semantic_drift: not a distinct trigger string; the drift extra emits the
#   same "goal_drift" label, so gating it separately would double-count.
INTENTIONALLY_UNGATED: tuple[str, ...] = ("near_duplicate_loop",)

# Harness config variants (issue #118). Every variant includes the standard
# flags above; the named opt-in detector is switched on top of them.
HARNESS_CONFIG_VARIANTS: tuple[str, ...] = (
    "standard",
    "goal_drift",
    "ml_ensemble",
)
DEFAULT_CONFIG_VARIANT = "standard"

# Committed healthy BaselineProfile the goal_drift variant replays against.
# Written by generate_fixtures.py next to the episode files it emits; when the
# scanned --fixtures directory has none, the copy in this repo's fixture
# directory is used so old call sites keep working.
GOAL_DRIFT_BASELINE_FILENAME = "goal_drift_baseline.json"
_HARNESS_DIR = Path(__file__).resolve().parent
_REPO_FIXTURES_DIR = _HARNESS_DIR / "fixtures"

# The opt-in flags and fixed threshold the labels require on top of
# Monitor.default(). The episode token budget must be a known number for
# budget_breach labels to be decidable; fixture episodes were generated
# against this exact value (see fixtures/generate_fixtures.py).
HARNESS_TOKEN_BUDGET = 50_000

# Evaluation schema reported in the JSON payload so downstream tooling can
# branch on the shape of the results. Bumped whenever the payload structure
# changes in a way a consumer would need to know about.
EVALUATION_SCHEMA = "snagline-detection-accuracy/v2"
EVALUATION_STAGE = 2

# Step indexing convention used by every onset / first-detection / latency
# number this harness emits: the ZERO-BASED position of the event in the
# episode's ``events`` list. Generated fixtures encode exactly this index in
# each event's ``step_id`` (``f"{episode_id}-s{i}"``), so an emitted risk's
# ``step_id`` maps back to an index by lookup.
STEP_INDEX_CONVENTION = "zero-based"

# Below this many latency samples a percentile is not meaningfully distinct
# from the sample maximum, so the table reports "n/a" instead. The raw value
# is still carried in JSON next to its sample count.
MEANINGFUL_PERCENTILE_N = 5


def harness_config(
    variant: str = DEFAULT_CONFIG_VARIANT, fixtures_dir: Path | None = None
) -> Config:
    """Monitor.default() configuration plus the flags the labels require.

    ``variant`` selects a config variant from :data:`HARNESS_CONFIG_VARIANTS`;
    unknown names raise ValueError so a typo cannot silently replay an episode
    under the wrong detector set. The ``goal_drift`` variant loads its healthy
    reference from ``<fixtures_dir>/goal_drift_baseline.json``, falling back to
    this repo's committed copy when ``fixtures_dir`` is None.
    """
    if variant not in HARNESS_CONFIG_VARIANTS:
        raise ValueError(
            f"unknown harness config variant {variant!r}; "
            f"expected one of {list(HARNESS_CONFIG_VARIANTS)}"
        )
    cfg = Config(
        token_runaway_enabled=True,
        episode_token_budget=HARNESS_TOKEN_BUDGET,
        meltdown_enabled=True,
        silent_abort_enabled=True,
        stagnation_enabled=True,
        side_effect_guard_enabled=True,
        compaction_tripwire_enabled=True,
        loop_cycle_enabled=True,
        loop_stall_enabled=True,
        idle_warn_seconds=10.0,
        max_episode_wall_seconds=120.0,
    )
    if variant == "goal_drift":
        from snagline.baseline import load_baseline

        base_dir: Path = (
            fixtures_dir if fixtures_dir is not None else _REPO_FIXTURES_DIR
        )  # type: ignore[assignment]
        baseline_path: Path = base_dir / GOAL_DRIFT_BASELINE_FILENAME
        if not baseline_path.is_file():
            raise ValueError(
                f"goal_drift variant needs a committed baseline fixture at "
                f"{baseline_path}; generate one with benchmarks/fixtures/generate_fixtures.py"
            )
        cfg.goal_drift_enabled = True
        cfg.goal_drift_baseline = load_baseline(str(baseline_path))
    elif variant == "ml_ensemble":
        cfg.ml_ensemble_enabled = True
    # Opt-in auto-calibration sweep (issue #101): SNAGLINE_CALIBRATION=auto
    # flips the harness onto calibrated thresholds without touching the
    # fixtures, labels, or scoring. SNAGLINE_CALIBRATION_BASELINE_PATH points
    # at a saved BaselineProfile JSON (fit it from known-healthy traffic).
    # Unset, the harness behaves exactly as before (hand-tuned thresholds).
    if os.environ.get("SNAGLINE_CALIBRATION", "").strip().lower() == "auto":
        cfg.calibration = "auto"
        calib_path = os.environ.get("SNAGLINE_CALIBRATION_BASELINE_PATH", "").strip()
        if calib_path:
            cfg.calibration_baseline_path = calib_path
    return cfg


class RiskCollector:
    """Minimal AlertSink that records every dispatched risk in memory."""

    def __init__(self) -> None:
        self.risks: list[FailureRisk] = []

    def emit(self, risk: FailureRisk) -> None:
        self.risks.append(risk)


def risk_to_label_trigger(risk: FailureRisk) -> str:
    """Map an emitted risk to its label-space trigger name.

    The meltdown detector signals both collapse shapes with the single
    ``meltdown`` trigger; its documented detail wording distinguishes them
    ("collapsed" below the low-entropy band, "spiked" above the high band).
    Every other trigger passes through unchanged.
    """
    if risk.trigger == "meltdown":
        if "collapsed" in risk.detail:
            return "meltdown_low"
        return "meltdown_high"
    return str(risk.trigger)


# --------------------------------------------------------------------------
# Fixture loading
# --------------------------------------------------------------------------

_EVENT_FIELDS = (
    "step_id",
    "episode_id",
    "timestamp",
    "action_type",
    "action_signature",
    "tool_name",
    "latency_ms",
    "error",
    "error_type",
    "tokens_in",
    "tokens_out",
    "side_effect",
    "metadata",
)


@dataclass(frozen=True, slots=True)
class Episode:
    """One validated fixture episode: id, optional label set, ordered events.

    ``onset_step`` is the zero-based index of the first event at which the
    fixture generator intentionally introduced the labeled failure condition
    (see fixtures/generate_fixtures.py). It is ``None`` for healthy controls
    and for labeled fixtures whose generator cannot identify such an event;
    latency cannot be computed for those episodes.
    """

    episode_id: str
    label_triggers: frozenset[str] | None  # None means healthy control
    events: tuple[StepEvent, ...]
    config_variant: str = DEFAULT_CONFIG_VARIANT
    onset_step: int | None = None


def _parse_event(raw: dict, envelope_id: str) -> StepEvent:
    unknown = set(raw) - set(_EVENT_FIELDS)
    if unknown:
        raise ValueError(
            f"event carries unknown fields {sorted(unknown)} "
            f"(metadata must not enter fixtures)"
        )
    missing = {"step_id", "timestamp", "action_type", "action_signature"} - set(raw)
    if missing:
        raise ValueError(f"event missing required fields {sorted(missing)}")
    raw_ep = raw.get("episode_id", envelope_id)
    if raw_ep != envelope_id:
        raise ValueError(
            f"event episode_id {raw_ep!r} does not match envelope {envelope_id!r}"
        )
    latency = raw.get("latency_ms")
    metadata = raw.get("metadata")
    if metadata is not None and not isinstance(metadata, dict):
        raise ValueError("metadata must be a dict when present")
    return StepEvent(
        step_id=str(raw["step_id"]),
        episode_id=envelope_id,
        timestamp=float(raw["timestamp"]),
        action_type=str(raw["action_type"]),
        action_signature=str(raw["action_signature"]),
        tool_name=raw.get("tool_name"),
        latency_ms=None if latency is None else float(latency),
        error=bool(raw.get("error", False)),
        error_type=raw.get("error_type"),
        tokens_in=raw.get("tokens_in"),
        tokens_out=raw.get("tokens_out"),
        side_effect=bool(raw.get("side_effect", False)),
        metadata=dict(metadata) if isinstance(metadata, dict) else {},
    )


def _parse_onset(raw: Mapping[str, object], n_events: int, source: str) -> int | None:
    """Validate a label's ``onset_step``.

    Absent means the fixture predates onsets (or the generator could not
    derive one), and latency is simply not reported for the episode. Present
    means the generator asserts the failure condition starts at this
    zero-based event index, so an out-of-range value is a generator bug and
    must fail loudly rather than silently degrading the latency numbers.
    """
    if "onset_step" not in raw:
        return None
    value = raw["onset_step"]
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{source}: onset_step must be an int or null, got {value!r}")
    if not 0 <= value < n_events:
        raise ValueError(
            f"{source}: onset_step {value} is outside the episode's {n_events} events"
        )
    return value


def parse_episode(line: str, source: str) -> Episode:
    """Parse and validate one JSONL line into an :class:`Episode`."""
    record = json.loads(line)
    ep_id = record.get("episode_id")
    if not isinstance(ep_id, str) or not ep_id:
        raise ValueError(f"{source}: episode_id must be a non-empty string")
    label = record.get("label")
    if label is None:
        label_triggers = None
        onset = None
    else:
        if not isinstance(label, dict) or "trigger" not in label:
            raise ValueError(f"{source}: label must be null or {{'trigger': ...}}")
        trig = label["trigger"]
        if trig not in SHIPPED_TRIGGERS:
            raise ValueError(f"{source}: unknown labeled trigger {trig!r}")
        label_triggers = frozenset({trig})
        onset = _parse_onset(label, len(record.get("events") or []), source)
    variant = record.get("config", DEFAULT_CONFIG_VARIANT)
    if variant not in HARNESS_CONFIG_VARIANTS:
        raise ValueError(f"{source}: unknown config variant {variant!r}")
    raw_events = record.get("events")
    if not isinstance(raw_events, list) or not raw_events:
        raise ValueError(f"{source}: events must be a non-empty list")
    events = tuple(_parse_event(e, ep_id) for e in raw_events)
    if onset is not None and onset >= len(events):
        raise ValueError(f"{source}: onset_step {onset} is outside the parsed episode")
    return Episode(ep_id, label_triggers, events, variant, onset)


def iter_fixtures(fixtures_dir: Path) -> Iterator[Episode]:
    """Yield episodes from every ``*.jsonl`` file, sorted for determinism."""
    for path in sorted(fixtures_dir.glob("*.jsonl")):
        for lineno, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if not line.strip():
                continue
            yield parse_episode(line, f"{path.name}:{lineno}")


# --------------------------------------------------------------------------
# Replay
# --------------------------------------------------------------------------


@dataclass(slots=True)
class EpisodeOutcome:
    """What fired on one episode versus what its label says.

    ``predicted`` is unchanged since issue #82: the SET of label-space
    triggers emitted at any point, so repeat emissions of one trigger on one
    episode count once and existing scoring is untouched.

    ``first_detection`` is the stage-2 addition: for each trigger that fired,
    the earliest zero-based event index whose ingest (or the finalize pass)
    emitted it. Repeated firing after that step never replaces it. It is
    keyed by the same label-space trigger names as ``predicted``, so the two
    always cover the same set of triggers.

    ``onset_step`` is carried through from the fixture label so the scorer can
    compute latency without re-reading the fixtures.
    """

    episode_id: str
    label_triggers: frozenset[str] | None
    predicted: set[str] = field(default_factory=set)
    first_detection: dict[str, int] = field(default_factory=dict)
    onset_step: int | None = None


def replay_episode(
    episode: Episode, fixtures_dir: Path | None = None
) -> EpisodeOutcome:
    """Replay one episode through a fresh default-configured Monitor.

    A fresh monitor per episode mirrors per-episode state isolation and keeps
    detector windows/CUSUM baselines from leaking across episodes. The
    episode's config variant decides which opt-in flags run (issue #118);
    ``fixtures_dir`` locates the committed goal-drift baseline for that
    variant, defaulting to this repo's own fixture directory.
    """
    collector = RiskCollector()
    sinks: list[AlertSink] = [collector]
    monitor = Monitor.default(
        config=harness_config(episode.config_variant, fixtures_dir), sinks=sinks
    )
    # Emitted risks carry the step_id of the event being processed; the
    # finalize pass (silent_abort) attributes its risk to the last ingested
    # event. Both resolve through this table to a zero-based event index, so
    # every detection can be pinned to a step in the fixture.
    step_index = {ev.step_id: i for i, ev in enumerate(episode.events)}
    for event in episode.events:
        monitor.ingest(event)
    # Finalize pass: judges silent-abort-style completion checks, then tears
    # down per-episode state.
    monitor.end_episode(episode.episode_id)
    outcome = EpisodeOutcome(
        episode.episode_id, episode.label_triggers, onset_step=episode.onset_step
    )
    for risk in collector.risks:
        trig = risk_to_label_trigger(risk)
        outcome.predicted.add(trig)
        idx = step_index.get(risk.step_id)
        if idx is None:
            # A risk whose step_id is not one of this episode's events: the
            # detection is still counted (predicted), but its step cannot be
            # pinned, so no latency is reported for it.
            continue
        prev = outcome.first_detection.get(trig)
        if prev is None or idx < prev:
            outcome.first_detection[trig] = idx
    return outcome


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------


def detection_latency(
    onset_step: int | None, first_detection_step: int | None
) -> int | None:
    """Steps between the injected failure and its first detection.

    ``first_detection_step - onset_step``. ``None`` (never a sentinel number)
    when the expected detector never fired, when no onset was recorded, or
    when the detection could not be pinned to a fixture event. Negative means
    the detector fired before the recorded onset: reported as-is rather than
    clamped, because that is a real signal about the onset derivation or the
    detector's precursor sensitivity.
    """
    if onset_step is None or first_detection_step is None:
        return None
    return first_detection_step - onset_step


def percentile(values: Sequence[float], p: float) -> float | None:
    """Nearest-rank percentile: ceil(p / 100 * n), clipped to n, 1-based.

    Returns ``None`` for an empty sample. Chosen over interpolation because
    latency samples are small whole steps and an interpolated percentile
    between two observed steps would report a step that never happened.
    """
    if not values:
        return None
    ordered = sorted(values)
    rank = max(1, min(len(ordered), math.ceil(p * len(ordered) / 100)))
    return float(ordered[rank - 1])


@dataclass(slots=True)
class TriggerStats:
    """Per-trigger one-vs-rest confusion counts plus detection-latency samples.

    ``tp``/``fp``/``fn`` and the derived precision/recall/F1 are unchanged
    since issue #82 -- they only ever look at those three cells, so the
    ``tn`` addition cannot move Stage-1 metrics. ``tn`` makes each trigger's
    confusion matrix complete: for trigger X, every episode lands in exactly
    one of the four cells (see :func:`score`).

    ``latencies`` holds one entry per episode where X was the EXPECTED trigger
    AND fired AND the fixture records an onset: the delay between the injected
    failure and its first correct detection. False-positive firings of X on
    other failure modes have no onset for X (the episode's onset belongs to a
    different failure), so they are excluded from these statistics; their
    first-detection steps are still reported per episode in the JSON/CSV
    output for later cross-firing analysis.
    """

    tp: int = 0
    fp: int = 0
    fn: int = 0
    tn: int = 0
    latencies: list[int] = field(default_factory=list)
    # How many episodes expecting this trigger carried a usable onset. Equal
    # to len(latencies) plus any expected-trigger hit that fired but could not
    # be pinned to a step.
    n_onset: int = 0

    @property
    def precision(self) -> float:
        denom = self.tp + self.fp
        return self.tp / denom if denom else 0.0

    @property
    def recall(self) -> float:
        denom = self.tp + self.fn
        return self.tp / denom if denom else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0

    @property
    def specificity(self) -> float:
        """TN / (TN + FP): the true-negative rate. Zero, not NaN, when the
        trigger never had a negative episode at all."""
        denom = self.tn + self.fp
        return self.tn / denom if denom else 0.0

    @property
    def mean_latency(self) -> float | None:
        return statistics.fmean(self.latencies) if self.latencies else None

    @property
    def median_latency(self) -> float | None:
        return statistics.median(self.latencies) if self.latencies else None

    @property
    def p95_latency(self) -> float | None:
        return percentile(self.latencies, 95)

    @property
    def n_episodes(self) -> int:
        """Every episode contributes to exactly one of the four cells."""
        return self.tp + self.fp + self.tn + self.fn

    @property
    def n_detected(self) -> int:
        """Episodes on which this trigger fired, expected or not."""
        return self.tp + self.fp


@dataclass(slots=True)
class ScoreReport:
    per_trigger: dict[str, TriggerStats]
    confusion: dict[tuple[str, str], int]  # (true-or-"healthy", predicted) -> n
    healthy_fired: int
    n_labeled: int
    n_healthy: int

    @property
    def macro_f1(self) -> float:
        supported = [
            s
            for t, s in self.per_trigger.items()
            if t in SHIPPED_TRIGGERS and (s.tp + s.fp + s.fn) > 0
        ]
        if not supported:
            return 0.0
        return sum(s.f1 for s in supported) / len(supported)


def score(outcomes: Sequence[EpisodeOutcome]) -> ScoreReport:
    """Aggregate outcomes into per-trigger TP/FP/FN/TN, latency samples, and a
    confusion map.

    TP/FP/FN semantics are unchanged since issue #82: labeled episodes
    contribute TP where prediction matches label, FN where the label never
    fired, and FP (with a confusion entry) where something else fired; healthy
    controls have no label to miss, so they can only contribute FP.

    TN completes each trigger's one-vs-rest confusion matrix. For trigger X an
    episode is a TN when X is NOT expected (either it expects another trigger,
    or it is an unlabeled healthy control) and X does NOT fire. The denominator
    is therefore every episode in the corpus: TP + FP + TN + FN == episode
    count for each trigger. Note what this does and does not mean -- it counts
    "X correctly stayed silent when X was not the label", which for a corpus
    with 41 healthy controls and 16 other failure modes is dominated by
    irrelevancy. It is reported because specificity needs it, not as a quality
    claim about X.

    Latency samples are collected only for expected-trigger hits on episodes
    whose fixture records an onset.
    """
    stats: dict[str, TriggerStats] = {t: TriggerStats() for t in SHIPPED_TRIGGERS}
    confusion: dict[tuple[str, str], int] = {}
    healthy_fired = 0
    n_labeled = 0
    n_healthy = 0

    for outcome in outcomes:
        if outcome.label_triggers is None:
            n_healthy += 1
            for trig in sorted(outcome.predicted):
                stats.setdefault(trig, TriggerStats()).fp += 1
                key = ("healthy", trig)
                confusion[key] = confusion.get(key, 0) + 1
            if outcome.predicted:
                healthy_fired += 1
            continue
        n_labeled += 1
        label = outcome.label_triggers
        pred = outcome.predicted
        for trig in label & pred:
            stats.setdefault(trig, TriggerStats()).tp += 1
        for trig in label - pred:
            stats.setdefault(trig, TriggerStats()).fn += 1
        for trig in pred - label:
            stats.setdefault(trig, TriggerStats()).fp += 1
            key = (sorted(label)[0], trig)
            confusion[key] = confusion.get(key, 0) + 1

    # Second pass for the TN cell only, now that the trigger universe is
    # final: every episode where X was not expected and did not fire.
    universe = list(stats)
    for outcome in outcomes:
        expected: frozenset[str] = outcome.label_triggers or frozenset()
        for trig in universe:
            if trig not in expected and trig not in outcome.predicted:
                stats[trig].tn += 1

    # Latency samples: expected trigger fired AND the fixture records an onset.
    for outcome in outcomes:
        expected_label = outcome.label_triggers
        if expected_label is None or outcome.onset_step is None:
            continue
        for trig in sorted(expected_label):
            s = stats[trig]
            s.n_onset += 1
            step = outcome.first_detection.get(trig)
            if step is None:
                continue  # never fired: an FN, so there is no latency to add
            latency = detection_latency(outcome.onset_step, step)
            if latency is not None:
                s.latencies.append(latency)

    return ScoreReport(stats, confusion, healthy_fired, n_labeled, n_healthy)


# --------------------------------------------------------------------------
# Output
# --------------------------------------------------------------------------


def format_table(report: ScoreReport) -> str:
    # Width is derived, not hardcoded (issue #223): side_effect_duplicate (21)
    # and wall_clock_budget (17) both overflowed the old fixed 16-char column
    # and pushed their numeric columns out of alignment with the header and
    # rules. Deriving from every row's trigger keeps the table aligned as
    # triggers are added.
    width = max(
        len("trigger"),
        *(len(t) for t in SHIPPED_TRIGGERS),
        *(len(t) for t in report.per_trigger),
    )
    lines: list[str] = []
    header = (
        f"{'trigger':<{width}} {'TP':>4} {'FP':>4} {'FN':>4} {'TN':>4}"
        f" {'precision':>10} {'recall':>8} {'f1':>7} {'specificity':>11}"
    )
    lines.append(header)
    lines.append("-" * len(header))
    for trig in SHIPPED_TRIGGERS:
        s = report.per_trigger.get(trig, TriggerStats())
        lines.append(
            f"{trig:<{width}} {s.tp:>4} {s.fp:>4} {s.fn:>4} {s.tn:>4}"
            f" {s.precision:>10.3f} {s.recall:>8.3f} {s.f1:>7.3f}"
            f" {s.specificity:>11.3f}"
        )
    # Triggers outside the shipped vocabulary still get honest rows.
    extra = sorted(set(report.per_trigger) - set(SHIPPED_TRIGGERS))
    for trig in extra:
        s = report.per_trigger[trig]
        lines.append(
            f"{trig:<{width}} {s.tp:>4} {s.fp:>4} {s.fn:>4} {s.tn:>4}"
            f" {s.precision:>10.3f} {s.recall:>8.3f} {s.f1:>7.3f}"
            f" {s.specificity:>11.3f}"
        )
    lines.append("-" * len(header))
    lines.append(f"macro-F1: {report.macro_f1:.3f}")
    lines.append(
        f"episodes: {report.n_labeled + report.n_healthy} "
        f"({report.n_labeled} labeled, {report.n_healthy} healthy controls)"
    )
    lines.append("confusion (firings on other data):")
    if report.confusion:
        for (truth, pred), count in sorted(report.confusion.items()):
            lines.append(f"  {truth} -> {pred}: {count}")
    else:
        lines.append("  (none)")
    lines.append(f"healthy controls that fired: {report.healthy_fired}")
    return "\n".join(lines)


def _fmt_optional(value: float | None, *, digits: int = 1) -> str:
    """Render a latency statistic, or n/a when there are no samples."""
    return "n/a" if value is None else f"{value:.{digits}f}"


def format_latency_table(report: ScoreReport) -> str:
    """Per-trigger detection latency: median and p95 steps after the onset.

    Printed as its own table (after :func:`format_table`) rather than more
    columns on the confusion table, because most triggers share one latency
    sample set and the widths would drown the confusion numbers. ``n/a`` means
    either the trigger never produced a latency sample (never fired as an
    expected trigger, or its fixtures carry no onset) or the sample is too
    small for a percentile to mean anything beyond the maximum.
    """
    width = max(
        len("trigger"),
        *(len(t) for t in SHIPPED_TRIGGERS),
        *(len(t) for t in report.per_trigger),
    )
    header = (
        f"{'trigger':<{width}} {'episodes':>8} {'detected':>8} {'onset':>6}"
        f" {'median':>8} {'p95':>8}"
    )
    lines: list[str] = [
        "detection latency (first detection step - onset step, "
        f"{STEP_INDEX_CONVENTION} steps):",
        header,
        "-" * len(header),
    ]
    for trig in SHIPPED_TRIGGERS:
        s = report.per_trigger.get(trig, TriggerStats())
        p95 = (
            "n/a"
            if len(s.latencies) < MEANINGFUL_PERCENTILE_N
            else _fmt_optional(s.p95_latency, digits=1)
        )
        lines.append(
            f"{trig:<{width}} {s.n_episodes:>8} {s.n_detected:>8}"
            f" {s.n_onset:>6} {_fmt_optional(s.median_latency):>8} {p95:>8}"
        )
    extra = sorted(set(report.per_trigger) - set(SHIPPED_TRIGGERS))
    for trig in extra:
        s = report.per_trigger[trig]
        p95 = (
            "n/a"
            if len(s.latencies) < MEANINGFUL_PERCENTILE_N
            else _fmt_optional(s.p95_latency, digits=1)
        )
        lines.append(
            f"{trig:<{width}} {s.n_episodes:>8} {s.n_detected:>8}"
            f" {s.n_onset:>6} {_fmt_optional(s.median_latency):>8} {p95:>8}"
        )
    lines.append(
        f"p95 is reported only with >= {MEANINGFUL_PERCENTILE_N} latency samples"
        f" (nearest-rank); median needs >= 1"
    )
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Machine-readable output
# --------------------------------------------------------------------------


def _git_commit_sha() -> str:
    """Best-effort HEAD SHA; empty when git is unavailable or not a repo.

    run() is called with the repo root as cwd in practice, but the harness can
    be copied out of a repo (a tarball, a wheel), so a failure here must not
    break the report.
    """
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    return out.stdout.strip() if out.returncode == 0 else ""


def episode_records(
    outcomes: Sequence[EpisodeOutcome],
) -> list[dict]:
    """One machine-readable record per episode.

    Carries the onset, the expected trigger, the first step each trigger fired
    (expected or not), the expected trigger's latency, and the episode's
    classification. The per-trigger confusion classification (TP/FP/FN/TN
    against this episode's expected trigger) is also included so the CSV can
    reproduce the aggregate table row by row.
    """
    records: list[dict] = []
    for outcome in outcomes:
        expected = (
            next(iter(outcome.label_triggers), None) if outcome.label_triggers else None
        )
        first = outcome.first_detection.get(expected) if expected else None
        if expected is None:
            classification = "FP" if outcome.predicted else "TN"
        elif expected in outcome.predicted:
            classification = "TP"
        else:
            classification = "FN"
        records.append(
            {
                "episode_id": outcome.episode_id,
                "expected_trigger": expected,
                "onset_step": outcome.onset_step,
                "first_detection_step": first,
                "detection_latency": detection_latency(outcome.onset_step, first),
                "detected_triggers": sorted(outcome.predicted),
                "first_detection": dict(sorted(outcome.first_detection.items())),
                "classification": classification,
            }
        )
    return records


def report_to_json(
    report: ScoreReport,
    outcomes: Sequence[EpisodeOutcome] | None = None,
    *,
    fixtures_dir: Path | None = None,
) -> dict:
    """Full machine-readable report.

    ``outcomes`` is required for the per-episode records; passing none yields
    the stage-1-style summary only, which keeps old call sites working.
    """
    payload = {
        "metadata": {
            "evaluation_schema": EVALUATION_SCHEMA,
            "evaluation_stage": EVALUATION_STAGE,
            "commit_sha": _git_commit_sha(),
            "python_version": platform.python_version(),
            "python_implementation": platform.python_implementation(),
            "platform": platform.platform(),
            "step_index_convention": STEP_INDEX_CONVENTION,
            "fixture_count": report.n_labeled + report.n_healthy,
            "labeled_count": report.n_labeled,
            "healthy_count": report.n_healthy,
            # Seeds are not persisted in the fixture files (stage-1 finding);
            # the generator's scheme is documented here so a replay is
            # reproducible without a per-episode seed field existing.
            "seeds": {
                "persisted_in_fixtures": False,
                "scheme": (
                    'random.Random(f"{episode_id}/seed") per episode in '
                    "benchmarks/fixtures/generate_fixtures.py"
                ),
            },
            "fixture_directory": str(fixtures_dir) if fixtures_dir else None,
        },
        "per_trigger": {
            trig: {
                "tp": s.tp,
                "fp": s.fp,
                "tn": s.tn,
                "fn": s.fn,
                "precision": round(s.precision, 6),
                "recall": round(s.recall, 6),
                "f1": round(s.f1, 6),
                "specificity": round(s.specificity, 6),
                "episodes": s.n_episodes,
                "detected": s.n_detected,
                "with_onset": s.n_onset,
                "latency_samples": len(s.latencies),
                "mean_latency": (
                    round(s.mean_latency, 6) if s.mean_latency is not None else None
                ),
                "median_latency": (
                    round(s.median_latency, 6) if s.median_latency is not None else None
                ),
                "p95_latency": (
                    round(s.p95_latency, 6) if s.p95_latency is not None else None
                ),
                "p95_meaningful": len(s.latencies) >= MEANINGFUL_PERCENTILE_N,
            }
            for trig, s in sorted(report.per_trigger.items())
        },
        "macro_f1": round(report.macro_f1, 6),
        "confusion": [
            {"true": truth, "predicted": pred, "count": count}
            for (truth, pred), count in sorted(report.confusion.items())
        ],
        "healthy_controls_fired": report.healthy_fired,
        "episodes": {
            "labeled": report.n_labeled,
            "healthy": report.n_healthy,
        },
        "definitions": {
            "onset_step": (
                "zero-based index of the first event at which the fixture "
                "generator intentionally introduces the labeled failure "
                "condition; null when the generator cannot derive one"
            ),
            "first_detection_step": (
                "earliest zero-based event index whose ingest (or the "
                "end_episode finalize pass) emitted this trigger; repeated "
                "firings never replace it"
            ),
            "detection_latency": "first_detection_step - onset_step, or null",
            "tn": (
                "one-vs-rest: the episode does not expect this trigger and the "
                "trigger does not fire; TP+FP+TN+FN equals the episode count"
            ),
            "latency_scope": (
                "latency statistics cover only expected-trigger hits on "
                "episodes with a recorded onset; cross-firing timings are in "
                "the per-episode first_detection map instead"
            ),
        },
    }
    if outcomes is not None:
        payload["episode_records"] = episode_records(outcomes)
    return payload


EPISODE_CSV_FIELDS = (
    "episode",
    "trigger",
    "expected_trigger",
    "onset_step",
    "first_detection_step",
    "detection_latency",
    "classification",
    "detected_triggers",
)

TRIGGER_CSV_FIELDS = (
    "trigger",
    "tp",
    "fp",
    "tn",
    "fn",
    "precision",
    "recall",
    "f1",
    "specificity",
    "episodes",
    "detected",
    "with_onset",
    "mean_latency",
    "median_latency",
    "p95_latency",
)


def episode_csv_rows(
    outcomes: Sequence[EpisodeOutcome],
) -> list[list]:
    """One row per (episode, trigger) for every trigger that was expected OR
    fired. True negatives -- the trigger was neither expected nor fired -- are
    the implicit complement and are omitted to keep the CSV compact; the
    aggregate trigger CSV carries the full TN count."""
    rows: list[list] = []
    for outcome in outcomes:
        expected = outcome.label_triggers or frozenset()
        detected = set(outcome.predicted)
        for trig in sorted(expected | detected):
            if trig in expected and trig in detected:
                classification = "TP"
            elif trig in expected:
                classification = "FN"
            else:
                classification = "FP"
            first = outcome.first_detection.get(trig)
            rows.append(
                [
                    outcome.episode_id,
                    trig,
                    sorted(expected)[0] if expected else "",
                    outcome.onset_step,
                    first,
                    detection_latency(outcome.onset_step, first)
                    if trig in expected
                    else None,
                    classification,
                    ";".join(sorted(detected)),
                ]
            )
    return rows


def trigger_csv_rows(report: ScoreReport) -> list[list]:
    rows: list[list] = []
    for trig in sorted(report.per_trigger):
        s = report.per_trigger[trig]
        rows.append(
            [
                trig,
                s.tp,
                s.fp,
                s.tn,
                s.fn,
                f"{s.precision:.6f}",
                f"{s.recall:.6f}",
                f"{s.f1:.6f}",
                f"{s.specificity:.6f}",
                s.n_episodes,
                s.n_detected,
                s.n_onset,
                f"{s.mean_latency:.6f}" if s.mean_latency is not None else "",
                f"{s.median_latency:.6f}" if s.median_latency is not None else "",
                f"{s.p95_latency:.6f}" if s.p95_latency is not None else "",
            ]
        )
    return rows


def write_report_files(
    report: ScoreReport,
    outcomes: Sequence[EpisodeOutcome],
    out_dir: Path,
    fixtures_dir: Path | None = None,
) -> dict[str, Path]:
    """Write the JSON report and both CSVs into ``out_dir`` for archival."""
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "detection_accuracy.json"
    payload = report_to_json(report, outcomes, fixtures_dir=fixtures_dir)
    json_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    episodes_path = out_dir / "detection_accuracy_episodes.csv"
    with episodes_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(EPISODE_CSV_FIELDS)
        writer.writerows(episode_csv_rows(outcomes))
    triggers_path = out_dir / "detection_accuracy_triggers.csv"
    with triggers_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(TRIGGER_CSV_FIELDS)
        writer.writerows(trigger_csv_rows(report))
    return {
        "json": json_path,
        "episodes_csv": episodes_path,
        "triggers_csv": triggers_path,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Precision/recall/F1, confusion, and detection-latency harness "
            "over labeled fixture episodes."
        )
    )
    parser.add_argument(
        "--fixtures",
        type=Path,
        default=Path("benchmarks/fixtures"),
        help="directory containing labeled *.jsonl episode files",
    )
    parser.add_argument(
        "--format",
        choices=("table", "json", "csv"),
        default="table",
        help="output format (csv: one row per episode/trigger)",
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=None,
        help="also write detection_accuracy.json and both CSVs to this dir",
    )
    ns = parser.parse_args(argv)

    outcomes = [replay_episode(ep, ns.fixtures) for ep in iter_fixtures(ns.fixtures)]
    report = score(outcomes)
    if ns.format == "json":
        payload = report_to_json(report, outcomes, fixtures_dir=ns.fixtures)
        print(json.dumps(payload, indent=2, sort_keys=True))
    elif ns.format == "csv":
        writer = csv.writer(sys.stdout)
        writer.writerow(EPISODE_CSV_FIELDS)
        writer.writerows(episode_csv_rows(outcomes))
    else:
        print(format_table(report))
        print()
        print(format_latency_table(report))
    if ns.report_dir is not None:
        write_report_files(report, outcomes, ns.report_dir, ns.fixtures)
    # False-positive gate for CI: any firing healthy control fails the run.
    return 1 if report.healthy_fired else 0


if __name__ == "__main__":
    sys.exit(main())
