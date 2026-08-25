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
  plus macro-F1 over all triggers with any support and a confusion summary
  listing which trigger fired on another trigger's data.
* Episodes with ``label: null`` are healthy controls: anything they emit is a
  false positive, and they can never contribute a false negative.

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
import json
import sys
from collections.abc import Iterator, Sequence
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
)

# The opt-in flags and fixed threshold the labels require on top of
# Monitor.default(). The episode token budget must be a known number for
# budget_breach labels to be decidable; fixture episodes were generated
# against this exact value (see fixtures/generate_fixtures.py).
HARNESS_TOKEN_BUDGET = 50_000


def harness_config() -> Config:
    """Monitor.default() configuration plus the flags the labels require."""
    return Config(
        token_runaway_enabled=True,
        episode_token_budget=HARNESS_TOKEN_BUDGET,
        meltdown_enabled=True,
        silent_abort_enabled=True,
    )


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
)


@dataclass(frozen=True, slots=True)
class Episode:
    """One validated fixture episode: id, optional label set, ordered events."""

    episode_id: str
    label_triggers: frozenset[str] | None  # None means healthy control
    events: tuple[StepEvent, ...]


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
    )


def parse_episode(line: str, source: str) -> Episode:
    """Parse and validate one JSONL line into an :class:`Episode`."""
    record = json.loads(line)
    ep_id = record.get("episode_id")
    if not isinstance(ep_id, str) or not ep_id:
        raise ValueError(f"{source}: episode_id must be a non-empty string")
    label = record.get("label")
    if label is None:
        label_triggers = None
    else:
        if not isinstance(label, dict) or "trigger" not in label:
            raise ValueError(f"{source}: label must be null or {{'trigger': ...}}")
        trig = label["trigger"]
        if trig not in SHIPPED_TRIGGERS:
            raise ValueError(f"{source}: unknown labeled trigger {trig!r}")
        label_triggers = frozenset({trig})
    raw_events = record.get("events")
    if not isinstance(raw_events, list) or not raw_events:
        raise ValueError(f"{source}: events must be a non-empty list")
    events = tuple(_parse_event(e, ep_id) for e in raw_events)
    return Episode(ep_id, label_triggers, events)


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
    """What fired on one episode versus what its label says."""

    episode_id: str
    label_triggers: frozenset[str] | None
    predicted: set[str] = field(default_factory=set)


def replay_episode(episode: Episode) -> EpisodeOutcome:
    """Replay one episode through a fresh default-configured Monitor.

    A fresh monitor per episode mirrors per-episode state isolation and keeps
    detector windows/CUSUM baselines from leaking across episodes.
    """
    collector = RiskCollector()
    sinks: list[AlertSink] = [collector]
    monitor = Monitor.default(config=harness_config(), sinks=sinks)
    for event in episode.events:
        monitor.ingest(event)
    # Finalize pass: judges silent-abort-style completion checks, then tears
    # down per-episode state.
    monitor.end_episode(episode.episode_id)
    outcome = EpisodeOutcome(episode.episode_id, episode.label_triggers)
    for risk in collector.risks:
        outcome.predicted.add(risk_to_label_trigger(risk))
    return outcome


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------


@dataclass(slots=True)
class TriggerStats:
    tp: int = 0
    fp: int = 0
    fn: int = 0

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
    """Aggregate outcomes into per-trigger TP/FP/FN plus a confusion map.

    Labeled episodes contribute TP where prediction matches label, FN where
    the label never fired, and FP (with a confusion entry) where something
    else fired. Healthy controls can only contribute FP: with no label there
    is nothing to miss, so they can never produce a false negative.
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

    return ScoreReport(stats, confusion, healthy_fired, n_labeled, n_healthy)


# --------------------------------------------------------------------------
# Output
# --------------------------------------------------------------------------


def format_table(report: ScoreReport) -> str:
    lines: list[str] = []
    header = f"{'trigger':<16} {'TP':>4} {'FP':>4} {'FN':>4} {'precision':>10} {'recall':>8} {'f1':>7}"
    lines.append(header)
    lines.append("-" * len(header))
    for trig in SHIPPED_TRIGGERS:
        s = report.per_trigger.get(trig, TriggerStats())
        lines.append(
            f"{trig:<16} {s.tp:>4} {s.fp:>4} {s.fn:>4}"
            f" {s.precision:>10.3f} {s.recall:>8.3f} {s.f1:>7.3f}"
        )
    # Triggers outside the shipped vocabulary still get honest rows.
    extra = sorted(set(report.per_trigger) - set(SHIPPED_TRIGGERS))
    for trig in extra:
        s = report.per_trigger[trig]
        lines.append(
            f"{trig:<16} {s.tp:>4} {s.fp:>4} {s.fn:>4}"
            f" {s.precision:>10.3f} {s.recall:>8.3f} {s.f1:>7.3f}"
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


def report_to_json(report: ScoreReport) -> dict:
    return {
        "per_trigger": {
            trig: {
                "tp": s.tp,
                "fp": s.fp,
                "fn": s.fn,
                "precision": round(s.precision, 6),
                "recall": round(s.recall, 6),
                "f1": round(s.f1, 6),
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
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Precision/recall/F1 harness over labeled fixture episodes."
    )
    parser.add_argument(
        "--fixtures",
        type=Path,
        default=Path("benchmarks/fixtures"),
        help="directory containing labeled *.jsonl episode files",
    )
    parser.add_argument(
        "--format",
        choices=("table", "json"),
        default="table",
        help="output format",
    )
    ns = parser.parse_args(argv)

    outcomes = [replay_episode(ep) for ep in iter_fixtures(ns.fixtures)]
    report = score(outcomes)
    if ns.format == "json":
        print(json.dumps(report_to_json(report), indent=2, sort_keys=True))
    else:
        print(format_table(report))
    # False-positive gate for CI: any firing healthy control fails the run.
    return 1 if report.healthy_fired else 0


if __name__ == "__main__":
    sys.exit(main())
