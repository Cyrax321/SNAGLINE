"""Healthy-run baseline fitting (project.md §5 / the `snagline baseline` CLI).

Fits a per-tool profile (latency mean/std/min/max and error rate) from a
JSONL trajectory of a *known-healthy* agent run. The profile is persisted as
JSON so later build phases (the `goal_drift` and `ml_ensemble` detectors) have
a reference to compare live traffic against.

This is intentionally dependency-free: the basic statistics are computed with
Welford-style online accumulators, so `snagline baseline` works with no extra
installed. Heavier model-based baselines (the `ml` extra) can build on top of
the same persisted profile later.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from typing import IO, Any

from snagline._input import open_input as _open_input
from snagline.events import StepEvent


def _usable_latency(latency_ms: float) -> bool:
    """Whether a latency sample is fit to enter the running moments (#438).

    A non-finite sample is not a measurement: NaN poisons the running sum
    (``mean_latency`` becomes NaN, ``std_latency`` collapses to 0.0 because
    ``max(0.0, NaN)`` is 0.0 in CPython) and the corrupted profile is then
    *persisted*, shared by every monitor and detector that loads it. A
    negative latency is not physical either. Both are dropped from the
    latency statistics while still counting the call and its error, so a
    single bad line in a healthy trajectory costs one sample, not the whole
    profile -- the fail-soft handling ``fit_baseline_from_jsonl``'s docstring
    already promises for malformed lines.
    """
    return (
        isinstance(latency_ms, (int, float))
        and math.isfinite(latency_ms)
        and latency_ms >= 0.0
    )


def _validate_moments(mean: Any, std: Any) -> tuple[float, float]:
    """Coerce persisted latency moments, rejecting values that poison consumers.

    The live fit path drops unusable samples via ``_usable_latency`` (issues
    #350/#438); the *load* path used on-disk numbers verbatim, so a profile
    written by hand, by an older version, or by a crashed fit carried NaN or
    negative moments straight into the detectors (issue #592). A NaN mean makes
    ``latency - mean`` NaN and every CUSUM comparison False, arming nothing;
    a negative mean makes every healthy latency look like an anomaly and pages
    on step 0. A negative std is not a spread at all. Coerce to float and
    reject loudly at load so the diagnostic names the profile, not a detector.
    """
    try:
        mean_f = float(mean)
        std_f = float(std)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"baseline moments must be numeric; got mean={mean!r}, std={std!r}"
        ) from exc
    if not math.isfinite(mean_f):
        raise ValueError(f"baseline mean_latency must be finite; got {mean_f!r}")
    if math.isfinite(std_f):
        if std_f < 0.0:
            raise ValueError(f"baseline std_latency must be >= 0; got {std_f!r}")
    else:
        raise ValueError(f"baseline std_latency must be finite; got {std_f!r}")
    if mean_f < 0.0:
        raise ValueError(f"baseline mean_latency must be >= 0; got {mean_f!r}")
    return mean_f, std_f


def _validate_count(
    value: Any, field: str, *, tool: str, limit: int | None = None
) -> int:
    """Coerce a persisted count, rejecting values that poison the rate channels.

    Sibling of ``_validate_moments`` for the *count* fields (issue #722): the
    live ``add()`` path only ever raises them, so a negative, non-finite, or
    non-integer value proves the file was hand-edited, written by an older
    version, or corrupted. A NaN ``count`` makes ``error_rate`` NaN, which
    flattens every ``error_rate`` comparison to False and silently disables
    the channel; a negative ``error_count`` moves the healthy reference rate
    below zero, so any healthy live rate pages critical. ``error_count`` and
    ``latency_count`` are subsets of ``count`` on every path that produces
    them, so a value above it is not a count of anything.
    """
    try:
        # int() raises ValueError for NaN and non-numeric strings, and
        # OverflowError for an infinite JSON literal -- all corruption the
        # diagnostic below should name, not propagate.
        n = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            f"baseline tool {tool!r}: {field} must be an integer; got {value!r}"
        ) from exc
    if n < 0:
        raise ValueError(f"baseline tool {tool!r}: {field} must be >= 0; got {n!r}")
    if limit is not None and n > limit:
        raise ValueError(
            f"baseline tool {tool!r}: {field} must be <= count ({limit}); got {n!r}"
        )
    return n


def _optional_latency(value: Any) -> float | None:
    """Validate a persisted optional latency bound, or None when absent.

    ``min_latency`` / ``max_latency`` are absent on legacy profiles and on tools
    that never saw a latency-bearing call, so ``None`` stays ``None``. A present
    bound must satisfy the same contract a live sample does (#438): non-finite
    and negative values are not measurements, and they are what the detectors
    clamp against (issue #592).
    """
    if value is None:
        return None
    if not _usable_latency(value):
        raise ValueError(
            f"baseline latency bound must be a finite, non-negative number; "
            f"got {value!r}"
        )
    return float(value)


@dataclass
class ToolBaseline:
    """Per-tool healthy-run statistics.

    ``count`` covers every observed tool_call; ``latency_count`` is the subset
    that carried ``latency_ms``. Keeping them separate lets error rates be
    measured even on streams whose steps report no timing (auto-calibration,
    issue #101), without skewing the latency moments.
    """

    tool_name: str
    count: int = 0
    latency_count: int = 0
    latency_sum: float = 0.0
    latency_sum_sq: float = 0.0
    error_count: int = 0
    min_latency: float | None = None
    max_latency: float | None = None

    def add(self, latency_ms: float | None, error: bool) -> None:
        self.count += 1
        if latency_ms is not None and _usable_latency(latency_ms):
            self.latency_count += 1
            self.latency_sum += latency_ms
            self.latency_sum_sq += latency_ms * latency_ms
            if self.min_latency is None or latency_ms < self.min_latency:
                self.min_latency = latency_ms
            if self.max_latency is None or latency_ms > self.max_latency:
                self.max_latency = latency_ms
        if error:
            self.error_count += 1

    @property
    def mean_latency(self) -> float:
        return self.latency_sum / self.latency_count if self.latency_count else 0.0

    @property
    def std_latency(self) -> float:
        if self.latency_count < 2:
            return 0.0
        var = (self.latency_sum_sq - self.latency_sum**2 / self.latency_count) / (
            self.latency_count - 1
        )
        return max(0.0, var) ** 0.5

    @property
    def error_rate(self) -> float:
        return self.error_count / self.count if self.count else 0.0

    def to_dict(self) -> dict:
        return {
            "tool_name": self.tool_name,
            "count": self.count,
            "latency_count": self.latency_count,
            "mean_latency": self.mean_latency,
            "std_latency": self.std_latency,
            "min_latency": self.min_latency,
            "max_latency": self.max_latency,
            "error_count": self.error_count,
            "error_rate": self.error_rate,
        }

    @classmethod
    def from_dict(cls, data: dict, name: str | None = None) -> ToolBaseline:
        # ``to_dict`` writes ``tool_name`` inside the entry, but a profile
        # hand-edited or written by an older version may name its tools only
        # via the outer ``tools`` map. Accept the containing key as a fallback
        # so such a profile loads and the #592 moment validation gets the
        # chance to run, instead of a bare KeyError from the load path.
        tool_name = data.get("tool_name", name)
        if not tool_name:
            raise ValueError(
                "baseline tool entry has no tool_name and was loaded without a "
                "key name to fall back on"
            )
        tb = cls(tool_name=tool_name)
        # Issue #722: the counts were restored verbatim while the moments got
        # validated (#592), so the same hand-edited or corrupted profile that
        # #592 closed the latency door on still walked in through the rate
        # door. A NaN ``count`` makes ``error_rate`` NaN and every comparison
        # against it False, silently disabling the error channel; a negative
        # ``error_count`` puts the healthy rate below zero and pages critical
        # on healthy traffic; a non-numeric ``count`` loaded fine and then
        # raised TypeError inside the first detector that read the rate.
        # Validate here, naming the tool, instead of trusting the file.
        tb.count = _validate_count(data.get("count"), "count", tool=tool_name)
        # Profiles written before latency_count existed counted only
        # latency-bearing calls, which was also the error-rate denominator;
        # defaulting latency_count to count reproduces those statistics.
        tb.latency_count = _validate_count(
            data.get("latency_count", tb.count),
            "latency_count",
            tool=tool_name,
            limit=tb.count,
        )
        mean = data.get("mean_latency", 0.0)
        std = data.get("std_latency", 0.0)
        # Issue #592: the load path used on-disk moments verbatim, so a
        # hand-edited or corrupted profile reached the detectors unvalidated.
        # A NaN mean makes ``latency > mean + k*std`` False forever -- the
        # latency CUSUM silently never arms -- and a negative mean makes the
        # error-rate and latency pages critical on step 0 of every episode,
        # because a healthy call's latency is above a negative baseline by
        # definition. Validate here rather than in every consumer; the live
        # fit path already rejects these at add_event (issues #350/#438).
        mean, std = _validate_moments(mean, std)
        tb.latency_sum = mean * tb.latency_count
        tb.latency_sum_sq = (
            std**2 * max(1, tb.latency_count - 1) + tb.latency_sum**2 / tb.latency_count
            if tb.latency_count
            else 0.0
        )
        tb.error_count = _validate_count(
            data.get("error_count", 0),
            "error_count",
            tool=tool_name,
            limit=tb.count,
        )
        tb.min_latency = _optional_latency(data.get("min_latency"))
        tb.max_latency = _optional_latency(data.get("max_latency"))
        return tb


@dataclass
class BaselineProfile:
    """Aggregated healthy-run profile across all tools.

    The embedding fields are optional and only written by the semantic
    goal-drift fitter (``snagline.drift``, the ``drift`` extra, issue #81).
    They carry an averaged unit-independent reference vector over healthy
    steps, never any text: profiles stay content-free on disk. Legacy JSON
    without these keys loads unchanged, and profiles fitted structurally
    serialize exactly as before (byte-identical zero-dep preset).
    """

    tools: dict[str, ToolBaseline] = field(default_factory=dict)
    total_steps: int = 0
    fitted_at: float = (
        0.0  # unix seconds when the profile was fitted; 0.0 means unknown (schema-v1)
    )
    # Semantic reference (optional, drift extra): mean of per-step embeddings
    # over the healthy trajectory, how many steps contributed, and which
    # model produced them (provenance only; never logged with content).
    embedding_centroid: list[float] | None = None
    embedding_count: int = 0
    embedding_model: str | None = None

    def add_event(self, event: StepEvent) -> None:
        # Coerce the numeric fields *before* any accumulator is touched. A
        # StepEvent built straight from JSON carries no type guarantees -- a
        # wrong-typed ``"latency_ms": "120ms"`` parses and constructs cleanly,
        # then dies in ToolBaseline.add's arithmetic (issue #540). Validating
        # up front means such an event is rejected whole rather than leaving
        # the profile half-updated (total_steps counted, tool stats missing).
        latency_ms = event.latency_ms
        if latency_ms is not None:
            latency_ms = float(latency_ms)
        error = bool(event.error)
        self.total_steps += 1
        if event.action_type == "tool_call":
            name = event.tool_name or "default"
            tb = self.tools.get(name)
            if tb is None:
                tb = ToolBaseline(tool_name=name)
                self.tools[name] = tb
            tb.add(latency_ms, error)

    def to_dict(self) -> dict:
        data: dict = {
            "version": 1,
            "total_steps": self.total_steps,
            "tools": {name: tb.to_dict() for name, tb in sorted(self.tools.items())},
        }
        if self.fitted_at:
            data["fitted_at"] = self.fitted_at
        # Emit semantic keys only when actually fitted so legacy structural
        # profiles keep their exact historical byte layout.
        if self.embedding_centroid is not None:
            data["embedding_model"] = self.embedding_model
            data["embedding_centroid"] = list(self.embedding_centroid)
            data["embedding_count"] = self.embedding_count
        return data

    @classmethod
    def from_dict(cls, data: dict) -> BaselineProfile:
        profile = cls(
            total_steps=data.get("total_steps", 0),
            fitted_at=float(data.get("fitted_at", 0.0)),
        )
        for name, tb in data.get("tools", {}).items():
            profile.tools[name] = ToolBaseline.from_dict(tb, name)
        centroid = data.get("embedding_centroid")
        if centroid is not None:
            profile.embedding_centroid = [float(v) for v in centroid]
            profile.embedding_count = int(data.get("embedding_count", 0))
            profile.embedding_model = data.get("embedding_model")
        return profile


def fit_baseline_from_jsonl(path: str) -> BaselineProfile:
    """Fit a ``BaselineProfile`` from a JSONL trajectory (mirrors replay's
    fail-soft line handling: malformed lines are skipped, not fatal)."""
    profile = BaselineProfile(fitted_at=time.time())
    with _open_input(path) as fh:
        for _lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                event = StepEvent(**obj)
                # add_event is inside the try on purpose: StepEvent has no
                # __post_init__ validation, so a line with a wrong-typed field
                # parses fine and only fails here. The arithmetic must be
                # covered too, not just the parsing (issue #540).
                profile.add_event(event)
            except (json.JSONDecodeError, TypeError, ValueError):
                # Fail-soft, like replay(): a bad line must not abort the fit.
                continue
    return profile


def save_baseline(profile: BaselineProfile, path: str) -> None:
    """Persist a ``BaselineProfile`` to ``path`` as JSON (round-trips with
    ``load_baseline`` via ``BaselineProfile.to_dict``). The written profile is
    content-free -- only aggregate per-tool statistics, never event payloads."""
    with open(path, "w", encoding="utf-8") as fh:
        _write_json(fh, profile.to_dict())


def load_baseline(path: str) -> BaselineProfile:
    """Load a ``BaselineProfile`` previously written by ``save_baseline``
    (the inverse of ``save_baseline``, via ``BaselineProfile.from_dict``)."""
    with _open_input(path) as fh:
        return BaselineProfile.from_dict(json.load(fh))


def _write_json(stream: IO[str], data: dict) -> None:
    json.dump(data, stream, indent=2, sort_keys=True)
    stream.write("\n")
