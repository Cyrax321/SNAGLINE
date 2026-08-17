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
from dataclasses import dataclass, field
from typing import IO

from snagline.events import StepEvent


@dataclass
class ToolBaseline:
    """Per-tool healthy-run statistics."""

    tool_name: str
    count: int = 0
    latency_sum: float = 0.0
    latency_sum_sq: float = 0.0
    error_count: int = 0
    min_latency: float | None = None
    max_latency: float | None = None

    def add(self, latency_ms: float, error: bool) -> None:
        self.count += 1
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
        return self.latency_sum / self.count if self.count else 0.0

    @property
    def std_latency(self) -> float:
        if self.count < 2:
            return 0.0
        var = (self.latency_sum_sq - self.latency_sum**2 / self.count) / (
            self.count - 1
        )
        return max(0.0, var) ** 0.5

    @property
    def error_rate(self) -> float:
        return self.error_count / self.count if self.count else 0.0

    def to_dict(self) -> dict:
        return {
            "tool_name": self.tool_name,
            "count": self.count,
            "mean_latency": self.mean_latency,
            "std_latency": self.std_latency,
            "min_latency": self.min_latency,
            "max_latency": self.max_latency,
            "error_count": self.error_count,
            "error_rate": self.error_rate,
        }

    @classmethod
    def from_dict(cls, data: dict) -> ToolBaseline:
        tb = cls(tool_name=data["tool_name"])
        tb.count = data["count"]
        tb.latency_sum = data.get("mean_latency", 0.0) * tb.count
        tb.latency_sum_sq = (
            data.get("std_latency", 0.0) ** 2 * max(1, tb.count - 1)
            + tb.latency_sum**2 / tb.count
            if tb.count
            else 0.0
        )
        tb.error_count = data.get("error_count", 0)
        tb.min_latency = data.get("min_latency")
        tb.max_latency = data.get("max_latency")
        return tb


@dataclass
class BaselineProfile:
    """Aggregated healthy-run profile across all tools."""

    tools: dict[str, ToolBaseline] = field(default_factory=dict)
    total_steps: int = 0

    def add_event(self, event: StepEvent) -> None:
        self.total_steps += 1
        if event.action_type == "tool_call" and event.latency_ms is not None:
            name = event.tool_name or "default"
            tb = self.tools.get(name)
            if tb is None:
                tb = ToolBaseline(tool_name=name)
                self.tools[name] = tb
            tb.add(event.latency_ms, event.error)

    def to_dict(self) -> dict:
        return {
            "version": 1,
            "total_steps": self.total_steps,
            "tools": {name: tb.to_dict() for name, tb in sorted(self.tools.items())},
        }

    @classmethod
    def from_dict(cls, data: dict) -> BaselineProfile:
        profile = cls(total_steps=data.get("total_steps", 0))
        for name, tb in data.get("tools", {}).items():
            profile.tools[name] = ToolBaseline.from_dict(tb)
        return profile


def fit_baseline_from_jsonl(path: str) -> BaselineProfile:
    """Fit a ``BaselineProfile`` from a JSONL trajectory (mirrors replay's
    fail-soft line handling: malformed lines are skipped, not fatal)."""
    profile = BaselineProfile()
    with open(path, encoding="utf-8") as fh:
        for _lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                event = StepEvent(**obj)
            except (json.JSONDecodeError, TypeError, ValueError):
                # Fail-soft, like replay(): a bad line must not abort the fit.
                continue
            profile.add_event(event)
    return profile


def save_baseline(profile: BaselineProfile, path: str) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        _write_json(fh, profile.to_dict())


def load_baseline(path: str) -> BaselineProfile:
    with open(path, encoding="utf-8") as fh:
        return BaselineProfile.from_dict(json.load(fh))


def _write_json(stream: IO[str], data: dict) -> None:
    json.dump(data, stream, indent=2, sort_keys=True)
    stream.write("\n")
