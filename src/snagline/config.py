"""All tunable thresholds for SNAGLINE's tier-1 detectors.

Centralizing thresholds here means ``Monitor.default()`` ships sensible
defaults and a caller can retune everything from one object without touching
detector constructors (project.md §5.4).
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Mapping
from dataclasses import dataclass, fields
from typing import Any, get_type_hints

from snagline.baseline import BaselineProfile

logger = logging.getLogger("snagline")


def _coerce(hint: type, value: str) -> Any:
    if hint is bool:
        return value.strip().lower() in ("1", "true", "yes", "on", "t")
    if hint is int:
        return int(value)
    if hint is float:
        return float(value)
    return value


def _load_toml(text: str) -> dict[str, Any]:
    try:
        import tomllib
    except ModuleNotFoundError:  # pragma: no cover - exercised only on <3.11
        raise RuntimeError(
            "TOML config requires Python 3.11+ (tomllib); use JSON or upgrade."
        ) from None
    return tomllib.loads(text)


@dataclass
class Config:
    # Loop detector
    loop_window_size: int = 12
    loop_repeat_threshold: int = 3

    # Error-cascade detector
    cascade_window_size: int = 10
    cascade_error_threshold: int = 3
    cascade_consecutive_threshold: int = 3
    # By default the cascade detector only counts *tool* failures. A LangChain
    # LLM 502 or a planning-chain error is infrastructure noise that should not
    # trip a "tool is failing" alert (issue #16). Flip this to True to count
    # every error-bearing step regardless of action_type.
    cascade_count_non_tool_errors: bool = False

    # Latency / CUSUM detector (used once that detector lands)
    cusum_k: float = 0.5
    cusum_h: float = 5.0
    # Warm-up: learn a baseline before alarming. Lowered from 20 to 5 so that
    # tools called only a handful of times are still monitored (issue #9) --
    # the frozen baseline + sigma floor make a single large spike alarm after
    # the warm-up rather than requiring several sustained ones.
    cusum_min_samples: int = 5
    # A perfectly stable baseline has sample std 0, which would make every
    # deviation infinite and force a div-by-zero guard that never fires. These
    # floors give the detector a meaningful deviation scale even for constant
    # baselines, so a single large spike alarms instead of requiring several.
    cusum_sigma_floor_abs: float = 1.0  # ms; never treat baseline std as smaller
    cusum_sigma_floor_rel: float = 0.05  # ... or smaller than 5% of the mean

    # Goal-drift detector (next phase, step 2). Compares a live run's per-tool
    # error rate / latency against a persisted healthy BaselineProfile and
    # flags meaningful deviation. Opt-in: enabled only when a baseline exists.
    goal_drift_enabled: bool = False
    goal_drift_error_tolerance: float = 0.1  # allow baseline error_rate + this
    goal_drift_latency_k: float = 3.0  # sigmas above baseline mean counts as drift
    goal_drift_min_samples: int = 10  # live steps before scoring an episode
    goal_drift_score_threshold: float = 0.5  # emit a risk above this score
    goal_drift_baseline: BaselineProfile | None = None  # healthy reference

    # ML ensemble detector (next phase, step 3). When enabled, Monitor.default
    # wraps the base detectors in a single MLOrchestrator so signals combine.
    ml_ensemble_enabled: bool = False
    ml_ensemble_score_threshold: float = 0.5  # emit a combined risk above this

    # Global
    fail_open: bool = True

    # --- 12-factor configuration (project.md §5.4, ATTACH_ANY_SYSTEM P0) -----
    @classmethod
    def from_env(
        cls, environ: Mapping[str, str] | None = None, prefix: str = "SNAGLINE_"
    ) -> Config:
        """Build a ``Config`` from environment variables (12-factor).

        Reads ``<prefix><FIELD>`` (case-insensitive) and overrides the matching
        scalar field. Unknown prefixes, unknown keys, and values that fail to
        coerce are ignored (logged at warning) rather than fatal, so a host can
        pass through unrelated environment without breaking startup.
        """
        environ = os.environ if environ is None else environ
        hints = get_type_hints(cls)
        overrides: dict[str, Any] = {}
        for key, value in environ.items():
            if not key.startswith(prefix):
                continue
            name = key[len(prefix) :].lower()
            if name not in hints:
                continue
            hint = hints[name]
            if hint in (bool, int, float, str):
                try:
                    overrides[name] = _coerce(hint, value)
                except ValueError:
                    logger.warning("snagline: ignoring bad env %s=%r", key, value)
        return cls(**overrides)

    @classmethod
    def load_file(cls, path: str) -> Config:
        """Load a ``Config`` from a JSON or TOML file.

        ``.json`` is parsed with the stdlib ``json`` module; ``.toml`` requires
        Python 3.11+ (``tomllib``). Unknown keys are ignored so a config file
        can carry extra metadata without breaking construction.
        """
        with open(path, encoding="utf-8") as fh:
            text = fh.read()
        if path.endswith(".toml"):
            data: dict[str, Any] = _load_toml(text)
        else:
            data = json.loads(text)
        valid = {f.name for f in fields(cls)}  # noqa: F821
        return cls(**{k: v for k, v in data.items() if k in valid})
