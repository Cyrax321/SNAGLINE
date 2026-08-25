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

    # --- Stagnation detector (issue #87) --------------------------------------
    # Opt-in novelty-rate tracker: flags an episode whose share of never-
    # before-seen action signatures collapses, i.e. the agent is busy but
    # discovering nothing. Distinct from the loop detector, which needs exact
    # repeats: near-duplicate actions with slightly varied arguments produce
    # fresh signatures and evade exact matching while still being stuck.
    # Default off so the zero-dependency preset and the published bench
    # numbers are untouched.
    stagnation_enabled: bool = False
    stagnation_window_size: int = 50  # steps per novelty window
    stagnation_min_novelty: float = 0.05  # stale when fewer than this share new
    stagnation_patience: int = 2  # consecutive stale windows before firing

    # Global
    fail_open: bool = True

    # --- Structured logging sink (issue #99) ---------------------------------
    # Emission format for the logging sink (``sinks/logging_sink.py``):
    # "text" keeps plain lines, "json" emits one compact JSON object per risk
    # with exactly the keys ts, episode_id, step_id, trigger, severity, score,
    # detail. Selectable via env ``SNAGLINE_LOG_FORMAT`` or a config-file
    # ``log_format`` key; values other than "text"/"json" are undefined.
    # Structure only: the emitted object never carries prompt/response content.
    log_format: str = "text"

    # --- Sidecar /metrics exposition (issue #98) -----------------------------
    # Format served by GET /metrics on the sidecar: "prometheus" serves text
    # exposition version 0.0.4 (the default), "classic" serves the legacy JSON
    # counters body. Per-request override via ?format=classic or
    # ?format=prometheus; environment override SNAGLINE_METRICS_FORMAT.
    metrics_format: str = "prometheus"

    # --- 12-factor configuration (project.md §5.4, ATTACH_ANY_SYSTEM P0) -----
    @classmethod
    def from_env_overrides(
        cls, environ: Mapping[str, str] | None = None, prefix: str = "SNAGLINE_"
    ) -> dict[str, Any]:
        """Return only the fields that ``environ`` actually sets, coerced.

        ``resolve`` needs the *set of keys present in the environment*, not just
        their values: once folded into a ``Config``, an environment variable
        whose value equals the built-in default is indistinguishable from an
        unset one, and comparing against a default instance would silently drop
        it (issue #66).

        Reads ``<prefix><FIELD>`` (case-insensitive). Unknown prefixes, unknown
        keys, and values that fail to coerce are ignored (logged at warning)
        rather than fatal, so a host can pass through unrelated environment
        without breaking startup.
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
        return overrides

    @classmethod
    def from_env(
        cls, environ: Mapping[str, str] | None = None, prefix: str = "SNAGLINE_"
    ) -> Config:
        """Build a ``Config`` from environment variables (12-factor).

        Reads ``<prefix><FIELD>`` (case-insensitive) and overrides the matching
        scalar field; every other field keeps its built-in default. See
        ``from_env_overrides`` for the same information as a dict of just the
        keys the environment set.
        """
        return cls(**cls.from_env_overrides(environ=environ, prefix=prefix))

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
        valid = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in valid})

    @classmethod
    def resolve(
        cls,
        path: str | None = None,
        environ: Mapping[str, str] | None = None,
        prefix: str = "SNAGLINE_",
    ) -> Config:
        """Build the effective ``Config`` by layering sources (12-factor).

        Precedence (lowest to highest): built-in defaults -> optional config
        file (``path``) -> environment variables (``prefix``). Environment
        overrides win over the file, which wins over defaults. Unknown file
        keys and unset environment keys do not change anything.

        This is the single entrypoint the CLI and host integrations should use
        so behavior is consistent across ``snagline serve``, ``watch``,
        ``replay``, and embedded use.
        """
        cfg = cls.load_file(path) if path else cls()
        # Apply exactly the keys the environment set. Comparing an env-derived
        # Config against a default instance instead would treat "set to the
        # default value" as "unset" and let the file win (issue #66) -- which is
        # precisely the case an operator hits when resetting a shared config
        # file back to stock behaviour from the environment.
        for name, value in cls.from_env_overrides(
            environ=environ, prefix=prefix
        ).items():
            setattr(cfg, name, value)
        return cfg
