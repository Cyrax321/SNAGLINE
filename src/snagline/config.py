"""All tunable thresholds for SNAGLINE's tier-1 detectors.

Centralizing thresholds here means ``Monitor.default()`` ships sensible
defaults and a caller can retune everything from one object without touching
detector constructors (project.md §5.4).
"""

from __future__ import annotations

import json
import logging
import os
import types
from collections.abc import Mapping
from dataclasses import dataclass, fields
from typing import Any, Union, get_args, get_origin, get_type_hints

from snagline.baseline import BaselineProfile

logger = logging.getLogger("snagline")


def _union_args(hint: Any) -> tuple[Any, ...] | None:
    """Return the member tuple when ``hint`` is a union type, else None.

    Covers both spellings across Python versions: PEP 604 ``X | Y``
    (``types.UnionType``, whose get_origin differs between versions) and
    ``typing.Union[X, Y]`` / ``Optional[X]``.
    """
    if isinstance(hint, types.UnionType):
        return get_args(hint)
    if get_origin(hint) is Union:
        return get_args(hint)
    return None


def _coercible_hint(hint: Any) -> Any:
    """Unwrap ``X | None`` to ``X`` when X is an env-coercible scalar.

    Lets optional scalar fields (for example the calibration baseline path,
    issue #101) be set from environment variables like plain scalars, while
    non-scalar optionals (object references such as a BaselineProfile) stay
    out of reach of string coercion.
    """
    args = _union_args(hint)
    if args is None or len(args) != 2 or type(None) not in args:
        return hint
    other = next(a for a in args if a is not type(None))
    if other in (bool, int, float, str):
        return other
    return hint


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

    # --- Loop hardening modes (issue #89, opt-in) ----------------------------
    # Each mode extends LoopDetector with one more failure shape beyond plain
    # single-signature repetition. All default off so the plain-loop path is
    # unchanged and the zero-dependency preset keeps its bench numbers.
    # Triggers emitted (API strings): "near_duplicate_loop", "cycle", "stall".
    loop_near_duplicate_enabled: bool = False  # collapse volatile ids, recount
    loop_cycle_enabled: bool = False  # A,B,A,B,... periodicity scan
    loop_cycle_window_size: int = 12  # window scanned for periodicity
    loop_cycle_min_period: int = 2  # shortest repeating period considered
    loop_cycle_max_period: int = 6  # longest repeating period considered
    loop_stall_enabled: bool = False  # identical signature, zero progress
    loop_stall_steps: int = 25  # consecutive identical steps before firing

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

    # Token-runaway detector (issue #84, opt-in). Sustained-burn CUSUM over
    # per-step token volume plus an optional hard per-episode budget envelope.
    # Needs adapters that report tokens_in/tokens_out; disabled by default so
    # the zero-configuration preset and its published bench numbers are
    # unchanged until the accuracy gate (#82) has validated thresholds.
    token_runaway_enabled: bool = False
    token_cusum_k: float = 0.5  # slack parameter
    token_cusum_h: float = 5.0  # alarm threshold
    token_min_samples: int = 20  # warm-up before sustained-burn alarms
    episode_token_budget: int | None = None  # total tokens; None disables envelope
    token_budget_warn_fraction: float = 0.8  # single warning at this fraction

    # Meltdown detector (issue #85, opt-in until the accuracy gate lands).
    # Sliding-window Shannon entropy over tool-call identities; flags both the
    # low-entropy rote-collapse shape and the high-entropy thrash shape
    # documented as "meltdown" in arXiv:2603.29231. Thresholds are bits and
    # were tuned against fixtures: uniform alternation over ~5 tools (~2.32
    # bits) stays silent, collapse onto one tool (<0.4 bits) and churn across
    # 12+ (~3.6 bits) fire.
    meltdown_enabled: bool = False
    meltdown_window_size: int = 20
    meltdown_low_entropy: float = 0.4  # bits; below this the window is rote
    meltdown_high_entropy: float = 2.8  # bits; above this the window thrashes
    meltdown_rearm_steps: int = 10  # in-band steps before re-arming

    # Silent-abort detector (issue #86, opt-in). Evaluated once at
    # end_episode(): fires when an episode's final step was an error-free bare
    # tool call instead of an output step -- the completion check from
    # arXiv:2608.02464 that caught 7/7 organic failures-of-omission there.
    silent_abort_enabled: bool = False

    # --- Side-effect guard detector (issue #88, opt-in) ----------------------
    # Duplicate detection for host-declared non-idempotent actions
    # (``StepEvent.side_effect``): a second identical (tool_name,
    # action_signature) pair within one episode fires "side_effect_duplicate"
    # immediately. Deliberately stricter than the loop detector, which waits
    # for ``loop_repeat_threshold`` hits inside a sliding window and targets
    # wasted-work loops; here one repeated payment/send/deploy is already the
    # incident. Default off so the zero-dependency preset and its published
    # bench numbers are untouched.
    side_effect_guard_enabled: bool = False
    side_effect_allowed_repeats: int = 1  # occurrences tolerated before firing
    side_effect_score: float = 0.9  # routes as critical severity

    # --- Opt-in auto-calibration from a fitted BaselineProfile (issue #101) --
    # calibration="auto" derives error-cascade thresholds and the CUSUM latency
    # reference from a healthy-run profile (see snagline.calibration) instead
    # of using the hand-tuned constants above. Without a usable profile the
    # hand-tuned defaults apply unchanged (never worse than today). Derived
    # cascade counts are clamp-limited to [2, hand-tuned default] so auto can
    # only ever become more sensitive than shipped behavior.
    calibration: str = "manual"  # "manual" | "auto"
    # False-alarm probability budget per window evaluation used when deriving
    # cascade thresholds. Small on purpose: tier-1 detectors evaluate every
    # step, so one window is implicitly tested many times per episode.
    calibration_alpha: float = 0.001
    # Healthy reference for calibration. Pass the fitted object directly, or
    # point calibration_baseline_path at a file written by save_baseline() /
    # `snagline baseline --save`. The object field cannot come from JSON/TOML
    # config files or the environment; use the path variant there.
    calibration_baseline: BaselineProfile | None = None
    calibration_baseline_path: str | None = None

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
            hint = _coercible_hint(hints[name])
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
