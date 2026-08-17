"""All tunable thresholds for SNAGLINE's tier-1 detectors.

Centralizing thresholds here means ``Monitor.default()`` ships sensible
defaults and a caller can retune everything from one object without touching
detector constructors (project.md §5.4).
"""

from __future__ import annotations

from dataclasses import dataclass

from snagline.baseline import BaselineProfile


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

    # Global
    fail_open: bool = True
