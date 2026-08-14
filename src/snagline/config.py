"""All tunable thresholds for SNAGLINE's tier-1 detectors.

Centralizing thresholds here means ``Monitor.default()`` ships sensible
defaults and a caller can retune everything from one object without touching
detector constructors (project.md §5.4).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Config:
    # Loop detector
    loop_window_size: int = 12
    loop_repeat_threshold: int = 3

    # Error-cascade detector
    cascade_window_size: int = 10
    cascade_error_threshold: int = 3
    cascade_consecutive_threshold: int = 3

    # Latency / CUSUM detector (used once that detector lands)
    cusum_k: float = 0.5
    cusum_h: float = 5.0
    cusum_min_samples: int = 20  # warm-up: learn baseline before alarming
    # A perfectly stable baseline has sample std 0, which would make every
    # deviation infinite and force a div-by-zero guard that never fires. These
    # floors give the detector a meaningful deviation scale even for constant
    # baselines, so a single large spike alarms instead of requiring several.
    cusum_sigma_floor_abs: float = 1.0  # ms; never treat baseline std as smaller
    cusum_sigma_floor_rel: float = 0.05  # ... or smaller than 5% of the mean

    # Global
    fail_open: bool = True
