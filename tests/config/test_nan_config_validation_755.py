import math

import pytest

from snagline.config import Config


@pytest.mark.parametrize(
    "knob",
    [
        "cusum_k",
        "token_cusum_k",
        "cusum_min_samples",
        "token_min_samples",
        "loop_window_size",
        "cascade_window_size",
        "loop_cycle_window_size",
        "meltdown_window_size",
        "meltdown_rearm_steps",
        "semantic_drift_window_size",
        "semantic_drift_cusum_h",
        "semantic_drift_cusum_k",
        "goal_drift_error_tolerance",
        "goal_drift_latency_k",
        "cusum_sigma_floor_abs",
        "cusum_sigma_floor_rel",
        "stagnation_window_size",
        "stagnation_patience",
        "side_effect_allowed_repeats",
        "cascade_consecutive_threshold",
        "cascade_error_threshold",
        "loop_repeat_threshold",
        "loop_stall_steps",
    ],
)
def test_config_rejects_nan_in_bound_validators(knob: str) -> None:
    """Issue #755: NaN must not pass bound validators in Config.__post_init__."""
    with pytest.raises(ValueError):
        Config(**{knob: math.nan})  # type: ignore[arg-type]


def test_config_valid_defaults_construct_cleanly() -> None:
    cfg = Config()
    assert cfg.cusum_k == 0.5
    assert cfg.cusum_h == 5.0
