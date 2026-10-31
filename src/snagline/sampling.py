"""Adaptive telemetry event rate sampler."""

from __future__ import annotations

import random


class AdaptiveSampler:
    def __init__(self, sample_rate: float = 1.0) -> None:
        self.sample_rate = max(0.0, min(1.0, float(sample_rate)))

    def should_sample(self) -> bool:
        if self.sample_rate >= 1.0:
            return True
        if self.sample_rate <= 0.0:
            return False
        return random.random() < self.sample_rate
