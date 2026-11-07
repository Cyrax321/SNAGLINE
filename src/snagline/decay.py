# Copyright 2026 SNAGLINE contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Risk score decay and half-life smoothing."""

from __future__ import annotations

import math


class ScoreDecay:
    def __init__(self, half_life_seconds: float = 300.0) -> None:
        if half_life_seconds <= 0.0:
            raise ValueError("half_life_seconds must be positive")
        self.half_life = half_life_seconds

    def decay_score(self, score: float, elapsed_seconds: float) -> float:
        if elapsed_seconds < 0.0:
            raise ValueError("elapsed_seconds cannot be negative")
        if elapsed_seconds == 0.0:
            return score
        decay_factor = math.pow(0.5, elapsed_seconds / self.half_life)
        return score * decay_factor

    def decay_batch(
        self, scores: dict[str, float], elapsed_seconds: float
    ) -> dict[str, float]:
        """Apply half-life decay across active score mappings."""
        return {k: self.decay_score(v, elapsed_seconds) for k, v in scores.items()}
