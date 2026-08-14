"""Console sink -- the default escalation target (zero dependency).

Writes each ``FailureRisk`` as a single JSON line to stderr. By design it only
ever serializes ``FailureRisk`` fields (ids, score, trigger, detail,
timestamp) -- never ``StepEvent.metadata`` -- so this sink cannot become an
accidental data-exfiltration path (project.md §11).
"""

from __future__ import annotations

import json
import sys
from typing import IO, Any

from snagline.risk import FailureRisk


class ConsoleSink:
    """Emits ``FailureRisk`` events as JSON lines to a stream (stderr by default)."""

    def __init__(self, stream: IO[str] | None = None) -> None:
        self._stream = stream if stream is not None else sys.stderr

    def emit(self, risk: FailureRisk) -> None:
        payload: dict[str, Any] = {
            "episode_id": risk.episode_id,
            "step_id": risk.step_id,
            "score": risk.score,
            "trigger": risk.trigger,
            "detail": risk.detail,
            "timestamp": risk.timestamp,
        }
        self._stream.write(json.dumps(payload) + "\n")
        self._stream.flush()
