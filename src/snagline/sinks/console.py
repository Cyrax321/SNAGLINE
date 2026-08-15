"""Console sink -- the default escalation target (zero dependency).

Writes each ``FailureRisk`` as a single JSON line. By default it goes to
stderr via a raw stream, but it can also be routed through the ``logging``
module (``logger=``) or to any text ``stream`` (stdout, a file, etc.) -- so
operators can fold SNAGLINE alerts into their existing log pipeline (issue #13).

By design it only ever serializes ``FailureRisk`` fields (ids, score, trigger,
detail, timestamp) -- never ``StepEvent.metadata`` -- so this sink cannot
become an accidental data-exfiltration path (project.md §11).
"""

from __future__ import annotations

import json
import logging
import sys
from typing import IO, Any, Optional

from snagline.risk import FailureRisk


class ConsoleSink:
    """Emits ``FailureRisk`` events as JSON lines.

    Target precedence:
      * ``logger=``  -> route through the given ``logging.Logger`` at ``level``
        (default WARNING). This is the recommended integration point for
        production deployments that already collect Python logs.
      * ``stream=``  -> write a raw JSON line to the given ``IO[str]`` (default
        ``sys.stderr``), as before.
    """

    def __init__(
        self,
        stream: Optional[IO[str]] = None,
        logger: Optional[logging.Logger] = None,
        level: int = logging.WARNING,
    ) -> None:
        self._stream = stream if stream is not None else sys.stderr
        self._logger = logger
        self._level = level

    def emit(self, risk: FailureRisk) -> None:
        payload: dict[str, Any] = {
            "episode_id": risk.episode_id,
            "step_id": risk.step_id,
            "score": risk.score,
            "trigger": risk.trigger,
            "detail": risk.detail,
            "timestamp": risk.timestamp,
        }
        line = json.dumps(payload)
        if self._logger is not None:
            self._logger.log(self._level, line)
        else:
            self._stream.write(line + "\n")
            self._stream.flush()
