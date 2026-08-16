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
from contextlib import suppress
from typing import IO, Any

from snagline.risk import FailureRisk

logger = logging.getLogger("snagline")


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
        stream: IO[str] | None = None,
        logger: logging.Logger | None = None,
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
            # A broken logging pipeline must never break ingest(); stay
            # fire-and-forget like every other AlertSink (issue #19).
            with suppress(Exception):  # pragma: no cover - host logger failure
                self._logger.log(self._level, line)
            return
        # The raw-stream path is fire-and-forget too: a closed pipe or invalid
        # file descriptor must not raise out of emit() and into the host's
        # ingest path, so we swallow write/flush errors and log once (issue #19).
        try:
            self._stream.write(line + "\n")
            self._stream.flush()
        except OSError:
            logger.warning(
                "snagline ConsoleSink: write to stream failed; dropping alert "
                "(fire-and-forget)"
            )
