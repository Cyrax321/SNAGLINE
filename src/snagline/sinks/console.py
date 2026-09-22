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
        if stream is not None:
            # Fail loudly at construction rather than dropping alerts one at a
            # time for the whole run. A binary stream -- ``open(p, "wb")``,
            # ``sys.stdout.buffer`` -- is an easy misconfiguration to reach for
            # while routing alerts to a file, and the failure is the worst kind
            # once it lands: the monitor runs cleanly and every alert is
            # discarded behind the fail-open contract, with the write raising
            # ``TypeError`` (not an ``OSError`` subclass) so the guard in
            # ``emit`` does not catch it either (issue #391).
            #
            # Probed with an empty write, which writes nothing: the only thing
            # under test is whether the stream accepts a ``str`` at all.
            #
            # Only the type mismatch is a rejection. ``OSError`` and the
            # ``ValueError`` a *closed* stream raises are runtime breakage
            # (closed pipe, bad descriptor, stream closed mid-run), not a
            # misconfiguration, and issue #327's contract -- now upstream --
            # is that such a sink still constructs and stays fire-and-forget
            # in ``emit``, dropping the alert and logging once. Rejecting it
            # here would break that contract for no benefit, since ``emit``
            # already handles it.
            try:
                stream.write("")
            except TypeError as exc:
                raise TypeError(
                    "ConsoleSink stream must be a writable text stream "
                    f"(got {type(stream).__name__}: {exc}); pass "
                    "`open(path, 'w')` for a file, or the logging module "
                    "via logger="
                ) from exc
            except (OSError, ValueError):
                # Falls through to the fire-and-forget path in ``emit``
                # (issues #19 and #327); the probe is only looking for a
                # type mismatch.
                pass
        self._stream = stream if stream is not None else sys.stderr
        self._logger = logger
        self._level = level
        self._fault_logged = False

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
        # A *closed* stream raises ValueError, not OSError -- "I/O operation on
        # closed file" -- so both shapes are caught (issue #327). A stream
        # whose *type* changed underneath us (binary, after construction)
        # raises TypeError, which is not an OSError subclass, so it is caught
        # too (issue #391).
        try:
            self._stream.write(line + "\n")
            self._stream.flush()
            self._fault_logged = False
        except (OSError, ValueError, TypeError):
            if not self._fault_logged:
                self._fault_logged = True
                logger.warning(
                    "snagline ConsoleSink: write to stream failed; dropping "
                    "alert (fire-and-forget); further failures are silent"
                )
