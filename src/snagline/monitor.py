"""The Monitor orchestrator -- fail-open by construction.

This is the heart of the adoption guarantee (project.md §1.2 / §4.6): a
monitoring library that can crash or stall the thing it monitors is a
non-starter. Therefore every ``observe`` and every ``emit`` call is wrapped so
that, by default (``fail_open=True``), any exception is logged and swallowed
rather than propagated. Only when ``fail_open=False`` is explicitly set do
exceptions propagate -- useful for tests and for callers who want strict
behavior.

The lock is per-instance, so a single Monitor can be shared across multiple
episodes / threads concurrently.
"""

from __future__ import annotations

import logging
import threading
from typing import List

from snagline.detectors.base import Detector
from snagline.events import StepEvent
from snagline.risk import FailureRisk
from snagline.sinks.base import AlertSink

logger = logging.getLogger("snagline")


class Monitor:
    """Runs every registered detector against each ingested event and
    dispatches any resulting ``FailureRisk`` to every registered sink.

    Fail-open: detector/sink exceptions are caught and logged, never
    propagated into the host agent, unless ``fail_open=False``.
    """

    def __init__(
        self,
        detectors: List[Detector],
        sinks: List[AlertSink],
        fail_open: bool = True,
    ) -> None:
        self._detectors = list(detectors)
        self._sinks = list(sinks)
        self._fail_open = fail_open
        self._lock = threading.Lock()

    def ingest(self, event: StepEvent) -> None:
        """Run all detectors against ``event`` and dispatch any risks.

        Never raises under the default ``fail_open=True``. With
        ``fail_open=False``, a detector or sink exception propagates.
        """
        with self._lock:
            for detector in self._detectors:
                try:
                    risk = detector.observe(event)
                except Exception:
                    logger.exception(
                        "snagline detector %s raised; ignoring (fail-open)",
                        getattr(detector, "name", repr(detector)),
                    )
                    if not self._fail_open:
                        raise
                    continue
                if risk is not None:
                    self._dispatch(risk)

    def _dispatch(self, risk: FailureRisk) -> None:
        for sink in self._sinks:
            try:
                sink.emit(risk)
            except Exception:
                logger.exception(
                    "snagline sink %s raised; ignoring (fail-open)",
                    type(sink).__name__,
                )
                if not self._fail_open:
                    raise

    async def ingest_async(self, event: StepEvent) -> None:
        """Thin async wrapper for async adapters (LangGraph/AutoGen async
        mode). Runs the same cheap sync path -- detectors are cheap enough
        that this is safe.
        """
        self.ingest(event)
