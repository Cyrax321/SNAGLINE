"""Extension point: the AlertSink protocol.

Sinks consume ``FailureRisk`` and escalate it (console, webhook, Slack, a
CONTINUUM ``REQUIRES_REVIEW`` event). They must never receive raw content: the
``FailureRisk`` carries no ``metadata`` field by design (project.md §11).
"""

from __future__ import annotations

from typing import Protocol

from snagline.risk import FailureRisk


class AlertSink(Protocol):
    """Protocol every sink (core or third-party) must satisfy."""

    def emit(self, risk: FailureRisk) -> None:
        """Emit one risk. Must be fire-and-forget and never block ingest()."""
        ...
