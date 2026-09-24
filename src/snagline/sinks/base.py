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


def format_sink_repr(type_name: str, **fields: object) -> str:
    """Build a compact, secret-free ``__repr__`` for a sink (issue #463).

    Sinks previously fell back to ``object.__repr__`` and printed as an opaque
    ``<...object at 0x...>``, which is useless when an operator inspects a
    monitor's sink list. Callers pass only the salient *non-secret* config:
    destination URLs and routing keys are credentials (issues #390/#404) and
    must never reach a repr, which can end up in a log line.
    """
    body = ", ".join(f"{name}={value!r}" for name, value in fields.items())
    return f"{type_name}({body})"
