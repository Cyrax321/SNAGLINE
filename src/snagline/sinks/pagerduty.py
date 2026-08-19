"""PagerDuty sink -- trigger PagerDuty Events API v2 incidents.

Zero dependency (stdlib ``urllib.request``). Posts a ``trigger`` event with a
mapped severity so on-call gets paged. Optional ``min_severity`` filter.

Privacy: only ``FailureRisk`` fields are transmitted, never raw content
(project.md §11).
"""

from __future__ import annotations

import json
import logging
import urllib.request
from typing import Any

from snagline.risk import (
    SEVERITY_CRITICAL,
    SEVERITY_INFO,
    SEVERITY_WARNING,
    FailureRisk,
)

logger = logging.getLogger("snagline")

_EVENTS_API = "https://events.pagerduty.com/v2/enqueue"

_SEVERITY_ORDER = {
    SEVERITY_INFO: 0,
    SEVERITY_WARNING: 1,
    SEVERITY_CRITICAL: 2,
}

# PagerDuty's severity vocabulary is a subset of ours.
_PD_SEVERITY = {
    SEVERITY_CRITICAL: "critical",
    SEVERITY_WARNING: "warning",
    SEVERITY_INFO: "info",
}


def _order(severity: str) -> int:
    return _SEVERITY_ORDER.get(severity, 1)


class PagerDutySink:
    """Trigger a PagerDuty incident for each ``FailureRisk``."""

    def __init__(
        self,
        routing_key: str,
        timeout: float = 2.0,
        source: str = "snagline",
        min_severity: str | None = None,
    ) -> None:
        self._key = routing_key
        self._timeout = timeout
        self._source = source
        self._min = min_severity

    def emit(self, risk: FailureRisk) -> None:
        if self._min is not None and _order(risk.severity) < _order(self._min):
            return
        pd_sev = _PD_SEVERITY.get(risk.severity, "warning")
        payload: dict[str, Any] = {
            "routing_key": self._key,
            "event_action": "trigger",
            "payload": {
                "summary": f"[{risk.severity}] {risk.trigger}: {risk.detail}",
                "source": self._source,
                "severity": pd_sev,
            },
            "custom_details": {
                "episode_id": risk.episode_id,
                "step_id": risk.step_id,
                "score": risk.score,
                "trigger": risk.trigger,
            },
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            _EVENTS_API,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                resp.read()
        except Exception:
            logger.exception(
                "snagline PagerDuty sink POST failed; ignoring (fail-open)"
            )
