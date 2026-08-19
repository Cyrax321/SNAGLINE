"""Slack sink -- post ``FailureRisk`` alerts to a Slack incoming webhook.

Zero dependency (stdlib ``urllib.request``), mirroring the webhook sink.
Fire-and-forget with a short timeout: ``emit`` never raises and never blocks
``ingest()`` for long. An optional ``min_severity`` filter lets a host route
only warnings/criticals to Slack while still sending everything elsewhere.

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

_SEVERITY_ORDER = {
    SEVERITY_INFO: 0,
    SEVERITY_WARNING: 1,
    SEVERITY_CRITICAL: 2,
}


def _order(severity: str) -> int:
    return _SEVERITY_ORDER.get(severity, 1)


class SlackSink:
    """POSTs each ``FailureRisk`` as a Slack message to ``webhook_url``."""

    def __init__(
        self,
        webhook_url: str,
        timeout: float = 2.0,
        min_severity: str | None = None,
    ) -> None:
        self._url = webhook_url
        self._timeout = timeout
        self._min = min_severity

    def emit(self, risk: FailureRisk) -> None:
        if self._min is not None and _order(risk.severity) < _order(self._min):
            return
        text = (
            f"[{risk.severity.upper()}] SNAGLINE failure detected\n"
            f"Trigger: {risk.trigger}\n"
            f"Episode: {risk.episode_id} (step {risk.step_id})\n"
            f"Score: {risk.score:.2f}\n"
            f"{risk.detail}"
        )
        payload: dict[str, Any] = {"text": text}
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            self._url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                resp.read()
        except Exception:
            logger.exception(
                "snagline Slack sink POST to %s failed; ignoring (fail-open)",
                self._url,
            )
