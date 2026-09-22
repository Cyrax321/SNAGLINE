"""Slack sink -- post ``FailureRisk`` alerts to a Slack incoming webhook.

Zero dependency (stdlib ``urllib.request``), mirroring the webhook sink.
Fire-and-forget with a short timeout: ``emit`` never raises and never blocks
``ingest()`` for long -- the ``timeout`` is a wall-clock deadline on the whole
POST (see ``bounded_post``), not just a per-socket-operation hint, and a
redirect is refused rather than followed, so a misrouting endpoint surfaces as
a logged failure instead of a silent success (issue #389). An optional
``min_severity`` filter lets a host route
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
from snagline.sinks.base import (
    bounded_post,
    describe_failure,
    redacted_destination,
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

    def __repr__(self) -> str:
        # The URL is the credential, so the default attribute-dump repr would
        # leak it into any diagnostic dump (issue #390).
        return f"SlackSink({redacted_destination(self._url)!r})"

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
            bounded_post(req, self._timeout)
        except Exception as exc:
            # The URL is the credential -- a Slack incoming webhook embeds its
            # secret as the final path segment -- and a failed POST is the
            # moment an operator goes looking in the logs. PagerDuty already
            # logs no routing key; this matches it (issue #390). The exception
            # is named by class only: a ``URLError`` embeds the URL in its
            # reason for some failures, and a traceback would carry it out
            # with the log line.
            logger.error(
                "snagline Slack sink POST to %s failed (%s); ignoring (fail-open)",
                redacted_destination(self._url),
                describe_failure(exc),
            )
