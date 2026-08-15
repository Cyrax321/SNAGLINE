"""Webhook sink -- POST ``FailureRisk`` JSON to any HTTP endpoint (zero dependency).

Uses only stdlib ``urllib.request`` (project.md §8). Fire-and-forget with a
short timeout: ``emit`` never raises and never blocks ``ingest()`` for long --
the Monitor's fail-open wrapper would swallow a raise anyway, but this sink
keeps its own failure handling so a dead endpoint stays silent even when the
Monitor runs with ``fail_open=False``.

Privacy: only ``FailureRisk`` fields are transmitted (score, trigger, ids,
detail, timestamp) -- never ``StepEvent.metadata`` (project.md §11).
"""

from __future__ import annotations

import json
import logging
import urllib.request
from typing import Any

from snagline.risk import FailureRisk

logger = logging.getLogger("snagline")


class WebhookSink:
    """POSTs each ``FailureRisk`` as JSON to ``url`` and ignores any failure."""

    def __init__(self, url: str, timeout: float = 2.0) -> None:
        self._url = url
        self._timeout = timeout

    def emit(self, risk: FailureRisk) -> None:
        payload: dict[str, Any] = {
            "episode_id": risk.episode_id,
            "step_id": risk.step_id,
            "score": risk.score,
            "trigger": risk.trigger,
            "detail": risk.detail,
            "timestamp": risk.timestamp,
        }
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
                "snagline webhook sink POST to %s failed; ignoring (fail-open)",
                self._url,
            )
