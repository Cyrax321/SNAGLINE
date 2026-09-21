"""Extension point: the AlertSink protocol, plus shared sink plumbing.

Sinks consume ``FailureRisk`` and escalate it (console, webhook, Slack, a
CONTINUUM ``REQUIRES_REVIEW`` event). They must never receive raw content: the
``FailureRisk`` carries no ``metadata`` field by design (project.md §11).
"""

from __future__ import annotations

import threading
import urllib.request
from typing import Any, Protocol
from urllib.parse import urlsplit

from snagline.risk import FailureRisk


class AlertSink(Protocol):
    """Protocol every sink (core or third-party) must satisfy."""

    def emit(self, risk: FailureRisk) -> None:
        """Emit one risk. Must be fire-and-forget and never block ingest()."""
        ...


def redacted_destination(url: str) -> str:
    """Return the destination URL without the parts that are the credential.

    A Slack incoming-webhook URL embeds its secret as the final path segment
    (``https://hooks.slack.com/services/T.../B.../<secret>``) and arbitrary
    webhook URLs routinely carry basic-auth credentials
    (``https://user:pass@host/``). Both are the credential itself, and a sink
    that cannot reach its destination is exactly the moment an operator goes
    looking in the logs -- so the raw URL must not reach them (issue #390).

    Only the scheme, host and port survive: enough to tell two sinks apart,
    not enough to authenticate as either.
    """
    parts = urlsplit(url)
    if not parts.scheme or not parts.hostname:
        # Not a usable destination at all; report it without echoing it back.
        return "<invalid destination url>"
    netloc = parts.netloc
    if "@" in netloc:
        # Drop the userinfo (``user:pass@``) -- it is the credential.
        netloc = netloc.rsplit("@", 1)[1]
    return f"{parts.scheme}://{netloc}/"


def bounded_post(request: urllib.request.Request, timeout: float) -> None:
    """POST ``request`` and drain the reply, bounded by a wall-clock deadline.

    ``urllib.request.urlopen``'s ``timeout=`` is applied per socket operation,
    and only *after* ``socket.create_connection`` has finished name
    resolution. Two gaps follow, and both leave the caller parked far past the
    budget it asked for: name resolution is not covered at all, so a slow or
    black-holed resolver holds the call for as long as it likes; and a server
    that trickles its body one byte every interval just under the timeout
    never trips a single read timeout. Issue #395 measured a 2.0 s budget
    taking 36.2 s on exactly that trickle, on a request the sink reported as
    successful. The network sinks are dispatched from the ingest path, so the
    stall is the host agent's step while the monitor believes its sink is
    fire-and-forget.

    Bound the whole exchange instead: run it on a short-lived daemon thread
    and give up when the deadline passes. An abandoned in-flight request is
    precisely the failure mode a fire-and-forget sink is built for, and a
    daemon thread dies with the process rather than joining it at shutdown.
    Bounding name resolution in pure stdlib would mean reimplementing the HTTP
    exchange by hand, which is far more new code to get wrong.

    Raises whatever the exchange raised once that is known, or ``TimeoutError``
    if the deadline passed first. Callers catch and log.
    """
    outcome: dict[str, Any] = {}

    def _post() -> None:
        try:
            with urllib.request.urlopen(request, timeout=timeout) as resp:
                resp.read()
        except Exception as exc:  # reported to the caller below
            outcome["error"] = exc

    # A recognisable name so a parked sink thread is identifiable in a
    # py-spy snapshot of a stuck agent, which is how this gets found.
    worker = threading.Thread(target=_post, daemon=True, name="snagline-sink-post")
    worker.start()
    worker.join(timeout)
    if worker.is_alive():
        raise TimeoutError(
            f"snagline sink POST did not finish within {timeout}s; abandoning it"
        )
    if "error" in outcome:
        raise outcome["error"]
