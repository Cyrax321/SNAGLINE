"""Extension point: the AlertSink protocol, plus shared sink plumbing.

Sinks consume ``FailureRisk`` and escalate it (console, webhook, Slack, a
CONTINUUM ``REQUIRES_REVIEW`` event). They must never receive raw content: the
``FailureRisk`` carries no ``metadata`` field by design (project.md §11).
"""

from __future__ import annotations

import urllib.error
from typing import Protocol
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
    try:
        parts = urlsplit(url)
    except ValueError:
        # An unbalanced bracket in the authority (``https://[::1``) makes
        # ``urlsplit`` itself raise, so the validity check below would never
        # run and the exception would escape through a sink's ``__repr__`` or
        # its fail-open log line -- the two callers this helper exists to keep
        # safe. Report the failure without echoing the input back.
        return "<invalid destination url>"
    if not parts.scheme or not parts.hostname:
        # Not a usable destination at all; report it without echoing it back.
        return "<invalid destination url>"
    netloc = parts.netloc
    if "@" in netloc:
        # Drop the userinfo (``user:pass@``) -- it is the credential.
        netloc = netloc.rsplit("@", 1)[1]
    return f"{parts.scheme}://{netloc}/"


def describe_failure(exc: BaseException) -> str:
    """Name a sink POST failure without repeating the exception's own text.

    ``URLError`` embeds the destination in its reason for some failures
    (``no host given: <url>``, ``unknown url type: <url>``), so a
    ``logger.exception`` call writes the credential into the log via the
    traceback -- including from inside the fail-open ``except`` block, where
    a second failure would be the last thing the operator is told (issue
    #390). The class name carries everything an operator can act on from a
    fire-and-forget sink, and an ``HTTPError``'s status code is an int, so it
    travels too.
    """
    if isinstance(exc, urllib.error.HTTPError):
        return f"HTTP {exc.code}"
    return type(exc).__name__
