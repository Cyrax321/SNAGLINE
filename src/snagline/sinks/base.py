"""Extension point: the AlertSink protocol, plus shared sink plumbing.

Sinks consume ``FailureRisk`` and escalate it (console, webhook, Slack, a
CONTINUUM ``REQUIRES_REVIEW`` event). They must never receive raw content: the
``FailureRisk`` carries no ``metadata`` field by design (project.md §11).
"""

from __future__ import annotations

import threading
import urllib.error
import urllib.request
from typing import Any, Protocol
from urllib.parse import urlsplit

from snagline.risk import FailureRisk


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """Refuse to follow a 3xx, so a redirect cannot silently drop the payload.

    urllib honours a ``301``/``302``/``303`` by re-issuing the request to the
    ``Location`` URL **as a GET with no body** (see
    ``HTTPRedirectHandler.redirect_request``), and returns the final 2xx to the
    caller. A sink that hits a redirect therefore reports a successful delivery
    while its payload travelled with the POST the server rejected, and went to a
    destination the server chose. A trailing-slash hop, an HTTP->HTTPS or proxy
    canonicalisation, or -- worst -- an auth redirect from an expired
    credential, all land here, and the last one turns a visible 401 into a
    silent misrouting (issue #389).

    Returning ``None`` makes ``http_error_30x`` give up, which falls through to
    ``HTTPDefaultErrorHandler`` and raises ``HTTPError`` for the 3xx; the sinks
    already log that fail-open. A ``307``/``308`` raises ``HTTPError`` already,
    because urllib preserves the method for those and refuses to rewrite a POST
    -- so this handler also makes the failure mode consistent across redirect
    codes instead of silent for exactly the common ones.
    """

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


# One shared opener: handlers carry no per-request state, so this is safe across
# the sink threads. Built rather than reusing the default opener so the policy
# is ours and not whatever the host process installed globally; proxies are
# still read from the environment by ``build_opener`` itself.
_opener = urllib.request.build_opener(_NoRedirect)


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


def bounded_post(
    request: urllib.request.Request,
    timeout: float,
    max_bytes: int | None = None,
) -> bytes:
    """POST ``request`` and return the reply body, bounded by a wall-clock deadline.

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

    Redirects are refused rather than followed, for the same reason: a followed
    3xx is reported as a 2xx delivery whose payload never arrived (see
    ``_NoRedirect``). For the halt webhook this is sharper still -- the reply
    is parsed into an enforcement directive, so a followed redirect would let
    the decision come from a server the operator never configured (issue #416).

    ``max_bytes`` caps the read, so a malicious or broken endpoint cannot make
    the exchange unbounded by streaming an endless body. The sinks pass
    ``None`` because they discard the reply anyway; the halt webhook passes
    its existing response cap.

    Raises whatever the exchange raised once that is known, or ``TimeoutError``
    if the deadline passed first. Callers catch and log.
    """
    outcome: dict[str, Any] = {}

    def _post() -> None:
        try:
            with _opener.open(request, timeout=timeout) as resp:
                outcome["body"] = (
                    resp.read(max_bytes) if max_bytes is not None else resp.read()
                )
        except Exception as exc:  # reported to the caller below
            outcome["error"] = exc

    # A recognisable name so a parked poster is identifiable in a py-spy
    # snapshot of a stuck agent, which is how this gets found.
    worker = threading.Thread(target=_post, daemon=True, name="snagline-bounded-post")
    worker.start()
    worker.join(timeout)
    if worker.is_alive():
        raise TimeoutError(
            f"snagline sink POST did not finish within {timeout}s; abandoning it"
        )
    if "error" in outcome:
        raise outcome["error"]
    return outcome.get("body", b"")
