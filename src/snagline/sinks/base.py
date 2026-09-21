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


class _SameHostRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Follow a 301/302/303 keeping the POST method and its body (issue #389).

    ``urllib``'s default handler rewrites *any* POST that meets one of those
    codes into a bodyless ``GET`` -- it even builds the replacement
    ``Request`` with ``method="GET"`` for 307/308, which the RFCs say must
    preserve the method -- and drops the content headers with the body. For a
    sink that is indistinguishable from success: the redirect target answers
    the ``GET`` with 200, ``emit`` never raises, and the alert was never
    delivered. An endpoint that moved, or one that expects the trailing slash
    the operator left off, silently stops paging.

    So the redirect is re-issued as a ``POST`` carrying the original body.
    Content-Type travels with it and Content-Length is recomputed, which is
    what makes a JSON body still arrive as JSON.

    One restriction, deliberately not lifted: the redirect is only followed
    when it points at the same scheme, host and port. These sinks POST a
    credential -- PagerDuty's ``routing_key`` is the body, a webhook URL can
    be the credential in the userinfo, and a Slack URL is one in the path --
    and a redirect to a different host hands that to whoever answers there.
    That is an exfiltration path, not a delivery path, so an off-host
    redirect raises instead. The sink logs it fail-open and the operator sees
    a nameable thing to fix, which is strictly better than the alert quietly
    vanishing -- which is what it does today, along with the body.
    """

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        if req.get_method() != "POST":
            return super().redirect_request(req, fp, code, msg, headers, newurl)
        newurl = newurl.replace(" ", "%20")
        target = urlsplit(newurl)
        origin = urlsplit(req.full_url)
        if (origin.scheme, origin.hostname, origin.port) != (
            target.scheme,
            target.hostname,
            target.port,
        ):
            # ``HTTPError`` because the caller chain treats it as the request's
            # outcome; the reason is written for the operator, since it is the
            # only text that reaches their log line.
            raise urllib.error.HTTPError(
                req.full_url,
                code,
                "refusing to follow a redirect to a different host "
                f"({redacted_destination(newurl)}); the POST body of these "
                "sinks is a credential, so it is not forwarded off the "
                "endpoint it was configured for",
                headers,
                fp,
            )
        # Content-Length is wrong the moment the body or encoding changes, and
        # urllib recomputes it from ``data``; everything else travels as-is so
        # a JSON body is still announced as JSON.
        kept = {k: v for k, v in req.headers.items() if k.lower() != "content-length"}
        return urllib.request.Request(
            newurl,
            data=req.data,
            headers=kept,
            origin_req_host=req.origin_req_host,
            unverifiable=True,
            method="POST",
        )


# One opener for every network sink: the handler chain is stateless, and
# building it per alert would allocate a fresh ``OpenerDirector`` per POST.
_opener = urllib.request.build_opener(_SameHostRedirectHandler)


def bounded_post(request: urllib.request.Request, timeout: float) -> None:
    """POST ``request`` and drain the reply, bounded by a wall-clock deadline.

    The POST also goes through a redirect handler that keeps the method and
    body across a 301/302/303 on the same host (issue #389) -- urllib's
    default would re-issue it as a bodyless GET, and the target's 200 answer
    would look like a delivered alert.

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
            with _opener.open(request, timeout=timeout) as resp:
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
