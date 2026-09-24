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


# Ceiling on sink POSTs in flight across the whole process at once.
#
# ``bounded_post`` gives up on a POST at its deadline but cannot cancel it: the
# worker thread -- and the socket it holds -- lives on until the exchange
# resolves on its own. Against an endpoint that is merely slow that is fine
# (the thread drains and exits shortly after), but against one that is *dead* --
# a black hole that neither answers nor resets -- every alert spawns a worker
# that never returns, and a monitor under load would accumulate one parked
# thread (and file descriptor) per emit without bound (issue #423).
#
# Cap the number that may be parked at once. The cap is deliberately generous:
# a healthy POST finishes well inside its short timeout and frees its slot at
# once, a transient resolver hiccup parks only a handful, and only a genuinely
# dead endpoint under sustained load ever walks it up to the ceiling. It is a
# single process-global guess -- it cannot know how many monitors or sinks
# share the interpreter -- so it is set high enough to bound memory and file
# descriptors without rejecting a realistic burst, and no higher. When it is
# full a POST is dropped rather than queued: holding an alert behind a wall of
# dead ones helps no one, and dropping keeps the caller's ingest step fast.
_MAX_INFLIGHT_POSTS = 64
_inflight_posts = threading.BoundedSemaphore(_MAX_INFLIGHT_POSTS)


class SinkBusyError(RuntimeError):
    """Raised when the in-flight sink-POST cap is full, so no POST was started.

    Distinct from ``TimeoutError`` -- which means a POST *ran* and overran its
    deadline -- a ``SinkBusyError`` means the POST was never attempted because
    too many earlier ones are still parked on a stalled endpoint. Callers log
    it fail-open exactly like any other delivery failure; ``describe_failure``
    names it by class, so it carries no destination URL into the log.
    """


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
    if the deadline passed first. Raises ``SinkBusyError`` without starting the
    POST at all when too many earlier ones are still parked (see
    ``_MAX_INFLIGHT_POSTS``). Callers catch and log.
    """
    if not _inflight_posts.acquire(blocking=False):
        # Every slot is occupied by a POST still parked on a stalled endpoint.
        # Refuse this one now rather than adding another abandoned thread; the
        # caller logs it fail-open and its ingest step stays fast.
        raise SinkBusyError(
            f"snagline sink POST pool is full ({_MAX_INFLIGHT_POSTS} in flight); "
            "dropping this delivery (fail-open)"
        )
    # Bind the pool we acquired from so the worker releases *that* object, not
    # whatever the module global happens to name when it finally exits: an
    # abandoned worker can outlive any reassignment of the global, and a
    # release aimed at a different semaphore than the acquire would corrupt
    # both counts.
    pool = _inflight_posts
    outcome: dict[str, Any] = {}

    def _post() -> None:
        try:
            with urllib.request.urlopen(request, timeout=timeout) as resp:
                resp.read()
        except Exception as exc:  # reported to the caller below
            outcome["error"] = exc
        finally:
            # Free the slot the moment this worker is done -- whether it
            # succeeded, failed, or was abandoned by a timed-out caller long
            # ago. Releasing here (not in the caller) is what bounds the parked
            # threads to the cap: an abandoned worker keeps its slot until it
            # actually finishes, so the ceiling counts live threads, not calls.
            pool.release()

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
