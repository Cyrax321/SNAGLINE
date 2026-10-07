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
# dead endpoint under sustained load ever walks it up to the ceiling. When it
# is full a POST is dropped rather than queued: holding an alert behind a wall
# of dead ones helps no one, and dropping keeps the caller's ingest step fast.
#
# The pool is *per caller*, not process-global (issue #559). A dead endpoint
# accepts the connection and never replies, so each POST it accepts parks a
# worker until its deadline -- and a worker is what holds the slot. One shared
# pool therefore lets a single dead destination exhaust the delivery budget of
# every other network sink in the process, and the monitor's halt webhook was
# in that same pool: a stalled user-configured escalation endpoint could
# suppress the enforcement directive, and the symptom looked in the logs like a
# healthy sink failing. Scoping the pool to each sink -- and giving the halt
# webhook its own -- means a dead endpoint can only starve itself. The
# thread/FD bound the cap exists to enforce is preserved: the pools are few
# (one per configured sink plus the halt webhook, all operator-chosen) and each
# is bounded, so the total ceiling scales with what was configured rather than
# with the alert rate.
_MAX_INFLIGHT_POSTS = 64

# The fallback pool for callers that do not supply their own -- the CLI's ad-hoc
# forward, and any third-party sink still using the pre-#559 signature. It is
# per-process and therefore has exactly the cross-caller coupling described
# above, but it keeps those callers bounded (issue #423) and unchanged; every
# in-tree network sink and the halt webhook now pass an explicit pool.
_inflight_posts = threading.BoundedSemaphore(_MAX_INFLIGHT_POSTS)


def new_inflight_pool(cap: int | None = None) -> threading.BoundedSemaphore:
    """Build an in-flight POST pool a sink keeps to itself (issue #559).

    A sink calls this once in its constructor and hands the result to every
    ``bounded_post`` call, so its parked workers can only ever consume its own
    delivery budget. ``cap`` defaults to ``_MAX_INFLIGHT_POSTS``; it is read at
    call time so a test (or a host) can resize it by patching the module
    global before the sink is built.
    """
    if cap is None:
        cap = _MAX_INFLIGHT_POSTS
    if cap < 1:
        cap = 1
    return threading.BoundedSemaphore(cap)


# Cap on how many reply-body bytes a network sink will read (issue #560). The
# sinks discard the reply, so ``bounded_post`` used to be called with
# ``max_bytes=None`` and read the whole thing first -- the wall-clock deadline
# bounds *time*, not *memory*, and a fast endless or chunked stream allocates
# without bound well inside a 2 s budget. The URL is operator-supplied, so this
# is a misconfigured or hostile endpoint driving allocation in the monitor's
# own process. 64 KiB matches the halt webhook's ``_MAX_HALT_RESPONSE_BYTES``
# (monitor.py): it is orders above any ack a sane endpoint returns, and the
# body is thrown away either way, so the cap is behaviour-preserving.
_MAX_SINK_RESPONSE_BYTES = 65_536


class SinkBusyError(RuntimeError):
    """Raised when the in-flight sink-POST cap is full, so no POST was started.

    Distinct from ``TimeoutError`` -- which means a POST *ran* and overran its
    deadline -- a ``SinkBusyError`` means the POST was never attempted because
    too many earlier ones are still parked on a stalled endpoint. Callers log
    it fail-open exactly like any other delivery failure; ``describe_failure``
    names it by class, so it carries no destination URL into the log.
    """


def bounded_post(
    request: urllib.request.Request,
    timeout: float,
    max_bytes: int | None = None,
    pool: threading.BoundedSemaphore | None = None,
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
    ``_MAX_SINK_RESPONSE_BYTES`` because they discard the reply anyway; the
    halt webhook passes its own response cap.

    ``pool`` is the in-flight cap this POST draws from, built by
    ``new_inflight_pool`` and owned by one caller. Passing one is what keeps a
    dead endpoint from starving the process's other sinks (issue #559): a sink
    that omits it shares the module fallback ``_inflight_posts`` and is bounded
    against itself rather than against the whole process.

    Raises whatever the exchange raised once that is known, or ``TimeoutError``
    if the deadline passed first. Raises ``SinkBusyError`` without starting the
    POST at all when too many earlier ones are still parked (see
    ``_MAX_INFLIGHT_POSTS``). Callers catch and log.
    """
    inflight = _inflight_posts if pool is None else pool
    if not inflight.acquire(blocking=False):
        # Every slot is occupied by a POST still parked on a stalled endpoint.
        # Refuse this one now rather than adding another abandoned thread; the
        # caller logs it fail-open and its ingest step stays fast. The pool is
        # the caller's own, so this message names no size: a per-sink pool is
        # not necessarily the shipped default, and the number is not actionable
        # from a log line anyway (``describe_failure`` carries the class only).
        raise SinkBusyError(
            "snagline sink POST pool is full; dropping this delivery (fail-open)"
        )
    # Bind the pool we acquired from so the worker releases *that* object, not
    # whatever the caller's reference happens to name when it finally exits: an
    # abandoned worker can outlive any reassignment, and a release aimed at a
    # different semaphore than the acquire would corrupt both counts.
    acquired = inflight
    outcome: dict[str, Any] = {}

    def _post() -> None:
        try:
            with _opener.open(request, timeout=timeout) as resp:
                outcome["body"] = (
                    resp.read(max_bytes) if max_bytes is not None else resp.read()
                )
        except Exception as exc:  # reported to the caller below
            outcome["error"] = exc
        finally:
            # Free the slot the moment this worker is done -- whether it
            # succeeded, failed, or was abandoned by a timed-out caller long
            # ago. Releasing here (not in the caller) is what bounds the parked
            # threads to the cap: an abandoned worker keeps its slot until it
            # actually finishes, so the ceiling counts live threads, not calls.
            acquired.release()

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
    return outcome.get("body", b"")
