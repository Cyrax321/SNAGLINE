"""Auto-instrumentation for the Anthropic SDK (ATTACH_ANY_SYSTEM P0).

Mirrors ``snagline.auto.openai``: wraps ``client.messages.create`` so each
call emits a ``StepEvent``. Import-safe (no-op when the SDK is absent) and
handles sync and async clients.

Global mode (issue #270) patches the *resource classes* --
``anthropic.resources.messages.{Messages, AsyncMessages}`` -- rather than
walking ``Anthropic.messages``: on modern Anthropic SDKs the client attribute
is a ``functools.cached_property`` descriptor, so the old class-attribute
walk resolved a descriptor, not a resource, and silently wrapped nothing.
"""

from __future__ import annotations

import inspect
import itertools
import logging
import time

from snagline.events import StepEvent, make_signature

try:  # optional dependency
    from anthropic import Anthropic, AsyncAnthropic  # type: ignore
except Exception:  # pragma: no cover - exercised only without the Anthropic SDK
    Anthropic = None  # type: ignore[assignment, misc]
    AsyncAnthropic = None  # type: ignore[assignment, misc]

logger = logging.getLogger("snagline")


def _emit(monitor, counter, model, tool_name, sig_text, start, error) -> None:
    latency = (time.time() - start) * 1000.0
    event = StepEvent(
        step_id=str(next(counter)),
        episode_id="anthropic-auto",
        timestamp=time.time(),
        action_type="tool_call",
        action_signature=make_signature("anthropic_call", model, sig_text),
        tool_name=tool_name,
        latency_ms=latency,
        error=error,
    )
    monitor.ingest(event)


def _is_async_call(original) -> bool:
    """True if calling ``original`` returns a coroutine that must be awaited.

    ``inspect.iscoroutinefunction`` alone misses async methods that carry a
    decorator (the SDK wraps ``create`` with ``@required_args``), which would
    misclassify them as sync: the wrapper would never await the call, record
    only the coroutine-creation latency, and report ``error=False`` for every
    failure. ``inspect.unwrap`` sees through ``__wrapped__`` chains; on plain
    (including undecorated async) functions it is the identity.
    """
    return inspect.iscoroutinefunction(original) or inspect.iscoroutinefunction(
        inspect.unwrap(original)
    )


def _wrap_one(monitor, original, tool_name):
    counter = itertools.count()
    is_async = _is_async_call(original)

    def _sync(*args, **kwargs):
        sig_text = str(kwargs.get("messages") or args)
        model = kwargs.get("model", "unknown")
        start = time.time()
        error = False
        try:
            result = original(*args, **kwargs)
        except Exception:
            error = True
            raise
        finally:
            _emit(monitor, counter, model, tool_name, sig_text, start, error)
        return result

    async def _async(*args, **kwargs):
        sig_text = str(kwargs.get("messages") or args)
        model = kwargs.get("model", "unknown")
        start = time.time()
        error = False
        try:
            result = await original(*args, **kwargs)
        except Exception:
            error = True
            raise
        finally:
            _emit(monitor, counter, model, tool_name, sig_text, start, error)
        return result

    return _async if is_async else _sync


def wrap_client(monitor, client):
    """Wrap ``client.messages.create`` in place. Returns the same client."""
    cur = getattr(client, "messages", None)
    if cur is None:
        logger.warning(
            "snagline.auto: wrap_client found no messages resource on %r", client
        )
        return client
    method = getattr(cur, "create", None)
    if method is None or not callable(method):
        logger.warning("snagline.auto: wrap_client found no create method on %r", cur)
        return client
    cur.create = _wrap_one(monitor, method, "anthropic.messages.create")
    return client


def _patch_resource_classes(monitor) -> int:
    """Patch ``Messages`` / ``AsyncMessages`` resource classes globally (#270).

    ``anthropic.resources.messages.Messages.create`` is what every
    ``Anthropic`` / ``AsyncAnthropic`` instance dispatches to, so wrapping it
    covers all clients, present and future, without touching the clients'
    ``cached_property`` surface.
    Returns the number of methods newly wrapped. Classes that already carry
    a snagline wrapper are skipped (re-instrumenting would double-count every
    call), so a second call against the same SDK returns -1 (already done).
    """
    patched = 0
    seen_wrapped = False
    try:
        from anthropic.resources.messages import (  # type: ignore
            AsyncMessages,
            Messages,
        )
    except Exception:  # pragma: no cover - defensive against SDK reshuffles
        Messages = AsyncMessages = None

    for cls in (Messages, AsyncMessages):
        if cls is None:
            continue
        original = cls.__dict__.get("create")
        if original is None or not callable(original):
            continue
        if getattr(original, "__snagline_wrapped__", False):
            seen_wrapped = True
            continue  # already instrumented; re-instrumenting would double-count
        wrapper = _wrap_one(monitor, original, "anthropic.messages.create")
        wrapper.__snagline_wrapped__ = True  # type: ignore[attr-defined]
        wrapper.__snagline_original__ = original  # type: ignore[attr-defined]
        cls.create = wrapper
        patched += 1
    # 0 new wraps + at least one already-wrapped class = fully instrumented,
    # not a failure. Distinguish that from "found nothing to patch at all".
    if patched == 0 and seen_wrapped:
        return -1
    return patched


def instrument_anthropic(monitor, client=None) -> bool:
    """Instrument the Anthropic SDK.

    If ``client`` is provided, only that instance is wrapped. Otherwise the
    SDK's resource classes (``Messages`` / ``AsyncMessages``) are patched
    globally, so every client -- present or future -- is observed. Returns
    True if anything was patched, False if the SDK is not importable or
    nothing could be patched.
    """
    if client is not None:
        wrap_client(monitor, client)
        return True
    if Anthropic is None:
        logger.warning("snagline.auto: Anthropic SDK not installed; nothing to patch")
        return False
    patched = _patch_resource_classes(monitor)
    if patched == 0:
        logger.warning(
            "snagline.auto: Anthropic SDK present but no create method could "
            "be patched; clients are NOT monitored"
        )
        return False
    return True
