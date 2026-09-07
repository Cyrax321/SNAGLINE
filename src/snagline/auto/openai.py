"""Auto-instrumentation for the OpenAI SDK (ATTACH_ANY_SYSTEM P0).

Wraps chat/completion ``create`` calls so each becomes a ``StepEvent`` fed to
a ``Monitor``, without the caller editing every call site. This is the real
"attach to any system" lever: ``import snagline.auto`` plus one call replaces
per-call instrumentation.

The module is import-safe: it imports fine without the OpenAI SDK installed,
and ``instrument_openai`` only patches the SDK when it is actually present (or
when a client is passed explicitly). Sync and async clients are both handled.

Global mode (issue #270) patches the *resource classes* -- ``Completions`` and
``AsyncCompletions`` -- rather than walking ``OpenAI.chat``: since OpenAI SDK
1.0 the client attributes are ``functools.cached_property`` descriptors, so the
old class-attribute walk resolved a descriptor, not a resource, and silently
wrapped nothing. The resource classes' ``create`` is what every instance call
dispatches to, so patching them covers all clients, current and future.
"""

from __future__ import annotations

import inspect
import itertools
import logging
import time

from snagline.events import StepEvent, make_signature

try:  # optional dependency
    from openai import AsyncOpenAI, OpenAI  # type: ignore
except Exception:  # pragma: no cover - exercised only without the OpenAI SDK
    OpenAI = None  # type: ignore[assignment, misc]
    AsyncOpenAI = None  # type: ignore[assignment, misc]

logger = logging.getLogger("snagline")


def _emit(monitor, counter, model, tool_name, sig_text, start, error) -> None:
    latency = (time.time() - start) * 1000.0
    event = StepEvent(
        step_id=str(next(counter)),
        episode_id="openai-auto",
        timestamp=time.time(),
        action_type="tool_call",
        action_signature=make_signature("openai_call", model, sig_text),
        tool_name=tool_name,
        latency_ms=latency,
        error=error,
    )
    monitor.ingest(event)


def _is_async_call(original) -> bool:
    """True if calling ``original`` returns a coroutine that must be awaited.

    ``inspect.iscoroutinefunction`` alone misses async methods that carry a
    decorator (the OpenAI/Anthropic SDKs wrap ``create`` with
    ``@required_args``), which would misclassify them as sync: the wrapper
    would never await the call, record only the coroutine-creation latency,
    and report ``error=False`` for every failure. ``inspect.unwrap`` sees
    through ``__wrapped__`` chains; on plain (including undecorated async)
    functions it is the identity.
    """
    return inspect.iscoroutinefunction(original) or inspect.iscoroutinefunction(
        inspect.unwrap(original)
    )


def _wrap_one(monitor, original, tool_name):
    counter = itertools.count()
    is_async = _is_async_call(original)

    def _sync(*args, **kwargs):
        sig_text = str(kwargs.get("messages") or kwargs.get("prompt") or args)
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
        sig_text = str(kwargs.get("messages") or kwargs.get("prompt") or args)
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
    """Wrap a given OpenAI client instance's create methods in place.

    Patches ``client.chat.completions.create`` and ``client.completions.create``
    (whichever exist) so each call emits a ``StepEvent``. Returns the same
    client for chaining.
    """
    patched = 0
    for path in ("chat.completions.create", "completions.create"):
        cur = client
        ok = True
        for part in path.split(".")[:-1]:
            cur = getattr(cur, part, None)
            if cur is None:
                ok = False
                break
        if not ok:
            continue
        name = path.split(".")[-1]
        method = getattr(cur, name, None)
        if method is None or not callable(method):
            continue
        setattr(cur, name, _wrap_one(monitor, method, "openai." + path))
        patched += 1
    if patched == 0:
        logger.warning(
            "snagline.auto: wrap_client found no create method to patch on %r",
            client,
        )
    return client


def _patch_resource_classes(monitor) -> int:
    """Patch the SDK's resource classes globally (issue #270).

    ``openai.resources.chat.completions.completions.{Completions,
    AsyncCompletions}`` and the legacy ``completions`` pair are the classes
    every ``OpenAI`` / ``AsyncOpenAI`` instance dispatches ``create`` to, so
    wrapping their methods covers all current and future clients without
    touching the clients' ``cached_property`` surface.

    Returns the number of methods newly wrapped. Classes that already carry
    a snagline wrapper are skipped (re-instrumenting would double-count every
    call), so a second call against the same SDK returns 0.
    """
    patched = 0
    seen_wrapped = False
    try:
        from openai.resources.chat.completions.completions import (  # type: ignore
            AsyncCompletions as ChatAsync,
        )
        from openai.resources.chat.completions.completions import (
            Completions as ChatSync,
        )
    except Exception:  # pragma: no cover - defensive against SDK reshuffles
        ChatSync = ChatAsync = None
    try:
        from openai.resources.completions import (  # type: ignore
            AsyncCompletions as LegacyAsync,
        )
        from openai.resources.completions import (
            Completions as LegacySync,
        )
    except Exception:  # pragma: no cover - defensive against SDK reshuffles
        LegacySync = LegacyAsync = None

    for cls, tool_name in [
        (ChatSync, "openai.chat.completions.create"),
        (ChatAsync, "openai.chat.completions.create"),
        (LegacySync, "openai.completions.create"),
        (LegacyAsync, "openai.completions.create"),
    ]:
        if cls is None:
            continue
        original = cls.__dict__.get("create")
        if original is None or not callable(original):
            continue
        if getattr(original, "__snagline_wrapped__", False):
            seen_wrapped = True
            continue  # already instrumented; re-instrumenting would double-count
        wrapper = _wrap_one(monitor, original, tool_name)
        wrapper.__snagline_wrapped__ = True  # type: ignore[attr-defined]
        wrapper.__snagline_original__ = original  # type: ignore[attr-defined]
        cls.create = wrapper
        patched += 1
    # 0 new wraps + at least one already-wrapped class = fully instrumented,
    # not a failure. Distinguish that from "found nothing to patch at all".
    if patched == 0 and seen_wrapped:
        return -1
    return patched


def instrument_openai(monitor, client=None) -> bool:
    """Instrument the OpenAI SDK.

    If ``client`` is provided, only that instance is wrapped. Otherwise the
    SDK's resource classes (``Completions`` / ``AsyncCompletions`` for both
    chat and legacy completions) are patched globally, so every client --
    present or future -- is observed. Returns True if anything was patched,
    False if the SDK is not importable or nothing could be patched.
    """
    if client is not None:
        wrap_client(monitor, client)
        return True
    if OpenAI is None:
        logger.warning("snagline.auto: OpenAI SDK not installed; nothing to patch")
        return False
    patched = _patch_resource_classes(monitor)
    if patched == 0:
        logger.warning(
            "snagline.auto: OpenAI SDK present but no create method could be "
            "patched; clients are NOT monitored"
        )
        return False
    return True
