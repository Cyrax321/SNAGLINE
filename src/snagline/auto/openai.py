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

Latency measurement and event timestamps use :func:`time.perf_counter`, not
:func:`time.time`, mirroring the explicit adapters (issue #155): the wall
clock advances in ~15.6 ms ticks on Windows on supported 3.10--3.12
interpreters, quantizing sub-tick latencies to zero, and is non-monotonic, so
a clock step mid-call fabricates negative or huge latencies.
``perf_counter`` has no meaningful epoch, so these timestamps are only
comparable within one process; detectors consume them solely as in-process
latency differences.
"""

from __future__ import annotations

import contextlib
import inspect
import itertools
import logging
import time
from typing import Any

from snagline.events import StepEvent, make_signature

try:  # optional dependency
    from openai import AsyncOpenAI, OpenAI  # type: ignore
except Exception:  # pragma: no cover - exercised only without the OpenAI SDK
    OpenAI = None  # type: ignore[assignment, misc]
    AsyncOpenAI = None  # type: ignore[assignment, misc]

logger = logging.getLogger("snagline")


def _emit(
    monitor,
    counter,
    model,
    tool_name,
    sig_text,
    start,
    error,
    error_type=None,
    tokens_in=None,
    tokens_out=None,
) -> None:
    now = time.perf_counter()
    event = StepEvent(
        step_id=str(next(counter)),
        episode_id="openai-auto",
        timestamp=now,
        action_type="tool_call",
        action_signature=make_signature("openai_call", model, sig_text),
        tool_name=tool_name,
        latency_ms=(now - start) * 1000.0,
        error=error,
        error_type=error_type,
        tokens_in=tokens_in,
        tokens_out=tokens_out,
    )
    monitor.ingest(event)


def _extract_tokens(result: Any) -> tuple[int | None, int | None]:
    try:
        usage = getattr(result, "usage", None)
        if usage is None and isinstance(result, dict):
            usage = result.get("usage")
        if usage is None:
            return None, None
        if isinstance(usage, dict):
            return usage.get("prompt_tokens"), usage.get("completion_tokens")
        return getattr(usage, "prompt_tokens", None), getattr(
            usage, "completion_tokens", None
        )
    except Exception:
        return None, None


def _is_stream_request(kwargs: dict) -> bool:
    return kwargs.get("stream") is True


def _is_stream_like(obj: Any) -> bool:
    return (
        hasattr(obj, "__iter__")
        or hasattr(obj, "__next__")
        or hasattr(obj, "__aiter__")
        or hasattr(obj, "__anext__")
    )


class _SyncStreamWrapper:
    """Iterate the wrapped stream, emitting one event at exhaustion.

    Streaming calls (``stream=True``) must NOT emit at ``create()`` return
    time -- that would record a ~0ms success before the first chunk arrives
    (issue #242). The event fires once the stream is exhausted, closed, or
    fails mid-flight, mirroring the explicit adapters.
    """

    def __init__(self, monitor, counter, model, tool_name, sig_text, start, stream):
        self._monitor = monitor
        self._counter = counter
        self._model = model
        self._tool_name = tool_name
        self._sig_text = sig_text
        self._start = start
        self._stream = stream
        self._iter = iter(stream)  # type: ignore[call-overload]
        self._last: Any = None
        self._emitted = False

    def __iter__(self):
        return self

    def __next__(self) -> Any:
        if self._emitted:
            raise StopIteration
        try:
            chunk = next(self._iter)
            self._last = chunk
            return chunk
        except StopIteration:
            self._emit(error=False, error_type=None)
            raise
        except Exception as e:
            self._emit(error=True, error_type=type(e).__name__)
            raise

    def close(self) -> None:
        if not self._emitted:
            self._emit(error=False, error_type=None)
        close = getattr(self._stream, "close", None)
        if callable(close):
            with contextlib.suppress(Exception):
                close()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._stream, name)

    def _emit(self, *, error: bool, error_type: str | None) -> None:
        if self._emitted:
            return
        self._emitted = True
        tokens_in, tokens_out = (None, None)
        if not error and self._last is not None:
            tokens_in, tokens_out = _extract_tokens(self._last)
            if tokens_in is None and tokens_out is None:
                tokens_in, tokens_out = _extract_tokens(self._stream)
        _emit(
            self._monitor,
            self._counter,
            self._model,
            self._tool_name,
            self._sig_text,
            self._start,
            error,
            error_type=error_type,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
        )


class _AsyncStreamWrapper:
    """Async twin of :class:`_SyncStreamWrapper`."""

    def __init__(self, monitor, counter, model, tool_name, sig_text, start, stream):
        self._monitor = monitor
        self._counter = counter
        self._model = model
        self._tool_name = tool_name
        self._sig_text = sig_text
        self._start = start
        self._stream = stream
        try:
            self._aiter = stream.__aiter__()  # type: ignore[union-attr]
        except Exception:
            self._aiter = aiter(stream)  # type: ignore[arg-type]
        self._last: Any = None
        self._emitted = False

    def __aiter__(self):
        return self

    async def __anext__(self) -> Any:
        if self._emitted:
            raise StopAsyncIteration
        try:
            chunk = await anext(self._aiter)  # type: ignore[arg-type]
            self._last = chunk
            return chunk
        except StopAsyncIteration:
            self._emit(error=False, error_type=None)
            raise
        except Exception as e:
            self._emit(error=True, error_type=type(e).__name__)
            raise

    async def aclose(self) -> None:
        if not self._emitted:
            self._emit(error=False, error_type=None)
        close = getattr(self._stream, "aclose", None) or getattr(
            self._stream, "close", None
        )
        if callable(close):
            with contextlib.suppress(Exception):
                res = close()
                if inspect.isawaitable(res):
                    await res

    def __getattr__(self, name: str) -> Any:
        return getattr(self._stream, name)

    def _emit(self, *, error: bool, error_type: str | None) -> None:
        if self._emitted:
            return
        self._emitted = True
        tokens_in, tokens_out = (None, None)
        if not error and self._last is not None:
            tokens_in, tokens_out = _extract_tokens(self._last)
            if tokens_in is None and tokens_out is None:
                tokens_in, tokens_out = _extract_tokens(self._stream)
        _emit(
            self._monitor,
            self._counter,
            self._model,
            self._tool_name,
            self._sig_text,
            self._start,
            error,
            error_type=error_type,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
        )


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
        start = time.perf_counter()
        try:
            result = original(*args, **kwargs)
        except Exception as e:
            _emit(
                monitor,
                counter,
                model,
                tool_name,
                sig_text,
                start,
                True,
                error_type=type(e).__name__,
            )
            raise
        if _is_stream_request(kwargs) and _is_stream_like(result):
            # Defer telemetry until exhaustion, close, or iteration error.
            return _SyncStreamWrapper(
                monitor, counter, model, tool_name, sig_text, start, result
            )
        _emit(monitor, counter, model, tool_name, sig_text, start, False)
        return result

    async def _async(*args, **kwargs):
        sig_text = str(kwargs.get("messages") or kwargs.get("prompt") or args)
        model = kwargs.get("model", "unknown")
        start = time.perf_counter()
        try:
            result = await original(*args, **kwargs)
        except Exception as e:
            _emit(
                monitor,
                counter,
                model,
                tool_name,
                sig_text,
                start,
                True,
                error_type=type(e).__name__,
            )
            raise
        if _is_stream_request(kwargs) and _is_stream_like(result):
            return _AsyncStreamWrapper(
                monitor, counter, model, tool_name, sig_text, start, result
            )
        _emit(monitor, counter, model, tool_name, sig_text, start, False)
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
