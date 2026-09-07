"""Auto-instrumentation for the Anthropic SDK (ATTACH_ANY_SYSTEM P0).

Mirrors ``snagline.auto.openai``: wraps ``client.messages.create`` so each
call emits a ``StepEvent``. Import-safe (no-op when the SDK is absent) and
handles sync and async clients.

Streaming calls (``stream=True``) do NOT emit at ``create()`` return time --
that would record a ~0ms success before the first chunk arrives (issue #242).
Instead the returned stream is wrapped, and the event is emitted once the
stream is exhausted, closed, or fails mid-flight, mirroring the explicit
adapters' ``_SyncStreamWrapper``/``_AsyncStreamWrapper``.
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
    from anthropic import Anthropic, AsyncAnthropic  # type: ignore
except Exception:  # pragma: no cover - exercised only without the Anthropic SDK
    Anthropic = None  # type: ignore[assignment, misc]
    AsyncAnthropic = None  # type: ignore[assignment, misc]

logger = logging.getLogger("snagline")

EPISODE_ID = "anthropic-auto"


def _extract_tokens(result: Any) -> tuple[int | None, int | None]:
    try:
        usage = getattr(result, "usage", None)
        if usage is None and isinstance(result, dict):
            usage = result.get("usage")
        if usage is None:
            return None, None
        if isinstance(usage, dict):
            # Anthropic: input_tokens / output_tokens
            return usage.get("input_tokens"), usage.get("output_tokens")
        return getattr(usage, "input_tokens", None), getattr(
            usage, "output_tokens", None
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
    metadata=None,
) -> None:
    latency = (time.time() - start) * 1000.0
    event = StepEvent(
        step_id=str(next(counter)),
        episode_id=EPISODE_ID,
        timestamp=time.time(),
        action_type="tool_call",
        action_signature=make_signature("anthropic_call", model, sig_text),
        tool_name=tool_name,
        latency_ms=latency,
        error=error,
        error_type=error_type,
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        metadata=metadata,
    )
    monitor.ingest(event)


class _SyncStreamWrapper:
    """Iterates the wrapped stream, emitting one event at exhaustion."""

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
            # Anthropic puts usage on the final message_delta chunk; some SDK
            # versions attach it to the stream object instead.
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
            metadata={"adapter": "anthropic-auto", "stream": True},
        )


class _AsyncStreamWrapper:
    """Async counterpart of ``_SyncStreamWrapper``."""

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
            metadata={"adapter": "anthropic-auto", "stream": True},
        )


def _wrap_one(monitor, original, tool_name):
    counter = itertools.count()
    is_async = inspect.iscoroutinefunction(original)

    def _sync(*args, **kwargs):
        sig_text = str(kwargs.get("messages") or args)
        model = kwargs.get("model", "unknown")
        start = time.time()
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
            # Defer telemetry until stream exhaustion, close, or error so a
            # streaming call is not recorded as a ~0ms success at open time.
            return _SyncStreamWrapper(
                monitor, counter, model, tool_name, sig_text, start, result
            )
        _emit(monitor, counter, model, tool_name, sig_text, start, False)
        return result

    async def _async(*args, **kwargs):
        sig_text = str(kwargs.get("messages") or args)
        model = kwargs.get("model", "unknown")
        start = time.time()
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
    """Wrap ``client.messages.create`` in place. Returns the same client."""
    cur = getattr(client, "messages", None)
    if cur is None:
        return client
    method = getattr(cur, "create", None)
    if method is None or not callable(method):
        return client
    cur.create = _wrap_one(monitor, method, "anthropic.messages.create")
    return client


def instrument_anthropic(monitor, client=None) -> bool:
    """Instrument the Anthropic SDK.

    If ``client`` is provided, only that instance is wrapped. Otherwise the
    installed ``anthropic.Anthropic`` / ``anthropic.AsyncAnthropic`` classes
    are patched globally. Returns True if anything was patched, False if the
    SDK is not importable.
    """
    if client is not None:
        wrap_client(monitor, client)
        return True
    if Anthropic is None:
        logger.warning("snagline.auto: Anthropic SDK not installed; nothing to patch")
        return False
    wrap_client(monitor, Anthropic)
    if AsyncAnthropic is not None:
        wrap_client(monitor, AsyncAnthropic)
    return True
