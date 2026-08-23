"""Anthropic adapter (project.md §6.5) -- explicit wrappers for raw SDK users.

Mirrors ``adapters/openai.py`` for ``anthropic.Anthropic``.

Usage::

    from snagline.adapters.anthropic import wrap_anthropic_client

    client = wrap_anthropic_client(monitor, anthropic_client, episode_id="ep-1")
    resp = client.messages.create(model="claude-3-5-sonnet-20241022", messages=[...])
"""

from __future__ import annotations

import contextlib
import inspect
import itertools
import time
from typing import Any

from snagline.events import StepEvent, make_signature

try:
    from anthropic import Anthropic, AsyncAnthropic  # type: ignore
except Exception:  # pragma: no cover
    Anthropic = None  # type: ignore
    AsyncAnthropic = None  # type: ignore


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


def observe_anthropic_call(
    monitor: Any,
    *,
    episode_id: str,
    model: str | None,
    messages: Any | None = None,
    latency_ms: float | None = None,
    error: bool = False,
    error_type: str | None = None,
    result: Any | None = None,
    step_id: str | None = None,
) -> StepEvent:
    sig_text = str(messages or "")
    sig = make_signature("tool_call", model or "anthropic", sig_text)
    tokens_in, tokens_out = (None, None)
    if result is not None:
        tokens_in, tokens_out = _extract_tokens(result)
    event = StepEvent(
        step_id=step_id or "anthropic",
        episode_id=episode_id,
        timestamp=time.time(),
        action_type="tool_call",
        action_signature=sig,
        tool_name=model or "anthropic",
        latency_ms=latency_ms,
        error=error,
        error_type=error_type,
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        metadata={"adapter": "anthropic"},
    )
    with contextlib.suppress(Exception):
        monitor.ingest(event)
    return event


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
    def __init__(
        self,
        monitor: Any,
        counter: Any,
        episode_id: str,
        model: Any,
        sig_text: str,
        start: float,
        stream: Any,
    ) -> None:
        self._monitor = monitor
        self._counter = counter
        self._episode_id = episode_id
        self._model = model
        self._sig_text = sig_text
        self._start = start
        self._stream = stream
        self._iter = iter(stream)  # type: ignore[call-overload]
        self._last: Any = None
        self._emitted = False

    def __iter__(self) -> Any:
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
        latency = (time.time() - self._start) * 1000.0
        tokens_in, tokens_out = (None, None)
        if not error and self._last is not None:
            tokens_in, tokens_out = _extract_tokens(self._last)
            if tokens_in is None and tokens_out is None:
                tokens_in, tokens_out = _extract_tokens(self._stream)
        sig = make_signature("tool_call", str(self._model), self._sig_text)
        event = StepEvent(
            step_id=str(next(self._counter)),
            episode_id=self._episode_id,
            timestamp=time.time(),
            action_type="tool_call",
            action_signature=sig,
            tool_name=str(self._model),
            latency_ms=latency,
            error=error,
            error_type=error_type,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            metadata={"adapter": "anthropic", "stream": True},
        )
        with contextlib.suppress(Exception):
            self._monitor.ingest(event)

    def __del__(self) -> None:
        if not self._emitted:
            with contextlib.suppress(Exception):
                self._emit(error=False, error_type=None)


class _AsyncStreamWrapper:
    def __init__(
        self,
        monitor: Any,
        counter: Any,
        episode_id: str,
        model: Any,
        sig_text: str,
        start: float,
        stream: Any,
    ) -> None:
        self._monitor = monitor
        self._counter = counter
        self._episode_id = episode_id
        self._model = model
        self._sig_text = sig_text
        self._start = start
        self._stream = stream
        try:
            self._aiter = stream.__aiter__()  # type: ignore[union-attr]
        except Exception:
            self._aiter = aiter(stream)  # type: ignore[arg-type]
        self._last: Any = None
        self._emitted = False

    def __aiter__(self) -> Any:
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
        latency = (time.time() - self._start) * 1000.0
        tokens_in, tokens_out = (None, None)
        if not error and self._last is not None:
            tokens_in, tokens_out = _extract_tokens(self._last)
            if tokens_in is None and tokens_out is None:
                tokens_in, tokens_out = _extract_tokens(self._stream)
        sig = make_signature("tool_call", str(self._model), self._sig_text)
        event = StepEvent(
            step_id=str(next(self._counter)),
            episode_id=self._episode_id,
            timestamp=time.time(),
            action_type="tool_call",
            action_signature=sig,
            tool_name=str(self._model),
            latency_ms=latency,
            error=error,
            error_type=error_type,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            metadata={"adapter": "anthropic", "stream": True},
        )
        with contextlib.suppress(Exception):
            self._monitor.ingest(event)


def _emit_now(
    monitor: Any,
    counter: Any,
    episode_id: str,
    model: Any,
    sig_text: str,
    start: float,
    *,
    error: bool,
    error_type: str | None,
    result: Any,
) -> None:
    latency = (time.time() - start) * 1000.0
    tokens_in, tokens_out = (None, None)
    if not error:
        tokens_in, tokens_out = _extract_tokens(result)
    sig = make_signature("tool_call", str(model), sig_text)
    event = StepEvent(
        step_id=str(next(counter)),
        episode_id=episode_id,
        timestamp=time.time(),
        action_type="tool_call",
        action_signature=sig,
        tool_name=str(model),
        latency_ms=latency,
        error=error,
        error_type=error_type,
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        metadata={"adapter": "anthropic"},
    )
    with contextlib.suppress(Exception):
        monitor.ingest(event)


def _wrap_one(monitor: Any, original: Any, episode_id: str, counter: Any) -> Any:
    is_async = inspect.iscoroutinefunction(original)

    def _sync(*args: Any, **kwargs: Any) -> Any:
        model = kwargs.get("model", "unknown")
        sig_text = str(kwargs.get("messages") or args)
        start = time.time()
        try:
            result = original(*args, **kwargs)
        except Exception as e:
            _emit_now(
                monitor,
                counter,
                episode_id,
                model,
                sig_text,
                start,
                error=True,
                error_type=type(e).__name__,
                result=None,
            )
            raise
        if _is_stream_request(kwargs) and _is_stream_like(result):
            return _SyncStreamWrapper(
                monitor, counter, episode_id, model, sig_text, start, result
            )
        _emit_now(
            monitor,
            counter,
            episode_id,
            model,
            sig_text,
            start,
            error=False,
            error_type=None,
            result=result,
        )
        return result

    async def _async(*args: Any, **kwargs: Any) -> Any:
        model = kwargs.get("model", "unknown")
        sig_text = str(kwargs.get("messages") or args)
        start = time.time()
        try:
            result = await original(*args, **kwargs)
        except Exception as e:
            _emit_now(
                monitor,
                counter,
                episode_id,
                model,
                sig_text,
                start,
                error=True,
                error_type=type(e).__name__,
                result=None,
            )
            raise
        if _is_stream_request(kwargs) and _is_stream_like(result):
            return _AsyncStreamWrapper(
                monitor, counter, episode_id, model, sig_text, start, result
            )
        _emit_now(
            monitor,
            counter,
            episode_id,
            model,
            sig_text,
            start,
            error=False,
            error_type=None,
            result=result,
        )
        return result

    return _async if is_async else _sync


def wrap_anthropic_client(
    monitor: Any, client: Any, episode_id: str = "anthropic"
) -> Any:
    """Wrap an ``Anthropic``/``AsyncAnthropic`` client instance explicitly.

    Patches ``client.messages.create`` in place on this instance only.
    Returns the same client for chaining.
    """
    counter = itertools.count()
    cur = getattr(client, "messages", None)
    if cur is None:
        return client
    method = getattr(cur, "create", None)
    if method is None or not callable(method):
        return client
    cur.create = _wrap_one(monitor, method, episode_id, counter)
    return client


def instrument_anthropic_explicit(
    monitor: Any, client: Any, episode_id: str = "anthropic"
) -> Any:
    return wrap_anthropic_client(monitor, client, episode_id=episode_id)
