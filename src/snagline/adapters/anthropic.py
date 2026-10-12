"""Anthropic adapter (project.md §6.5) -- explicit wrappers for raw SDK users.

Mirrors ``adapters/openai.py`` for ``anthropic.Anthropic``.

Latency measurement and event timestamps use :func:`time.perf_counter`, not
:func:`time.time`: the wall clock advances in ~15.6 ms ticks on Windows,
quantizing sub-tick latencies to zero (issue #155). ``perf_counter`` has no
meaningful epoch, so these timestamps are only comparable within one process;
detectors consume them solely as in-process latency differences.

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
from typing import Any, Literal

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


def _stream_usage(chunk: Any) -> tuple[int | None, int | None]:
    """Pull (input_tokens, output_tokens) contributions from one stream event.

    Anthropic splits usage across the stream and never repeats it on the final
    event (issue #530, ported from the auto-adapter in #602): ``message_start``
    carries ``input_tokens`` (plus the initial ``output_tokens``) on
    ``event.message.usage``, and ``message_delta`` carries the running
    ``output_tokens`` on ``event.usage``. ``message_stop`` -- the last event,
    i.e. what ``_last`` ends up holding -- and the ``Stream`` object carry no
    usage at all, so probing only those always yielded ``None`` and every
    streaming call reported zero tokens. We read usage metadata only, never
    message content.
    """
    try:
        message = getattr(chunk, "message", None)
        usage = getattr(message, "usage", None) if message is not None else None
        if usage is None:
            usage = getattr(chunk, "usage", None)
        if usage is None and isinstance(chunk, dict):
            inner = chunk.get("message")
            usage = (
                inner.get("usage") if isinstance(inner, dict) else chunk.get("usage")
            )
        if usage is None:
            return None, None
        if isinstance(usage, dict):
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
    side_effect: bool = False,
) -> StepEvent:
    sig_text = str(messages or "")
    sig = make_signature("tool_call", model or "anthropic", sig_text)
    tokens_in, tokens_out = (None, None)
    if result is not None:
        tokens_in, tokens_out = _extract_tokens(result)
    event = StepEvent(
        step_id=step_id or "anthropic",
        episode_id=episode_id,
        timestamp=time.perf_counter(),
        action_type="tool_call",
        action_signature=sig,
        tool_name=model or "anthropic",
        latency_ms=latency_ms,
        error=error,
        error_type=error_type,
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        metadata={"adapter": "anthropic"},
        side_effect=side_effect,
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
        # Usage accumulated across the stream: Anthropic puts input_tokens on
        # message_start and output_tokens on message_delta, and neither _last
        # nor the Stream carries any (issue #602, ported from #530).
        self._tokens_in: int | None = None
        self._tokens_out: int | None = None

    def __iter__(self) -> Any:
        return self

    def __next__(self) -> Any:
        if self._emitted:
            raise StopIteration
        try:
            chunk = next(self._iter)
            self._last = chunk
            self._track_usage(chunk)
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

    def __enter__(self) -> _SyncStreamWrapper:
        return self

    def __exit__(self, *exc_info: object) -> Literal[False]:
        # An exception escaping the body is the observed call's visible
        # outcome; close() alone would record it as a clean success, and
        # failures surfaced through proxied stream methods (``s.text()``)
        # are invisible to __next__. See openai.py for the full rationale.
        if exc_info[1] is not None:
            self._emit(error=True, error_type=type(exc_info[1]).__name__)
        self.close()
        return False

    def __getattr__(self, name: str) -> Any:
        return getattr(self._stream, name)

    def _emit(self, *, error: bool, error_type: str | None) -> None:
        if self._emitted:
            return
        self._emitted = True
        latency = (time.perf_counter() - self._start) * 1000.0
        tokens_in, tokens_out = (None, None)
        if not error:
            # Prefer usage accumulated across the stream (#530 / #602):
            # Anthropic puts input_tokens on message_start and output_tokens on
            # message_delta, and the last event (_last) / the Stream object
            # carry none. Fall back to the last chunk / stream for shapes that
            # do (e.g. a dict or an SDK that stamps usage on the terminal chunk).
            tokens_in, tokens_out = self._tokens_in, self._tokens_out
            if tokens_in is None and tokens_out is None:
                tokens_in, tokens_out = _extract_tokens(self._last)
                if tokens_in is None and tokens_out is None:
                    tokens_in, tokens_out = _extract_tokens(self._stream)
        sig = make_signature("tool_call", str(self._model), self._sig_text)
        event = StepEvent(
            step_id=str(next(self._counter)),
            episode_id=self._episode_id,
            timestamp=time.perf_counter(),
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

    def _track_usage(self, chunk: Any) -> None:
        ti, to = _stream_usage(chunk)
        if ti is not None:
            self._tokens_in = ti
        if to is not None:
            self._tokens_out = to

    # No __del__: a stream dropped without reaching exhaustion, close(), or an
    # iteration error never completed, so it is deliberately not recorded.
    # Emitting from __del__ marked such a stream error=False -- a phantom success
    # that understated the error rate -- and stamped latency_ms as time-until-GC,
    # a garbage sample that poisoned the LatencyAnomalyDetector baseline for the
    # tool. The async wrapper below and the snagline.auto stream wrappers never
    # emitted on abandonment; dropping this __del__ makes every surface agree and
    # keeps the metric to calls that actually completed (issue #531).


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
        # Usage accumulated across the stream, same rationale as the sync
        # wrapper (issue #602).
        self._tokens_in: int | None = None
        self._tokens_out: int | None = None

    def __aiter__(self) -> Any:
        return self

    async def __anext__(self) -> Any:
        if self._emitted:
            raise StopAsyncIteration
        try:
            chunk = await anext(self._aiter)  # type: ignore[arg-type]
            self._last = chunk
            self._track_usage(chunk)
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

    async def __aenter__(self) -> _AsyncStreamWrapper:
        return self

    async def __aexit__(self, *exc_info: object) -> Literal[False]:
        # Same treatment as the sync twin (see openai.py).
        if exc_info[1] is not None:
            self._emit(error=True, error_type=type(exc_info[1]).__name__)
        await self.aclose()
        return False

    def __getattr__(self, name: str) -> Any:
        return getattr(self._stream, name)

    def _emit(self, *, error: bool, error_type: str | None) -> None:
        if self._emitted:
            return
        self._emitted = True
        latency = (time.perf_counter() - self._start) * 1000.0
        tokens_in, tokens_out = (None, None)
        if not error:
            # Prefer usage accumulated across the stream (#530 / #602):
            # Anthropic puts input_tokens on message_start and output_tokens on
            # message_delta, and the last event (_last) / the Stream object
            # carry none. Fall back to the last chunk / stream for shapes that
            # do (e.g. a dict or an SDK that stamps usage on the terminal chunk).
            tokens_in, tokens_out = self._tokens_in, self._tokens_out
            if tokens_in is None and tokens_out is None:
                tokens_in, tokens_out = _extract_tokens(self._last)
                if tokens_in is None and tokens_out is None:
                    tokens_in, tokens_out = _extract_tokens(self._stream)
        sig = make_signature("tool_call", str(self._model), self._sig_text)
        event = StepEvent(
            step_id=str(next(self._counter)),
            episode_id=self._episode_id,
            timestamp=time.perf_counter(),
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

    def _track_usage(self, chunk: Any) -> None:
        ti, to = _stream_usage(chunk)
        if ti is not None:
            self._tokens_in = ti
        if to is not None:
            self._tokens_out = to


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
    latency = (time.perf_counter() - start) * 1000.0
    tokens_in, tokens_out = (None, None)
    if not error:
        tokens_in, tokens_out = _extract_tokens(result)
    sig = make_signature("tool_call", str(model), sig_text)
    event = StepEvent(
        step_id=str(next(counter)),
        episode_id=episode_id,
        timestamp=time.perf_counter(),
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


def _is_async_call(original: Any) -> bool:
    """True if calling ``original`` returns a coroutine that must be awaited.

    ``inspect.iscoroutinefunction`` alone misses async methods that carry a
    decorator: the Anthropic SDK wraps ``create`` with ``@required_args``, whose
    wrapper is a plain sync function that returns the wrapped coroutine. The
    explicit adapter then took the sync branch, never awaited the call, and
    recorded the coroutine-creation latency as a 0 ms success -- with
    ``error=False`` even when the real request failed, because the failure is
    raised by the await, which never ran (issue #588). ``inspect.unwrap`` sees
    through the ``__wrapped__`` chain; on plain (including undecorated async)
    functions it is the identity.
    """
    return inspect.iscoroutinefunction(original) or inspect.iscoroutinefunction(
        inspect.unwrap(original)
    )


def _wrap_one(monitor: Any, original: Any, episode_id: str, counter: Any) -> Any:
    is_async = _is_async_call(original)

    def _sync(*args: Any, **kwargs: Any) -> Any:
        model = kwargs.get("model", "unknown")
        sig_text = str(kwargs.get("messages") or args)
        start = time.perf_counter()
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
        start = time.perf_counter()
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
    if getattr(method, "__snagline_wrapped__", False):
        # Already instrumented by a previous wrap of this instance; wrapping
        # the wrapper would ingest every host call twice.
        return client
    wrapper = _wrap_one(monitor, method, episode_id, counter)
    wrapper.__snagline_wrapped__ = True  # type: ignore[attr-defined]
    cur.create = wrapper
    return client


def instrument_anthropic_explicit(
    monitor: Any, client: Any, episode_id: str = "anthropic"
) -> Any:
    return wrap_anthropic_client(monitor, client, episode_id=episode_id)
