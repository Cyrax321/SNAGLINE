"""OpenAI adapter (project.md §6.5) -- explicit wrappers for raw SDK users.

This is the "raw loop" concrete form: a user with ``openai.OpenAI`` and no
orchestration framework. Unlike ``snagline.auto.openai`` which monkeypatches
the SDK globally, this module provides explicit, per-client wrappers that are
obvious at the call site and never hide control flow.

Usage::

    from snagline.adapters.openai import wrap_openai_client

    client = wrap_openai_client(monitor, openai_client, episode_id="ep-1")
    resp = client.chat.completions.create(model="gpt-4o", messages=[...])

or low-level observation without wrapping::

    from snagline.adapters.openai import observe_openai_call
    observe_openai_call(monitor, episode_id="ep", model="gpt-4o", error=False, latency_ms=120)

The adapter is duck-typed and import-safe: it works without ``openai``
installed and never hard-couples to a specific SDK release. All
``StepEvent`` construction is fail-open and never raises into the host.

Latency measurement and event timestamps use :func:`time.perf_counter`, not
:func:`time.time`: the wall clock advances in ~15.6 ms ticks on Windows,
quantizing sub-tick latencies to zero (issue #155). ``perf_counter`` has no
meaningful epoch, so these timestamps are only comparable within one process;
detectors consume them solely as in-process latency differences.
"""

from __future__ import annotations

import contextlib
import inspect
import itertools
import time
from typing import Any

from snagline.events import StepEvent, make_signature

try:
    from openai import AsyncOpenAI, OpenAI  # type: ignore
except Exception:  # pragma: no cover
    OpenAI = None  # type: ignore
    AsyncOpenAI = None  # type: ignore


def _extract_tokens(result: Any) -> tuple[int | None, int | None]:
    # OpenAI response: result.usage.{prompt_tokens, completion_tokens, total_tokens}
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


def observe_openai_call(
    monitor: Any,
    *,
    episode_id: str,
    model: str | None,
    messages: Any | None = None,
    prompt: Any | None = None,
    latency_ms: float | None = None,
    error: bool = False,
    error_type: str | None = None,
    result: Any | None = None,
    step_id: str | None = None,
    side_effect: bool = False,
) -> StepEvent:
    """Observe a single OpenAI call as a ``StepEvent`` and ingest it.

    ``messages``/``prompt`` are hashed into the signature, not stored.
    If ``result`` is given, tokens are extracted from ``result.usage``.
    Returns the ingested event. ``side_effect=True`` marks the call as a
    non-idempotent action (issue #88); set it only from your own knowledge
    of what the call does, never heuristically.
    """
    sig_text = str(messages or prompt or "")
    # Include model + message shape hash; exclude volatile content already hashed.
    sig = make_signature("tool_call", model or "openai", sig_text)
    tokens_in, tokens_out = (None, None)
    if result is not None:
        tokens_in, tokens_out = _extract_tokens(result)
    event = StepEvent(
        step_id=step_id or "openai",
        episode_id=episode_id,
        timestamp=time.perf_counter(),
        action_type="tool_call",
        action_signature=sig,
        tool_name=model or "openai",
        latency_ms=latency_ms,
        error=error,
        error_type=error_type,
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        metadata={"adapter": "openai"},
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
        latency = (time.perf_counter() - self._start) * 1000.0
        tokens_in, tokens_out = (None, None)
        if not error and self._last is not None:
            tokens_in, tokens_out = _extract_tokens(self._last)
            # OpenAI streaming with include_usage also exposes usage on final chunk
            if tokens_in is None and tokens_out is None:
                # Some SDKs attach usage to the wrapper stream object itself
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
            metadata={"adapter": "openai", "stream": True},
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
        # aiter() is 3.10+
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
        latency = (time.perf_counter() - self._start) * 1000.0
        tokens_in, tokens_out = (None, None)
        if not error and self._last is not None:
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
            metadata={"adapter": "openai", "stream": True},
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
        metadata={"adapter": "openai"},
    )
    with contextlib.suppress(Exception):
        monitor.ingest(event)


def _wrap_one(monitor: Any, original: Any, episode_id: str, counter: Any) -> Any:
    is_async = inspect.iscoroutinefunction(original)

    def _sync(*args: Any, **kwargs: Any) -> Any:
        model = kwargs.get("model", "unknown")
        sig_text = str(kwargs.get("messages") or kwargs.get("prompt") or args)
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
            # Defer telemetry until stream exhaustion, close, or iteration error
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
        sig_text = str(kwargs.get("messages") or kwargs.get("prompt") or args)
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


def wrap_openai_client(monitor: Any, client: Any, episode_id: str = "openai") -> Any:
    """Wrap an ``OpenAI``/``AsyncOpenAI`` client instance explicitly.

    Patches ``client.chat.completions.create`` and ``client.completions.create``
    in place on *this* instance only (no global monkeypatch). Returns the same
    client for chaining. Safe to call even if the SDK is not installed or the
    client shape is mocked.
    """
    counter = itertools.count()
    for path in ("chat.completions.create", "completions.create"):
        cur: Any = client
        ok = True
        parts = path.split(".")
        for part in parts[:-1]:
            cur = getattr(cur, part, None)
            if cur is None:
                ok = False
                break
        if not ok:
            continue
        method = getattr(cur, parts[-1], None)
        if method is None or not callable(method):
            continue
        setattr(cur, parts[-1], _wrap_one(monitor, method, episode_id, counter))
    return client


def instrument_openai_explicit(
    monitor: Any, client: Any, episode_id: str = "openai"
) -> Any:
    """Alias for ``wrap_openai_client`` covering the explicit-wrapper naming."""
    return wrap_openai_client(monitor, client, episode_id=episode_id)
