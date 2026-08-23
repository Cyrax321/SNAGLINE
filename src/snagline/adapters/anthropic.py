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
        return getattr(usage, "input_tokens", None), getattr(usage, "output_tokens", None)
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


def _wrap_one(monitor: Any, original: Any, episode_id: str, counter: Any) -> Any:
    is_async = inspect.iscoroutinefunction(original)

    def _sync(*args: Any, **kwargs: Any) -> Any:
        model = kwargs.get("model", "unknown")
        sig_text = str(kwargs.get("messages") or args)
        start = time.time()
        error = False
        error_type = None
        result = None
        try:
            result = original(*args, **kwargs)
        except Exception as e:
            error = True
            error_type = type(e).__name__
            raise
        finally:
            latency = (time.time() - start) * 1000.0
            tokens_in, tokens_out = _extract_tokens(result) if not error else (None, None)
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
        return result

    async def _async(*args: Any, **kwargs: Any) -> Any:
        model = kwargs.get("model", "unknown")
        sig_text = str(kwargs.get("messages") or args)
        start = time.time()
        error = False
        error_type = None
        result = None
        try:
            result = await original(*args, **kwargs)
        except Exception as e:
            error = True
            error_type = type(e).__name__
            raise
        finally:
            latency = (time.time() - start) * 1000.0
            tokens_in, tokens_out = _extract_tokens(result) if not error else (None, None)
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
        return result

    return _async if is_async else _sync


def wrap_anthropic_client(monitor: Any, client: Any, episode_id: str = "anthropic") -> Any:
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


def instrument_anthropic_explicit(monitor: Any, client: Any, episode_id: str = "anthropic") -> Any:
    return wrap_anthropic_client(monitor, client, episode_id=episode_id)
