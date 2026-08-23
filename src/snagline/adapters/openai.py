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
) -> StepEvent:
    """Observe a single OpenAI call as a ``StepEvent`` and ingest it.

    ``messages``/``prompt`` are hashed into the signature, not stored.
    If ``result`` is given, tokens are extracted from ``result.usage``.
    Returns the ingested event.
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
        timestamp=time.time(),
        action_type="tool_call",
        action_signature=sig,
        tool_name=model or "openai",
        latency_ms=latency_ms,
        error=error,
        error_type=error_type,
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        metadata={"adapter": "openai"},
    )
    with contextlib.suppress(Exception):
        monitor.ingest(event)
    return event


def _wrap_one(monitor: Any, original: Any, episode_id: str, counter: Any) -> Any:
    is_async = inspect.iscoroutinefunction(original)

    def _sync(*args: Any, **kwargs: Any) -> Any:
        model = kwargs.get("model", "unknown")
        sig_text = str(kwargs.get("messages") or kwargs.get("prompt") or args)
        start = time.time()
        error = False
        error_type: str | None = None
        result: Any = None
        try:
            result = original(*args, **kwargs)
        except Exception as e:
            error = True
            error_type = type(e).__name__
            raise
        finally:
            latency = (time.time() - start) * 1000.0
            tokens_in, tokens_out = (
                _extract_tokens(result) if not error else (None, None)
            )
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
                metadata={"adapter": "openai"},
            )
            with contextlib.suppress(Exception):
                monitor.ingest(event)
        return result

    async def _async(*args: Any, **kwargs: Any) -> Any:
        model = kwargs.get("model", "unknown")
        sig_text = str(kwargs.get("messages") or kwargs.get("prompt") or args)
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
            tokens_in, tokens_out = (
                _extract_tokens(result) if not error else (None, None)
            )
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
                metadata={"adapter": "openai"},
            )
            with contextlib.suppress(Exception):
                monitor.ingest(event)
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
