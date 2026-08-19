"""Auto-instrumentation for the Anthropic SDK (ATTACH_ANY_SYSTEM P0).

Mirrors ``snagline.auto.openai``: wraps ``client.messages.create`` so each
call emits a ``StepEvent``. Import-safe (no-op when the SDK is absent) and
handles sync and async clients.
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


def _wrap_one(monitor, original, tool_name):
    counter = itertools.count()
    is_async = inspect.iscoroutinefunction(original)

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
