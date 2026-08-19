"""Auto-instrumentation for the OpenAI SDK (ATTACH_ANY_SYSTEM P0).

Wraps chat/completion ``create`` calls so each becomes a ``StepEvent`` fed to
a ``Monitor``, without the caller editing every call site. This is the real
"attach to any system" lever: ``import snagline.auto`` plus one call replaces
per-call instrumentation.

The module is import-safe: it imports fine without the OpenAI SDK installed,
and ``instrument_openai`` only patches the SDK when it is actually present (or
when a client is passed explicitly). Sync and async clients are both handled.
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


def _wrap_one(monitor, original, tool_name):
    counter = itertools.count()
    is_async = inspect.iscoroutinefunction(original)

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
        method = getattr(cur, path.split(".")[-1], None)
        if method is None or not callable(method):
            continue
        setattr(cur, path.split(".")[-1], _wrap_one(monitor, method, "openai." + path))
    return client


def instrument_openai(monitor, client=None) -> bool:
    """Instrument the OpenAI SDK.

    If ``client`` is provided, only that instance is wrapped. Otherwise the
    installed ``openai.OpenAI`` / ``openai.AsyncOpenAI`` classes are patched
    globally (so every future client is observed). Returns True if anything
    was patched, False if the SDK is not importable.
    """
    if client is not None:
        wrap_client(monitor, client)
        return True
    if OpenAI is None:
        logger.warning("snagline.auto: OpenAI SDK not installed; nothing to patch")
        return False
    wrap_client(monitor, OpenAI)
    if AsyncOpenAI is not None:
        wrap_client(monitor, AsyncOpenAI)
    return True
