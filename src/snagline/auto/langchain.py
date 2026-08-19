"""Auto-instrumentation for LangChain (ATTACH_ANY_SYSTEM P0).

Wraps the common ``invoke`` / ``generate`` entrypoints on a LangChain model
or chain so each call emits a ``StepEvent``. Import-safe: a no-op when
LangChain is absent, and handles synchronous and asynchronous methods.
"""

from __future__ import annotations

import inspect
import itertools
import logging
import time

from snagline.events import StepEvent, make_signature

logger = logging.getLogger("snagline")

# Entrypoints we patch on a model or chain instance.
_LANGCHAIN_METHODS = ("invoke", "generate", "ainvoke", "agenerate")


def _emit(monitor, counter, model, tool_name, sig_text, start, error) -> None:
    latency = (time.time() - start) * 1000.0
    event = StepEvent(
        step_id=str(next(counter)),
        episode_id="langchain-auto",
        timestamp=time.time(),
        action_type="tool_call",
        action_signature=make_signature("langchain_call", model, sig_text),
        tool_name=tool_name,
        latency_ms=latency,
        error=error,
    )
    monitor.ingest(event)


def _wrap_one(monitor, original, tool_name):
    counter = itertools.count()
    is_async = inspect.iscoroutinefunction(original)

    def _sync(*args, **kwargs):
        sig_text = str(kwargs.get("input") or kwargs.get("prompts") or args)
        model = getattr(original, "__self__", None)
        model_name = getattr(model, "model_name", None) or getattr(
            model, "model", "langchain"
        )
        start = time.time()
        error = False
        try:
            result = original(*args, **kwargs)
        except Exception:
            error = True
            raise
        finally:
            _emit(monitor, counter, model_name, tool_name, sig_text, start, error)
        return result

    async def _async(*args, **kwargs):
        sig_text = str(kwargs.get("input") or kwargs.get("prompts") or args)
        model = getattr(original, "__self__", None)
        model_name = getattr(model, "model_name", None) or getattr(
            model, "model", "langchain"
        )
        start = time.time()
        error = False
        try:
            result = await original(*args, **kwargs)
        except Exception:
            error = True
            raise
        finally:
            _emit(monitor, counter, model_name, tool_name, sig_text, start, error)
        return result

    return _async if is_async else _sync


def wrap_client(monitor, client):
    """Patch the invoke/generate entrypoints on ``client`` in place."""
    for name in _LANGCHAIN_METHODS:
        method = getattr(client, name, None)
        if method is None or not callable(method):
            continue
        setattr(client, name, _wrap_one(monitor, method, "langchain." + name))
    return client


def instrument_langchain(monitor, client=None) -> bool:
    """Instrument LangChain.

    If ``client`` is provided, only that instance is wrapped. Otherwise the
    installed ``langchain`` base classes are patched globally (best effort).
    Returns True if anything was patched, False if LangChain is absent.
    """
    if client is not None:
        wrap_client(monitor, client)
        return True
    try:  # pragma: no cover - exercised only with LangChain installed
        from langchain.chains.base import Chain  # type: ignore
        from langchain_core.language_models import BaseLanguageModel  # type: ignore
    except Exception:
        logger.warning("snagline.auto: LangChain not installed; nothing to patch")
        return False
    wrap_client(monitor, BaseLanguageModel)
    wrap_client(monitor, Chain)
    return True
