"""Auto-instrumentation for LangChain (ATTACH_ANY_SYSTEM P0).

Wraps the common ``invoke`` / ``generate`` entrypoints on a LangChain model
or chain so each call emits a ``StepEvent``. Import-safe: a no-op when
LangChain is absent, and handles synchronous and asynchronous methods.

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

import inspect
import itertools
import logging
import time

from snagline.events import StepEvent, make_signature

logger = logging.getLogger("snagline")

# Entrypoints we patch on a model or chain instance.
_LANGCHAIN_METHODS = ("invoke", "generate", "ainvoke", "agenerate")


def _emit(monitor, counter, model, tool_name, sig_text, start, error) -> None:
    now = time.perf_counter()
    event = StepEvent(
        step_id=str(next(counter)),
        episode_id="langchain-auto",
        timestamp=now,
        action_type="tool_call",
        action_signature=make_signature("langchain_call", model, sig_text),
        tool_name=tool_name,
        latency_ms=(now - start) * 1000.0,
        error=error,
    )
    monitor.ingest(event)


def _wrap_one(monitor, original, tool_name):
    # Idempotent: wrapping an already-wrapped callable stacks a second layer
    # and double-counts every call (issue #336). The OpenAI/Anthropic modules
    # share the same sentinel, so one host instrumenting through more than one
    # of these paths stays at one event per call.
    if getattr(original, "__snagline_wrapped__", False):
        return original
    counter = itertools.count()
    is_async = inspect.iscoroutinefunction(original)

    def _sync(*args, **kwargs):
        sig_text = str(kwargs.get("input") or kwargs.get("prompts") or args)
        model = getattr(original, "__self__", None)
        model_name = getattr(model, "model_name", None) or getattr(
            model, "model", "langchain"
        )
        start = time.perf_counter()
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
        start = time.perf_counter()
        error = False
        try:
            result = await original(*args, **kwargs)
        except Exception:
            error = True
            raise
        finally:
            _emit(monitor, counter, model_name, tool_name, sig_text, start, error)
        return result

    wrapper = _async if is_async else _sync
    # Mark the wrapper so a second wrap_client does not stack another layer
    # (issue #336); the guard in _wrap_one reads this attribute.
    wrapper.__snagline_wrapped__ = True  # type: ignore[attr-defined,union-attr]
    return wrapper


def wrap_client(monitor, client):
    """Patch the invoke/generate entrypoints on ``client`` in place."""
    _patch_client(monitor, client)
    return client


def _patch_client(monitor, client) -> int:
    """Counting core of :func:`wrap_client` (issue #321).

    Returns the number of methods newly wrapped, so the instrument_*
    entrypoints can honour their documented "True if anything was patched"
    contract on the explicit-client path.  Already-wrapped entrypoints are
    left alone: a second layer would double-count every call (issue #336).
    """
    patched = 0
    for name in _LANGCHAIN_METHODS:
        method = getattr(client, name, None)
        if method is None or not callable(method):
            continue
        # _wrap_one returns the existing wrapper unchanged when the method is
        # already instrumented, so setattr is a harmless no-op that keeps the
        # entrypoint visible for other paths.
        setattr(client, name, _wrap_one(monitor, method, "langchain." + name))
        patched += 1
    if patched == 0:
        logger.warning(
            "snagline.auto: wrap_client found no langchain method to patch on %r",
            client,
        )
    return patched


def instrument_langchain(monitor, client=None) -> bool:
    """Instrument LangChain.

    If ``client`` is provided, only that instance is wrapped. Otherwise the
    installed ``langchain`` base classes are patched globally (best effort).
    Returns True if anything was patched, False if LangChain is absent.
    """
    if client is not None:
        return _patch_client(monitor, client) != 0
    try:  # pragma: no cover - exercised only with LangChain installed
        from langchain.chains.base import Chain  # type: ignore
        from langchain_core.language_models import BaseLanguageModel  # type: ignore
    except Exception:
        logger.warning("snagline.auto: LangChain not installed; nothing to patch")
        return False
    wrap_client(monitor, BaseLanguageModel)
    wrap_client(monitor, Chain)
    return True
