"""Auto-instrumentation for LangChain (ATTACH_ANY_SYSTEM P0).

Wraps the common ``invoke`` / ``generate`` entrypoints on a LangChain model
or chain so each call emits a ``StepEvent``. Import-safe: a no-op when
LangChain is absent, and handles synchronous and asynchronous methods.

Global mode (issue #339) patches the base classes that the declared
dependency actually provides. The ``langchain`` extra ships only
``langchain-core``, so the targets are ``BaseChatModel``, ``BaseLLM`` and
their shared parent ``BaseLanguageModel`` from there;
``langchain.chains.base.Chain`` remains as a fallback for langchain < 1.0,
where chains were a distinct class. In langchain >= 1.0 that module no longer
exists, and importing it aborted the whole entrypoint with a wrong "LangChain
not installed" message while langchain-core was perfectly installed.

``BaseLLM`` must be a target in its own right: it is a sibling of
``BaseChatModel``, not a subclass, and it re-declares ``invoke`` / ``ainvoke``
to route to ``generate_prompt`` / ``agenerate_prompt``. Patching only
``BaseLanguageModel`` left every raw completion-model call unobserved while
``instrument_langchain()`` still reported True.

``Runnable`` itself is deliberately NOT a target. ``invoke`` / ``ainvoke`` are
defined there, but patching the base would also wrap every non-model runnable
-- output parsers, retrievers, prompts -- and a composed chain calls each of
its components' ``invoke`` in turn, so one logical call would emit one event
per component. The language-model bases cover LLM calls without that nesting
noise; chain objects that are pure runnables are out of scope for global mode
and remain coverable per-client via ``instrument_langchain(monitor,
client=chain)``.

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

# Entrypoints per-client mode patches on a model or chain instance. ``generate``
# / ``agenerate`` were removed from ``BaseLanguageModel`` in langchain-core 1.x
# and now live only on ``BaseChatModel``; they are kept for per-client mode,
# where the caller explicitly asked to wrap that one object.
_LANGCHAIN_METHODS = ("invoke", "ainvoke", "generate", "agenerate")

# Global mode patches only the outermost entrypoints. ``BaseChatModel.invoke``
# delegates to ``self.generate`` and ``ainvoke`` to ``self.agenerate`` (both
# defined on that same class in langchain-core 1.x), so wrapping all four
# would emit two events for a single user-facing call. A caller who wants the
# batch entrypoints too can use per-client mode (issue #339).
_LANGCHAIN_GLOBAL_METHODS = ("invoke", "ainvoke")


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


def _resolve_model(original, args):
    """Return ``(model, call_args)`` for a wrapped call.

    ``original`` is a *bound* method when the target was a client instance, so
    ``__self__`` names the model and ``args`` holds only the call input. When
    the target was a *class* -- global mode -- ``getattr(cls, name)`` returns
    an unbound function: ``__self__`` is absent, and the instance arrives as
    ``args[0]`` once Python binds the wrapper through the descriptor protocol.
    Reading the model from ``__self__`` there yielded ``None`` (so every global
    event reported model ``"langchain"``) and left the instance inside ``args``,
    baking its ``repr`` into the action signature.
    """
    bound = getattr(original, "__self__", None)
    if bound is not None:
        return bound, args
    if not args:
        return None, args
    return args[0], args[1:]


def _model_name(model):
    return getattr(model, "model_name", None) or getattr(model, "model", "langchain")


def _wrap_one(monitor, original, tool_name):
    counter = itertools.count()
    is_async = inspect.iscoroutinefunction(original)

    def _sync(*args, **kwargs):
        model, call_args = _resolve_model(original, args)
        sig_text = str(kwargs.get("input") or kwargs.get("prompts") or call_args)
        model_name = _model_name(model)
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
        model, call_args = _resolve_model(original, args)
        sig_text = str(kwargs.get("input") or kwargs.get("prompts") or call_args)
        model_name = _model_name(model)
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

    return _async if is_async else _sync


def _patch_target(monitor, target, methods=_LANGCHAIN_METHODS) -> int:
    """Patch ``methods`` on ``target`` (instance *or* class).

    Returns the number of entrypoints that were patched. Callers use the count
    to tell "patched something" from "the SDK is present but exposes none of
    the entrypoints we look for" (issue #339), and it is how the instrument_*
    entrypoints honour their documented "True if anything was patched" contract
    on the explicit-client path (issue #321).
    """
    patched = 0
    for name in methods:
        method = getattr(target, name, None)
        if method is None or not callable(method):
            continue
        setattr(target, name, _wrap_one(monitor, method, "langchain." + name))
        patched += 1
    if patched == 0:
        logger.warning(
            "snagline.auto: wrap_client found no langchain method to patch on %r",
            target,
        )
    return patched


def wrap_client(monitor, client):
    """Patch the invoke/generate entrypoints on ``client`` in place.

    Returns the same client for chaining.
    """
    _patch_target(monitor, client)
    return client


def _global_targets() -> tuple[list[type], bool]:
    """Classes global mode should patch, and whether langchain-core is present.

    The ``langchain`` extra declares only ``langchain-core``, so the primary
    targets come from there. ``invoke`` / ``ainvoke`` are defined on
    ``Runnable`` and inherited by chains and models; ``BaseChatModel`` and
    ``BaseLLM`` however *override* both, so patching ``Runnable`` alone would
    miss every model -- attribute lookup finds the subclass's own entrypoint
    first. The two model bases and their shared parent are all patched; the
    bases are disjoint (``BaseChatModel`` and ``BaseLLM`` are siblings), and
    each overrides ``invoke`` to delegate to a *generation* method rather than
    to the parent's ``invoke``, so an instance resolves to exactly one patched
    entrypoint via its MRO and no call is double-counted.

    The top-level ``langchain`` package is a fallback for langchain < 1.0,
    whose ``Chain`` lives at ``langchain.chains.base``. That module was removed
    in the 1.0 restructure, and importing it used to take the whole global
    entrypoint down with a wrong "LangChain not installed" message (issue
    #339).

    Returns ``(targets, core_present)``: ``core_present`` distinguishes three
    states, only the first of which deserves "not installed" -- a genuinely
    absent dependency. A package that imports but exposes none of the classes,
    or whose own imports fail, is present-but-unusable and is reported by the
    caller as a monitoring failure instead (issue #339).
    """
    targets: list[type] = []
    # Probe the package itself first, so a present-but-reshaped SDK cannot be
    # mistaken for an absent one: both surface as ImportError from the class
    # imports below, but only a genuinely missing dependency fails this one
    # (issue #339).
    core_present = False
    try:
        import langchain_core  # type: ignore # noqa: F401 (presence probe)

        core_present = True
    except ModuleNotFoundError as exc:
        # Only a missing ``langchain_core`` itself means "not installed". A
        # ModuleNotFoundError naming some *other* module means the SDK is
        # present but one of its own imports is broken -- an unusable install,
        # not an absent one, and reporting "not installed" would send the
        # operator looking for a pip install that changes nothing.
        core_present = exc.name != "langchain_core"
    except ImportError:
        # A plain ImportError (e.g. a broken optional dependency inside the
        # package) is likewise installed-but-unusable, not absent.
        core_present = True
    try:
        from langchain_core.language_models import (  # type: ignore
            BaseLanguageModel,
        )
        from langchain_core.language_models.chat_models import (  # type: ignore
            BaseChatModel,
        )
        from langchain_core.language_models.llms import (  # type: ignore
            BaseLLM,
        )

        targets.append(BaseChatModel)
        # ``BaseLLM`` (raw completion models) is a *sibling* of BaseChatModel,
        # not a subclass, and it re-declares ``invoke`` / ``ainvoke`` -- they
        # resolve to ``generate_prompt`` / ``agenerate_prompt``, never to
        # ``BaseLanguageModel.invoke``. Patching only the shared base left
        # every completion-model call unobserved while instrument_langchain()
        # still reported True.
        targets.append(BaseLLM)
        targets.append(BaseLanguageModel)
    except ImportError:
        pass
    try:  # langchain < 1.0: chains were a distinct class, not just Runnables
        from langchain.chains.base import Chain  # type: ignore

        targets.append(Chain)
    except ImportError:
        pass
    return targets, core_present


def instrument_langchain(monitor, client=None) -> bool:
    """Instrument LangChain.

    If ``client`` is provided, only that instance is wrapped. Otherwise the
    installed ``langchain`` / ``langchain-core`` base classes are patched
    globally, so the supported model bases and legacy ``Chain`` instances --
    present or future -- are observed. Chains that are pure ``Runnable``
    compositions are out of scope here and need per-client mode; see the module
    docstring for why ``Runnable`` itself is not patched.
    Returns True if anything was patched, False if LangChain is absent or no
    entrypoint could be found.
    """
    if client is not None:
        return _patch_target(monitor, client) != 0
    targets, core_present = _global_targets()
    if not core_present:
        # Genuinely absent: the declared dependency is not installed. This is
        # the only case that deserves "not installed"; a present-but-reshaped
        # SDK must be reported as a monitoring failure instead (issue #339).
        logger.warning("snagline.auto: langchain-core not installed; nothing to patch")
        return False
    patched = 0
    for cls in targets:
        patched += _patch_target(monitor, cls, _LANGCHAIN_GLOBAL_METHODS)
    if patched == 0:
        logger.warning(
            "snagline.auto: langchain-core installed but no invoke/ainvoke "
            "entrypoint could be patched; LangChain is NOT monitored"
        )
        return False
    return True
