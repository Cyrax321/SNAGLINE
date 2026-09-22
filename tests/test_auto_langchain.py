"""Tests for LangChain auto-instrumentation (ATTACH_ANY_SYSTEM P0)."""

from __future__ import annotations

import asyncio
import sys
import types

import pytest

from snagline.auto.langchain import instrument_langchain, wrap_client
from snagline.events import make_signature


class _SpyMonitor:
    def __init__(self):
        self.events: list = []

    def ingest(self, event) -> None:
        self.events.append(event)


class _FakeLLM:
    model_name = "gpt-4o"

    def invoke(self, prompt, **kw):
        return "ok"

    def generate(self, prompts, **kw):
        return "result"


def test_wrap_client_records_invoke_and_generate():
    mon = _SpyMonitor()
    llm = wrap_client(mon, _FakeLLM())
    assert llm.invoke("hello") == "ok"
    assert llm.generate(["a", "b"]) == "result"
    assert len(mon.events) == 2
    tools = {e.tool_name for e in mon.events}
    assert tools == {"langchain.invoke", "langchain.generate"}
    for ev in mon.events:
        assert ev.error is False
        assert ev.latency_ms is not None


def test_wrap_client_records_error_and_propagates():
    class _BoomLLM:
        def invoke(self, *a, **kw):
            raise RuntimeError("boom")

    mon = _SpyMonitor()
    llm = wrap_client(mon, _BoomLLM())
    raised = False
    try:
        llm.invoke("hi")
    except RuntimeError:
        raised = True
    assert raised
    assert len(mon.events) == 1
    assert mon.events[0].error is True


def _force_absent(monkeypatch):
    """Make both langchain packages unimportable for the duration of a test.

    ``langchain-core`` is an optional dependency and CI installs it, so
    absence cannot be assumed: ``delitem`` alone just re-imports it from disk,
    and a package nothing imported yet is not in ``sys.modules`` at all. A
    ``None`` sentinel under every relevant name makes the import fail outright.
    """
    names = {
        n
        for n in sys.modules
        if n == "langchain_core"
        or n.startswith("langchain_core.")
        or n == "langchain"
        or n.startswith("langchain.")
    }
    # The top-level names are sentinelled unconditionally: an installed-but-
    # never-imported package has no sys.modules entry to catch in the loop.
    names.update(("langchain_core", "langchain"))
    for name in names:
        monkeypatch.setitem(sys.modules, name, None)


def test_instrument_langchain_without_sdk_is_safe_noop(monkeypatch, caplog):
    """Issue #339: this test used to pass in CI *because* the broken
    ``langchain.chains`` import made an installed SDK look absent."""
    _force_absent(monkeypatch)
    mon = _SpyMonitor()
    with caplog.at_level("WARNING"):
        assert instrument_langchain(mon) is False
    assert "nothing to patch" in caplog.text


def test_instrument_langchain_with_explicit_client():
    mon = _SpyMonitor()
    assert instrument_langchain(mon, client=_FakeLLM()) is True


# --- global mode (issue #339) -------------------------------------------------
# ``langchain.chains`` was removed in the langchain 1.0 restructure, so the old
# import aborted the whole entrypoint with a wrong "LangChain not installed"
# message while langchain-core was installed. These reproduce both SDK shapes
# with local fakes: the 1.x base classes under ``langchain_core``, and the 0.x
# ``langchain.chains.base.Chain`` fallback.


def _delegating_chat_base():
    """A chat-model base whose ``invoke`` delegates to ``generate`` -- the
    shape that makes patching all four entrypoints double-count (one event for
    the user's ``invoke``, one for the internal ``generate``)."""

    class BaseChatModel:
        model_name = "chat-base"

        def invoke(self, input, **kw):
            return self.generate(input)

        async def ainvoke(self, input, **kw):
            return await self.agenerate(input)

        def generate(self, messages):
            return "chat-ok"

        async def agenerate(self, messages):
            return "chat-async-ok"

    class BaseLLM:
        # A *sibling* of BaseChatModel, not a subclass: it re-declares the
        # entrypoints and routes them elsewhere than the shared parent's, so
        # patching only BaseLanguageModel never intercepts it.
        model_name = "completion-base"

        def invoke(self, input, **kw):
            return self.generate_prompt(input)

        async def ainvoke(self, input, **kw):
            return await self.agenerate_prompt(input)

        def generate_prompt(self, prompts):
            return "llm-ok"

        async def agenerate_prompt(self, prompts):
            return "llm-async-ok"

    class BaseLanguageModel:
        model_name = "lm-base"

        def invoke(self, input, **kw):
            return "lm-ok"

        async def ainvoke(self, input, **kw):
            return "lm-async-ok"

    return BaseChatModel, BaseLLM, BaseLanguageModel


@pytest.fixture
def fake_langchain_core(monkeypatch):
    """Install a fake ``langchain_core`` with the 1.x class layout, and make
    sure the 0.x top-level ``langchain`` package is not importable."""

    chat_cls, llm_cls, lm_cls = _delegating_chat_base()
    lm_mod = types.ModuleType("langchain_core.language_models")
    lm_mod.BaseLanguageModel = lm_cls  # type: ignore[attr-defined]
    chat_mod = types.ModuleType("langchain_core.language_models.chat_models")
    chat_mod.BaseChatModel = chat_cls  # type: ignore[attr-defined]
    llm_mod = types.ModuleType("langchain_core.language_models.llms")
    llm_mod.BaseLLM = llm_cls  # type: ignore[attr-defined]
    _force_absent(monkeypatch)
    monkeypatch.setitem(
        sys.modules, "langchain_core", types.ModuleType("langchain_core")
    )
    monkeypatch.setitem(sys.modules, "langchain_core.language_models", lm_mod)
    monkeypatch.setitem(
        sys.modules, "langchain_core.language_models.chat_models", chat_mod
    )
    monkeypatch.setitem(sys.modules, "langchain_core.language_models.llms", llm_mod)
    return chat_mod, llm_mod, lm_mod


@pytest.fixture
def fake_langchain_0x(monkeypatch, fake_langchain_core):
    """Add the pre-1.0 ``langchain.chains.base.Chain`` on top of the fake core."""

    class Chain:
        def invoke(self, input, **kw):
            return "chain-ok"

    chains_mod = types.ModuleType("langchain.chains.base")
    chains_mod.Chain = Chain  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "langchain", types.ModuleType("langchain"))
    monkeypatch.setitem(
        sys.modules, "langchain.chains", types.ModuleType("langchain.chains")
    )
    monkeypatch.setitem(sys.modules, "langchain.chains.base", chains_mod)
    return chains_mod


def test_global_mode_patches_langchain_core_bases(fake_langchain_core):
    mon = _SpyMonitor()
    assert instrument_langchain(mon) is True, "an installed SDK must not report absent"
    chat, llm, lm = fake_langchain_core

    # Sync call through a chat-model instance emits exactly once, even though
    # BaseChatModel.invoke delegates to self.generate.
    assert chat.BaseChatModel().invoke("hi") == "chat-ok"
    assert len(mon.events) == 1
    assert mon.events[0].tool_name == "langchain.invoke"
    assert mon.events[0].error is False

    # The completion base is patched too, not just the chat one.
    assert llm.BaseLLM().invoke("hi") == "llm-ok"
    assert len(mon.events) == 2

    # The shared parent is patched as well.
    assert lm.BaseLanguageModel().invoke("hi") == "lm-ok"
    assert len(mon.events) == 3

    # Async leg.
    assert asyncio.run(lm.BaseLanguageModel().ainvoke("hi")) == "lm-async-ok"
    assert len(mon.events) == 4


def test_global_mode_patches_the_completion_base(fake_langchain_core):
    """``BaseLLM`` re-declares ``invoke`` / ``ainvoke`` and is a sibling of
    ``BaseChatModel``, so patching only ``BaseLanguageModel`` left every
    completion-model call invisible while instrument_langchain() reported
    True."""
    _, llm, _ = fake_langchain_core
    mon = _SpyMonitor()
    assert instrument_langchain(mon) is True

    assert llm.BaseLLM().invoke("hi") == "llm-ok"
    assert len(mon.events) == 1
    assert mon.events[0].tool_name == "langchain.invoke"

    assert asyncio.run(llm.BaseLLM().ainvoke("hi")) == "llm-async-ok"
    assert len(mon.events) == 2


def test_global_mode_does_not_wrap_the_delegation_target(fake_langchain_core):
    """``BaseChatModel.invoke`` calls ``self.generate`` and ``BaseLLM.invoke``
    calls ``self.generate_prompt``; patching those too would emit a second
    event for the same user-facing call. Global mode wraps only the outermost
    entrypoints."""
    chat, llm, _ = fake_langchain_core
    mon = _SpyMonitor()
    assert instrument_langchain(mon) is True
    assert chat.BaseChatModel().generate("hi") == "chat-ok"
    assert llm.BaseLLM().generate_prompt("hi") == "llm-ok"
    assert mon.events == [], "generation methods are an implementation detail"


def test_global_mode_async_does_not_double_count(fake_langchain_core):
    chat, llm, _ = fake_langchain_core
    mon = _SpyMonitor()
    assert instrument_langchain(mon) is True
    assert asyncio.run(chat.BaseChatModel().ainvoke("hi")) == "chat-async-ok"
    assert asyncio.run(llm.BaseLLM().ainvoke("hi")) == "llm-async-ok"
    assert len(mon.events) == 2


def test_global_mode_falls_back_to_langchain_chains(
    fake_langchain_core, fake_langchain_0x
):
    """langchain < 1.0: ``Chain`` still exists at ``langchain.chains.base`` and
    is patched alongside the core bases."""
    chains = fake_langchain_0x
    mon = _SpyMonitor()
    assert instrument_langchain(mon) is True
    assert chains.Chain().invoke("hi") == "chain-ok"
    assert len(mon.events) == 1


def test_global_mode_distinguishes_absent_from_reshaped(monkeypatch, caplog):
    """A genuinely missing dependency and a present-but-reshaped one must not
    share one message: "not installed" for the former, a monitoring failure for
    the latter (issue #339)."""
    _force_absent(monkeypatch)
    with caplog.at_level("WARNING", logger="snagline"):
        assert instrument_langchain(_SpyMonitor()) is False
    assert any("langchain-core not installed" in r.message for r in caplog.records)

    # Present, but the class layout moved: patching nothing is a failure, not
    # silence -- and the message must not claim the SDK is absent.
    caplog.clear()
    core_mod = types.ModuleType("langchain_core")
    lm_mod = types.ModuleType("langchain_core.language_models")  # no classes
    monkeypatch.setitem(sys.modules, "langchain_core", core_mod)
    monkeypatch.setitem(sys.modules, "langchain_core.language_models", lm_mod)
    with caplog.at_level("WARNING", logger="snagline"):
        assert instrument_langchain(_SpyMonitor()) is False
    assert any("NOT monitored" in r.message for r in caplog.records)
    assert not any("not installed" in r.message for r in caplog.records)


def test_global_mode_distinguishes_absent_from_broken(monkeypatch, caplog):
    """``import langchain_core`` can fail without the package being missing:
    a broken transitive dependency raises ``ModuleNotFoundError`` naming some
    *other* module. That is an unusable install, not an absent one, so it must
    not be reported as "not installed" (issue #339)."""

    # Make ``import langchain_core`` fail the way a broken transitive dep does:
    # the package is discoverable (so it is installed) but its own import
    # raises for a module that is not langchain_core itself.
    real_import = __import__

    def _fake_import(name, *a, **kw):
        if name == "langchain_core":
            raise ModuleNotFoundError("No module named 'pydantic'", name="pydantic")
        return real_import(name, *a, **kw)

    _force_absent(monkeypatch)  # drop any cached, healthy copy first
    monkeypatch.setitem(sys.modules, "langchain_core", None)
    monkeypatch.setattr("builtins.__import__", _fake_import)

    with caplog.at_level("WARNING", logger="snagline"):
        assert instrument_langchain(_SpyMonitor()) is False
    assert not any("not installed" in r.message for r in caplog.records), (
        "a broken transitive dep is not an absent dependency"
    )
    assert any("NOT monitored" in r.message for r in caplog.records)


def test_global_mode_records_the_model_name(fake_langchain_core):
    """Patching a *class* yields an unbound ``original``, so ``__self__`` is
    absent and the instance arrives as ``args[0]``. Reading the model name from
    ``__self__`` there returned ``None`` and every global event was attributed
    to ``"langchain"`` instead of the model."""
    chat, llm, lm = fake_langchain_core
    mon = _SpyMonitor()
    assert instrument_langchain(mon) is True

    chat.BaseChatModel().invoke("hi")
    llm.BaseLLM().invoke("hi")
    lm.BaseLanguageModel().invoke("hi")

    assert [e.action_signature for e in mon.events] == [
        make_signature("langchain_call", name, "('hi',)")
        for name in ("chat-base", "completion-base", "lm-base")
    ]


def test_global_mode_excludes_the_instance_from_the_signature(fake_langchain_core):
    """The instance reaching ``args[0]`` in global mode is an implementation
    detail of the call, not part of the input: leaving it in ``args`` baked its
    ``repr`` into ``action_signature``, so two calls with identical input
    produced two different signatures and defeated dedup."""
    chat, _llm, _lm = fake_langchain_core
    mon = _SpyMonitor()
    assert instrument_langchain(mon) is True

    chat.BaseChatModel().invoke("hi")
    chat.BaseChatModel().invoke("hi")  # a *different* instance, same input

    assert len(mon.events) == 2
    assert mon.events[0].action_signature == mon.events[1].action_signature
