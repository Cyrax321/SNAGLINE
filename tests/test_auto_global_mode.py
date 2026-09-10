"""Global-mode auto-instrumentation tests (issue #270).

The real OpenAI/Anthropic SDKs expose their resources through
``functools.cached_property`` descriptors on the client classes
(``OpenAI.chat``, ``Anthropic.messages``), and wrap ``create`` with
decorators like ``@required_args`` that hide the underlying ``async def``.
``instrument_openai`` / ``instrument_anthropic`` in global mode used to walk
the *client* class attributes, which resolves a descriptor (not a resource),
wraps nothing, and still returns ``True`` -- a completely unmonitored process
reporting success.

These tests reproduce both SDK shapes with local fakes (no SDK install
needed): a ``cached_property`` client surface and decorated ``create``
methods carrying ``__wrapped__``. They pin the fixed behavior: global mode
patches the resource classes' ``create``, sync and async calls both emit
events with correct error flags, and re-instrumenting is idempotent.

Fresh resource classes are minted per test: the wrapper sets a
``__snagline_wrapped__`` marker on the class attribute, so reusing one class
across tests would leak "already instrumented" state.
"""

from __future__ import annotations

import asyncio
import functools
import inspect
import sys
import types

import pytest

import snagline.auto.anthropic as anth_mod
import snagline.auto.openai as oai_mod
from snagline.auto.anthropic import instrument_anthropic
from snagline.auto.openai import instrument_openai


class _SpyMonitor:
    def __init__(self):
        self.events: list = []

    def ingest(self, event) -> None:
        self.events.append(event)


def _required_args_fake(fn):
    """Stand-in for the SDKs' ``@required_args``: a decorator that hides the
    wrapped function's coroutine nature from ``iscoroutinefunction``."""

    def inner(*args, **kwargs):
        return fn(*args, **kwargs)

    inner.__wrapped__ = fn  # type: ignore[attr-defined]
    return inner


def _fresh_resource_classes():
    """A fresh sync/async resource-class pair with the SDK's real shape."""

    class Sync:
        @_required_args_fake
        def create(self, *, model="gpt", messages=None, **kw):
            return "sync-ok"

    class Async:
        @_required_args_fake
        async def create(self, *, model="gpt", messages=None, **kw):
            await asyncio.sleep(0)
            return "async-ok"

    return Sync, Async


class _ChatHolder:
    def __init__(self, resource):
        self.completions = resource


class _FakeSdkClient:
    """Client whose resource access mirrors the cached_property shape."""

    def __init__(self, resource):
        self._resource = resource

    @functools.cached_property
    def chat(self):
        return _ChatHolder(self._resource)

    @functools.cached_property
    def messages(self):
        return self._resource


@pytest.fixture
def fake_openai_sdk(monkeypatch):
    """Install a fake ``openai`` package shaped like the real one."""
    sync_cls, async_cls = _fresh_resource_classes()
    chat_mod = types.ModuleType("openai.resources.chat.completions.completions")
    chat_mod.Completions = sync_cls  # type: ignore[attr-defined]
    chat_mod.AsyncCompletions = async_cls  # type: ignore[attr-defined]
    legacy_mod = types.ModuleType("openai.resources.completions")
    legacy_mod.Completions = sync_cls  # type: ignore[attr-defined]
    legacy_mod.AsyncCompletions = async_cls  # type: ignore[attr-defined]
    for name in (
        "openai",
        "openai.resources",
        "openai.resources.chat",
        "openai.resources.chat.completions",
        "openai.resources.chat.completions.completions",
        "openai.resources.completions",
    ):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    monkeypatch.setitem(
        sys.modules, "openai.resources.chat.completions.completions", chat_mod
    )
    monkeypatch.setitem(sys.modules, "openai.resources.completions", legacy_mod)
    return chat_mod, legacy_mod


@pytest.fixture
def fake_anthropic_sdk(monkeypatch):
    sync_cls, async_cls = _fresh_resource_classes()
    mod = types.ModuleType("anthropic.resources.messages")
    mod.Messages = sync_cls  # type: ignore[attr-defined]
    mod.AsyncMessages = async_cls  # type: ignore[attr-defined]
    for name in (
        "anthropic",
        "anthropic.resources",
        "anthropic.resources.messages",
    ):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    monkeypatch.setitem(sys.modules, "anthropic.resources.messages", mod)
    return mod


@pytest.fixture
def openai_present(monkeypatch):
    """Make the module-level SDK sentinels non-None so global mode runs."""
    monkeypatch.setattr(oai_mod, "OpenAI", object)
    monkeypatch.setattr(oai_mod, "AsyncOpenAI", object)


@pytest.fixture
def anthropic_present(monkeypatch):
    monkeypatch.setattr(anth_mod, "Anthropic", object)
    monkeypatch.setattr(anth_mod, "AsyncAnthropic", object)


# --- _is_async_call sees through decorators ----------------------------------


def test_is_async_call_detects_decorated_async():
    @_required_args_fake
    async def f():
        return None

    assert inspect.iscoroutinefunction(f) is False, "fixture must hide async-ness"
    assert oai_mod._is_async_call(f) is True
    assert anth_mod._is_async_call(f) is True


def test_is_async_call_plain_sync_and_async():
    def s():
        return 1

    async def a():
        return 1

    assert oai_mod._is_async_call(s) is False
    assert oai_mod._is_async_call(a) is True


# --- wrap_client warns when it patches nothing -------------------------------


def test_wrap_client_warns_when_nothing_patched(caplog):
    class _Bare:
        pass

    with caplog.at_level("WARNING", logger="snagline"):
        oai_mod.wrap_client(None, _Bare())
    assert any("no create method" in r.message for r in caplog.records)


# --- global mode: patches the resource classes, not the client walk ----------


def test_instrument_openai_global_patches_resource_classes(
    fake_openai_sdk, openai_present
):
    mon = _SpyMonitor()
    # The pre-#270 walk resolved the cached_property descriptor and wrapped
    # nothing; the fixed global mode patches the resource classes.
    assert instrument_openai(mon) is True
    chat, legacy = fake_openai_sdk
    assert "create" in chat.Completions.__dict__
    assert getattr(chat.Completions.create, "__snagline_wrapped__", False) is True
    assert getattr(chat.AsyncCompletions.create, "__snagline_wrapped__", False) is True
    assert getattr(legacy.Completions.create, "__snagline_wrapped__", False) is True

    # Sync calls through resource instances emit events.
    out = chat.Completions().create(model="gpt", messages=[{"role": "user"}])
    assert out == "sync-ok"
    assert len(mon.events) == 1
    assert mon.events[0].error is False
    assert mon.events[0].tool_name == "openai.chat.completions.create"

    # Async leg: the wrapper must await the decorated async create.
    result = asyncio.run(chat.AsyncCompletions().create(model="gpt", messages=[]))
    assert result == "async-ok"
    assert len(mon.events) == 2
    assert mon.events[1].error is False


def test_instrument_openai_global_async_error_flag(fake_openai_sdk, openai_present):
    """A failing decorated async create must surface error=True: the sync
    misclassification would have swallowed it (never awaited, never raised)."""
    chat, _ = fake_openai_sdk

    class _Boom:
        @_required_args_fake
        async def create(self, **kw):
            raise RuntimeError("boom")

    chat.AsyncCompletions = _Boom
    mon = _SpyMonitor()
    assert instrument_openai(mon) is True
    with pytest.raises(RuntimeError):
        asyncio.run(chat.AsyncCompletions().create(model="gpt", messages=[]))
    assert len(mon.events) == 1
    assert mon.events[0].error is True


def test_instrument_openai_global_is_idempotent(fake_openai_sdk, openai_present):
    mon = _SpyMonitor()
    assert instrument_openai(mon) is True
    # First wrap count: one wrapper layer on each class create.
    chat, _ = fake_openai_sdk
    assert chat.Completions.create.__qualname__.count("_wrap_one") == 1

    # Re-instrumenting must not double-wrap (double events per call).
    assert instrument_openai(_SpyMonitor()) is True
    assert chat.Completions.create.__qualname__.count("_wrap_one") == 1
    chat.Completions().create(model="gpt", messages=[])
    assert len(mon.events) == 1, "single event per call, not one per wrapper layer"


def test_instrument_anthropic_global_patches_resource_classes(
    fake_anthropic_sdk, anthropic_present
):
    mon = _SpyMonitor()
    assert instrument_anthropic(mon) is True
    resource = fake_anthropic_sdk
    assert getattr(resource.Messages.create, "__snagline_wrapped__", False) is True
    assert getattr(resource.AsyncMessages.create, "__snagline_wrapped__", False) is True

    out = resource.Messages().create(model="claude", messages=[{"role": "user"}])
    assert out == "sync-ok"
    assert len(mon.events) == 1
    assert mon.events[0].tool_name == "anthropic.messages.create"

    result = asyncio.run(resource.AsyncMessages().create(model="claude", messages=[]))
    assert result == "async-ok"
    assert len(mon.events) == 2


def test_instrument_global_returns_false_when_nothing_patchable(
    monkeypatch, openai_present, caplog
):
    # SDK sentinel present, but resource classes carry no create method.
    resource = types.ModuleType("openai.resources.chat.completions.completions")
    resource.Completions = type("NoCreate", (), {})  # type: ignore[attr-defined]
    resource.AsyncCompletions = type("NoCreate", (), {})  # type: ignore[attr-defined]
    legacy = types.ModuleType("openai.resources.completions")
    legacy.Completions = type("NoCreate", (), {})  # type: ignore[attr-defined]
    legacy.AsyncCompletions = type("NoCreate", (), {})  # type: ignore[attr-defined]
    for name in (
        "openai",
        "openai.resources",
        "openai.resources.chat",
        "openai.resources.chat.completions",
        "openai.resources.chat.completions.completions",
        "openai.resources.completions",
    ):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    monkeypatch.setitem(
        sys.modules, "openai.resources.chat.completions.completions", resource
    )
    monkeypatch.setitem(sys.modules, "openai.resources.completions", legacy)

    with caplog.at_level("WARNING", logger="snagline"):
        result = instrument_openai(_SpyMonitor())
    assert result is False, "must not claim success when nothing was patched"
    assert any("NOT monitored" in r.message for r in caplog.records)


def test_instrument_global_does_not_walk_client_cached_property(
    fake_openai_sdk, openai_present
):
    """The exact #270 failure shape: a client whose resource attribute is a
    cached_property descriptor. The fixed global mode patches the resource
    classes, so calls through a descriptor-shaped client still emit."""
    mon = _SpyMonitor()
    chat, _ = fake_openai_sdk
    assert instrument_openai(mon) is True
    client = _FakeSdkClient(chat.Completions())
    out = client.chat.completions.create(model="gpt", messages=[])
    assert out == "sync-ok"
    assert len(mon.events) == 1
