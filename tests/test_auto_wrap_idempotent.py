"""Per-client wrapping must be idempotent (issue #336).

``wrap_client`` patches a bound method on a client instance. It never marked
its own output, so calling it twice -- or calling it on a client whose class
global mode had already patched -- stacked a second wrapper and every single
call emitted two events, one per layer. The detector layer is counting
distinct LLM calls; a double layer silently doubles measured latency samples
and halves every per-call budget.

This module covers the per-client-against-itself case; the global/per-client
composition is exercised in ``test_auto_global_mode.py``.
"""

from __future__ import annotations

import pytest

from snagline.auto.anthropic import wrap_client as wrap_anthropic
from snagline.auto.openai import wrap_client as wrap_openai


class _SpyMonitor:
    def __init__(self):
        self.events: list = []

    def ingest(self, event) -> None:
        self.events.append(event)


def _fresh_openai_client():
    """Fresh classes per client: a shared class attribute would leak wrapper
    state across tests, exactly the contamination the marker exists to track."""

    class _Completions:
        def create(self, *, model="gpt", messages=None, prompt=None, **kw):
            return "ok"

    class _Chat:
        completions = _Completions()

    class _Client:
        chat = _Chat()
        completions = _Completions()

    return _Client()


def _fresh_anthropic_client():
    class _Messages:
        def create(self, *, model="claude", messages=None, **kw):
            return "ok"

    class _Client:
        messages = _Messages()

    return _Client()


@pytest.mark.parametrize(
    "wrap, make, call",
    [
        (wrap_openai, _fresh_openai_client, lambda c: c.chat.completions.create()),
        (wrap_anthropic, _fresh_anthropic_client, lambda c: c.messages.create()),
    ],
)
def test_wrap_client_twice_emits_one_event_per_call(wrap, make, call):
    mon = _SpyMonitor()
    client = make()
    wrap(mon, client)
    wrap(mon, client)  # second layer must not stack
    call(client)
    assert len(mon.events) == 1, "two wrap_client layers double-count every call"


@pytest.mark.parametrize(
    "wrap, make, call",
    [
        (wrap_openai, _fresh_openai_client, lambda c: c.chat.completions.create()),
        (wrap_anthropic, _fresh_anthropic_client, lambda c: c.messages.create()),
    ],
)
def test_wrap_client_twice_keeps_every_subsequent_call_single(wrap, make, call):
    """The doubled count compounds: without the guard, *every* call paid for
    both layers, not just the first."""
    mon = _SpyMonitor()
    client = make()
    wrap(mon, client)
    wrap(mon, client)
    for _ in range(5):
        call(client)
    assert len(mon.events) == 5


def test_wrap_client_marks_its_own_output():
    """The guard reads the marker off the callable it is handed, so the
    per-client wrapper must set it on itself -- not rely on global mode having
    set it on some class attribute elsewhere."""
    mon = _SpyMonitor()
    client = _fresh_openai_client()
    wrap_openai(mon, client)
    assert (
        getattr(client.chat.completions.create, "__snagline_wrapped__", False) is True
    )


def test_wrap_one_returns_the_existing_wrapper_verbatim():
    from snagline.auto.openai import _wrap_one

    def original():
        return None

    first = _wrap_one(_SpyMonitor(), original, "t")
    # The marker lives on the wrapper, so the guard sees it only when the
    # wrapper itself is handed back -- which is what wrap_client resolves on
    # its second call.
    second = _wrap_one(_SpyMonitor(), first, "t")
    assert first is second, "re-wrapping must return the same wrapper, not a layer"


# --- LangChain (issue #336 names it as the odd one out) ----------------------
# Its ``_wrap_one`` never set the sentinel at all, so neither ``wrap_client``
# nor the other modules' guards could see its output.


class _FakeLLM:
    model_name = "gpt-4o"

    def invoke(self, prompt, **kw):
        return "ok"

    async def ainvoke(self, prompt, **kw):
        return "ok-async"


def test_langchain_wrap_client_twice_emits_one_event_per_call():
    from snagline.auto.langchain import wrap_client as wrap_langchain

    mon = _SpyMonitor()
    llm = wrap_langchain(mon, _FakeLLM())
    wrap_langchain(mon, llm)
    assert llm.invoke("hi") == "ok"
    assert len(mon.events) == 1, "two wrap_client layers double-count every call"


def test_langchain_wrap_client_twice_async_emits_one_event_per_call():
    import asyncio

    from snagline.auto.langchain import wrap_client as wrap_langchain

    async def go():
        mon = _SpyMonitor()
        llm = wrap_langchain(mon, _FakeLLM())
        wrap_langchain(mon, llm)
        assert await llm.ainvoke("hi") == "ok-async"
        assert len(mon.events) == 1

    asyncio.run(go())


# --- Already-wrapped is success, not a warning -------------------------------


def test_wrap_client_stays_quiet_when_everything_is_already_wrapped(caplog):
    """Issue #336 note: a second call that finds everything already wrapped is
    success, not the ``found no create method`` warning case -- the same
    distinction ``_patch_resource_classes`` makes by returning -1."""
    mon = _SpyMonitor()
    client = _fresh_openai_client()
    wrap_openai(mon, client)
    with caplog.at_level("WARNING", logger="snagline"):
        wrap_openai(mon, client)
    assert not any("no create method" in r.message for r in caplog.records)
