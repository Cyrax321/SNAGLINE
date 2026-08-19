"""Tests for LangChain auto-instrumentation (ATTACH_ANY_SYSTEM P0)."""

from __future__ import annotations

from snagline.auto.langchain import instrument_langchain, wrap_client


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


def test_instrument_langchain_without_sdk_is_safe_noop():
    mon = _SpyMonitor()
    assert instrument_langchain(mon) is False


def test_instrument_langchain_with_explicit_client():
    mon = _SpyMonitor()
    assert instrument_langchain(mon, client=_FakeLLM()) is True
