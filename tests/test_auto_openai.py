"""Tests for OpenAI auto-instrumentation (ATTACH_ANY_SYSTEM P0).

Exercises wrap_client with a fake client that mimics the OpenAI SDK surface,
so no real SDK is required. Also confirms instrument_openai is a safe no-op
when the SDK is absent.
"""

from __future__ import annotations

from snagline.auto.openai import instrument_openai, wrap_client


class _SpyMonitor:
    def __init__(self):
        self.events: list = []

    def ingest(self, event) -> None:
        self.events.append(event)


class _FakeCompletions:
    def create(self, *, model="gpt", messages=None, prompt=None, **kw):
        return "ok"


class _FakeChat:
    def __init__(self):
        self.completions = _FakeCompletions()


class _FakeClient:
    """Per-instance resources: ``wrap_client`` mutates the resource objects it
    hands out, so class-level attributes would leak a wrapper (and its bound
    monitor) across tests -- a later test's events would land in the earlier
    test's spy, or arrive twice."""

    def __init__(self):
        self.chat = _FakeChat()
        self.completions = _FakeCompletions()


def test_wrap_client_records_on_success():
    mon = _SpyMonitor()
    client = wrap_client(mon, _FakeClient())
    out = client.chat.completions.create(model="gpt-4o", messages=[{"role": "user"}])
    assert out == "ok"
    assert len(mon.events) == 1
    ev = mon.events[0]
    assert ev.tool_name == "openai.chat.completions.create"
    assert ev.error is False
    assert ev.latency_ms is not None


def test_wrap_client_records_both_paths():
    mon = _SpyMonitor()
    client = wrap_client(mon, _FakeClient())
    client.chat.completions.create(model="gpt-4o", messages=[{"role": "user"}])
    client.completions.create(model="gpt-4o", prompt="hi")
    assert len(mon.events) == 2
    assert mon.events[1].tool_name == "openai.completions.create"


def test_wrap_client_records_error_and_propagates():
    class _BoomCompletions:
        def create(self, **kw):
            raise RuntimeError("boom")

    class _BoomClient:
        chat = type("C", (), {"completions": _BoomCompletions()})()

    mon = _SpyMonitor()
    client = wrap_client(mon, _BoomClient())
    raised = False
    try:
        client.chat.completions.create(model="gpt-4o", messages=[{"role": "user"}])
    except RuntimeError:
        raised = True
    assert raised, "exception must propagate"
    assert len(mon.events) == 1
    assert mon.events[0].error is True


def test_instrument_openai_without_sdk_is_safe_noop(monkeypatch, caplog):
    # The SDK-absent branch only runs when the module-level import failed, so
    # force it regardless of whether openai happens to be installed in this
    # venv (issue #295) -- otherwise this test silently exercises the
    # "SDK installed" path the explicit-client test below already covers.
    monkeypatch.setattr("snagline.auto.openai.OpenAI", None)
    monkeypatch.setattr("snagline.auto.openai.AsyncOpenAI", None)
    mon = _SpyMonitor()
    with caplog.at_level("WARNING"):
        assert instrument_openai(mon) is False
    assert "nothing to patch" in caplog.text


def test_instrument_openai_with_explicit_client():
    mon = _SpyMonitor()
    assert instrument_openai(mon, client=_FakeClient()) is True
