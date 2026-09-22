"""Tests for Anthropic auto-instrumentation (ATTACH_ANY_SYSTEM P0)."""

from __future__ import annotations

from snagline.auto.anthropic import instrument_anthropic, wrap_client


class _SpyMonitor:
    def __init__(self):
        self.events: list = []

    def ingest(self, event) -> None:
        self.events.append(event)


class _FakeMessages:
    def create(self, *, model="claude", messages=None, **kw):
        return "ok"


class _FakeClient:
    """Per-instance resource: ``wrap_client`` mutates the object it finds, so a
    class-level attribute would leak a wrapper bound to another test's monitor
    across tests."""

    def __init__(self):
        self.messages = _FakeMessages()


def test_wrap_client_records_on_success():
    mon = _SpyMonitor()
    client = wrap_client(mon, _FakeClient())
    out = client.messages.create(model="claude-3-5-sonnet", messages=[{"role": "user"}])
    assert out == "ok"
    assert len(mon.events) == 1
    ev = mon.events[0]
    assert ev.tool_name == "anthropic.messages.create"
    assert ev.error is False
    assert ev.latency_ms is not None


def test_wrap_client_missing_messages_attr_is_safe():
    mon = _SpyMonitor()

    class _Bare:
        pass

    wrap_client(mon, _Bare())
    assert mon.events == []


def test_wrap_client_records_error_and_propagates():
    class _BoomMessages:
        def create(self, **kw):
            raise RuntimeError("boom")

    class _BoomClient:
        messages = _BoomMessages()

    mon = _SpyMonitor()
    client = wrap_client(mon, _BoomClient())
    raised = False
    try:
        client.messages.create(model="claude", messages=[{"role": "user"}])
    except RuntimeError:
        raised = True
    assert raised, "exception must propagate"
    assert len(mon.events) == 1
    assert mon.events[0].error is True


def test_instrument_anthropic_without_sdk_is_safe_noop(monkeypatch, caplog):
    # Force the SDK-absent branch regardless of whether the anthropic package
    # is installed in this venv (issue #295).
    monkeypatch.setattr("snagline.auto.anthropic.Anthropic", None)
    monkeypatch.setattr("snagline.auto.anthropic.AsyncAnthropic", None)
    mon = _SpyMonitor()
    with caplog.at_level("WARNING"):
        assert instrument_anthropic(mon) is False
    assert "nothing to patch" in caplog.text


def test_instrument_anthropic_with_explicit_client():
    mon = _SpyMonitor()
    assert instrument_anthropic(mon, client=_FakeClient()) is True
