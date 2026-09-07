"""Tests for OpenAI auto-instrumentation (ATTACH_ANY_SYSTEM P0).

Exercises wrap_client with a fake client that mimics the OpenAI SDK surface,
so no real SDK is required. Also confirms instrument_openai is a safe no-op
when the SDK is absent.

The streaming tests pin the deferred-emission contract (issue #242): with
``stream=True`` nothing is ingested when ``create()`` returns; exactly one
event lands when the stream is exhausted, closed, or fails mid-flight.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

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
    completions = _FakeCompletions()


class _FakeClient:
    chat = _FakeChat()
    completions = _FakeCompletions()


def _fake_chat_client(create):
    """A minimal OpenAI-shaped client whose chat.completions.create is ``create``."""
    return SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )


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


def test_instrument_openai_without_sdk_is_safe_noop():
    mon = _SpyMonitor()
    assert instrument_openai(mon) is False


def test_instrument_openai_with_explicit_client():
    mon = _SpyMonitor()
    assert instrument_openai(mon, client=_FakeClient()) is True


# --- deferred stream emission (issue #242) ------------------------------------


def _stream(chunks, delay=0.0):
    import time as _time

    class _S:
        def __init__(self):
            self.closed = False

        def __iter__(self):
            return iter(self._gen())

        def _gen(self):
            for c in chunks:
                if delay:
                    _time.sleep(delay)
                yield c

        def close(self):
            self.closed = True

    return _S()


def test_openai_auto_stream_defers_until_exhaustion():
    """create() returns immediately for a stream; ingesting then would record
    a ~0ms success. Nothing is ingested until the stream is exhausted."""
    mon = _SpyMonitor()

    def fake_create(**kw):
        assert kw.get("stream") is True
        return _stream(
            [
                SimpleNamespace(usage=None),
                SimpleNamespace(
                    usage=SimpleNamespace(prompt_tokens=5, completion_tokens=7)
                ),
            ],
            delay=0.05,
        )

    client = wrap_client(mon, _fake_chat_client(fake_create))
    stream = client.chat.completions.create(
        model="gpt-4o", messages=[{"role": "u"}], stream=True
    )
    assert len(mon.events) == 0, "event must not be emitted at stream-open"
    chunks = list(stream)
    assert len(chunks) == 2
    assert len(mon.events) == 1, "exactly one event after exhaustion"
    ev = mon.events[0]
    assert ev.error is False
    assert ev.error_type is None
    assert ev.tokens_in == 5
    assert ev.tokens_out == 7
    # Latency reflects the streamed duration, not the create() return time.
    assert ev.latency_ms > 40.0


def test_openai_auto_stream_error_mid_flight_is_recorded_as_error():
    mon = _SpyMonitor()

    def fake_create(**kw):
        def _gen():
            yield SimpleNamespace(usage=None)
            raise RuntimeError("mid-stream 502")

        class _S:
            def __iter__(self):
                return _gen()

        return _S()

    client = wrap_client(mon, _fake_chat_client(fake_create))
    stream = client.chat.completions.create(model="gpt-4o", messages=[], stream=True)
    assert len(mon.events) == 0
    try:
        list(stream)
        raise AssertionError("iteration should raise")
    except RuntimeError:
        pass
    assert len(mon.events) == 1
    assert mon.events[0].error is True
    assert mon.events[0].error_type == "RuntimeError"


def test_openai_auto_stream_close_emits_and_closes_underlying():
    mon = _SpyMonitor()
    inner = _stream([SimpleNamespace(usage=None)])

    def fake_create(**kw):
        return inner

    client = wrap_client(mon, _fake_chat_client(fake_create))
    stream = client.chat.completions.create(model="gpt-4o", messages=[], stream=True)
    assert len(mon.events) == 0
    next(iter(stream))  # consume one chunk
    stream.close()
    assert len(mon.events) == 1
    assert inner.closed, "underlying stream.close() must still be called"


def test_openai_auto_non_stream_still_immediate():
    mon = _SpyMonitor()
    client = wrap_client(mon, _FakeClient())
    out = client.chat.completions.create(model="gpt-4o", messages=[{"role": "user"}])
    assert out == "ok"
    assert len(mon.events) == 1


def test_openai_auto_async_stream_defers_until_exhaustion():
    mon = _SpyMonitor()

    async def fake_create(**kw):
        chunks = [
            SimpleNamespace(usage=SimpleNamespace(prompt_tokens=3, completion_tokens=4))
        ]

        class _AS:
            def __init__(self):
                self._i = 0

            def __aiter__(self):
                return self

            async def __anext__(self):
                if self._i >= len(chunks):
                    raise StopAsyncIteration
                c = chunks[self._i]
                self._i += 1
                return c

        return _AS()

    from snagline.auto.openai import _wrap_one

    wrapped = _wrap_one(mon, fake_create, "openai.chat.completions.create")

    async def run():
        stream = await wrapped(model="gpt-4o", messages=[], stream=True)
        assert len(mon.events) == 0
        chunks = [c async for c in stream]
        assert len(chunks) == 1
        assert len(mon.events) == 1
        ev = mon.events[0]
        assert ev.error is False
        assert ev.tokens_in == 3
        assert ev.tokens_out == 4

    asyncio.run(run())
