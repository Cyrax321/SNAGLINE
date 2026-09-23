"""The explicit adapter stream wrappers must honour ``with``/``async with`` and
the client wrappers must be idempotent (issues #426, #427)."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from snagline.adapters.anthropic import wrap_anthropic_client
from snagline.adapters.openai import wrap_openai_client


class _Mon:
    def __init__(self):
        self.events = []

    def ingest(self, event):
        self.events.append(event)


def _openai_client(mon, create):
    client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )
    wrap_openai_client(mon, client, episode_id="ep")
    return client


def _anthropic_client(mon, create):
    client = SimpleNamespace(messages=SimpleNamespace(create=create))
    wrap_anthropic_client(mon, client, episode_id="ep")
    return client


class _SyncStream:
    """A stream that only offers ``__iter__``/``close`` -- no context manager."""

    def __init__(self, chunks):
        self._chunks = list(chunks)
        self.closed = False

    def __iter__(self):
        return iter(self._chunks)

    def close(self):
        self.closed = True


class _AsyncStream:
    def __init__(self, chunks):
        self._chunks = list(chunks)
        self._idx = 0
        self.closed = False

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self._idx >= len(self._chunks):
            raise StopAsyncIteration
        chunk = self._chunks[self._idx]
        self._idx += 1
        return chunk

    async def aclose(self):
        self.closed = True


def test_openai_sync_stream_supports_with():
    """``with client.chat.completions.create(..., stream=True) as s:`` works."""
    mon = _Mon()
    raw = _SyncStream([SimpleNamespace(usage=None), SimpleNamespace(usage=None)])
    client = _openai_client(mon, lambda *a, **k: raw)

    stream = client.chat.completions.create(model="gpt-4o", messages=[], stream=True)
    with stream as s:
        chunks = list(s)

    assert chunks == raw._chunks
    assert raw.closed
    assert len(mon.events) == 1
    assert mon.events[0].error is False


def test_anthropic_sync_stream_supports_with():
    mon = _Mon()
    raw = _SyncStream([SimpleNamespace(usage=None)])
    client = _anthropic_client(mon, lambda *a, **k: raw)

    stream = client.messages.create(model="claude-3", messages=[], stream=True)
    with stream as s:
        assert list(s) == raw._chunks

    assert raw.closed
    assert len(mon.events) == 1


def test_openai_with_body_exception_is_recorded_as_error():
    """An exception escaping the ``with`` body is the call's visible outcome and
    must not be recorded as a clean success."""
    mon = _Mon()
    raw = _SyncStream([SimpleNamespace(usage=None)])
    client = _openai_client(mon, lambda *a, **k: raw)

    stream = client.chat.completions.create(model="gpt-4o", messages=[], stream=True)
    with pytest.raises(RuntimeError):
        with stream as s:
            next(iter(s))
            raise RuntimeError("host body died")

    assert len(mon.events) == 1
    assert mon.events[0].error is True
    assert mon.events[0].error_type == "RuntimeError"


def test_openai_async_stream_supports_async_with():
    mon = _Mon()
    raw = _AsyncStream([SimpleNamespace(usage=None), SimpleNamespace(usage=None)])

    async def create(*a, **k):
        return raw

    client = _openai_client(mon, create)

    async def run():
        stream = await client.chat.completions.create(
            model="gpt-4o", messages=[], stream=True
        )
        async with stream as s:
            out = [c async for c in s]
        assert out == raw._chunks

    asyncio.run(run())
    assert raw.closed
    assert len(mon.events) == 1
    assert mon.events[0].error is False


def test_anthropic_async_stream_supports_async_with():
    mon = _Mon()
    raw = _AsyncStream([SimpleNamespace(usage=None)])

    async def create(*a, **k):
        return raw

    client = _anthropic_client(mon, create)

    async def run():
        stream = await client.messages.create(
            model="claude-3", messages=[], stream=True
        )
        async with stream as s:
            assert [c async for c in s] == raw._chunks

    asyncio.run(run())
    assert raw.closed
    assert len(mon.events) == 1


def test_wrap_openai_client_is_idempotent():
    """Double-wrapping must not ingest one host call twice (#427)."""
    mon = _Mon()
    calls = []

    def create(*a, **k):
        calls.append(k)
        return SimpleNamespace(usage=None)

    client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )
    wrap_openai_client(mon, client, episode_id="ep")
    first = client.chat.completions.create
    wrap_openai_client(mon, client, episode_id="ep")

    # The wrapper installed by the first wrap is still in place, not re-wrapped.
    assert client.chat.completions.create is first
    client.chat.completions.create(model="gpt-4o", messages=[], stream=False)
    assert len(calls) == 1
    assert len(mon.events) == 1


def test_wrap_anthropic_client_is_idempotent():
    mon = _Mon()
    calls = []

    def create(*a, **k):
        calls.append(k)
        return SimpleNamespace(usage=None)

    client = SimpleNamespace(messages=SimpleNamespace(create=create))
    wrap_anthropic_client(mon, client, episode_id="ep")
    first = client.messages.create
    wrap_anthropic_client(mon, client, episode_id="ep")

    assert client.messages.create is first
    client.messages.create(model="claude-3", messages=[], stream=False)
    assert len(calls) == 1
    assert len(mon.events) == 1
