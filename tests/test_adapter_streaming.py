"""Smoke tests for deferred streaming wrappers."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

from snagline.adapters.anthropic import wrap_anthropic_client
from snagline.adapters.openai import wrap_openai_client


class _Sink:
    def __init__(self):
        self.risks = []
        self.events = []

    def emit(self, risk):
        self.risks.append(risk)


class _Mon:
    def __init__(self):
        self.events = []

    def ingest(self, event):
        self.events.append(event)


def _mock_openai_stream(chunks):
    class S:
        def __init__(self, chunks):
            self._chunks = chunks
            self.closed = False

        def __iter__(self):
            return iter(self._chunks)

        def close(self):
            self.closed = True

    return S(chunks)


def _mock_anthropic_stream(chunks):
    class S:
        def __init__(self, chunks):
            self._chunks = chunks

        def __iter__(self):
            return iter(self._chunks)

    return S(chunks)


def test_openai_sync_stream_defers_until_exhaustion():
    mon = _Mon()

    def fake_create(*args, **kwargs):
        # simulate OpenAI returning a stream when stream=True
        assert kwargs.get("stream") is True
        return _mock_openai_stream(
            [
                SimpleNamespace(usage=None),
                SimpleNamespace(
                    usage=SimpleNamespace(prompt_tokens=5, completion_tokens=7)
                ),
            ]
        )

    client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=fake_create))
    )
    wrap_openai_client(mon, client, episode_id="ep-stream")
    stream = client.chat.completions.create(
        model="gpt-4o", messages=[{"role": "user", "content": "hi"}], stream=True
    )
    # not yet ingested
    assert len(mon.events) == 0
    chunks = list(stream)
    assert len(chunks) == 2
    # now ingested once after exhaustion
    assert len(mon.events) == 1
    ev = mon.events[0]
    assert ev.tool_name == "gpt-4o"
    assert ev.tokens_in == 5
    assert ev.tokens_out == 7
    assert ev.latency_ms is not None


def test_openai_sync_stream_error_during_iteration():
    mon = _Mon()

    def fake_create(*args, **kwargs):
        class ErrStream:
            def __iter__(self):
                return self

            def __next__(self):
                raise RuntimeError("boom")

        return ErrStream()

    client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=fake_create))
    )
    wrap_openai_client(mon, client, episode_id="ep-stream")
    stream = client.chat.completions.create(model="gpt-4o", messages=[], stream=True)
    try:
        next(iter(stream))
        assert False, "should raise"
    except RuntimeError:
        pass
    assert len(mon.events) == 1
    assert mon.events[0].error is True


def test_openai_sync_non_stream_still_immediate():
    mon = _Mon()

    def fake_create(*args, **kwargs):
        return SimpleNamespace(
            usage=SimpleNamespace(prompt_tokens=1, completion_tokens=2)
        )

    client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=fake_create))
    )
    wrap_openai_client(mon, client, episode_id="ep")
    res = client.chat.completions.create(model="gpt-4o", messages=[], stream=False)
    assert res is not None
    assert len(mon.events) == 1
    assert mon.events[0].error is False


def test_anthropic_sync_stream_defers():
    mon = _Mon()

    def fake_create(*args, **kwargs):
        return _mock_anthropic_stream(
            [
                SimpleNamespace(usage=None),
                SimpleNamespace(usage=SimpleNamespace(input_tokens=3, output_tokens=4)),
            ]
        )

    client = SimpleNamespace(messages=SimpleNamespace(create=fake_create))
    wrap_anthropic_client(mon, client, episode_id="ep-a")
    stream = client.messages.create(model="claude-3", messages=[], stream=True)
    assert len(mon.events) == 0
    list(stream)
    assert len(mon.events) == 1
    assert mon.events[0].tokens_in == 3


def test_openai_async_stream_defers():
    mon = _Mon()

    async def fake_create(*args, **kwargs):
        class AStream:
            def __init__(self, chunks):
                self._chunks = chunks
                self._idx = 0

            def __aiter__(self):
                return self

            async def __anext__(self):
                if self._idx >= len(self._chunks):
                    raise StopAsyncIteration
                c = self._chunks[self._idx]
                self._idx += 1
                return c

        return AStream(
            [
                SimpleNamespace(usage=None),
                SimpleNamespace(
                    usage=SimpleNamespace(prompt_tokens=2, completion_tokens=3)
                ),
            ]
        )

    client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=fake_create))
    )
    wrap_openai_client(mon, client, episode_id="ep-async")

    # fake_create is async, so wrapper will be async
    async def run():
        stream = await client.chat.completions.create(
            model="gpt-4o", messages=[], stream=True
        )
        assert len(mon.events) == 0
        out = []
        async for c in stream:  # type: ignore
            out.append(c)
        assert len(out) == 2
        assert len(mon.events) == 1

    asyncio.run(run())
