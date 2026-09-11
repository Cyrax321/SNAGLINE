"""Issue #242: auto wrappers must defer emission to stream exhaustion.

With ``stream=True``, ``create()`` returns almost immediately, so emitting
in a ``finally`` at return time recorded a ~0ms success before the first
chunk arrived -- and the raw stream came back unwrapped, so mid-iteration
failures were never observed.
"""

from __future__ import annotations

import asyncio
from typing import Any

from snagline.auto.anthropic import wrap_client as wrap_anthropic
from snagline.auto.openai import wrap_client as wrap_openai


class _SpyMonitor:
    def __init__(self) -> None:
        self.events: list[Any] = []

    def ingest(self, event: Any) -> None:
        self.events.append(event)


class _SyncStream:
    def __init__(self, chunks: list[Any], fail_after: int | None = None) -> None:
        self._chunks = chunks
        self._fail_after = fail_after
        self._yielded = 0
        self.closed = False

    def __iter__(self):
        return self

    def __next__(self):
        if self._fail_after is not None and self._yielded >= self._fail_after:
            raise RuntimeError("mid-iteration boom")
        if not self._chunks:
            raise StopIteration
        self._yielded += 1
        return self._chunks.pop(0)

    def close(self) -> None:
        self.closed = True


class _AsyncStream:
    def __init__(self, chunks: list[Any]) -> None:
        self._chunks = chunks
        self.closed = False

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self._chunks:
            raise StopAsyncIteration
        return self._chunks.pop(0)

    async def aclose(self) -> None:
        self.closed = True


def _openai_client(stream: Any):
    class _Completions:
        def create(self, **kw: Any) -> Any:
            return stream

    class _Client:
        chat = type("C", (), {"completions": _Completions()})()

    return _Client()


def _anthropic_client(stream: Any):
    class _Messages:
        def create(self, **kw: Any) -> Any:
            return stream

    return type("C", (), {"messages": _Messages()})()


def test_openai_stream_emits_nothing_until_exhausted() -> None:
    mon = _SpyMonitor()
    client = wrap_openai(mon, _openai_client(_SyncStream(["a", "b"])))
    out = client.chat.completions.create(model="m", messages=[], stream=True)
    assert mon.events == [], "emit at stream-open records a ~0ms false success"
    assert list(out) == ["a", "b"]
    assert len(mon.events) == 1
    assert mon.events[0].error is False
    assert mon.events[0].latency_ms >= 0.0


def test_openai_stream_failure_is_observed() -> None:
    mon = _SpyMonitor()
    client = wrap_openai(mon, _openai_client(_SyncStream(["a"], fail_after=1)))
    out = client.chat.completions.create(model="m", messages=[], stream=True)
    assert next(out) == "a"
    try:
        next(out)
        raise AssertionError("stream should have raised")
    except RuntimeError:
        pass
    assert len(mon.events) == 1
    assert mon.events[0].error is True
    assert mon.events[0].error_type == "RuntimeError"


def test_openai_stream_close_emits_once_and_delegates() -> None:
    mon = _SpyMonitor()
    inner = _SyncStream(["a", "b"])
    client = wrap_openai(mon, _openai_client(inner))
    out = client.chat.completions.create(model="m", messages=[], stream=True)
    out.close()
    assert inner.closed
    assert len(mon.events) == 1
    assert mon.events[0].error is False


def test_anthropic_stream_emits_nothing_until_exhausted() -> None:
    mon = _SpyMonitor()
    client = wrap_anthropic(mon, _anthropic_client(_SyncStream(["x"])))
    out = client.messages.create(model="m", messages=[], stream=True)
    assert mon.events == []
    assert list(out) == ["x"]
    assert len(mon.events) == 1
    assert mon.events[0].error is False


def test_non_stream_calls_still_emit_immediately() -> None:
    mon = _SpyMonitor()
    client = wrap_openai(mon, _openai_client({"ok": True}))
    assert client.chat.completions.create(model="m", messages=[]) == {"ok": True}
    assert len(mon.events) == 1


def test_async_stream_defers_to_exhaustion() -> None:
    async def go() -> None:
        mon = _SpyMonitor()
        writes: list[str] = []

        class _AStream:
            def __init__(self) -> None:
                self._n = 0

            def __aiter__(self):
                return self

            async def __anext__(self):
                self._n += 1
                if self._n > 2:
                    raise StopAsyncIteration
                writes.append(f"c{self._n}")
                return writes[-1]

        async def _create(**kw: Any) -> Any:
            return _AStream()

        from snagline.auto.openai import _wrap_one

        wrapped = _wrap_one(mon, _create, "openai.messages.create")
        out = await wrapped(model="m", messages=[], stream=True)
        assert mon.events == []
        seen = [c async for c in out]
        assert seen == ["c1", "c2"]
        assert len(mon.events) == 1
        assert mon.events[0].error is False

    asyncio.run(go())


def test_async_stream_close_emits() -> None:
    async def go() -> None:
        mon = _SpyMonitor()
        inner = _AsyncStream(["z"])

        async def _create(**kw: Any) -> Any:
            return inner

        from snagline.auto.openai import _wrap_one

        wrapped = _wrap_one(mon, _create, "openai.messages.create")
        out = await wrapped(model="m", messages=[], stream=True)
        await out.aclose()
        assert inner.closed
        assert len(mon.events) == 1

    asyncio.run(go())
