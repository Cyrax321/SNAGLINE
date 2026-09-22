"""Issue #242: auto wrappers must defer emission to stream exhaustion.

With ``stream=True``, ``create()`` returns almost immediately, so emitting
in a ``finally`` at return time recorded a ~0ms success before the first
chunk arrived -- and the raw stream came back unwrapped, so mid-iteration
failures were never observed.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from snagline.auto import anthropic as anth
from snagline.auto import openai as oai
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


# --- Context-manager form (issue #335) ---------------------------------------
# ``with stream as s:`` is the streaming idiom both SDKs document. Implicit
# special-method lookup for ``with`` resolves on the type, never through
# ``__getattr__``, so the wrappers' attribute proxying could not supply the
# protocol: the form raised TypeError and emitted zero events -- a monitored
# call that was not monitored at all.


def test_openai_stream_context_manager_form() -> None:
    mon = _SpyMonitor()
    inner = _SyncStream(["a", "b"])
    client = wrap_openai(mon, _openai_client(inner))
    with client.chat.completions.create(model="m", messages=[], stream=True) as out:
        assert mon.events == [], "__enter__ must not emit early"
        assert list(out) == ["a", "b"]
    # Iteration already emitted at exhaustion; __exit__ only closes.
    assert inner.closed, "__exit__ must close the underlying stream"
    assert len(mon.events) == 1
    assert mon.events[0].error is False


def test_openai_stream_context_manager_emits_when_block_skips_iteration() -> None:
    mon = _SpyMonitor()
    inner = _SyncStream(["a", "b"])
    client = wrap_openai(mon, _openai_client(inner))
    with client.chat.completions.create(model="m", messages=[], stream=True) as out:
        next(out)  # exhaust neither the stream nor the wrapper
    assert inner.closed
    assert len(mon.events) == 1, (
        "__exit__ must emit once when nothing iterated to the end"
    )
    assert mon.events[0].error is False


def test_openai_stream_context_manager_propagates_block_errors() -> None:
    # An exception escaping the ``with`` body is the observed call's visible
    # outcome: the stream never finished, so __exit__ must not let close()
    # record a clean success. The block error still propagates (__exit__
    # returns False), and the emit is a no-op when the stream already
    # emitted at exhaustion.
    mon = _SpyMonitor()
    inner = _SyncStream(["a"])
    client = wrap_openai(mon, _openai_client(inner))
    with pytest.raises(RuntimeError, match="caller boom"):
        with client.chat.completions.create(model="m", messages=[], stream=True):
            raise RuntimeError("caller boom")
    assert inner.closed
    assert len(mon.events) == 1
    assert mon.events[0].error is True
    assert mon.events[0].error_type == "RuntimeError"


def test_anthropic_stream_context_manager_form() -> None:
    mon = _SpyMonitor()
    inner = _SyncStream(["x", "y"])
    client = wrap_anthropic(mon, _anthropic_client(inner))
    with client.messages.create(model="m", messages=[], stream=True) as out:
        assert mon.events == []
        assert list(out) == ["x", "y"]
    assert inner.closed
    assert len(mon.events) == 1
    assert mon.events[0].error is False


def test_anthropic_stream_context_manager_emits_on_block_error() -> None:
    mon = _SpyMonitor()
    inner = _SyncStream(["x"])
    client = wrap_anthropic(mon, _anthropic_client(inner))
    with pytest.raises(RuntimeError):
        with client.messages.create(model="m", messages=[], stream=True):
            raise RuntimeError("caller boom")
    assert inner.closed
    assert len(mon.events) == 1
    assert mon.events[0].error is True
    assert mon.events[0].error_type == "RuntimeError"


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


# --- async context-manager form (issue #335) ---------------------------------


@pytest.mark.parametrize("module", [oai, anth], ids=["openai", "anthropic"])
def test_async_stream_context_manager_form_is_observed(module) -> None:
    """``async with (await create(...)) as s:`` -- the async twin. ``__getattr__``
    never participates in dunder lookup, so without real ``__aenter__`` /
    ``__aexit__`` on the wrapper this raised TypeError and emitted nothing.
    """

    async def go() -> None:
        mon = _SpyMonitor()
        inner = _AsyncStream(["a", "b"])

        async def _create(**kw: Any) -> Any:
            return inner

        wrapped = module._wrap_one(mon, _create, "create")
        out = await wrapped(model="m", messages=[], stream=True)
        async with out as s:
            assert mon.events == [], "__aenter__ must not emit early"
            assert [c async for c in s] == ["a", "b"]
        # Iteration already emitted at exhaustion; __aexit__ only closes.
        assert inner.closed, "__aexit__ must close the underlying stream"
        assert len(mon.events) == 1
        assert mon.events[0].error is False

    asyncio.run(go())


@pytest.mark.parametrize("module", [oai, anth], ids=["openai", "anthropic"])
def test_async_stream_context_manager_emits_when_block_skips_iteration(module) -> None:
    """``__aexit__`` must still emit once when the block never finishes the
    stream -- otherwise a caller that opens a stream and abandons it inside the
    context manager records nothing."""

    async def go() -> None:
        mon = _SpyMonitor()
        inner = _AsyncStream(["a", "b"])

        async def _create(**kw: Any) -> Any:
            return inner

        wrapped = module._wrap_one(mon, _create, "create")
        out = await wrapped(model="m", messages=[], stream=True)
        async with out as s:
            await anext(s)  # exhaust neither the stream nor the wrapper
        assert inner.closed
        assert len(mon.events) == 1
        assert mon.events[0].error is False

    asyncio.run(go())


@pytest.mark.parametrize("module", [oai, anth], ids=["openai", "anthropic"])
def test_async_stream_context_manager_records_block_error(module) -> None:
    """The async twin of the sync block-error case: an exception escaping the
    body must be recorded as the call's failure, not a clean success."""

    async def go() -> None:
        mon = _SpyMonitor()
        inner = _AsyncStream(["a", "b"])

        async def _create(**kw: Any) -> Any:
            return inner

        wrapped = module._wrap_one(mon, _create, "create")
        out = await wrapped(model="m", messages=[], stream=True)
        with pytest.raises(RuntimeError, match="caller boom"):
            async with out as s:
                await anext(s)
                raise RuntimeError("caller boom")
        assert inner.closed
        assert len(mon.events) == 1
        assert mon.events[0].error is True
        assert mon.events[0].error_type == "RuntimeError"

    asyncio.run(go())


def test_openai_stream_context_manager_error_after_exhaustion_stays_success() -> None:
    # When the stream already emitted at exhaustion the error emit is a
    # no-op: the call genuinely completed, so a caller exception raised by
    # post-processing after the loop must not flip it to a failure.
    mon = _SpyMonitor()
    inner = _SyncStream(["a"])
    client = wrap_openai(mon, _openai_client(inner))
    with pytest.raises(RuntimeError, match="after"):
        with client.chat.completions.create(model="m", messages=[], stream=True) as out:
            assert list(out) == ["a"]
            raise RuntimeError("after exhaustion")
    assert inner.closed
    assert len(mon.events) == 1
    assert mon.events[0].error is False
