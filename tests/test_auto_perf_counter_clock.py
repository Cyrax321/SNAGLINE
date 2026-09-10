"""Issue #155 follow-up: the ``snagline.auto`` wrappers measure on perf_counter.

PR #161 migrated every module in ``snagline.adapters`` off ``time.time()``
(issue #155): on Windows the wall clock advances in ~15.6 ms ticks on the
supported 3.10--3.12 interpreters, so a call shorter than one tick recorded
``latency_ms == 0.0`` and starved the CUSUM latency detector of usable
samples. The auto-instrumentation wrappers were never part of that audit and
kept reading ``time.time``.

These tests reuse the deterministic scripted-clock pattern from
``tests/adapters/test_perf_counter_clock.py``: script ``time.perf_counter``
itself and never touch ``time.time``. If a wrapper ever reads the wall clock,
the scripted readings are never consumed and the assertions fail, so the
regression cannot hide behind platform timing or Python-version clock
precision.
"""

from __future__ import annotations

import time
from typing import Any

import pytest

from snagline.auto.anthropic import wrap_client as wrap_anthropic
from snagline.auto.langchain import wrap_client as wrap_langchain
from snagline.auto.openai import wrap_client as wrap_openai


class _SpyMonitor:
    def __init__(self) -> None:
        self.events: list[Any] = []

    def ingest(self, event: Any) -> None:
        self.events.append(event)


def _script(*values: float):
    """Scripted clock: one value per expected reading, then fail loudly."""
    reads = iter(values)
    return lambda: next(reads)


# Sub-millisecond deltas built from binary-exact floats (mirrors the adapters'
# regression tests): 2**-10 seconds is exactly representable, so the asserted
# latency_ms holds bit-for-bit on every platform.
T0 = 1024.0
SUB_MS_S = 2**-10  # 0.9765625 ms


def test_openai_auto_wrapper_preserves_sub_ms_latency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Read order per successful call: start, then one reading in _emit shared
    # by the latency delta and the event timestamp.
    monkeypatch.setattr(time, "perf_counter", _script(T0, T0 + SUB_MS_S))
    mon = _SpyMonitor()

    class _Completions:
        def create(self, **kw: Any) -> str:
            return "ok"

    class _Client:
        chat = type("C", (), {"completions": _Completions()})()

    client = wrap_openai(mon, _Client())
    client.chat.completions.create(model="gpt-4o", messages=[{"role": "user"}])

    e = mon.events[0]
    assert e.latency_ms == pytest.approx(SUB_MS_S * 1000.0, rel=1e-12)
    assert e.latency_ms > 0.0  # exactly what the Windows tick quantized to zero
    assert e.timestamp == T0 + SUB_MS_S


def test_anthropic_auto_wrapper_preserves_sub_ms_latency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(time, "perf_counter", _script(T0, T0 + SUB_MS_S))
    mon = _SpyMonitor()

    class _Messages:
        def create(self, **kw: Any) -> str:
            return "ok"

    client = wrap_anthropic(mon, type("C", (), {"messages": _Messages()})())
    client.messages.create(model="claude", messages=[{"role": "user"}])

    e = mon.events[0]
    assert e.latency_ms == pytest.approx(SUB_MS_S * 1000.0, rel=1e-12)
    assert e.latency_ms > 0.0
    assert e.timestamp == T0 + SUB_MS_S


def test_langchain_auto_wrapper_preserves_sub_ms_latency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(time, "perf_counter", _script(T0, T0 + SUB_MS_S))
    mon = _SpyMonitor()

    class _Chain:
        def invoke(self, *args: Any, **kw: Any) -> str:
            return "ok"

    client = wrap_langchain(mon, _Chain())
    client.invoke({"input": "hi"})

    e = mon.events[0]
    assert e.latency_ms == pytest.approx(SUB_MS_S * 1000.0, rel=1e-12)
    assert e.latency_ms > 0.0
    assert e.timestamp == T0 + SUB_MS_S


def test_openai_auto_wrapper_error_path_reads_perf_counter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The finally-emitted event on a raising call must also come from the
    # monotonic clock: a wall clock that stepped backwards mid-call would
    # fabricate a negative latency (issue #155's second failure mode).
    monkeypatch.setattr(time, "perf_counter", _script(T0, T0 + SUB_MS_S))
    mon = _SpyMonitor()

    class _Boom:
        def create(self, **kw: Any) -> str:
            raise RuntimeError("boom")

    client = wrap_openai(
        mon, type("C", (), {"chat": type("B", (), {"completions": _Boom()})()})()
    )
    with pytest.raises(RuntimeError):
        client.chat.completions.create(model="gpt-4o", messages=[{"role": "user"}])

    e = mon.events[0]
    assert e.error is True
    assert e.latency_ms == pytest.approx(SUB_MS_S * 1000.0, rel=1e-12)
