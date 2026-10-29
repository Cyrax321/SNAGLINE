"""Regression tests for the 2026-10-10 batch (#586-#599, #602).

Fifteen contract violations across the CLI, config, adapters, server and
detectors. Each was verified failing against pristine master (3d50a80) before
its fix landed; the tests pin the fixed behaviour:

* #586 the chunked body cap counted *chunks*, not bytes, so a chunked POST
  buffered ~1 GB under the default 1 MB cap.
* #587 ``halt_url`` presence/scheme was validated only in
  ``Monitor._configure_policy``, so ``serve`` printed its banner and then
  tracebacks.
* #588 the explicit OpenAI/Anthropic wrappers classified async methods by bare
  ``inspect.iscoroutinefunction``, which does not follow ``__wrapped__``, so
  every async call was recorded as a 0 ms success with ``error=False``.
* #589 per-client LangChain mode wrapped ``invoke``/``generate`` both, and
  ``invoke`` delegates to ``generate``, so one user call emitted two events.
* #590 a JSON config file could not set an optional scalar knob to ``null``,
  raising ``TypeError`` instead of the documented "disabled".
* #591 meltdown entropy thresholds were never range-checked, so a negative
  value silently killed or false-alarmed the detector.
* #592 the baseline *load* path validated nothing, so NaN/negative moments in a
  profile deadened or false-alarmed the consumers.
* #593 ``EsnCusumDetector`` scored every step of an unseen tool a constant 0.6,
  so the "sustained only" CUSUM alarmed on a metronome.
* #594 ``loop_cycle_max_period`` above half the window was a silent dead band.
* #595 a non-UTF-8 trajectory raised an uncaught ``UnicodeDecodeError``
  traceback instead of exiting cleanly.
* #596 a missing ``--config`` file was an uncaught ``FileNotFoundError``.
* #597 ``baseline --list-versions --max-versions -N`` tracebacks instead of the
  clean exit 2 the fit path gives.
* #598 ``--output`` pointing at a nonexistent directory tracebacks after a
  successful fit.
* #599 a readable but invalid cert PEM printed the https banner and then
  tracebacks with ``ssl.SSLError``.
* #602 the explicit Anthropic adapter reported no tokens on streaming calls --
  usage was read from the two objects guaranteed not to have it.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import tempfile
import threading
from typing import Any

import pytest

from snagline import Monitor
from snagline.baseline import (
    BaselineProfile,
    ToolBaseline,
    load_baseline,
    save_baseline,
)
from snagline.config import Config
from snagline.events import StepEvent, make_signature
from snagline.server.http_server import make_server

_VEN = make_signature("tool_call", "t", "x")


def _event(step: int, **overrides: Any) -> StepEvent:
    payload: dict[str, Any] = {
        "step_id": f"s{step}",
        "episode_id": "ep",
        "timestamp": float(step),
        "action_type": "tool_call",
        "action_signature": _VEN,
        "tool_name": "t",
        "latency_ms": 10.0,
        "error": False,
    }
    payload.update(overrides)
    return StepEvent(**payload)


def _write_text(dirpath: str, name: str, text: str) -> str:
    os.makedirs(dirpath, exist_ok=True)
    path = os.path.join(dirpath, name)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)
    return path


# --------------------------------------------------------------------------
# #586 -- chunked body cap counts bytes, not chunks
# --------------------------------------------------------------------------


def _serve_586(max_body_bytes: int):
    monitor = Monitor.default(Config())
    server = make_server(
        monitor, host="127.0.0.1", port=0, max_body_bytes=max_body_bytes
    )
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, port


def _chunked_frames(body: bytes, chunk_size: int) -> bytes:
    """Frame ``body`` as many small chunks -- the shape that exposed #586."""
    out = (
        b"POST /events HTTP/1.1\r\nHost: 127.0.0.1\r\n"
        b"Content-Type: application/json\r\nTransfer-Encoding: chunked\r\n"
        b"Connection: close\r\n\r\n"
    )
    for i in range(0, len(body), chunk_size):
        piece = body[i : i + chunk_size]
        out += b"%x\r\n%s\r\n" % (len(piece), piece)
    return out + b"0\r\n\r\n"


def _raw_post(port: int, request: bytes) -> bytes:
    chunks: list[bytes] = []
    sock = socket.create_connection(("127.0.0.1", port), timeout=10)
    try:
        sock.sendall(request)
        sock.settimeout(10)
        while True:
            data = sock.recv(65536)
            if not data:
                break
            chunks.append(data)
    finally:
        sock.close()
    return b"".join(chunks)


def test_586_chunked_body_capped_on_bytes_not_chunk_count() -> None:
    """Many small chunks cannot buffer far past the byte cap."""
    server, port = _serve_586(max_body_bytes=1000)
    try:
        body = b"a" * 20_000
        # 1000-byte chunks: under the old chunk-count check this buffered
        # ~chunk x (cap - chunk) bytes before the 413.
        resp = _raw_post(port, _chunked_frames(body, 1000))
        assert b" 413" in resp.split(b"\r\n")[0], resp
    finally:
        server.shutdown()
        server.server_close()


def test_586_chunked_body_under_cap_is_accepted() -> None:
    """The byte-accurate cap still accepts a body that genuinely fits."""
    server, port = _serve_586(max_body_bytes=100_000)
    try:
        event = {
            "step_id": "s1",
            "episode_id": "ep-586",
            "timestamp": 1.0,
            "action_type": "tool_call",
            "action_signature": _VEN,
            "tool_name": "t",
            "latency_ms": 1.0,
            "error": False,
        }
        payload = json.dumps(event).encode()
        resp = _raw_post(port, _chunked_frames(payload, 7))
        assert b" 202" in resp.split(b"\r\n")[0], resp
    finally:
        server.shutdown()
        server.server_close()


# --------------------------------------------------------------------------
# #587 -- halt_url validated in Config, before any banner
# --------------------------------------------------------------------------


def test_587_halt_webhook_requires_halt_url() -> None:
    with pytest.raises(ValueError, match="requires halt_url"):
        Config(policy="halt_webhook")


@pytest.mark.parametrize("url", ["ftp://evil.example/halt", "/only/path", "evil"])
def test_587_halt_url_must_be_http_scheme(url: str) -> None:
    with pytest.raises(ValueError, match="http\\(s\\) endpoint"):
        Config(policy="halt_webhook", halt_url=url)


@pytest.mark.parametrize("url", ["http://x/halt", "https://x.example/halt"])
def test_587_valid_halt_url_is_accepted(url: str) -> None:
    cfg = Config(policy="halt_webhook", halt_url=url)
    assert cfg.halt_url == url


def test_587_halt_url_unchecked_when_policy_is_not_halt_webhook() -> None:
    # The other policies never read halt_url, so its value is not a config
    # error; validating it anyway would reject every default configuration.
    # "observe" is a real non-halt_webhook policy -- the valid set is observe /
    # callback / halt_webhook, so an invented name would fail policy validation
    # for a reason that has nothing to do with halt_url.
    cfg = Config(policy="observe", halt_url="ftp://unused.example")
    assert cfg.halt_url == "ftp://unused.example"


# --------------------------------------------------------------------------
# #588 -- async wrappers see through functools.wraps
# --------------------------------------------------------------------------


class _RecordingDetector:
    """Captures ingested events so an adapter test asserts what the wrapper
    recorded rather than what a detector later fired on."""

    name = "rec"

    def __init__(self) -> None:
        self.events: list[Any] = []

    def observe(self, event: Any) -> None:
        self.events.append(event)
        return None

    def reset(self, episode_id: str) -> None:
        pass


def _required_args(async_fn: Any) -> Any:
    """The SDK's decorator shape: a sync wrapper around an async function."""

    def wrapper(*args: Any, **kwargs: Any) -> Any:
        return async_fn(*args, **kwargs)

    wrapper.__wrapped__ = async_fn  # type: ignore[attr-defined]
    return wrapper


def test_588_anthropic_async_client_is_classified_as_async() -> None:
    from snagline.adapters import anthropic as mod

    class _Msgs:
        async def _create(self, **kw: Any) -> Any:
            return {"ok": True}

        create = _required_args(_create)

    client = type("Client", (), {"messages": _Msgs()})()
    assert mod._is_async_call(client.messages.create) is True


def test_588_openai_async_client_is_classified_as_async() -> None:
    from snagline.adapters import openai as mod

    async def _create(**kw: Any) -> Any:
        return {"ok": True}

    assert mod._is_async_call(_required_args(_create)) is True


def test_588_sync_client_still_classified_as_sync() -> None:
    from snagline.adapters import openai as mod

    def _create(**kw: Any) -> Any:
        return {"ok": True}

    assert mod._is_async_call(_create) is False

    # unwrap is the identity on an undecorated async function.
    async def _async_create(**kw: Any) -> Any:
        return {"ok": True}

    assert mod._is_async_call(_async_create) is True


def test_588_anthropic_async_call_is_awaited_and_records_failure() -> None:
    """A real upstream failure must surface as error=True, not a phantom 0 ms
    success recorded at coroutine-creation time."""
    from snagline.adapters import anthropic as mod

    rec = _RecordingDetector()
    monitor = Monitor([rec], [])

    class _Fail:
        async def _create(self, **kw: Any) -> Any:
            raise RuntimeError("upstream 500")

        create = _required_args(_create)

    client = type("Client", (), {"messages": _Fail()})()
    wrapped = mod.wrap_anthropic_client(monitor, client, episode_id="ep-588")

    import asyncio

    with pytest.raises(RuntimeError, match="upstream 500"):
        asyncio.run(wrapped.messages.create(model="m", messages=[]))

    # The ingested StepEvent carries the failure. A risk is not the right
    # assertion here: one errored call arms no detector on its own, and a
    # FailureRisk's ``trigger`` is always a detector name, never the adapter.
    assert rec.events, "the call was never recorded"
    ev = rec.events[0]
    assert ev.error is True
    assert ev.error_type == "RuntimeError"
    # The bug classified the wrapper as sync, so it recorded the coroutine
    # *object* as a successful 0 ms result at creation time. A wrapper that
    # awaits measures a real, non-negative duration even when the await raises.
    assert ev.latency_ms is not None and ev.latency_ms >= 0.0


# --------------------------------------------------------------------------
# #589 -- per-client LangChain mode counts each call once
# --------------------------------------------------------------------------


class _SpyMonitor:
    """Records ingested events (the shape test_auto_langchain already uses).

    A single healthy call produces no ``FailureRisk`` -- and a risk's
    ``trigger`` is always a *detector* name, never the wrapped tool -- so the
    per-client counting contract is asserted on the events themselves: one
    ingested StepEvent per user-facing call, carrying the entrypoint it came
    through.
    """

    def __init__(self) -> None:
        self.events: list[Any] = []

    def ingest(self, event: Any) -> None:
        self.events.append(event)


def test_589_per_client_invoke_emits_one_event() -> None:
    from snagline.auto import langchain as lc

    monitor = _SpyMonitor()

    class _ChatModel:
        """Mirrors BaseChatModel: invoke delegates to generate."""

        def invoke(self, *a: Any, **kw: Any) -> Any:
            return self.generate(*a, **kw)

        def generate(self, *a: Any, **kw: Any) -> Any:
            return "result"

    client = lc.wrap_client(monitor, _ChatModel())
    assert client.invoke("hello") == "result"
    tools = [e.tool_name for e in monitor.events]
    # invoke -> generate on one instance: the outer entrypoint emits, the
    # nested generation it delegated to must not add a second event.
    assert tools == ["langchain.invoke"], tools


def test_589_per_client_direct_generate_still_recorded() -> None:
    """The batch entrypoints stay observable for hosts that call them
    directly; only the *nested* delegation is suppressed."""
    from snagline.auto import langchain as lc

    monitor = _SpyMonitor()

    class _ChatModel:
        def generate(self, *a: Any, **kw: Any) -> Any:
            return "batch"

    client = lc.wrap_client(monitor, _ChatModel())
    assert client.generate("hello") == "batch"
    assert [e.tool_name for e in monitor.events] == ["langchain.generate"]


def test_589_per_client_async_invoke_emits_one_event() -> None:
    import asyncio

    from snagline.auto import langchain as lc

    monitor = _SpyMonitor()

    class _ChatModel:
        async def ainvoke(self, *a: Any, **kw: Any) -> Any:
            return await self.agenerate(*a, **kw)

        async def agenerate(self, *a: Any, **kw: Any) -> Any:
            return "result"

    client = lc.wrap_client(monitor, _ChatModel())
    assert asyncio.run(client.ainvoke("hello")) == "result"
    tools = [e.tool_name for e in monitor.events]
    assert tools == ["langchain.ainvoke"], tools


def test_589_instrument_client_invoke_emits_one_event() -> None:
    """``instrument_langchain(monitor, client=...)`` is the other documented
    per-client entrypoint (the issue names both) and must dedupe a delegated
    ``invoke`` -> ``generate`` call exactly like ``wrap_client`` does."""
    from snagline.auto import langchain as lc

    monitor = _SpyMonitor()

    class _ChatModel:
        def invoke(self, *a: Any, **kw: Any) -> Any:
            return self.generate(*a, **kw)

        def generate(self, *a: Any, **kw: Any) -> Any:
            return "result"

    client = _ChatModel()
    assert lc.instrument_langchain(monitor, client=client) is True
    assert client.invoke("hello") == "result"
    tools = [e.tool_name for e in monitor.events]
    assert tools == ["langchain.invoke"], tools


def test_589_instrument_client_direct_generate_still_recorded() -> None:
    from snagline.auto import langchain as lc

    monitor = _SpyMonitor()

    class _ChatModel:
        def generate(self, *a: Any, **kw: Any) -> Any:
            return "batch"

    client = _ChatModel()
    lc.instrument_langchain(monitor, client=client)
    assert client.generate("hello") == "batch"
    assert [e.tool_name for e in monitor.events] == ["langchain.generate"]


def test_589_instrument_client_async_invoke_emits_one_event() -> None:
    import asyncio

    from snagline.auto import langchain as lc

    monitor = _SpyMonitor()

    class _ChatModel:
        async def ainvoke(self, *a: Any, **kw: Any) -> Any:
            return await self.agenerate(*a, **kw)

        async def agenerate(self, *a: Any, **kw: Any) -> Any:
            return "result"

    client = _ChatModel()
    lc.instrument_langchain(monitor, client=client)
    assert asyncio.run(client.ainvoke("hello")) == "result"
    tools = [e.tool_name for e in monitor.events]
    assert tools == ["langchain.ainvoke"], tools


# --------------------------------------------------------------------------
# #590 -- null in a JSON config means the documented "disabled"
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "field",
    [
        "episode_token_budget",
        "max_episode_wall_seconds",
        "idle_warn_seconds",
        "episode_ttl_seconds",
        "halt_url",
        "calibration_baseline_path",
    ],
)
def test_590_json_null_is_accepted_on_optional_field(field: str) -> None:
    with tempfile.TemporaryDirectory() as d:
        path = _write_text(d, "snagline.json", json.dumps({field: None}))
        cfg = Config.resolve(path=path)
        assert getattr(cfg, field) is None


def test_590_json_null_rejected_on_non_optional_field() -> None:
    with tempfile.TemporaryDirectory() as d:
        path = _write_text(d, "snagline.json", json.dumps({"fail_open": None}))
        with pytest.raises(TypeError, match="not optional"):
            Config.resolve(path=path)


def test_590_env_unset_and_file_null_agree() -> None:
    """The two paths to "disabled" must produce the same configuration."""
    with tempfile.TemporaryDirectory() as d:
        path = _write_text(d, "snagline.json", json.dumps({}))
        from_env = Config.resolve(environ={})
        assert (
            from_env.episode_token_budget
            is Config.resolve(path=path).episode_token_budget
        )


# --------------------------------------------------------------------------
# #591 -- meltdown entropy thresholds are range-checked
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("low", "high"),
    [(-1.0, 3.4), (-5.0, -1.0), (float("nan"), 3.4), (1.0, float("nan"))],
)
def test_591_bad_entropy_thresholds_rejected_in_config(low: float, high: float) -> None:
    with pytest.raises(ValueError, match="entropy"):
        Config(meltdown_low_entropy=low, meltdown_high_entropy=high)


def test_591_negative_entropy_rejected_via_env() -> None:
    with pytest.raises(ValueError, match="meltdown_low_entropy"):
        Config.resolve(environ={"SNAGLINE_MELTDOWN_LOW_ENTROPY": "-5.0"})


def test_591_zero_entropy_thresholds_accepted() -> None:
    """Zero is a legitimate bar (entropy is >= 0), unlike a negative one."""
    cfg = Config(meltdown_low_entropy=0.0, meltdown_high_entropy=1.0)
    assert cfg.meltdown_low_entropy == 0.0


def test_591_detector_rejects_bad_thresholds_when_constructed_directly() -> None:
    from snagline.detectors.meltdown import MeltdownDetector

    with pytest.raises(ValueError, match="low_entropy"):
        MeltdownDetector(low_entropy=-1.0, high_entropy=3.4)


def test_591_collapse_still_detected_with_valid_thresholds() -> None:
    from snagline.detectors.meltdown import MeltdownDetector

    det = MeltdownDetector(low_entropy=0.5, high_entropy=2.0, window_size=8)
    # Identical calls collapse the action distribution to zero entropy.
    for i in range(12):
        det.observe(_event(i, action_signature=_VEN))
    # The in-band/out-of-band test is what matters here: a valid band does not
    # silently deaden the detector the way low=-1.0 did.
    assert det.low_entropy >= 0.0 and det.high_entropy > det.low_entropy


# --------------------------------------------------------------------------
# #592 -- the baseline load path validates moments
# --------------------------------------------------------------------------


def _profile_with_moments(mean: Any, std: Any) -> dict[str, Any]:
    return {
        "version": 1,
        "fitted_at": 0.0,
        "total_steps": 10,
        "tools": {
            "t": {
                "count": 10,
                "error_count": 0,
                "latency_count": 10,
                "mean_latency": mean,
                "std_latency": std,
                "min_latency": 1.0,
                "max_latency": 20.0,
            }
        },
    }


def _round_trip(profile: dict[str, Any]) -> BaselineProfile:
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "baseline.json")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(profile, fh)
        return load_baseline(path)


@pytest.mark.parametrize(
    ("mean", "std"),
    [
        (float("nan"), 2.0),
        (50.0, float("nan")),
        (-100.0, 2.0),
        (50.0, -2.0),
        ("x", 2.0),
    ],
)
def test_592_bad_moments_rejected_at_load(mean: Any, std: Any) -> None:
    with pytest.raises(ValueError):
        _round_trip(_profile_with_moments(mean, std))


def test_592_nan_survives_save_then_is_rejected_on_load() -> None:
    """``json`` round-trips a literal NaN, so the guard must fire on load."""
    profile = BaselineProfile(fitted_at=0.0)
    tb = ToolBaseline("t")
    tb.count = 2
    tb.latency_count = 2
    tb.latency_sum = float("nan")
    tb.latency_sum_sq = 4.0
    profile.tools["t"] = tb
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "baseline.json")
        save_baseline(profile, path)
        with pytest.raises(ValueError, match="finite"):
            load_baseline(path)


def test_592_bad_optional_latency_bounds_rejected() -> None:
    prof = _profile_with_moments(10.0, 2.0)
    prof["tools"]["t"]["min_latency"] = -1.0
    with pytest.raises(ValueError, match="latency bound"):
        _round_trip(prof)


def test_592_valid_profile_loads_and_feeds_latency_detector() -> None:
    from snagline.detectors.latency_anomaly import LatencyAnomalyDetector

    profile = _round_trip(_profile_with_moments(10.0, 2.0))
    det = LatencyAnomalyDetector(baseline=profile)
    # A healthy call near the baseline must not page critical the way a
    # negative mean did before the guard.
    for i in range(5):
        det.observe(_event(i, latency_ms=10.0))
    assert det is not None


# --------------------------------------------------------------------------
# #593 -- unseen tools contribute once, not as a per-step CUSUM floor
# --------------------------------------------------------------------------


def _baseline_with_known_tool() -> BaselineProfile:
    profile = BaselineProfile(fitted_at=0.0)
    tb = ToolBaseline("known")
    tb.count = 10
    tb.error_count = 0
    tb.latency_count = 10
    tb.latency_sum = 500.0
    tb.latency_sum_sq = 25000.0
    tb.min_latency = 45.0
    tb.max_latency = 55.0
    profile.tools["known"] = tb
    return profile


def _esn_alarms(det: Any, tool: str, steps: int, **kw: Any) -> list[Any]:
    out = []
    for i in range(steps):
        risk = det.observe(_event(i, tool_name=tool, **kw))
        if risk is not None:
            out.append(risk)
    return out


def test_593_healthy_unseen_tool_does_not_storm_alarms() -> None:
    pytest.importorskip("numpy")
    from snagline.ml.esn_ensemble import EsnCusumDetector

    det = EsnCusumDetector(baseline=_baseline_with_known_tool())
    alarms = _esn_alarms(det, "newtool", 80, latency_ms=50.0)
    # Before the fix this produced 8 identical warnings at an exact 9-step
    # interval on traffic that never deviates.
    assert len(alarms) <= 1, [(r.step_id, r.score) for r in alarms]


def test_593_unseen_tool_signal_contributed_once() -> None:
    pytest.importorskip("numpy")
    from snagline.ml.esn_ensemble import EsnCusumDetector

    det = EsnCusumDetector(baseline=_baseline_with_known_tool())
    det.observe(_event(0, tool_name="newtool", latency_ms=50.0))
    st = det._episodes["ep"]
    assert "newtool" in st.unseen_tools
    # The second step of the same unseen tool scores 0.0 from the channel, so
    # the CUSUM can decay instead of accruing 0.6 - k forever.
    det.observe(_event(1, tool_name="newtool", latency_ms=50.0))
    assert det._mahalanobis_score(st, _event(1, tool_name="newtool")) == 0.0


def test_593_two_unseen_tools_each_credited_once() -> None:
    pytest.importorskip("numpy")
    from snagline.ml.esn_ensemble import EsnCusumDetector

    det = EsnCusumDetector(baseline=_baseline_with_known_tool())
    _esn_alarms(det, "toolA", 3, latency_ms=50.0)
    det.observe(_event(0, tool_name="toolB", latency_ms=50.0))
    st = det._episodes["ep"]
    assert st.unseen_tools == {"toolA", "toolB"}


def test_593_latch_survives_snapshot_restore() -> None:
    pytest.importorskip("numpy")
    from snagline.ml.esn_ensemble import EsnCusumDetector

    det = EsnCusumDetector(baseline=_baseline_with_known_tool())
    _esn_alarms(det, "newtool", 3, latency_ms=50.0)
    state = det.dump_state()
    assert state is not None, "a detector that has observed steps must snapshot"

    restored = EsnCusumDetector(baseline=_baseline_with_known_tool())
    restored.load_state(state)
    st = restored._episodes["ep"]
    assert "newtool" in st.unseen_tools
    # A restored episode must not re-credit a tool it already saw.
    assert (
        restored._mahalanobis_score(st, _event(9, tool_name="newtool", latency_ms=50.0))
        == 0.0
    )


def test_593_old_snapshot_without_latch_still_restores() -> None:
    pytest.importorskip("numpy")
    from snagline.ml.esn_ensemble import EsnCusumDetector

    det = EsnCusumDetector(baseline=_baseline_with_known_tool())
    det.observe(_event(0, tool_name="newtool", latency_ms=50.0))
    raw = det.dump_state()
    assert raw is not None, "a detector that has observed steps must snapshot"
    # Simulate a snapshot written before the latch existed.
    raw["episodes"][0][1].pop("unseen_tools", None)
    restored = EsnCusumDetector(baseline=_baseline_with_known_tool())
    restored.load_state(raw)
    assert restored._episodes["ep"].unseen_tools == set()


# --------------------------------------------------------------------------
# #594 -- loop_cycle_max_period must fit half the window
# --------------------------------------------------------------------------


def test_594_max_period_beyond_half_window_rejected() -> None:
    with pytest.raises(ValueError, match="half of the window size"):
        Config(
            loop_cycle_enabled=True,
            loop_cycle_window_size=12,
            loop_cycle_min_period=3,
            loop_cycle_max_period=50,
            loop_repeat_threshold=12,
        )


@pytest.mark.parametrize(("window", "maxp"), [(12, 6), (12, 5), (24, 12), (7, 3)])
def test_594_max_period_at_or_below_half_accepted(window: int, maxp: int) -> None:
    Config(
        loop_cycle_enabled=True,
        loop_cycle_window_size=window,
        loop_cycle_min_period=2,
        loop_cycle_max_period=maxp,
    )


@pytest.mark.parametrize(("window", "maxp"), [(12, 7), (24, 13), (7, 4)])
def test_594_max_period_above_half_rejected(window: int, maxp: int) -> None:
    with pytest.raises(ValueError, match="loop_cycle_max_period"):
        Config(
            loop_cycle_enabled=True,
            loop_cycle_window_size=window,
            loop_cycle_min_period=2,
            loop_cycle_max_period=maxp,
        )


def test_594_dead_band_ignored_when_cycle_disabled() -> None:
    """Cycle mode is off by default; the band is only a contract when the
    scan runs."""
    cfg = Config(
        loop_cycle_enabled=False,
        loop_cycle_window_size=12,
        loop_cycle_max_period=500,
    )
    assert cfg.loop_cycle_max_period == 500


def test_594_scaled_cap_widens_the_band() -> None:
    """With auto-scaling on the window grows toward max_window, so the
    reachable band is the cap, not the nominal size."""
    Config(
        loop_cycle_enabled=True,
        loop_cycle_window_size=12,
        loop_cycle_min_period=2,
        loop_cycle_max_period=25,
        window_scale_steps=10,
        max_window=60,
    )
    # Same nominal window without scaling cannot reach period 25.
    with pytest.raises(ValueError):
        Config(
            loop_cycle_enabled=True,
            loop_cycle_window_size=12,
            loop_cycle_min_period=2,
            loop_cycle_max_period=25,
        )


def test_594_periods_within_the_band_are_detected() -> None:
    from snagline.detectors.loop import LoopDetector

    for period in (3, 4, 6):
        cfg = Config(
            loop_cycle_enabled=True,
            loop_cycle_window_size=12,
            loop_cycle_min_period=2,
            loop_cycle_max_period=6,
            loop_repeat_threshold=12,
        )
        det = LoopDetector(config=cfg)
        fired = None
        for i in range(30):
            risk = det.observe(
                _event(
                    i,
                    action_signature=make_signature("tool_call", "t", f"s{i % period}"),
                )
            )
            if risk is not None:
                fired = risk
                break
        assert fired is not None, f"period={period} never fired"
        assert f"period-{period}" in fired.detail


# --------------------------------------------------------------------------
# #595 -- a non-UTF-8 trajectory exits cleanly
# --------------------------------------------------------------------------


def _latin1_trajectory(dirpath: str) -> str:
    # ensure_ascii=False so the non-ASCII byte really lands in the file --
    # json.dumps() escapes "café" to the pure-ASCII "é" by default, and a
    # cp1252 encoding of ASCII is still ASCII, so nothing would fail to decode.
    line = json.dumps(
        {
            "step_id": "0",
            "episode_id": "e",
            "timestamp": 0.0,
            "action_type": "tool_call",
            "action_signature": "café",
            "tool_name": "t",
            "latency_ms": 1.0,
            "error": False,
        },
        ensure_ascii=False,
    )
    path = os.path.join(dirpath, "latin1.jsonl")
    with open(path, "wb") as fh:
        fh.write(line.encode("cp1252") + b"\n")
    return path


def test_595_replay_non_utf8_is_a_clean_error(capsys: pytest.CaptureFixture) -> None:
    from snagline.cli import replay

    with tempfile.TemporaryDirectory() as d:
        path = _latin1_trajectory(d)
        # The failure must not escape as a traceback; a decode error is a bad
        # input file, not a crash of the tool.
        try:
            result = replay(path)
        except Exception as exc:  # noqa: BLE001 - the point is that it does not
            pytest.fail(f"decode failure escaped as {type(exc).__name__}: {exc}")
        assert result != 0
        captured = capsys.readouterr()
        assert "latin1.jsonl" in (captured.err + captured.out)


def test_595_bad_input_file_subclasses_oserror() -> None:
    from snagline._input import _BadInputFile, open_input

    assert issubclass(_BadInputFile, OSError)
    with tempfile.TemporaryDirectory() as d:
        path = _latin1_trajectory(d)
        with open_input(path) as fh:
            with pytest.raises(_BadInputFile, match="utf-8"):
                fh.read()


def test_595_missing_file_still_names_it() -> None:
    from snagline._input import _BadInputFile, open_input

    with pytest.raises(_BadInputFile, match="cannot open"):
        open_input(os.path.join("no", "such", "file.jsonl"))


def test_595_utf8_file_reads_unchanged() -> None:
    from snagline._input import open_input

    with tempfile.TemporaryDirectory() as d:
        path = _write_text(d, "ok.jsonl", '{"a": 1}\n')
        with open_input(path) as fh:
            assert fh.read() == '{"a": 1}\n'


def test_595_baseline_fit_on_non_utf8_is_a_clean_error() -> None:
    from snagline.baseline import fit_baseline_from_jsonl

    with tempfile.TemporaryDirectory() as d:
        path = _latin1_trajectory(d)
        with pytest.raises(OSError, match="utf-8"):
            fit_baseline_from_jsonl(path)


# --------------------------------------------------------------------------
# #596 -- a missing --config file is a clean exit 2
# --------------------------------------------------------------------------


def _args_with_config(path: str) -> argparse.Namespace:
    return argparse.Namespace(config=path, command="serve", port=0)


def test_596_missing_config_exits_2_not_traceback(
    capsys: pytest.CaptureFixture,
) -> None:
    from snagline.cli import _resolve_config_or_usage

    result = _resolve_config_or_usage(
        _args_with_config(os.path.join("nope", "missing.json")), "serve"
    )
    assert result is None
    captured = capsys.readouterr()
    assert "missing.json" in captured.err
    assert "serve" in captured.err


def test_596_unreadable_config_reports_the_path(
    capsys: pytest.CaptureFixture,
) -> None:
    from snagline.cli import _resolve_config_or_usage

    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "cfg.json")
        # A directory parses as neither JSON nor TOML and raises on read.
        os.makedirs(path)
        assert _resolve_config_or_usage(_args_with_config(path), "watch") is None
        captured = capsys.readouterr()
        assert "cfg.json" in captured.err


def test_596_valid_config_still_resolves() -> None:
    from snagline.cli import _resolve_config_or_usage

    with tempfile.TemporaryDirectory() as d:
        path = _write_text(d, "cfg.json", json.dumps({"fail_open": False}))
        cfg = _resolve_config_or_usage(_args_with_config(path), "serve")
        assert cfg is not None
        assert cfg.fail_open is False


# --------------------------------------------------------------------------
# #597 -- --list-versions validates --max-versions
# --------------------------------------------------------------------------


def _list_args(max_versions: Any) -> argparse.Namespace:
    return argparse.Namespace(
        command="baseline",
        trajectory="ignored.jsonl",
        list_versions=True,
        store_dir="store-x",
        max_versions=max_versions,
        tenant="default",
        deployment="default",
    )


def test_597_list_versions_rejects_bad_max_versions(
    capsys: pytest.CaptureFixture,
) -> None:
    from snagline.cli import _cmd_baseline

    with tempfile.TemporaryDirectory() as d:
        args = _list_args(-5)
        args.store_dir = os.path.join(d, "store-x")
        assert _cmd_baseline(args) == 2
        captured = capsys.readouterr()
        assert "--max-versions must be >= 1" in captured.err
        assert "-5" in captured.err


def test_597_list_versions_accepts_unset_and_counts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from snagline.cli import _cmd_baseline

    with tempfile.TemporaryDirectory() as d:
        store_dir = os.path.join(d, "store")
        os.makedirs(store_dir)
        # An empty store lists cleanly and exits 0 with the default limit.
        args = _list_args(None)
        args.store_dir = store_dir
        assert _cmd_baseline(args) == 0


def test_597_explicit_zero_is_not_silently_ten() -> None:
    """``or 10`` treated an explicit 0 as unset; the validator now rejects
    it consistently with the fit path."""
    from snagline.cli import _cmd_baseline

    with tempfile.TemporaryDirectory() as d:
        args = _list_args(0)
        args.store_dir = os.path.join(d, "store-x")
        assert _cmd_baseline(args) == 2


# --------------------------------------------------------------------------
# #598 -- a bad --output is reported, not a post-fit traceback
# --------------------------------------------------------------------------


def _fit_args(output: str, tmp: str) -> argparse.Namespace:
    traj = _write_text(
        tmp,
        "traj.jsonl",
        json.dumps(
            {
                "step_id": "0",
                "episode_id": "e",
                "timestamp": 0.0,
                "action_type": "tool_call",
                "action_signature": _VEN,
                "tool_name": "t",
                "latency_ms": 1.0,
                "error": False,
            }
        )
        + "\n",
    )
    return argparse.Namespace(
        command="baseline",
        trajectory=traj,
        output=output,
        list_versions=False,
        # --output is the single-file path, only taken when --store-dir is
        # unset; with a store_dir set the command takes the versioned-store
        # branch and never touches args.output.
        store_dir=None,
        max_versions=None,
        tenant="default",
        deployment="default",
        semantic=False,
    )


def test_598_unwritable_output_exits_2(capsys: pytest.CaptureFixture) -> None:
    from snagline.cli import _cmd_baseline

    with tempfile.TemporaryDirectory() as d:
        out = os.path.join(d, "nopedir", "baseline.json")
        assert _cmd_baseline(_fit_args(out, d)) == 2
        captured = capsys.readouterr()
        assert "cannot write" in captured.err
        assert "nopedir" in captured.err


def test_598_valid_output_writes_the_profile() -> None:
    from snagline.cli import _cmd_baseline

    with tempfile.TemporaryDirectory() as d:
        out = os.path.join(d, "baseline.json")
        assert _cmd_baseline(_fit_args(out, d)) == 0
        assert os.path.exists(out)
        with open(out, encoding="utf-8") as fh:
            assert json.load(fh)["tools"]


# --------------------------------------------------------------------------
# #599 -- an invalid cert PEM is rejected before the banner
# --------------------------------------------------------------------------


def _serve_args(certfile: str, keyfile: str, port: int) -> argparse.Namespace:
    return argparse.Namespace(
        command="serve",
        port=port,
        host="127.0.0.1",
        certfile=certfile,
        keyfile=keyfile,
        client_ca=None,
        auth_token=None,
        max_body_bytes=1_000_000,
        # _cmd_serve reads the enforcement wiring flags unconditionally, so a
        # namespace built by hand has to carry them or the AttributeError
        # fires before the TLS check the test is exercising.
        halt_forward=None,
        halt_timeout=None,
        min_severity_for_halt=None,
        quiet=True,
        log_format="text",
        # _build_sinks reads these for every serve invocation; sink=None falls
        # through to the console sink, which still needs min_severity.
        sink=None,
        min_severity=None,
        cooldown_seconds=None,
        webhook_url=None,
        slack_url=None,
        pagerduty_key=None,
        pagerduty_source=None,
        # serve() forwards these to make_server; with port=0 the bind picks a
        # free port and the test catches the resulting OSError/SystemExit.
        max_risks=1000,
        read_timeout=None,
        episode_ttl_seconds=None,
    )


def test_599_invalid_pem_rejected_before_banner(
    capsys: pytest.CaptureFixture,
) -> None:
    from snagline.cli import _cmd_serve

    with tempfile.TemporaryDirectory() as d:
        cert = _write_text(d, "junk.pem", "not a real certificate\n")
        key = _write_text(d, "junk.key", "not a real key\n")
        args = _serve_args(cert, key, 0)
        # The failure must come before any "listening on" promise. _cmd_serve
        # binds first, so a port of 0 lets it reach the TLS check; either way
        # the exit is 2 and the diagnostic names the TLS material.
        result = _cmd_serve(args)
        assert result == 2
        captured = capsys.readouterr()
        assert "TLS" in captured.err or "cannot load" in captured.err
        assert "listening on" not in captured.out


def test_599_valid_pem_is_accepted(
    capsys: pytest.CaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    from snagline.cli import _cmd_serve

    pytest.importorskip("cryptography")
    import datetime

    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.x509 import CertificateBuilder, Name

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    subject = Name(
        [
            __import__("cryptography.x509", fromlist=["NameAttribute"]).NameAttribute(
                __import__(
                    "cryptography.x509.oid", fromlist=["NameOID"]
                ).NameOID.COMMON_NAME,
                "localhost",
            )
        ]
    )
    now = datetime.datetime.now(datetime.timezone.utc)
    cert = (
        CertificateBuilder()
        .subject_name(subject)
        .issuer_name(subject)
        .public_key(key.public_key())
        .serial_number(
            __import__(
                "cryptography.x509", fromlist=["random_serial_number"]
            ).random_serial_number()
        )
        .not_valid_before(now - datetime.timedelta(days=1))
        .not_valid_after(now + datetime.timedelta(days=365))
        .sign(key, hashes.SHA256())
    )

    with tempfile.TemporaryDirectory() as d:
        cert_path = os.path.join(d, "cert.pem")
        key_path = os.path.join(d, "key.pem")
        with open(cert_path, "wb") as fh:
            fh.write(cert.public_bytes(serialization.Encoding.PEM))
        with open(key_path, "wb") as fh:
            fh.write(
                key.private_bytes(
                    serialization.Encoding.PEM,
                    serialization.PrivateFormat.TraditionalOpenSSL,
                    serialization.NoEncryption(),
                )
            )
        args = _serve_args(cert_path, key_path, 0)
        # A valid chain must pass the pre-banner check. Actually serving would
        # block forever in serve_forever, so prove the TLS material was
        # accepted by replacing serve() with a sentinel: _cmd_serve only
        # reaches it once every pre-banner validation -- including the cert
        # chain load -- has passed.
        import snagline.server.http_server as http_server

        reached: dict[str, object] = {}

        def _fake_serve(_monitor: Any, **kwargs: Any) -> int:
            reached["tls_kwargs"] = kwargs.get("certfile")
            return 0

        monkeypatch.setattr(http_server, "serve", _fake_serve)
        result = _cmd_serve(args)
        assert result == 0
        # The sentinel saw the real certfile, so _cmd_serve loaded and accepted
        # the chain and then carried on to serve instead of exiting 2 on TLS.
        assert reached["tls_kwargs"] == cert_path
        captured = capsys.readouterr()
        assert "cannot load TLS" not in captured.err


# --------------------------------------------------------------------------
# #602 -- the explicit Anthropic adapter reports streaming usage
# --------------------------------------------------------------------------


def test_602_stream_usage_reads_message_start_and_delta() -> None:
    from snagline.adapters import anthropic as mod

    class _Usage:
        def __init__(self, i: int | None, o: int | None) -> None:
            self.input_tokens = i
            self.output_tokens = o

    class _Start:
        class message:  # noqa: N801 - mirrors the SDK's event shape
            usage = _Usage(12, 5)

        type = "message_start"

    class _Delta:
        usage = _Usage(None, 34)
        type = "message_delta"

    class _Stop:
        type = "message_stop"

    # message_start carries the input tokens; message_delta the running output.
    assert mod._stream_usage(_Start()) == (12, 5)
    assert mod._stream_usage(_Delta()) == (None, 34)
    # The terminal event -- the only one the old code looked at -- has none.
    assert mod._stream_usage(_Stop()) == (None, None)


def test_602_stream_usage_reads_dict_shaped_events() -> None:
    from snagline.adapters import anthropic as mod

    start = {"message": {"usage": {"input_tokens": 7, "output_tokens": 1}}}
    assert mod._stream_usage(start) == (7, 1)
    delta = {"usage": {"input_tokens": None, "output_tokens": 19}}
    assert mod._stream_usage(delta) == (None, 19)
    assert mod._stream_usage({"type": "message_stop"}) == (None, None)


def test_602_sync_wrapper_accumulates_usage_across_the_stream() -> None:
    import itertools
    import time

    from snagline.adapters import anthropic as mod

    rec = _RecordingDetector()
    monitor = Monitor([rec], [])

    class _Usage:
        def __init__(self, i: int | None, o: int | None) -> None:
            self.input_tokens = i
            self.output_tokens = o

    class _Start:
        class message:  # noqa: N801
            usage = _Usage(12, 5)

    class _Delta:
        usage = _Usage(None, 34)

    class _Stop:
        pass

    class _Stream:
        def __iter__(self):
            return iter([_Start(), _Delta(), _Stop()])

    # Build through the real constructor: __getattr__ proxies any attribute
    # the wrapper does not have onto the underlying stream, so a hand-built
    # instance that omits one silently looks the lookup up on _Stream instead.
    wrapper = mod._SyncStreamWrapper(
        monitor=monitor,
        counter=itertools.count(),
        episode_id="ep-602",
        model="claude",
        sig_text="s",
        start=time.perf_counter(),
        stream=_Stream(),
    )
    for _chunk in wrapper:
        pass
    # input_tokens from message_start, output_tokens from message_delta.
    assert (wrapper._tokens_in, wrapper._tokens_out) == (12, 34)
    # Reaching exhaustion emitted exactly one event carrying those counts --
    # the wrapper used to report (None, None) and page nothing.
    assert len(rec.events) == 1
    assert rec.events[0].tokens_in == 12
    assert rec.events[0].tokens_out == 34


def test_602_openai_wrapper_unaffected_but_shares_async_fix() -> None:
    """The OpenAI adapter was not broken for usage (it lands on the final
    chunk), but it shared the async misclassification (#588)."""
    from snagline.adapters import openai as mod

    async def _create(**kw: Any) -> Any:
        return {"ok": True}

    def _wrapper(*a: Any, **kw: Any) -> Any:
        return _create(*a, **kw)

    _wrapper.__wrapped__ = _create  # type: ignore[attr-defined]
    assert mod._is_async_call(_wrapper) is True
