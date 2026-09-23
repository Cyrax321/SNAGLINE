"""Regression tests for issues #434-#438.

Five silent contract violations in the ingest and configuration path, each
verified failing against pristine master before the fix landed:

* #434 a ``Transfer-Encoding: chunked`` POST was read as an empty body and
  answered 400 "invalid StepEvent JSON" -- the JSON was fine.
* #435 ``SemanticGoalDriftDetector.load_state`` accepted a running sum fitted
  under a different embedding dimension than the live baseline, then wedged
  (IndexError swallowed by fail-open) for the rest of the run.
* #436 a repeat/cascade count threshold wider than its sliding window is
  unreachable on every input, permanently blinding the detector.
* #437 values from a config file were used verbatim, so ``{"fail_open":
  "false"}`` stayed a truthy string and kept fail-open on.
* #438 a non-finite ``latency_ms`` was absorbed into the running moments and
  poisoned the *persisted* baseline every later monitor loads.
"""

from __future__ import annotations

import json
import math
import socket
import threading
from typing import Any

import pytest

from snagline import Monitor
from snagline.baseline import BaselineProfile, fit_baseline_from_jsonl, save_baseline
from snagline.config import Config
from snagline.detectors.error_cascade import ErrorCascadeDetector
from snagline.detectors.loop import LoopDetector
from snagline.drift.goal_drift import SemanticGoalDriftDetector
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
        "latency_ms": 1.0,
        "error": False,
    }
    payload.update(overrides)
    return StepEvent(**payload)


def _event_line(step: int, **overrides: Any) -> str:
    payload: dict[str, Any] = {
        "step_id": f"s{step}",
        "episode_id": "ep",
        "timestamp": float(step),
        "action_type": "tool_call",
        "action_signature": f"a{step}",
        "tool_name": "t",
        "latency_ms": 1.0,
        "error": False,
        "tokens_in": 10,
        "tokens_out": 5,
    }
    payload.update(overrides)
    return json.dumps(payload)


# ---------------------------------------------------------------- #434 ----


def _serve(max_body_bytes: int | None = None) -> tuple[Any, int]:
    kwargs: dict[str, Any] = {}
    if max_body_bytes is not None:
        kwargs["max_body_bytes"] = max_body_bytes
    server = make_server(Monitor.default(), host="127.0.0.1", port=0, **kwargs)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, server.server_address[1]


def _raw_request(port: int, request: bytes) -> bytes:
    """Send a hand-built request and return the raw response bytes."""
    sock = socket.create_connection(("127.0.0.1", port), timeout=5)
    try:
        sock.sendall(request)
        chunks: list[bytes] = []
        while True:
            data = sock.recv(4096)
            if not data:
                break
            chunks.append(data)
    finally:
        sock.close()
    return b"".join(chunks)


def _chunked_request(target: str, body: bytes, extra: bytes = b"") -> bytes:
    """Frame ``body`` as a single Transfer-Encoding: chunked request."""
    return (
        f"POST {target} HTTP/1.1\r\nHost: 127.0.0.1\r\n"
        "Content-Type: application/json\r\nTransfer-Encoding: chunked\r\n"
        f"Connection: close\r\n".encode()
        + extra
        + b"\r\n"
        + b"%x\r\n%s\r\n" % (len(body), body)
        + b"0\r\n\r\n"
    )


def test_434_chunked_post_is_ingested() -> None:
    server, port = _serve()
    try:
        event = {
            "step_id": "s1",
            "episode_id": "ep-chunked",
            "timestamp": 1.0,
            "action_type": "tool_call",
            "action_signature": make_signature("tool_call", "t", "{}"),
            "tool_name": "t",
            "latency_ms": 5.0,
            "error": False,
        }
        resp = _raw_request(
            port, _chunked_request("/events", json.dumps(event).encode())
        )
        status = resp.split(b"\r\n")[0]
        assert b" 202" in status, status
        assert json.loads(resp.split(b"\r\n\r\n", 1)[1])["status"] == "ingested"
    finally:
        server.shutdown()
        server.server_close()


def test_434_chunked_multi_chunk_body_is_reassembled() -> None:
    """A body split across chunks must reassemble to the original bytes."""
    server, port = _serve()
    try:
        head = json.dumps(
            {
                "step_id": "s1",
                "episode_id": "ep-split",
                "timestamp": 1.0,
                "action_type": "tool_call",
                "action_signature": "aaaa1111bbbb2222",
                "tool_name": "t",
                "latency_ms": 1.0,
                "error": False,
            }
        )
        body = head.encode()
        request = (
            b"POST /events HTTP/1.1\r\nHost: 127.0.0.1\r\n"
            b"Content-Type: application/json\r\nTransfer-Encoding: chunked\r\n"
            b"Connection: close\r\n\r\n"
            + b"%x\r\n%s\r\n" % (5, body[:5])
            + b"%x\r\n%s\r\n" % (len(body) - 5, body[5:])
            + b"0\r\n\r\n"
        )
        resp = _raw_request(port, request)
        assert b" 202" in resp.split(b"\r\n")[0], resp
    finally:
        server.shutdown()
        server.server_close()


def test_434_chunked_body_still_capped() -> None:
    """A chunked body cannot stream around the max_body_bytes cap."""
    server, port = _serve(max_body_bytes=1000)
    try:
        # A valid event padded past the cap on a field StepEvent actually
        # carries, so it is the size limit and not malformed JSON that fires.
        event = {
            "step_id": "s1",
            "episode_id": "ep-big",
            "timestamp": 1.0,
            "action_type": "tool_call",
            "action_signature": "a" * 5000,
            "tool_name": "t",
            "latency_ms": 1.0,
            "error": False,
        }
        resp = _raw_request(
            port, _chunked_request("/events", json.dumps(event).encode())
        )
        assert b" 413" in resp.split(b"\r\n")[0], resp
    finally:
        server.shutdown()
        server.server_close()


def test_434_malformed_chunk_framing_is_400() -> None:
    server, port = _serve()
    try:
        request = (
            b"POST /events HTTP/1.1\r\nHost: 127.0.0.1\r\n"
            b"Content-Type: application/json\r\nTransfer-Encoding: chunked\r\n"
            b"Connection: close\r\n\r\n"
            b"not-hex\r\n{}\r\n0\r\n\r\n"
        )
        resp = _raw_request(port, request)
        assert b" 400" in resp.split(b"\r\n")[0], resp
    finally:
        server.shutdown()
        server.server_close()


# ---------------------------------------------------------------- #435 ----


def _drift_detector(dim: int) -> SemanticGoalDriftDetector:
    cfg = Config(semantic_drift_enabled=True, semantic_drift_min_samples=2)
    # Baseline centroid and live embedder agree on the *new* dimension; only
    # the restored snapshot is stale, which is the wedge the live path cannot
    # see (it compares embedder vs centroid, and those match).
    profile = BaselineProfile(tools={}, embedding_centroid=[0.0] * dim)
    return SemanticGoalDriftDetector(
        config=cfg, baseline=profile, embedder=lambda event: [1.0] * dim
    )


def test_435_dimension_mismatched_restore_does_not_wedge() -> None:
    detector = _drift_detector(4)
    detector.load_state(
        {"episodes": {"ep": {"sum": [1.0, 0.0, 0.0], "n": 5, "debt": 0.0}}}
    )
    # No exception, and no swallowed-per-step failure: the stale sum is
    # dropped and the episode re-accumulates from scratch in the live
    # dimension rather than indexing out of range on every step.
    for i in range(6):
        detector.observe(_event(i))
    state = detector._episodes["ep"]
    assert state.sum is not None
    assert len(state.sum) == 4
    assert state.n == 6  # warm-up restarted, not inherited from the stale run


def test_435_matching_dimension_restore_is_kept() -> None:
    detector = _drift_detector(4)
    detector.load_state(
        {"episodes": {"ep": {"sum": [1.0, 0.0, 0.0, 0.0], "n": 5, "debt": 0.0}}}
    )
    state = detector._episodes["ep"]
    assert state.sum == [1.0, 0.0, 0.0, 0.0]
    assert state.n == 5


def test_435_mismatched_episode_does_not_evict_matching_episodes() -> None:
    detector = _drift_detector(4)
    detector.load_state(
        {
            "episodes": {
                "bad": {"sum": [1.0, 0.0], "n": 3, "debt": 0.0},
                "good": {"sum": [1.0, 0.0, 0.0, 0.0], "n": 7, "debt": 0.25},
            }
        }
    )
    assert detector._episodes["good"].sum == [1.0, 0.0, 0.0, 0.0]
    assert detector._episodes["good"].n == 7
    assert detector._episodes["bad"].sum is None


# ---------------------------------------------------------------- #436 ----


def test_436_loop_repeat_threshold_wider_than_window_rejected() -> None:
    with pytest.raises(ValueError, match="loop_repeat_threshold"):
        Config(loop_window_size=4, loop_repeat_threshold=99)


def test_436_cascade_error_threshold_wider_than_window_rejected() -> None:
    with pytest.raises(ValueError, match="cascade_error_threshold"):
        Config(cascade_window_size=4, cascade_error_threshold=99)


def test_436_cycle_window_below_two_periods_rejected() -> None:
    with pytest.raises(ValueError, match="loop_cycle_window_size"):
        Config(
            loop_cycle_enabled=True, loop_cycle_window_size=3, loop_cycle_min_period=2
        )


def test_436_threshold_at_window_edge_is_accepted() -> None:
    # The boundary is reachable: a window entirely filled with one signature.
    cfg = Config(loop_window_size=4, loop_repeat_threshold=4)
    detector = LoopDetector(config=cfg)
    risks = [detector.observe(_event(i)) for i in range(4)]
    assert any(r is not None and r.trigger == "loop" for r in risks)


def test_436_scaled_window_lets_threshold_grow_to_the_cap() -> None:
    """With scaling on the window grows toward max_window, so a threshold
    above the nominal size but inside the cap is reachable, not dead."""
    cfg = Config(
        window_scale_steps=10,
        max_window=64,
        loop_window_size=4,
        loop_repeat_threshold=32,
    )
    assert cfg.loop_repeat_threshold == 32


def test_436_cascade_density_rule_fires_at_the_window_edge() -> None:
    """The boundary is reachable: a window entirely full of flagged steps.
    That is what the guard measures against -- 4 fires, 99 never could."""
    cfg = Config(
        cascade_window_size=4,
        cascade_error_threshold=4,
        cascade_consecutive_threshold=99,
    )
    detector = ErrorCascadeDetector(config=cfg)
    fired = [detector.observe(_event(i, error=True)) for i in range(40)]
    assert any(r is not None for r in fired)


# ---------------------------------------------------------------- #437 ----


def _write_config(values: dict[str, Any], tmp_path: Any) -> str:
    path = str(tmp_path / "cfg.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(values, fh)
    return path


def test_437_quoted_false_is_coerced_to_false(tmp_path: Any) -> None:
    path = _write_config({"fail_open": "false"}, tmp_path)
    cfg = Config.load_file(path)
    assert cfg.fail_open is False


def test_437_quoted_false_reaches_the_monitor(tmp_path: Any) -> None:
    """The operator asked to disable fail-open; it must actually be off."""
    path = _write_config({"fail_open": "false"}, tmp_path)
    monitor = Monitor.default(config=Config.load_file(path))
    assert monitor._fail_open is False


def test_437_quoted_int_is_coerced(tmp_path: Any) -> None:
    path = _write_config({"max_live_episodes": "5000"}, tmp_path)
    assert Config.load_file(path).max_live_episodes == 5000


def test_437_layered_resolve_coerces_the_same_way(tmp_path: Any) -> None:
    path = _write_config({"fail_open": "false"}, tmp_path)
    assert Config.resolve(path).fail_open is False


def test_437_zero_for_bool_is_coerced(tmp_path: Any) -> None:
    """``{"fail_open": 0}`` is falsy and used to silently turn fail-open off
    with no error; coercion makes the intent explicit instead of accidental."""
    path = _write_config({"fail_open": 0}, tmp_path)
    assert Config.load_file(path).fail_open is False


def test_437_unparseable_int_is_rejected(tmp_path: Any) -> None:
    path = _write_config({"max_live_episodes": "not-a-number"}, tmp_path)
    with pytest.raises(ValueError, match="max_live_episodes"):
        Config.load_file(path)


def test_437_wrong_scalar_type_is_rejected(tmp_path: Any) -> None:
    path = _write_config({"max_live_episodes": [1, 2]}, tmp_path)
    with pytest.raises(TypeError, match="max_live_episodes"):
        Config.load_file(path)


def test_437_native_types_pass_through(tmp_path: Any) -> None:
    path = _write_config({"max_live_episodes": 5000, "fail_open": False}, tmp_path)
    cfg = Config.load_file(path)
    assert cfg.max_live_episodes == 5000
    assert cfg.fail_open is False


def test_437_unknown_keys_still_ignored(tmp_path: Any) -> None:
    """The fail-soft contract for metadata-carrying config files is kept."""
    path = _write_config({"comment": "unrelated metadata", "fail_open": True}, tmp_path)
    cfg = Config.load_file(path)
    assert cfg.fail_open is True
    assert not hasattr(cfg, "comment")


# ---------------------------------------------------------------- #438 ----


def _nan_trajectory(tmp_path: Any) -> str:
    path = str(tmp_path / "nan_traj.jsonl")
    with open(path, "w", encoding="utf-8") as fh:
        for i in range(10):
            fh.write(_event_line(i, latency_ms=float("nan")) + "\n")
    return path


def test_438_nan_latency_does_not_poison_the_profile(tmp_path: Any) -> None:
    profile = fit_baseline_from_jsonl(_nan_trajectory(tmp_path))
    tool = profile.tools["t"]
    assert math.isfinite(tool.mean_latency)
    assert math.isfinite(tool.std_latency)
    assert math.isfinite(tool.min_latency if tool.min_latency is not None else 0.0)


def test_438_bad_samples_are_excluded_good_samples_counted(tmp_path: Any) -> None:
    path = str(tmp_path / "mixed.jsonl")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(_event_line(0, latency_ms=5.0) + "\n")
        fh.write(_event_line(1, latency_ms=float("nan")) + "\n")
        fh.write(_event_line(2, latency_ms=15.0) + "\n")
    tool = fit_baseline_from_jsonl(path).tools["t"]
    assert tool.latency_count == 2
    assert tool.count == 3  # the call still happened; only its timing was bad
    assert tool.mean_latency == 10.0


def test_438_negative_latency_is_also_dropped() -> None:
    from snagline.baseline import ToolBaseline

    tool = ToolBaseline(tool_name="t")
    tool.add(-5.0, error=False)
    assert tool.latency_count == 0
    assert tool.count == 1


def test_438_inf_latency_is_dropped() -> None:
    from snagline.baseline import ToolBaseline

    tool = ToolBaseline(tool_name="t")
    tool.add(float("inf"), error=False)
    assert tool.latency_count == 0


def test_438_profile_persists_without_nan(tmp_path: Any) -> None:
    profile = fit_baseline_from_jsonl(_nan_trajectory(tmp_path))
    out = str(tmp_path / "base.json")
    save_baseline(profile, out)
    with open(out, encoding="utf-8") as fh:
        assert "NaN" not in fh.read()
    # A good sample afterwards accumulates cleanly, not onto a poisoned sum.
    tool = profile.tools["t"]
    tool.add(5.0, error=False)
    assert tool.mean_latency == 5.0
    assert tool.latency_count == 1


def test_438_bad_sample_still_records_its_error() -> None:
    from snagline.baseline import ToolBaseline

    tool = ToolBaseline(tool_name="t")
    tool.add(float("nan"), error=True)
    assert tool.error_count == 1
    assert tool.count == 1
    assert tool.latency_count == 0
