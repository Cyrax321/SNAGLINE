"""Batch #664-#670: seven verified bugs against master (968dc19).

One named test per issue:

* #664 an unsupported method with a malformed Content-Length got two status
  lines on one connection
* #665 ``baseline <file> --store-dir`` tracebacks on a missing trajectory
  file
* #666 ``serve`` prints its listening banner and then tracebacks when the
  address cannot be bound
* #667 the Authorization Bearer scheme is matched case-sensitively
* #668 ``--max-risks 0`` silently retained one risk instead of none
* #669 ``EsnCusumDetector.load_state`` dropped every restored episode's
  readout and pre-advance context
* #670 the ESN Mahalanobis score returned NaN for an overflowing latency
"""

from __future__ import annotations

import io
import os
import socket
import sys
import tempfile

import pytest

from snagline.cli import main
from snagline.monitor import Monitor
from snagline.server.http_server import make_handler, make_server


def _run(argv):
    """Run main() with stdout+stderr captured, returning (code, combined)."""
    old_out, old_err = sys.stdout, sys.stderr
    out, err = io.StringIO(), io.StringIO()
    sys.stdout, sys.stderr = out, err
    try:
        code = main(argv)
        return code, out.getvalue() + err.getvalue()
    finally:
        sys.stdout, sys.stderr = old_out, old_err


# ---------------------------------------------------------------- #664 ----


def _start_server(**kwargs):
    server = make_server(Monitor([], []), host="127.0.0.1", port=0, **kwargs)
    import threading

    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, int(server.server_address[1])


def _read_all(sock):
    chunks: list[bytes] = []
    while True:
        data = sock.recv(65536)
        if not data:
            break
        chunks.append(data)
    return b"".join(chunks)


def _require_ml_extra():
    """The ESN detectors live in snagline.ml, which needs the optional 'ml'
    extra (numpy). Guard them so this batch runs on the core matrix legs and
    skips cleanly there; the dedicated ML leg exercises them for real."""
    pytest.importorskip("numpy", reason="ml extra (numpy) not installed")


def test_664_bad_content_length_on_an_unsupported_method_answers_once():
    """``_parse_content_length`` answers 400 itself; ``_method_not_allowed``
    must not write a second status line behind it."""
    server, port = _start_server()
    try:
        sock = socket.create_connection(("127.0.0.1", port), timeout=10)
        try:
            sock.sendall(
                b"PUT /events HTTP/1.0\r\nHost: localhost\r\n"
                b"Content-Length: abc\r\n\r\n"
            )
            response = _read_all(sock)
        finally:
            sock.close()
    finally:
        server.shutdown()
        server.server_close()
    assert response.count(b"HTTP/1.0 4") == 1, response[:300]
    assert b"invalid Content-Length" in response


def test_664_a_well_formed_unsupported_method_still_answers_once():
    """The guard must not break the normal 405/404 path."""
    server, port = _start_server()
    try:
        sock = socket.create_connection(("127.0.0.1", port), timeout=10)
        try:
            sock.sendall(
                b"PUT /events HTTP/1.0\r\nHost: localhost\r\nContent-Length: 0\r\n\r\n"
            )
            response = _read_all(sock)
        finally:
            sock.close()
    finally:
        server.shutdown()
        server.server_close()
    assert response.count(b"HTTP/1.0 4") == 1, response[:300]
    # /events exists, so it is a 405 naming the allowed methods.
    assert b"HTTP/1.0 405" in response
    assert b"Allow:" in response


# ---------------------------------------------------------------- #665 ----


def test_665_baseline_store_dir_missing_file_is_a_clean_exit_2():
    """The --store-dir path must share the fit path's contract: one-line
    diagnostic and exit 2, not a FileNotFoundError traceback."""
    with tempfile.TemporaryDirectory() as d:
        missing = os.path.join(d, "nope", "missing.jsonl")
        code, err = _run(["baseline", missing, "--store-dir", d])
    assert code == 2, err
    assert "cannot open" in err
    assert missing in err
    assert "Traceback" not in err


def test_665_baseline_plain_fit_path_still_exits_2_too():
    """The pre-existing contract the fix aligns with."""
    with tempfile.TemporaryDirectory() as d:
        missing = os.path.join(d, "nope", "missing.jsonl")
        code, err = _run(["baseline", missing])
    assert code == 2, err
    assert "cannot open" in err
    assert "Traceback" not in err


# ---------------------------------------------------------------- #666 ----


def test_666_serve_reports_a_bind_failure_without_a_traceback(monkeypatch):
    """The banner already printed, so a bind failure must surface as a
    one-line diagnostic and a non-zero exit -- not a traceback. The listener
    is stubbed to raise OSError, so the test depends on no host networking
    and no occupied port."""
    import snagline.server.http_server as http_server

    def _unbindable(*args: object, **kwargs: object) -> object:
        raise OSError("cannot assign requested address")

    monkeypatch.setattr(http_server, "make_server", _unbindable)
    code, err = _run(["serve", "--host", "127.0.0.1", "--port", "0"])
    assert code != 0, "bind failure must not exit 0"
    assert "Traceback" not in err
    assert "cannot start listener" in err


# ---------------------------------------------------------------- #667 ----


def test_667_the_bearer_scheme_token_is_case_insensitive():
    """RFC 7235 §2.1: auth-scheme is a case-insensitive token, so ``bearer``
    and ``BEARER`` authenticate exactly like ``Bearer``."""
    server, port = _start_server(auth_token="sekret")
    try:
        for scheme in ("bearer", "BEARER", "Bearer"):
            sock = socket.create_connection(("127.0.0.1", port), timeout=10)
            try:
                # /health is deliberately open, so only /metrics actually
                # reaches the Authorization gate this test is about.
                sock.sendall(
                    f"GET /metrics HTTP/1.0\r\nHost: localhost\r\n"
                    f"Authorization: {scheme} sekret\r\n\r\n".encode()
                )
                response = _read_all(sock)
            finally:
                sock.close()
            assert response.startswith(b"HTTP/1.0 200"), (scheme, response[:120])
    finally:
        server.shutdown()
        server.server_close()


def test_667_the_wrong_token_is_still_rejected():
    """Case-insensitivity of the scheme must not weaken the digest check."""
    server, port = _start_server(auth_token="sekret")
    try:
        sock = socket.create_connection(("127.0.0.1", port), timeout=10)
        try:
            sock.sendall(
                b"GET /metrics HTTP/1.0\r\nHost: localhost\r\n"
                b"Authorization: bearer wrong\r\n\r\n"
            )
            response = _read_all(sock)
        finally:
            sock.close()
    finally:
        server.shutdown()
        server.server_close()
    assert response.startswith(b"HTTP/1.0 401"), response[:120]


# ---------------------------------------------------------------- #668 ----


def test_668_max_risks_zero_retains_nothing():
    """``--max-risks 0`` must mean an empty ring, not the old silent
    max(1, max_risks) clamp that kept exactly one risk."""
    handler = make_handler(Monitor.default(), max_risks=0)
    # make_handler is typed as the bare handler class; the deque is attached
    # dynamically by the factory, so read it through getattr.
    assert getattr(handler, "snagline_risks").maxlen == 0


def test_668_a_negative_max_risks_is_rejected():
    """deque(maxlen=<0>) is unbounded, so a negative cap must be refused at
    construction rather than silently dropping the bound."""
    with pytest.raises(ValueError):
        make_handler(Monitor.default(), max_risks=-1)


def test_668_cli_rejects_a_negative_max_risks_before_the_banner():
    code, err = _run(["serve", "--max-risks", "-1"])
    assert code == 2, err
    assert "--max-risks" in err
    assert "listening on" not in err


# ---------------------------------------------------------------- #669 ----


def _ev(i, episode="ep1", latency=10.0, error=False, signature=None):
    from snagline.events import StepEvent, make_signature

    return StepEvent(
        step_id=f"s{i}",
        episode_id=episode,
        timestamp=float(i),
        action_type="tool_call",
        action_signature=signature or make_signature("tool_call", "t", f"s{i % 4}"),
        tool_name="t",
        latency_ms=latency,
        error=error,
    )


def _profile():
    from snagline.baseline import BaselineProfile, ToolBaseline

    prof = BaselineProfile()
    tb = ToolBaseline(tool_name="t")
    for v in (10.0, 10.1, 10.2):
        tb.add(v, error=False)
    prof.tools["t"] = tb
    return prof


def test_669_load_state_keeps_the_episode_readout_and_context():
    """A restored episode must keep the readout it had already solved and the
    pre-advance context its next prediction is judged from -- not be pushed
    back into warm-up to relearn from the live stream."""
    _require_ml_extra()
    from snagline.ml.esn_ensemble import EsnCusumDetector

    d = EsnCusumDetector(baseline=_profile(), warmup_steps=20)
    for i in range(60):
        d.observe(_ev(i))
    live = d._episodes["ep1"]
    assert live.beta is not None
    assert live.context_prev is not None
    snap = d.dump_state()
    assert snap is not None

    restored = EsnCusumDetector(baseline=_profile(), warmup_steps=20)
    restored.load_state(snap)
    got = restored._episodes["ep1"]
    assert got.beta is not None, "the solved readout was dropped on restore"
    assert got.context_prev is not None, "the pre-advance context was dropped"
    assert got.warm_n == live.warm_n


def test_669_a_restored_episode_scores_without_rewarming():
    """End-to-end shape of the defect: without refitting, a short anomaly
    must still register on the restored detector instead of being swallowed
    by a restart into warm-up."""
    _require_ml_extra()
    from snagline.ml.esn_ensemble import EsnCusumDetector

    d = EsnCusumDetector(baseline=_profile(), warmup_steps=20)
    for i in range(60):
        d.observe(_ev(i))
    snap = d.dump_state()
    assert snap is not None

    restored = EsnCusumDetector(baseline=_profile(), warmup_steps=20)
    restored.load_state(snap)
    got = restored._episodes["ep1"]
    # Warm-up is done and the readout is live, so a hot anomaly scores
    # instead of returning the warm-up constant 0.0.
    score = restored._esn_anomaly(got, restored._features(_ev(61, latency=9000.0)))
    assert score > 0.0


# ---------------------------------------------------------------- #670 ----


def test_670_the_mahalanobis_score_saturates_instead_of_going_nan():
    """An overflowing z-score must saturate at 1.0, not produce a NaN that
    ``max()`` silently discards."""
    _require_ml_extra()
    from snagline.events import StepEvent
    from snagline.ml.esn_ensemble import EsnCusumDetector

    d = EsnCusumDetector(baseline=_profile())
    # The score is per-episode (it consults the episode's unseen-tool latch),
    # so the episode state must exist; one observe creates it.
    d.observe(_ev(0))
    st = d._episodes["ep1"]
    for lat in (1e100, 1e308, float("inf")):
        ev = StepEvent(
            step_id="s1",
            episode_id="ep1",
            timestamp=1.0,
            action_type="tool_call",
            action_signature="sig",
            tool_name="t",
            latency_ms=lat,
            error=False,
        )
        assert d._mahalanobis_score(st, ev) == pytest.approx(1.0), lat


def test_670_a_finite_latency_still_scores_below_one():
    """The saturation must not flatten the normal range into its cap."""
    _require_ml_extra()
    from snagline.events import StepEvent
    from snagline.ml.esn_ensemble import EsnCusumDetector

    d = EsnCusumDetector(baseline=_profile())
    d.observe(_ev(0))
    st = d._episodes["ep1"]
    ev = StepEvent(
        step_id="s1",
        episode_id="ep1",
        timestamp=1.0,
        action_type="tool_call",
        action_signature="sig",
        tool_name="t",
        latency_ms=200.0,
        error=False,
    )
    score = d._mahalanobis_score(st, ev)
    assert 0.0 < score < 1.0
