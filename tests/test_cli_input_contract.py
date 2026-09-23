"""CLI contract bugs: missing input files (#428), --semantic on retrain (#429),
and the serve banner promising a listener that never bound (#430)."""

from __future__ import annotations

import json
import os
import sys
import tempfile

import pytest

from snagline.cli import main


def _traj(path, lines):
    with open(path, "w", encoding="utf-8") as fh:
        for obj in lines:
            fh.write(json.dumps(obj) + "\n")
    return path


def _run(argv):
    """Run main() with stdout+stderr captured, returning (code, combined)."""
    import io

    old_out, old_err = sys.stdout, sys.stderr
    out, err = io.StringIO(), io.StringIO()
    sys.stdout, sys.stderr = out, err
    try:
        code = main(argv)
        return code, out.getvalue() + err.getvalue()
    finally:
        sys.stdout, sys.stderr = old_out, old_err


# ---------------------------------------------------------------- #428 ----


def test_replay_missing_file_is_a_clean_exit_not_a_traceback():
    with tempfile.TemporaryDirectory() as d:
        missing = os.path.join(d, "nope", "missing.jsonl")
        code, err = _run(["replay", missing])
    assert code == 2, err
    assert "cannot open" in err
    assert missing in err
    assert "Traceback" not in err


def test_replay_existing_file_still_works():
    with tempfile.TemporaryDirectory() as d:
        path = _traj(
            os.path.join(d, "t.jsonl"),
            [
                {
                    "step_id": "1",
                    "episode_id": "e",
                    "timestamp": 1.0,
                    "action_type": "tool_call",
                    "action_signature": "s1",
                }
            ],
        )
        code, err = _run(["replay", path, "--quiet"])
    assert code == 0, err


def test_watch_missing_file_is_a_clean_exit_not_a_traceback():
    with tempfile.TemporaryDirectory() as d:
        missing = os.path.join(d, "gone.jsonl")
        code, err = _run(["watch", "--file", missing])
    assert code == 2, err
    assert "cannot open" in err
    assert missing in err
    assert "Traceback" not in err


def test_baseline_missing_input_is_a_clean_exit_not_a_traceback():
    with tempfile.TemporaryDirectory() as d:
        missing = os.path.join(d, "absent.jsonl")
        out = os.path.join(d, "baseline.json")
        code, err = _run(["baseline", missing, "--output", out])
    assert code == 2, err
    assert "cannot open" in err
    assert missing in err
    assert "Traceback" not in err


# ---------------------------------------------------------------- #429 ----


def test_semantic_flag_is_rejected_on_baseline_retrain():
    """`--semantic` on the retrain path used to be accepted and then dropped,
    silently persisting a structural baseline."""
    with tempfile.TemporaryDirectory() as d:
        store = os.path.join(d, "store")
        jsonl = _traj(os.path.join(d, "traj.jsonl"), [])
        code, err = _run(
            [
                "baseline",
                "retrain",
                "--store-dir",
                store,
                "--jsonl",
                jsonl,
                "--semantic",
            ]
        )
    assert code == 2, err
    assert "--semantic is not supported on the retrain path" in err


def test_baseline_retrain_without_semantic_is_unaffected():
    with tempfile.TemporaryDirectory() as d:
        store = os.path.join(d, "store")
        jsonl = _traj(
            os.path.join(d, "traj.jsonl"),
            [
                {
                    "step_id": "1",
                    "episode_id": "e",
                    "timestamp": 1.0,
                    "action_type": "tool_call",
                    "action_signature": "tool-a",
                    "latency_ms": 12.0,
                }
            ],
        )
        code, err = _run(
            ["baseline", "retrain", "--store-dir", store, "--jsonl", jsonl]
        )
    assert code == 0, err
    assert "stored version" in err


# ---------------------------------------------------------------- #430 ----


@pytest.mark.parametrize("port", [-1, 65536, 100000])
def test_serve_rejects_out_of_range_port_before_the_banner(port):
    code, err = _run(["serve", "--port", str(port)])
    assert code == 2, err
    assert "--port must be between 0 and 65535" in err
    assert str(port) in err
    # The banner must not have promised a listener that cannot exist.
    assert "listening on" not in err


def test_serve_accepts_port_zero_as_an_ephemeral_bind(monkeypatch):
    """0 stays the supported way to let the OS choose a free port (the server
    tests use it as their placeholder), so the validator must not reject it."""
    captured = {}

    def _fake_serve(monitor, **kwargs):
        captured.update(kwargs)

    monkeypatch.setattr("snagline.server.http_server.serve", _fake_serve)
    code, err = _run(["serve", "--port", "0"])
    assert code == 0, err
    assert captured["port"] == 0


def test_serve_rejects_unreadable_certfile_before_the_banner():
    with tempfile.TemporaryDirectory() as d:
        missing = os.path.join(d, "cert.pem")
        key = os.path.join(d, "key.pem")
        with open(key, "w", encoding="utf-8") as fh:
            fh.write("key")
        code, err = _run(
            ["serve", "--certfile", missing, "--keyfile", key, "--port", "8799"]
        )
    assert code == 2, err
    assert "cannot read --certfile" in err
    assert missing in err
    assert "listening on" not in err


def test_serve_rejects_unreadable_keyfile_before_the_banner():
    with tempfile.TemporaryDirectory() as d:
        cert = os.path.join(d, "cert.pem")
        missing = os.path.join(d, "key.pem")
        with open(cert, "w", encoding="utf-8") as fh:
            fh.write("cert")
        code, err = _run(
            ["serve", "--certfile", cert, "--keyfile", missing, "--port", "8799"]
        )
    assert code == 2, err
    assert "cannot read --keyfile" in err
    assert "listening on" not in err
