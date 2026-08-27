"""Tests for the healthy-run baseline fitter and the `snagline baseline` CLI."""

from __future__ import annotations

import json

from snagline.baseline import (
    BaselineProfile,
    ToolBaseline,
    fit_baseline_from_jsonl,
    load_baseline,
    save_baseline,
)
from snagline.cli import main


def _event(tool, latency, error=False, episode="ep"):
    return {
        "step_id": "0",
        "episode_id": episode,
        "timestamp": 1.0,
        "action_type": "tool_call",
        "action_signature": "sig",
        "tool_name": tool,
        "latency_ms": latency,
        "error": error,
    }


def test_tool_baseline_statistics():
    tb = ToolBaseline(tool_name="search")
    for lat in (100.0, 110.0, 90.0):
        tb.add(lat, error=False)
    tb.add(100.0, error=True)
    assert tb.count == 4
    assert tb.mean_latency == 100.0
    # sample std of [100,110,90,100]: mean 100, var 200/3, std ~8.165
    assert abs(tb.std_latency - 8.165) < 0.1
    assert tb.error_count == 1
    assert tb.error_rate == 0.25
    assert tb.min_latency == 90.0
    assert tb.max_latency == 110.0


def test_fit_baseline_aggregates_per_tool(tmp_path):
    traj = tmp_path / "healthy.jsonl"
    rows = [
        _event("search", 100.0),
        _event("search", 120.0),
        _event("lookup", 50.0),
        # non-tool events and missing latency are ignored for tool stats
        {
            "step_id": "1",
            "episode_id": "ep",
            "timestamp": 1.0,
            "action_type": "message",
            "action_signature": "m",
            "error": False,
        },
    ]
    traj.write_text("\n".join(json.dumps(r) for r in rows) + "\n")

    profile = fit_baseline_from_jsonl(str(traj))
    assert profile.total_steps == 4
    assert set(profile.tools) == {"search", "lookup"}
    assert abs(profile.tools["search"].mean_latency - 110.0) < 1e-6
    assert abs(profile.tools["lookup"].mean_latency - 50.0) < 1e-6


def test_baseline_skips_malformed_lines(tmp_path):
    traj = tmp_path / "mixed.jsonl"
    traj.write_text(json.dumps(_event("search", 100.0)) + "\n" + "not json\n")
    profile = fit_baseline_from_jsonl(str(traj))
    assert "search" in profile.tools


def test_baseline_save_and_load_roundtrip(tmp_path):
    profile = BaselineProfile()
    for lat in (100.0, 110.0, 90.0):
        profile.add_event(type("E", (), _event("search", lat))())
    out = tmp_path / "baseline.json"
    save_baseline(profile, str(out))
    loaded = load_baseline(str(out))
    assert loaded.total_steps == profile.total_steps
    assert loaded.tools["search"].count == 3
    assert abs(loaded.tools["search"].mean_latency - 100.0) < 1e-6
    assert (
        abs(loaded.tools["search"].std_latency - profile.tools["search"].std_latency)
        < 1e-6
    )


def test_cli_baseline_command(tmp_path, capsys):
    traj = tmp_path / "healthy.jsonl"
    traj.write_text(
        "\n".join(json.dumps(_event("search", lat)) for lat in (100.0, 120.0, 110.0))
        + "\n"
    )
    out = tmp_path / "baseline.json"
    rc = main(["baseline", str(traj), "--output", str(out)])
    assert rc == 0
    assert out.exists()
    out_text = capsys.readouterr().out
    assert "fitted 1 tool(s)" in out_text
    loaded = load_baseline(str(out))
    assert "search" in loaded.tools


def test_fitted_at_roundtrip_and_old_files(tmp_path):
    # New files carry fitted_at; old schema-v1 files without it load as 0.0.
    traj = tmp_path / "h.jsonl"
    traj.write_text(json.dumps(_event("search", 100.0)) + "\n")
    profile = fit_baseline_from_jsonl(str(traj))
    assert profile.fitted_at > 0
    out = tmp_path / "baseline.json"
    save_baseline(profile, str(out))
    data = json.loads(out.read_text())
    assert "fitted_at" in data
    loaded = load_baseline(str(out))
    assert loaded.fitted_at == profile.fitted_at
    # Simulate a pre-128 file without the key.
    old = {k: v for k, v in data.items() if k != "fitted_at"}
    old_path = tmp_path / "old.json"
    old_path.write_text(json.dumps(old))
    old_loaded = load_baseline(str(old_path))
    assert old_loaded.fitted_at == 0.0
    assert old_loaded.tools["search"].mean_latency == 100.0
