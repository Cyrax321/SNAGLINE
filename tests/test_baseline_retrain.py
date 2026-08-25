"""Scheduled baseline retrain contract (issue #102).

Covers the ``snagline baseline retrain`` CLI mode and the
``retrain_from_jsonl`` store primitive: round-trip through BaselineStore
versioning, crash-safe atomic pointer flip, newest-window selection, the
``--max-age`` staleness warning (fires when stale AND stays silent when
fresh), graceful failure paths, and unchanged behavior of the legacy flat
``snagline baseline <trajectory>`` form.
"""

from __future__ import annotations

import json
import os
import time

import pytest

from snagline.baseline_store import (
    BaselineStore,
    capture_from_jsonl,
    retrain_from_jsonl,
)
from snagline.cli import _active_baseline_age, main


class BaselineProfileStub:
    """Minimal profile stand-in: the store only persists its dict shape."""

    def __init__(self) -> None:
        self.total_steps = 1

    @property
    def tools(self) -> dict:
        return {}

    def to_dict(self) -> dict:
        return {"version": 1, "total_steps": 1, "tools": {}}


def _window(path, tool: str, latencies: list[float]) -> str:
    rows = [
        {
            "step_id": str(i),
            "episode_id": "ep",
            "timestamp": 1.0 + i,
            "action_type": "tool_call",
            "action_signature": "SIG",
            "tool_name": tool,
            "latency_ms": latency,
        }
        for i, latency in enumerate(latencies)
    ]
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return str(path)


def test_retrain_round_trips_through_versioning(tmp_path):
    """Refit lands as a NEW version; old version stays rollback-loadable."""
    store_dir = tmp_path / "store"
    win_a = _window(tmp_path / "a.jsonl", "search", [100.0, 102.0, 104.0, 106.0, 108.0])
    win_b = _window(tmp_path / "b.jsonl", "search", [200.0, 202.0, 204.0, 206.0, 208.0])

    store = BaselineStore(str(store_dir))
    v1 = capture_from_jsonl(store, win_a, tenant="acme", deployment="prod")
    v2 = retrain_from_jsonl(store, win_b, tenant="acme", deployment="prod")

    # The bump really produced a distinct, newer version id.
    assert v2 != v1
    assert store.list_versions("acme", "prod") == [v1, v2]
    # Active profile now reflects the NEW window (mean of B computed by hand).
    active = store.load("acme", "prod")
    assert active is not None
    assert active.total_steps == 5
    assert active.tools["search"].mean_latency == 204.0
    assert active.tools["search"].count == 5
    assert active.tools["search"].error_rate == 0.0
    # Rollback path: the pre-retrain version still holds window A's stats.
    old = store.load_version("acme", "prod", v1)
    assert old is not None
    assert old.tools["search"].mean_latency == 104.0


def test_retrain_pointer_flip_is_crash_safe(tmp_path, monkeypatch):
    """A crash between history write and pointer flip loses nothing.

    Fault injection on stdlib plumbing only (os.replace): the second rename
    (the latest.json pointer flip) fails. The active baseline must remain the
    previous complete version, byte-identical, and a later retrain recovers.
    """
    import snagline.baseline_store as bs

    store_dir = tmp_path / "store"
    win_a = _window(tmp_path / "a.jsonl", "search", [100.0] * 4)
    win_b = _window(tmp_path / "b.jsonl", "search", [300.0] * 4)

    store = BaselineStore(str(store_dir))
    v1 = capture_from_jsonl(store, win_a, tenant="t", deployment="d")
    pointer = store_dir / "t" / "d" / "latest.json"
    before = pointer.read_bytes()

    real_replace = bs.os.replace
    calls = {"n": 0}

    def flaky_replace(src, dst):
        calls["n"] += 1
        if calls["n"] == 2:
            raise OSError("simulated crash before pointer flip")
        return real_replace(src, dst)

    monkeypatch.setattr(bs.os, "replace", flaky_replace)
    with pytest.raises(OSError, match="simulated crash"):
        retrain_from_jsonl(store, win_b, tenant="t", deployment="d")
    monkeypatch.undo()

    # Pointer untouched: active baseline is still the complete old version.
    assert pointer.read_bytes() == before
    active = store.load("t", "d")
    assert active is not None
    assert active.tools["search"].mean_latency == 100.0
    # The history entry for the interrupted attempt may exist as an orphan;
    # it must be a complete JSON file either way (never torn).
    versions_before_recovery = store.list_versions("t", "d")
    for vid in versions_before_recovery:
        raw = (store_dir / "t" / "d" / "versions" / f"{vid}.json").read_bytes()
        json.loads(raw)
    # Recovery: a later retrain flips the pointer to the new window cleanly.
    v_new = retrain_from_jsonl(store, win_b, tenant="t", deployment="d")
    recovered = store.load("t", "d")
    assert recovered is not None
    assert recovered.tools["search"].mean_latency == 300.0
    assert v_new != v1


def test_cli_retrain_jsonl_round_trip(tmp_path, capsys):
    store_dir = tmp_path / "store"
    win_b = _window(tmp_path / "w.jsonl", "fetch", [50.0, 52.0, 54.0])

    rc = main(
        [
            "baseline",
            "retrain",
            "--store-dir",
            str(store_dir),
            "--tenant",
            "acme",
            "--deployment",
            "prod",
            "--jsonl",
            win_b,
        ]
    )

    assert rc == 0
    out = capsys.readouterr().out
    assert "stored version" in out
    assert "acme/prod" in out
    store = BaselineStore(str(store_dir))
    active = store.load("acme", "prod")
    assert active is not None
    assert active.tools["fetch"].mean_latency == 52.0
    assert active.total_steps == 3


def test_cli_retrain_windows_dir_picks_newest_by_mtime(tmp_path, capsys):
    store_dir = tmp_path / "store"
    older = tmp_path / "win-0001.jsonl"
    newer = tmp_path / "win-0002.jsonl"
    _window(older, "search", [100.0] * 3)
    _window(newer, "search", [400.0] * 3)
    # Fixed timestamps, no sleeps: make mtimes unambiguous.
    old_ts, new_ts = 1600000000, 1600000600
    os.utime(older, (old_ts, old_ts))
    os.utime(newer, (new_ts, new_ts))

    rc = main(
        [
            "baseline",
            "retrain",
            "--store-dir",
            str(store_dir),
            "--windows-dir",
            str(tmp_path),
        ]
    )

    assert rc == 0
    out = capsys.readouterr().out
    assert "win-0002.jsonl" in out  # picked the newer window explicitly
    active = BaselineStore(str(store_dir)).load("default", "default")
    assert active is not None
    assert active.tools["search"].mean_latency == 400.0


def test_cli_retrain_max_age_warns_on_stale_baseline(tmp_path, capsys):
    store_dir = tmp_path / "store"
    win_a = _window(tmp_path / "a.jsonl", "search", [100.0] * 3)
    win_b = _window(tmp_path / "b.jsonl", "search", [110.0] * 3)
    store = BaselineStore(str(store_dir))
    # Fixed ancient version id: wall-clock unix time far in the past.
    capture_from_jsonl(
        store, win_a, tenant="acme", deployment="prod", version="1000000000.0"
    )

    rc = main(
        [
            "baseline",
            "retrain",
            "--store-dir",
            str(store_dir),
            "--tenant",
            "acme",
            "--deployment",
            "prod",
            "--jsonl",
            win_b,
            "--max-age",
            "3600",
        ]
    )

    assert rc == 0
    err = capsys.readouterr().err
    assert "WARNING" in err
    assert "acme/prod" in err
    assert "retrain cadence" in err


def test_cli_retrain_max_age_silent_when_fresh(tmp_path, capsys):
    store_dir = tmp_path / "store"
    win_a = _window(tmp_path / "a.jsonl", "search", [100.0] * 3)
    win_b = _window(tmp_path / "b.jsonl", "search", [110.0] * 3)
    store = BaselineStore(str(store_dir))
    capture_from_jsonl(
        store,
        win_a,
        tenant="acme",
        deployment="prod",
        version=f"{time.time():.6f}",
    )

    rc = main(
        [
            "baseline",
            "retrain",
            "--store-dir",
            str(store_dir),
            "--tenant",
            "acme",
            "--deployment",
            "prod",
            "--jsonl",
            win_b,
            "--max-age",
            "3600",
        ]
    )

    assert rc == 0
    err = capsys.readouterr().err
    assert "WARNING" not in err


def test_cli_retrain_usage_and_input_errors(tmp_path, capsys):
    store_dir = tmp_path / "store"
    win = _window(tmp_path / "w.jsonl", "search", [100.0] * 2)

    # Missing --store-dir.
    assert main(["baseline", "retrain", "--jsonl", win]) == 2
    # Neither nor both window inputs.
    assert main(["baseline", "retrain", "--store-dir", str(store_dir)]) == 2
    assert (
        main(
            [
                "baseline",
                "retrain",
                "--store-dir",
                str(store_dir),
                "--jsonl",
                win,
                "--windows-dir",
                str(tmp_path),
            ]
        )
        == 2
    )
    # Empty windows directory: resolvable inputs, no window found.
    empty = tmp_path / "empty"
    empty.mkdir()
    assert (
        main(
            [
                "baseline",
                "retrain",
                "--store-dir",
                str(store_dir),
                "--windows-dir",
                str(empty),
            ]
        )
        == 3
    )
    # Unreadable window file: graceful exit, no traceback.
    assert (
        main(
            [
                "baseline",
                "retrain",
                "--store-dir",
                str(store_dir),
                "--jsonl",
                str(tmp_path / "missing.jsonl"),
            ]
        )
        == 3
    )
    captured = capsys.readouterr()
    combined = captured.out + captured.err
    assert "no .jsonl window files found" in combined
    assert "cannot read" in combined


def test_active_baseline_age_skips_non_numeric_ids(tmp_path):
    store = BaselineStore(str(tmp_path / "store"))
    profile = BaselineProfileStub()
    store.save(profile, tenant="t", deployment="d", version="v-custom")
    # Only custom ids: age is unknowable, helper reports None (fail-open).
    assert _active_baseline_age(store, "t", "d") is None
    # Numeric timestamped id alongside custom ids: age comes from the newest
    # numeric one only.
    now_id = f"{time.time():.6f}"
    store.save(profile, tenant="t", deployment="d", version=now_id)
    age = _active_baseline_age(store, "t", "d")
    assert age is not None
    assert 0.0 <= age < 60.0


def test_legacy_flat_baseline_form_unchanged(tmp_path):
    """'retrain' dispatch must not disturb plain trajectory fitting."""
    traj = _window(tmp_path / "h.jsonl", "search", [100.0, 100.0, 100.0])
    out = tmp_path / "baseline.json"

    rc = main(["baseline", traj, "--output", str(out)])

    assert rc == 0
    data = json.loads(out.read_text())
    assert data["tools"]["search"]["mean_latency"] == 100.0
