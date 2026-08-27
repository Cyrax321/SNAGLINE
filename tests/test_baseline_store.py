"""Tests for the versioned, per-tenant BaselineStore (P1 item 6)."""

from __future__ import annotations

import json

from snagline.baseline import BaselineProfile
from snagline.baseline_store import (
    BaselineStore,
    capture_from_jsonl,
)


def _trajectory(tmp_path):
    path = tmp_path / "healthy.jsonl"
    rows = [
        {
            "step_id": str(i),
            "episode_id": "ep",
            "timestamp": 1.0 + i,
            "action_type": "tool_call",
            "action_signature": "SIG",
            "tool_name": "search",
            "latency_ms": 100.0 + i,
        }
        for i in range(5)
    ]
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return str(path)


def test_store_save_and_load_latest(tmp_path):
    store = BaselineStore(str(tmp_path / "store"))
    profile = BaselineProfile()
    profile.total_steps = 5
    store.save(profile, tenant="acme", deployment="prod")
    loaded = store.load(tenant="acme", deployment="prod")
    assert loaded is not None
    assert loaded.total_steps == 5


def test_store_versions_and_rollback(tmp_path):
    store = BaselineStore(str(tmp_path / "store"))
    v1 = store.save(BaselineProfile(), tenant="t", deployment="d")
    p2 = BaselineProfile()
    p2.total_steps = 9
    v2 = store.save(p2, tenant="t", deployment="d")
    versions = store.list_versions("t", "d")
    assert versions == [v1, v2]
    # load() returns the newest.
    assert store.load("t", "d").total_steps == 9  # type: ignore[union-attr]
    # A specific older version can still be fetched.
    assert store.load_version("t", "d", v1).total_steps == 0  # type: ignore[union-attr]


def test_store_isolation_by_tenant(tmp_path):
    store = BaselineStore(str(tmp_path / "store"))
    p = BaselineProfile()
    p.total_steps = 3
    store.save(p, tenant="acme", deployment="prod")
    # Different tenant has nothing.
    assert store.load(tenant="other", deployment="prod") is None


def test_store_prunes_old_versions(tmp_path):
    store = BaselineStore(str(tmp_path / "store"), max_versions=3)
    for _ in range(5):
        store.save(BaselineProfile(), tenant="t", deployment="d")
    assert len(store.list_versions("t", "d")) == 3


def test_capture_from_jsonl_fits_and_stores(tmp_path):
    store = BaselineStore(str(tmp_path / "store"))
    version = capture_from_jsonl(
        store, _trajectory(tmp_path), tenant="acme", deployment="prod"
    )
    loaded = store.load_version("acme", "prod", version)
    assert loaded is not None
    assert loaded.total_steps == 5
    assert "search" in loaded.tools
