"""Tests for the versioned, per-tenant BaselineStore (P1 item 6)."""

from __future__ import annotations

import json

import pytest

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


def test_prune_is_chronological_not_lexicographic(tmp_path):
    # Issue #451 defect 1: variable-width custom ids pruned by string sort
    # deletes the wrong version. "9" sorts after "10"/"11" lexicographically,
    # so lexicographic pruning kept "9" (the actual oldest) and dropped "10".
    store = BaselineStore(str(tmp_path / "store"), max_versions=2)
    for v in ("9", "10", "11"):
        store.save(BaselineProfile(), tenant="t", deployment="d", version=v)
    assert sorted(store.list_versions("t", "d")) == ["10", "11"]


def test_prune_never_deletes_the_just_written_version(tmp_path):
    # Issue #451 defect 1, sharpest case: at max_versions=1, writing "b" then
    # "a" must keep "a" (the newest), not delete it. Lexicographic pruning
    # deleted "a.json" while latest.json still resolved to it -- the newest
    # baseline was unrecoverable for rollback.
    store = BaselineStore(str(tmp_path / "store"), max_versions=1)
    store.save(BaselineProfile(), tenant="t", deployment="d", version="b")
    p = BaselineProfile()
    p.total_steps = 7
    store.save(p, tenant="t", deployment="d", version="a")
    assert store.list_versions("t", "d") == ["a"]
    rolled_back = store.load_version("t", "d", "a")
    assert rolled_back is not None and rolled_back.total_steps == 7


@pytest.mark.parametrize("bad", ["a/b", "../../evil", "..", ".", "  ", "a\\b", "-x"])
def test_save_rejects_unsafe_version_ids(tmp_path, bad):
    # Issue #451 defect 2: a version id with a path separator crashed save()
    # mid-write *after* advertising the id, and ".." escaped the scope dir.
    # Reject up front, before any file is written.
    store = BaselineStore(str(tmp_path / "store"))
    with pytest.raises(ValueError, match="version id"):
        store.save(BaselineProfile(), tenant="t", deployment="d", version=bad)
    # Nothing leaked outside the scope's versions/ dir (and no partial write).
    stray = [
        p
        for p in (tmp_path / "store").rglob("*.json")
        if p.parent.name != "versions" and p.name != "latest.json"
    ]
    assert stray == [], f"unsafe id wrote outside versions/: {stray}"


def test_save_empty_version_falls_back_to_safe_default(tmp_path):
    # An empty id is falsy and folds to the synthesized time-based default,
    # which is path-safe -- so it must NOT raise (the reject path is for
    # truthy-but-unsafe ids like whitespace or separators).
    store = BaselineStore(str(tmp_path / "store"))
    v = store.save(BaselineProfile(), tenant="t", deployment="d", version="")
    assert store.load_version("t", "d", v) is not None


def test_load_version_rejects_unsafe_id(tmp_path):
    # The read path builds the same on-disk path, so ".." on load must not
    # traverse out of the scope dir either.
    store = BaselineStore(str(tmp_path / "store"))
    store.save(BaselineProfile(), tenant="t", deployment="d", version="v1")
    with pytest.raises(ValueError, match="version id"):
        store.load_version("t", "d", "../../../etc/passwd")
