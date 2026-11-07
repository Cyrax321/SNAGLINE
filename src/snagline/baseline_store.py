# Copyright 2026 SNAGLINE contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Versioned, per-tenant baseline store (ATTACH_ANY_SYSTEM P1, item 6).

The `goal_drift` detector compares live traffic against a *healthy* reference
profile. In production that reference must be (a) scoped per tenant/deployment
(because "healthy" latency differs across customers and environments) and
(b) versioned, so a retrained baseline can be rolled back if it regresses.

This module is the storage layer: a file-backed ``BaselineStore`` that keeps
the latest profile plus a bounded history of past versions under a root
directory. Heavier backends (DB, object store) can implement the same shape
later; the core stays stdlib-only.
"""

from __future__ import annotations

import copy
import json
import os
import re
import threading
import time
from pathlib import Path
from typing import IO

from snagline.baseline import BaselineProfile, fit_baseline_from_jsonl

# A stored version id becomes a single on-disk filename (``{version}.json``) and
# a sort key for retention. It must therefore be one path-safe component: no
# separators (which would crash ``save`` mid-write once the nested parent is
# missing, or escape the scope dir with ``..``) and no leading ``.``/``-``
# (which rules out ``.``, ``..`` and hidden files). The synthesized default
# ``f"{time.time():.6f}"`` matches this (issue #451).
_SAFE_VERSION_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")


def _validate_version_id(version: str) -> str:
    """Return ``version`` if it is a single path-safe component, else raise.

    Fails loudly at the top of the write/read path instead of mid-write (a
    ``/`` leaves a save half-done after it has advertised the id) or silently
    out of tree (``..`` escapes the scope directory).
    """
    if not isinstance(version, str) or _SAFE_VERSION_RE.fullmatch(version) is None:
        raise ValueError(
            f"invalid baseline version id {version!r}: expected a single "
            r"path-safe component matching [A-Za-z0-9][A-Za-z0-9._-]* "
            r"(no '/', '\', '..', or leading '.'/'-')"
        )
    return version


def _write_json(stream: IO[str], data: dict) -> None:
    json.dump(data, stream, indent=2, sort_keys=True)
    stream.write("\n")


def _atomic_write_json(path: Path, data: dict) -> None:
    """Write ``data`` as JSON so readers see either the old or new file.

    The payload lands in a sibling temp file first, is flushed and fsynced,
    then moved into place with ``os.replace`` (atomic on POSIX and Windows).
    A crash mid-write can therefore never leave a torn or half-written JSON
    file behind; issue #102 relies on this for the version bump.
    """
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        _write_json(fh, data)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


class BaselineStore:
    """File-backed, versioned baseline store keyed by (tenant, deployment)."""

    def __init__(self, root_dir: str, max_versions: int = 10) -> None:
        self._root = Path(root_dir)
        # Issue #332: a limit <= 0 makes _prune delete every version file,
        # including the one this same save() call just wrote, so the store ends
        # up with no history and no pointer while save() still reported success.
        # Retention has no meaningful zero (``latest.json`` is the pointer, so
        # "keep only the latest" is max_versions=1), so reject it at the door.
        if max_versions < 1:
            raise ValueError(f"max_versions must be >= 1; got {max_versions!r}")
        self._max_versions = max_versions

    # --- paths ---------------------------------------------------------------
    def _scope_dir(self, tenant: str, deployment: str) -> Path:
        return self._root / tenant / deployment

    def _version_path(self, tenant: str, deployment: str, version: str) -> Path:
        _validate_version_id(version)
        return self._scope_dir(tenant, deployment) / "versions" / f"{version}.json"

    # --- write ---------------------------------------------------------------
    def save(
        self,
        profile: BaselineProfile,
        tenant: str = "default",
        deployment: str = "default",
        version: str | None = None,
        max_versions: int | None = None,
    ) -> str:
        """Persist ``profile`` and return the version id used.

        Writes both a timestamped history entry and a ``latest.json`` pointer.
        Both files are written atomically (temp + fsync + rename): the history
        entry first, then the pointer flip, so a reader of ``latest.json``
        always sees one complete profile, never a partial write. Old versions
        beyond ``max_versions`` (this call, else the store default) are pruned
        oldest-first by write time, and the version written by *this* call is
        never pruned.

        Raises ``ValueError`` when the effective retention is < 1 (issue #332).
        A per-call override is checked here rather than only at construction,
        because ``save(max_versions=0)`` on a well-configured store would
        otherwise delete the version this call just wrote and hand back a
        version id that no longer resolves. A ``version`` id that is not a
        single path-safe component is rejected before anything is written
        (issue #451).
        """
        version = version or f"{time.time():.6f}"
        limit = max_versions if max_versions is not None else self._max_versions
        # Checked before any file is touched (issue #332): a limit < 1 would
        # make _prune delete the version this call writes, so a failed save
        # must not leave a half-applied store behind either.
        if limit < 1:
            raise ValueError(
                f"max_versions must be >= 1; got {limit!r} (passed to save())"
            )
        # Validate before any filesystem effect: a bad id must not create the
        # versions dir, write a history entry, or flip the latest pointer.
        _validate_version_id(version)
        scope = self._scope_dir(tenant, deployment)
        versions_dir = scope / "versions"
        versions_dir.mkdir(parents=True, exist_ok=True)

        # History entry first: durable even if the process dies before the
        # pointer flip, in which case latest.json still resolves to the
        # previous complete version (issue #102).
        _atomic_write_json(
            self._version_path(tenant, deployment, version), profile.to_dict()
        )
        # Atomic pointer flip to the new version.
        _atomic_write_json(scope / "latest.json", profile.to_dict())

        self._prune(tenant, deployment, limit, keep=version)
        return version

    def _prune(
        self, tenant: str, deployment: str, limit: int, keep: str | None = None
    ) -> None:
        versions_dir = self._scope_dir(tenant, deployment) / "versions"
        if not versions_dir.exists():
            return
        # Order by write time (``st_mtime_ns``), not by filename: string sort
        # only equals chronological order for the fixed-width default ids, so
        # variable-width custom ids ("9" vs "10") pruned lexicographically
        # deleted the wrong version (issue #451). Filename is a stable
        # tiebreak for the rare same-timestamp case.
        keep_name = f"{keep}.json" if keep is not None else None
        entries: list[tuple[int, str, Path]] = []
        for p in versions_dir.glob("*.json"):
            if p.name == keep_name:
                continue  # never prune the version just written
            try:
                mtime = p.stat().st_mtime_ns
            except OSError:
                continue
            entries.append((mtime, p.name, p))
        entries.sort(key=lambda e: (e[0], e[1]))
        # The kept version (if present) occupies one retention slot, so the
        # number to delete counts it against the limit -- but it is never a
        # deletion candidate. At limit < 1 this keeps only the just-written
        # version rather than deleting it, matching "never prune the id just
        # written".
        reserved = 1 if keep_name and (versions_dir / keep_name).exists() else 0
        excess = max(0, len(entries) + reserved - limit)
        for _, _, p in entries[:excess]:
            p.unlink(missing_ok=True)

    # --- read ----------------------------------------------------------------
    def load(
        self, tenant: str = "default", deployment: str = "default"
    ) -> BaselineProfile | None:
        """Return the latest profile, or None if nothing has been stored."""
        latest = self._scope_dir(tenant, deployment) / "latest.json"
        if not latest.exists():
            return None
        return BaselineProfile.from_dict(json.loads(latest.read_text(encoding="utf-8")))

    def load_version(
        self, tenant: str, deployment: str, version: str
    ) -> BaselineProfile | None:
        path = self._version_path(tenant, deployment, version)
        if not path.exists():
            return None
        return BaselineProfile.from_dict(json.loads(path.read_text(encoding="utf-8")))

    def list_versions(
        self, tenant: str = "default", deployment: str = "default"
    ) -> list[str]:
        versions_dir = self._scope_dir(tenant, deployment) / "versions"
        if not versions_dir.exists():
            return []
        # Chronological (write-time) order, oldest first -- the same key
        # ``_prune`` uses, so listing and retention agree even for
        # variable-width custom ids (issue #451). ``_newest_stored_age``
        # reads the newest via ``reversed(...)`` and relies on this order.
        entries: list[tuple[int, str]] = []
        for p in versions_dir.glob("*.json"):
            try:
                mtime = p.stat().st_mtime_ns
            except OSError:
                continue
            entries.append((mtime, p.name[:-5]))
        entries.sort(key=lambda e: (e[0], e[1]))
        return [name for _, name in entries]


def capture_from_jsonl(
    store: BaselineStore,
    trajectory_path: str,
    tenant: str = "default",
    deployment: str = "default",
    version: str | None = None,
    max_versions: int | None = None,
) -> str:
    """Fit a baseline from a healthy-run JSONL trajectory and store it.

    Returns the stored version id. ``max_versions`` overrides the store's
    retention for this write.
    """
    return store.save(
        fit_baseline_from_jsonl(trajectory_path),
        tenant=tenant,
        deployment=deployment,
        version=version,
        max_versions=max_versions,
    )


def retrain_from_jsonl(
    store: BaselineStore,
    window_path: str,
    tenant: str = "default",
    deployment: str = "default",
    max_versions: int | None = None,
) -> str:
    """Refit a baseline from a JSONL window and atomically bump the store.

    This is the library-side primitive behind ``snagline baseline retrain``
    (issue #102): fit a fresh ``BaselineProfile`` from the newest healthy-run
    window, then persist it as a new timestamped version whose pointer flip is
    atomic (see ``BaselineStore.save``). Returns the new version id; the
    previous version stays loadable for rollback.
    """
    return store.save(
        fit_baseline_from_jsonl(window_path),
        tenant=tenant,
        deployment=deployment,
        max_versions=max_versions,
    )


class BaselineCollector:
    """Live auto-capture building block for P1 item 6.

    A host feeds it every ``StepEvent`` during a *known-healthy* run; whenever
    it decides the run is a good reference (a cadence it owns -- e.g. nightly,
    or after N steps), it calls ``commit()`` to persist a versioned baseline.

    ``snapshot()`` returns an independent copy, not the live accumulator: the
    name promises a point-in-time view, but returning the profile itself meant
    a caller that inspected *and mutated* the result (``p.tools.clear()`` while
    deciding whether to ``commit()``) corrupted the profile a later
    ``commit()`` persisted, and a reader iterating it raced ``observe()`` with
    no lock on either side (issue #357). ``observe`` / ``snapshot`` / ``commit``
    share a leaf lock held only over local work -- never across the fsyncing
    save -- so a concurrent ingest cannot reshape a profile mid-copy or
    mid-serialization either.

    Fail-open: a ``commit`` with no store configured is a no-op rather than an
    error, so the collector is safe to drop into any pipeline.
    """

    def __init__(
        self,
        store: BaselineStore | None = None,
        tenant: str = "default",
        deployment: str = "default",
        max_versions: int | None = None,
    ) -> None:
        self._store = store
        self._tenant = tenant
        self._deployment = deployment
        self._max_versions = max_versions
        self._profile = BaselineProfile()
        self._lock = threading.Lock()

    def observe(self, event) -> None:
        with self._lock:
            self._profile.add_event(event)

    def snapshot(self) -> BaselineProfile:
        # deepcopy, not the live object: every other snapshot-style accessor in
        # the codebase (SidecarMetricsCollector.snapshot, Monitor.snapshot, the
        # detectors' dump_state) copies, and this one used to hand out the very
        # object observe() keeps mutating (issue #357).
        with self._lock:
            return copy.deepcopy(self._profile)

    def commit(self, version: str | None = None) -> str | None:
        if self._store is None:
            return None
        with self._lock:
            # Record fit time so --max-age works even with custom version ids.
            self._profile.fitted_at = time.time()
            # Copy before the (slow, fsyncing) save: the lock is not held
            # across disk I/O, and a concurrent observe() must not be able to
            # reshape the profile while save() serializes it.
            to_save = copy.deepcopy(self._profile)
        return self._store.save(
            to_save,
            tenant=self._tenant,
            deployment=self._deployment,
            version=version,
            max_versions=self._max_versions,
        )
