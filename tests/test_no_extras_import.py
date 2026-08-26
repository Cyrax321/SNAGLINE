"""No-extras import smoke check (project.md §1.1, issue #92 gate).

Bare ``import snagline`` and every module touched by a change must import on
a stdlib-only interpreter: no numpy, no sentence-transformers, nothing. The
optional extras stay behind lazy, guarded imports inside functions.
"""

from __future__ import annotations

import importlib
import subprocess
import sys

NEW_OR_TOUCHED_MODULES = (
    "snagline",
    "snagline.cli",
    "snagline.config",
    "snagline.monitor",
    "snagline.risk",
    "snagline.sinks.heartbeat",
    "snagline.detectors.windowing",
    "snagline.detectors.loop",
    "snagline.detectors.error_cascade",
    "snagline.detectors.latency_anomaly",
    "snagline.detectors.meltdown",
    "snagline.detectors.stagnation",
)


def test_bare_import_works_stdlib_only() -> None:
    script = ";".join(f"import {m}" for m in NEW_OR_TOUCHED_MODULES)
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            f"import sys; {script}; "
            "banned = {'numpy', 'sklearn', 'sentence_transformers', 'redis'};"
            "assert not (set(sys.modules) & banned), 'extra leaked into core';"
            "print('ok')",
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "ok"


def test_touched_modules_import_in_process() -> None:
    for name in NEW_OR_TOUCHED_MODULES:
        assert importlib.import_module(name) is not None
