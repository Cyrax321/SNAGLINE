"""HeartbeatSink: external liveness evidence (issue #92).

Properties under test: the liveness file is created on first touch and its
mtime advances on every later touch; a missing directory is handled fail-open;
an unwritable path never raises into the host (logged once instead).
"""

from __future__ import annotations

import os
import time

from snagline.sinks.heartbeat import HeartbeatSink


def test_touch_creates_file_then_advances_mtime(tmp_path) -> None:
    hb_path = tmp_path / "hb"
    hb = HeartbeatSink(str(hb_path))
    hb.touch()
    assert hb_path.exists()
    first = os.stat(hb_path).st_mtime_ns
    time.sleep(0.01)
    hb.touch()
    second = os.stat(hb_path).st_mtime_ns
    assert second > first


def test_missing_directory_created_fail_open(tmp_path) -> None:
    nested = tmp_path / "var" / "run" / "snagline" / "hb"
    hb = HeartbeatSink(str(nested))
    hb.touch()  # must not raise
    assert nested.exists()


def test_unwritable_path_never_raises(tmp_path) -> None:
    # A directory where the file should be: every touch fails at the OS level.
    blocker = tmp_path / "blocker"
    blocker.mkdir()
    hb = HeartbeatSink(str(blocker))
    for _ in range(3):
        hb.touch()  # logged once, swallowed forever; never raises


def test_emit_is_also_a_touch(tmp_path) -> None:
    hb_path = tmp_path / "hb"
    hb = HeartbeatSink(str(hb_path))

    from snagline.risk import FailureRisk

    hb.emit(FailureRisk("ep", "s", 0.5, "loop", "", 0.0))
    assert hb_path.exists()


def test_no_content_ever_written(tmp_path) -> None:
    hb_path = tmp_path / "hb"
    HeartbeatSink(str(hb_path)).touch()
    assert hb_path.read_bytes() == b""
