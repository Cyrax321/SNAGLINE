"""External liveness evidence (issue #92).

In-band detectors can never see a hung host: no events means no ``observe()``
calls, so silence itself is invisible to anything living inside the event
stream. The heartbeat closes that hole from the outside: it touches a
liveness file's mtime on every ingest, and any external supervisor (cron,
systemd timer, k8s probe, CONTINUUM's sidecar) alerts when the mtime goes
stale. The watcher owns the alerting decision; snagline just leaves evidence
of life.

Stdlib only (~five lines of effect), zero new failure surface: every touch is
guarded so a missing directory or an unwritable path logs once and never
raises into the host (project.md §1.2). No content is ever written -- the
file carries nothing but its own mtime.
"""

from __future__ import annotations

import logging
import os

from snagline.risk import FailureRisk

logger = logging.getLogger("snagline")


class HeartbeatSink:
    """Touch a liveness file's mtime so silence becomes externally detectable.

    Driven per-ingest by the CLI (``snagline watch --heartbeat PATH``) because
    "no risks" is exactly the signal: wiring it as a passive AlertSink would
    only prove that alarms flow, not that the host is alive. ``emit`` touches
    anyway, so the object may also sit in a sinks list without harm.
    """

    def __init__(self, path: str) -> None:
        self._path = path
        self._fault_logged = False

    def touch(self) -> None:
        """Best-effort mtime bump; fail-open, logged once until it recovers."""
        try:
            try:
                # Fast path: the file exists (two syscalls, no content).
                os.utime(self._path, None)
            except FileNotFoundError:
                # Missing file or directory: create parents once, create the
                # file, stamp it. Handled fail-open like every other branch.
                parent = os.path.dirname(self._path) or "."
                os.makedirs(parent, exist_ok=True)
                fd = os.open(self._path, os.O_WRONLY | os.O_CREAT, 0o644)
                os.close(fd)
                os.utime(self._path, None)
            self._fault_logged = False
        except OSError as exc:
            # Unwritable path, read-only filesystem, whatever: never raise
            # into the host agent, just say so once (issue #14 style).
            if not self._fault_logged:
                self._fault_logged = True
                logger.warning(
                    "snagline: heartbeat %s unavailable (%s); continuing "
                    "without liveness evidence",
                    self._path,
                    exc,
                )

    def emit(self, risk: FailureRisk) -> None:
        """AlertSink compatibility: a risk flowing is also evidence of life."""
        self.touch()
