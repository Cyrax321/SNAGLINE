"""The Monitor orchestrator -- fail-open by construction.

This is the heart of the adoption guarantee (project.md §1.2 / §4.6): a
monitoring library that can crash or stall the thing it monitors is a
non-starter. Therefore every ``observe`` and every ``emit`` call is wrapped so
that, by default (``fail_open=True``), any exception is logged and swallowed
rather than propagated. Only when ``fail_open=False`` is explicitly set do
exceptions propagate -- useful for tests and for callers who want strict
behavior.

The lock is per-instance, so a single Monitor can be shared across multiple
episodes / threads concurrently.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from typing import Any

from snagline.config import Config
from snagline.detectors.base import Detector
from snagline.events import StepEvent
from snagline.risk import FailureRisk
from snagline.sinks.base import AlertSink
from snagline.state import StateBackend, default_state_backend

logger = logging.getLogger("snagline")

# Bump when the snapshot payload shape changes incompatibly (issue #91).
SNAPSHOT_FORMAT_VERSION = 1


def _detector_key(index: int, detector: Any) -> str:
    """Stable per-position key for a detector inside a snapshot payload."""
    return f"{index}:{getattr(detector, 'name', type(detector).__name__)}"


class MonitorMetrics:
    """Self-observability counters for a Monitor (P3, item 10).

    Lets an operator see, without touching the host, how much traffic the
    monitor is handling and whether its detectors/sinks are faulting. Counts
    are incremented under a dedicated lock so concurrent ingest stays safe.
    """

    def __init__(self) -> None:
        self.events_ingested = 0
        self.risks_emitted = 0
        self.detector_errors = 0
        self.sink_errors = 0

    def as_dict(self) -> dict:
        return {
            "events_ingested": self.events_ingested,
            "risks_emitted": self.risks_emitted,
            "detector_errors": self.detector_errors,
            "sink_errors": self.sink_errors,
        }


class Monitor:
    """Runs every registered detector against each ingested event and
    dispatches any resulting ``FailureRisk`` to every registered sink.

    Fail-open: detector/sink exceptions are caught and logged, never
    propagated into the host agent, unless ``fail_open=False``.

    Concurrency is sharded by ``episode_id`` via a ``StateBackend``: two
    distinct episodes can be observed concurrently, while a single episode's
    events serialize. This removes the single global ingest lock that would
    bottleneck a high-throughput service (P1, item 4).
    """

    def __init__(
        self,
        detectors: list[Detector],
        sinks: list[AlertSink],
        fail_open: bool = True,
        state_backend: StateBackend | None = None,
    ) -> None:
        self._detectors = list(detectors)
        self._sinks = list(sinks)
        self._fail_open = fail_open
        self._state = state_backend or default_state_backend()
        # A faulty detector or sink must not spam the logs on every single step
        # (issue #14) -- log each distinct fault exactly once.
        self._fault_lock = threading.Lock()
        self._fault_logged: set[str] = set()
        self._metrics = MonitorMetrics()
        self._metrics_lock = threading.Lock()

    def ingest(self, event: StepEvent) -> None:
        """Run all detectors against ``event`` and dispatch any risks.

        Never raises under the default ``fail_open=True``. With
        ``fail_open=False``, a detector or sink exception propagates.

        Detection runs under a per-instance lock, but dispatch to sinks happens
        *outside* that lock (issue #8): a slow or network-bound sink (e.g. the
        webhook sink) must not block concurrent ``ingest`` calls, and a sink
        that re-enters ``ingest`` must not deadlock.
        """
        risks: list[FailureRisk] = []
        with self._state.episode_lock(event.episode_id):
            for detector in self._detectors:
                try:
                    risk = detector.observe(event)
                except Exception:
                    self._incr("detector_errors")
                    self._log_fault_once(
                        f"detector {getattr(detector, 'name', repr(detector))} "
                        "raised; ignoring (fail-open)"
                    )
                    if not self._fail_open:
                        raise
                    continue
                if risk is not None:
                    risks.append(risk)
        with self._metrics_lock:
            self._metrics.events_ingested += 1
            self._metrics.risks_emitted += len(risks)
        for risk in risks:
            self._dispatch(risk)

    def _dispatch(self, risk: FailureRisk) -> None:
        for sink in self._sinks:
            try:
                sink.emit(risk)
            except Exception:
                self._incr("sink_errors")
                self._log_fault_once(
                    f"sink {type(sink).__name__} raised; ignoring (fail-open)"
                )
                if not self._fail_open:
                    raise

    def _incr(self, name: str) -> None:
        with self._metrics_lock:
            setattr(self._metrics, name, getattr(self._metrics, name) + 1)

    def metrics(self) -> dict:
        """Return a snapshot of self-observability counters (P3, item 10)."""
        with self._metrics_lock:
            return self._metrics.as_dict()

    def _log_fault_once(self, key: str) -> None:
        """Log a fail-open fault, but only the first time we see this exact
        message (issue #14). Avoids dumping a traceback on every step when a
        detector or sink is consistently broken."""
        with self._fault_lock:
            if key in self._fault_logged:
                return
            self._fault_logged.add(key)
        logger.error("snagline: %s", key)

    async def ingest_async(self, event: StepEvent) -> None:
        """Thin async wrapper for async adapters (LangGraph/AutoGen async
        mode). Runs the same cheap sync path -- detectors are cheap enough
        that this is safe.
        """
        self.ingest(event)

    def end_episode(self, episode_id: str) -> None:
        """Signal that ``episode_id`` has finished; judge it, then clear state.

        Two phases:

        1. *Judgment* -- detectors exposing ``finalize(episode_id)`` (the
           ``EpisodeFinalizer`` duck-typed extension point, issue #86) are
           called once; e.g. the silent-abort completion check can only decide
           now that no output step ever came. Returned risks are dispatched
           like any other.
        2. *Teardown* -- every detector's ``reset(episode_id)`` drops
           per-episode state, and the state backend releases the episode.

        Fail-open throughout: a ``finalize`` or ``reset`` exception is logged,
        never propagated (unless ``fail_open=False``). Dispatch happens
        outside the episode lock, mirroring ``ingest``.
        """
        finalized: list[FailureRisk] = []
        with self._state.episode_lock(episode_id):
            for detector in self._detectors:
                finalize = getattr(detector, "finalize", None)
                if callable(finalize):
                    try:
                        risk = finalize(episode_id)
                    except Exception:
                        self._log_fault_once(
                            f"detector {getattr(detector, 'name', repr(detector))} "
                            "finalize raised; ignoring (fail-open)"
                        )
                        if not self._fail_open:
                            raise
                    else:
                        if risk is not None:
                            finalized.append(risk)
                try:
                    detector.reset(episode_id)
                except Exception:
                    self._log_fault_once(
                        f"detector {getattr(detector, 'name', repr(detector))} "
                        "reset raised; ignoring (fail-open)"
                    )
                    if not self._fail_open:
                        raise
            # Inside the lock: no other thread can be in this episode's
            # critical section, so dropping the entry cannot strand a holder.
            release = getattr(self._state, "release", None)
            if callable(release):
                try:
                    release(episode_id)
                except Exception:
                    self._log_fault_once(
                        f"state backend {type(self._state).__name__} release "
                        "raised; ignoring (fail-open)"
                    )
                    if not self._fail_open:
                        raise
        if finalized:
            with self._metrics_lock:
                self._metrics.risks_emitted += len(finalized)
            for risk in finalized:
                self._dispatch(risk)

    # --- Restart-survivable state (issue #91) --------------------------------
    #
    # All per-episode detector state used to live only in instance memory, so
    # a deploy or crash reset every window and CUSUM baseline mid-episode.
    # Snapshots are plain stdlib JSON (never pickle) written atomically via
    # tmp-file + os.replace. Restore is a *setup-time* operation: malformed
    # payloads raise rather than fail open, because a monitor that silently
    # starts blank is precisely the quiet misbehavior this feature exists to
    # prevent. Runtime detection itself stays fail-open as always.

    def snapshot_dict(self) -> dict[str, Any]:
        """Return all restorable detector/sink state as a JSON-ready dict."""
        detectors: dict[str, Any] = {}
        for i, detector in enumerate(self._detectors):
            dump = getattr(detector, "dump_state", None)
            detectors[_detector_key(i, detector)] = dump() if callable(dump) else None
        sinks: dict[str, Any] = {}
        for i, sink in enumerate(self._sinks):
            dump = getattr(sink, "dump_state", None)
            sinks[f"{i}:{type(sink).__name__}"] = dump() if callable(dump) else None
        return {
            "format_version": SNAPSHOT_FORMAT_VERSION,
            "detectors": detectors,
            "sinks": sinks,
        }

    def snapshot(self, path: str) -> None:
        """Atomically write a JSON snapshot of all detector/sink state."""
        payload = json.dumps(self.snapshot_dict(), indent=2, sort_keys=True)
        tmp = f"{path}.tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            fh.write(payload)
        os.replace(tmp, path)

    def restore_dict(self, data: dict[str, Any], strict_names: bool = False) -> None:
        """Restore detector/sink state produced by ``snapshot_dict``.

        ``strict_names=True`` additionally requires the current detector
        composition to match the snapshot exactly (order included); the
        default is tolerant: matching entries are applied by key, missing
        detectors are skipped with a warning, and states without a home are
        ignored with a warning.
        """
        version = data.get("format_version")
        if version != SNAPSHOT_FORMAT_VERSION:
            raise ValueError(
                f"snapshot format_version {version!r} incompatible with "
                f"{SNAPSHOT_FORMAT_VERSION}"
            )
        dumped_detectors: dict[str, Any] = data.get("detectors") or {}
        consumed: set[str] = set()
        for i, detector in enumerate(self._detectors):
            load = getattr(detector, "load_state", None)
            if not callable(load):
                continue
            key = _detector_key(i, detector)
            matched_key: str | None = None
            entry = dumped_detectors.get(key)
            if entry is not None:
                matched_key = key
            else:
                # Fallback: same-name state recorded at a different position.
                name = getattr(detector, "name", None)
                if name is not None:
                    for k, v in dumped_detectors.items():
                        if (
                            k not in consumed
                            and k.endswith(f":{name}")
                            and v is not None
                        ):
                            entry = v
                            matched_key = k
                            break
            if entry is None or matched_key is None:
                continue
            load(entry)
            consumed.add(matched_key)
        orphaned = {
            k
            for k, v in dumped_detectors.items()
            if v is not None and k not in consumed
        }
        if orphaned:
            logger.warning(
                "snagline: snapshot carried state for %d unknown detector "
                "slot(s); ignored",
                len(orphaned),
            )
        if strict_names:
            expected = [k.split(":", 1)[1] for k in sorted(dumped_detectors)]
            current = [getattr(d, "name", type(d).__name__) for d in self._detectors]
            if expected != current:
                raise ValueError(
                    "snapshot detector composition mismatch: "
                    f"snapshot={expected} monitor={current}"
                )
        dumped_sinks: dict[str, Any] = data.get("sinks") or {}
        for i, sink in enumerate(self._sinks):
            load = getattr(sink, "load_state", None)
            key = f"{i}:{type(sink).__name__}"
            if callable(load) and dumped_sinks.get(key) is not None:
                load(dumped_sinks[key])

    def restore(self, path: str, strict_names: bool = False) -> None:
        """Load a JSON snapshot written by :meth:`snapshot`."""
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        self.restore_dict(data, strict_names=strict_names)

    @classmethod
    def default(
        cls,
        config: Config | None = None,
        sinks: list[AlertSink] | None = None,
        state_backend: StateBackend | None = None,
    ) -> Monitor:
        """Construct a zero-configuration Monitor with sensible defaults.

        Wires up the three tier-1 detectors that ship in this build
        (loop, error-cascade, and the latency-anomaly/CUSUM detector) and,
        unless ``sinks`` is given, the console sink.
        """
        from snagline.detectors.error_cascade import ErrorCascadeDetector
        from snagline.detectors.goal_drift import GoalDriftDetector
        from snagline.detectors.latency_anomaly import LatencyAnomalyDetector
        from snagline.detectors.loop import LoopDetector
        from snagline.detectors.meltdown import MeltdownDetector
        from snagline.detectors.ml_ensemble import MLOrchestrator
        from snagline.detectors.silent_abort import SilentAbortDetector
        from snagline.detectors.token_runaway import TokenRunawayDetector
        from snagline.sinks.console import ConsoleSink

        cfg = config or Config()
        base: list[Any] = [
            LoopDetector(config=cfg),
            ErrorCascadeDetector(config=cfg),
            LatencyAnomalyDetector(config=cfg),
        ]
        # Goal-drift is opt-in: only when explicitly enabled and a baseline is
        # supplied, so the zero-dependency default is unchanged (step 2).
        if cfg.goal_drift_enabled and cfg.goal_drift_baseline is not None:
            base.append(GoalDriftDetector(baseline=cfg.goal_drift_baseline, config=cfg))
        # Horizon detectors are opt-in too (issues #84/#85/#86): each needs
        # telemetry or validation the zero-config preset does not promise.
        if cfg.token_runaway_enabled:
            base.append(TokenRunawayDetector(config=cfg))
        if cfg.meltdown_enabled:
            base.append(MeltdownDetector(config=cfg))
        if cfg.silent_abort_enabled:
            base.append(SilentAbortDetector(config=cfg))
        if cfg.ml_ensemble_enabled:
            # Combine the base detectors into one orchestrated signal (step 3).
            detectors: list[Detector] = [MLOrchestrator(base, config=cfg)]
        else:
            detectors = list(base)
        chosen_sinks: list[AlertSink] = sinks if sinks is not None else [ConsoleSink()]
        monitor = cls(detectors, chosen_sinks, fail_open=cfg.fail_open)
        # Set after construction so subclasses with a fixed __init__ signature
        # (e.g. test doubles) still work; base Monitor.__init__ also sets it.
        monitor._state = state_backend or default_state_backend()
        return monitor
