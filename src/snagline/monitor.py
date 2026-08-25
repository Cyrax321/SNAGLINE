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

import logging
import threading
from typing import Any

from snagline.config import Config
from snagline.detectors.base import Detector
from snagline.events import StepEvent
from snagline.risk import FailureRisk
from snagline.sinks.base import AlertSink
from snagline.state import StateBackend, default_state_backend

logger = logging.getLogger("snagline")


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

    def add_sink(self, sink: AlertSink) -> None:
        """Register one more sink at runtime (issue #122).

        The sink joins dispatch under the same rules as construction: every
        registered sink sees every dispatched risk, in registration order,
        and its exceptions are swallowed exactly like any other sink's
        (fail-open). Safe to call while ingests run from other threads;
        dispatch tolerates a concurrent append.
        """
        self._sinks.append(sink)

    def remove_sink(self, sink: AlertSink) -> bool:
        """Unregister ``sink`` (issue #122); returns True when removed.

        Matching follows ``list.remove`` semantics, so a sink registered
        twice must be removed once per registration. Returns False when the
        sink was never registered.
        """
        try:
            self._sinks.remove(sink)
        except ValueError:
            return False
        return True

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
        """Signal that ``episode_id`` has finished; clear its per-episode state.

        Calls ``reset(episode_id)`` on every detector so loop windows, CUSUM
        baselines, and cascade counters for that episode are dropped, then asks
        the state backend to release whatever it holds for that id. This is the
        teardown hook adapters call when an agent run completes; without it,
        per-episode state would accumulate for the life of the Monitor.

        Backends are only asked to release if they implement it -- a backend
        written against the narrower ``StateBackend`` protocol is unaffected.

        Fail-open: a detector's ``reset`` exception is logged, never propagated.
        """
        with self._state.episode_lock(episode_id):
            for detector in self._detectors:
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
        from snagline.detectors.ml_ensemble import MLOrchestrator
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
