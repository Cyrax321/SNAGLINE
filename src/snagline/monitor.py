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
import time
import urllib.request
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlsplit

from snagline.calibration import (
    CalibrationPlan,
    build_plan,
    resolve_baseline_profile,
)
from snagline.config import Config, validate_policy
from snagline.detectors.base import Detector
from snagline.events import StepEvent
from snagline.risk import FailureRisk
from snagline.sinks.base import AlertSink
from snagline.state import StateBackend, default_state_backend

logger = logging.getLogger("snagline")

# Bump when the snapshot payload shape changes incompatibly (issue #91).
SNAPSHOT_FORMAT_VERSION = 1

# --- Enforcement policy (issue #93) -----------------------------------------
#
# Optional escalation layer that runs AFTER the sinks on every dispatched risk
# (documented ordering: detectors -> sinks -> policy), outside every episode
# lock. Three policies:
#
#   "observe"      detection only; today's behavior, zero added work.
#   "callback"     invoke a host-supplied callable wrapped fail-open.
#   "halt_webhook" POST the FailureRisk JSON to halt_url and surface the
#                  response directive as ``monitor.last_directive``.
#
# The halt webhook NEVER raises into the host loop and never blocks other
# episodes' ingest: on timeout, error, or dead endpoint the directive simply
# stays/defaults to "continue" (fail-open). We deliberately do not pause
# anything ourselves; the host decides what its own loop does with the
# directive (project.md §1.2).
DEFAULT_POLICY = "observe"
HALT_ACTION_CONTINUE = "continue"
HALT_ACTION_PAUSE = "pause"
VALID_HALT_ACTIONS = (HALT_ACTION_CONTINUE, HALT_ACTION_PAUSE)
# Cap on how many response-body bytes one halt consultation will read: the
# endpoint holds control, but a runaway body still cannot balloon memory.
_MAX_HALT_RESPONSE_BYTES = 65_536


@dataclass(frozen=True, slots=True)
class HaltDirective:
    """The latest control directive returned by the halt webhook (issue #93).

    ``action`` is ``"continue"`` or ``"pause"``; ``reason`` is the short
    string the endpoint supplied (structure only, no event content);
    ``timestamp`` is when the directive was received. Immutable and swapped
    atomically under a lock, so concurrent multi-episode use always reads a
    complete directive.
    """

    action: str = HALT_ACTION_CONTINUE
    reason: str = ""
    timestamp: float = 0.0


class _EpisodeClock:
    """Per-episode time-axis state (issue #92).

    Every field derives ONLY from ``StepEvent.timestamp`` values seen in
    ``ingest()`` -- no wall-clock reads anywhere, so replaying a trajectory
    reproduces identical risks. One instance per active episode.
    """

    __slots__ = ("last_ts", "elapsed", "idle_fired", "warned", "breached")

    def __init__(self, first_ts: float) -> None:
        self.last_ts = first_ts
        self.elapsed = 0.0  # sum of positive inter-event deltas
        self.idle_fired = False  # "idle_gap" fires once per episode
        self.warned = False  # budget warning fires once per episode
        self.breached = False  # budget breach fires once per episode


def _detector_key(index: int, detector: Any) -> str:
    """Stable per-position key for a detector inside a snapshot payload."""
    return f"{index}:{getattr(detector, 'name', type(detector).__name__)}"


def _auto_calibration_plan(cfg: Config) -> CalibrationPlan | None:
    """Resolve the auto-calibration plan for ``cfg``, fail-open (issue #101).

    Returns None unless ``cfg.calibration == "auto"`` AND a usable baseline
    resolves; every failure mode (unknown value, missing baseline, unreadable
    file, derivation error) logs and falls back to the hand-tuned defaults so
    monitoring can never become worse than today because of calibration.
    """
    if str(getattr(cfg, "calibration", "") or "").strip().lower() != "auto":
        return None
    try:
        profile = resolve_baseline_profile(cfg)
    except Exception as exc:
        logger.warning(
            "snagline: auto-calibration baseline unavailable (%s); "
            "using hand-tuned thresholds",
            exc,
        )
        return None
    if profile is None:
        logger.info(
            "snagline: calibration=auto without a BaselineProfile; "
            "keeping hand-tuned thresholds"
        )
        return None
    try:
        return build_plan(profile, cfg)
    except Exception as exc:
        logger.warning(
            "snagline: auto-calibration failed (%s); using hand-tuned thresholds",
            exc,
        )
        return None


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
        # Enforcement faults (issue #93): callback raises and halt-webhook
        # timeouts/errors, counted separately from sink errors.
        self.policy_errors = 0

    def as_dict(self) -> dict:
        return {
            "events_ingested": self.events_ingested,
            "risks_emitted": self.risks_emitted,
            "detector_errors": self.detector_errors,
            "sink_errors": self.sink_errors,
            "policy_errors": self.policy_errors,
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

    Enforcement (issue #93): with ``policy="callback"`` an ``on_risk``
    callable is invoked after the sinks, wrapped fail-open; with
    ``policy="halt_webhook"`` every risk scoring at or above
    ``min_severity_for_halt`` is POSTed to ``halt_url`` and the response
    directive is surfaced thread-safely as :attr:`last_directive`. The
    default ``policy="observe"`` keeps construction byte-identical to the
    pre-#93 behavior: no callback, no network, zero overhead.
    """

    def __init__(
        self,
        detectors: list[Detector],
        sinks: list[AlertSink],
        fail_open: bool = True,
        state_backend: StateBackend | None = None,
        policy: str = DEFAULT_POLICY,
        on_risk: Callable[[FailureRisk], None] | None = None,
        halt_url: str | None = None,
        halt_timeout_s: float = 0.25,
        min_severity_for_halt: float = 0.8,
        config: Config | None = None,
    ) -> None:
        self._detectors = list(detectors)
        self._sinks = list(sinks)
        self._fail_open = fail_open
        self._state = state_backend or default_state_backend()
        # Horizon-scale time axis (issue #92). Inert unless one of the opt-in
        # horizon knobs is set on the config; all state lives in ``_clocks``,
        # keyed by episode, derived purely from event timestamps.
        cfg = config or Config()
        self._max_wall_seconds = cfg.max_episode_wall_seconds
        self._warn_fraction = cfg.warn_fraction
        self._idle_warn_seconds = cfg.idle_warn_seconds
        self._clocks: dict[str, _EpisodeClock] = {}
        # Per-episode retention cap (issue #184): bounded LRU of live episode
        # ids. Every ingest touches the entry; when the cap is exceeded the
        # least-recently-seen id is evicted silently (no finalize risks) via
        # the same teardown as end_episode. Explicit end_episode still frees
        # immediately; this is the safety net for hosts that never call it.
        # No content retention: keys are episode ids only.
        self._max_live_episodes: int = int(getattr(cfg, "max_live_episodes", 10_000))
        if self._max_live_episodes < 1:
            self._max_live_episodes = 10_000
        self._live_episodes: OrderedDict[str, None] = OrderedDict()
        self._live_lock = threading.Lock()
        # A faulty detector or sink must not spam the logs on every single step
        # (issue #14) -- log each distinct fault exactly once.
        self._fault_lock = threading.Lock()
        self._fault_logged: set[str] = set()
        self._metrics = MonitorMetrics()
        self._metrics_lock = threading.Lock()
        self._configure_policy(
            policy=policy,
            on_risk=on_risk,
            halt_url=halt_url,
            halt_timeout_s=halt_timeout_s,
            min_severity_for_halt=min_severity_for_halt,
        )

    def _configure_policy(
        self,
        *,
        policy: str = DEFAULT_POLICY,
        on_risk: Callable[[FailureRisk], None] | None = None,
        halt_url: str | None = None,
        halt_timeout_s: float = 0.25,
        min_severity_for_halt: float = 0.8,
    ) -> None:
        """Validate and install the enforcement configuration (issue #93).

        Called from ``__init__`` and re-callable from ``Monitor.default`` so
        subclasses with a fixed ``__init__`` signature keep working (same
        rationale as the post-construction ``_state`` assignment there).
        Invalid values are configuration errors and raise here -- loudly, at
        setup time, like log_format (issue #119) -- never at ingest time.
        """
        normalized = validate_policy(policy)
        if normalized == "callback" and on_risk is None:
            logger.warning(
                "snagline: policy='callback' without an on_risk callable; "
                "no enforcement action will run"
            )
        if normalized == "halt_webhook":
            if not halt_url:
                raise ValueError("policy='halt_webhook' requires halt_url")
            if halt_timeout_s <= 0:
                raise ValueError("halt_timeout_s must be positive")
            # Control-plane endpoint: restrict to http(s). urllib would happily
            # attempt other schemes (file://, ftp://); a typo or injection in
            # operator config must fail loudly here, not probe odd handlers.
            scheme = urlsplit(str(halt_url)).scheme.strip().lower()
            if scheme not in ("http", "https"):
                raise ValueError(
                    f"halt_url must be an http(s) endpoint; got scheme {scheme!r}"
                )
        # Risk scores live in [0, 1]; a threshold outside that range would
        # silently disable or permanently enable halting.
        if not 0.0 <= float(min_severity_for_halt) <= 1.0:
            raise ValueError("min_severity_for_halt must be within [0, 1]")
        self._policy = normalized
        self._on_risk = on_risk
        self._halt_url = halt_url
        self._halt_timeout_s = halt_timeout_s
        self._min_severity_for_halt = min_severity_for_halt
        self._directive_lock = threading.Lock()
        self._last_directive = HaltDirective()

    @property
    def policy(self) -> str:
        """The active enforcement policy (issue #93)."""
        return self._policy

    @property
    def halt_url(self) -> str | None:
        """The halt webhook endpoint, or None when not in halt_webhook mode."""
        return self._halt_url

    @property
    def last_directive(self) -> HaltDirective:
        """The latest halt-webhook directive, thread-safe (issue #93).

        Starts at action="continue" so consumers always read a valid
        directive; on timeout/error/dead endpoint it stays at or falls back
        to continue (fail-open).
        """
        with self._directive_lock:
            return self._last_directive

    def ingest(self, event: StepEvent) -> None:
        """Run all detectors against ``event`` and dispatch any risks.

        Never raises under the default ``fail_open=True``. With
        ``fail_open=False``, a detector or sink exception propagates.

        Detection runs under a per-instance lock, but dispatch to sinks happens
        *outside* that lock (issue #8): a slow or network-bound sink (e.g. the
        webhook sink) must not block concurrent ``ingest`` calls, and a sink
        that re-enters ``ingest`` must not deadlock. The enforcement policy
        runs after the sinks, also outside the lock (issue #93).

        Region contract (issue #92): the time-axis head below is the ONLY place
        that touches event timestamps for idle/budget decisions, and it reads no
        wall clock, so replay stays deterministic. The dispatch tail belongs to
        the sinks/policy layer and is left untouched by time-axis logic.
        """
        # Per-episode retention cap (issue #184): touch LRU and evict if over
        # cap. Done outside the episode lock so eviction of a different id
        # does not deadlock. LRU by last-seen is safe: an active episode is
        # by definition recently seen and cannot be evicted out from under
        # itself. Eviction is silent (no finalize) and fail-open.
        evicted: str | None = None
        with self._live_lock:
            if event.episode_id in self._live_episodes:
                self._live_episodes.move_to_end(event.episode_id)
            else:
                self._live_episodes[event.episode_id] = None
                if len(self._live_episodes) > self._max_live_episodes:
                    evicted, _ = self._live_episodes.popitem(last=False)
        if evicted is not None and evicted != event.episode_id:
            self._evict_episode(evicted)
        risks: list[FailureRisk] = []
        with self._state.episode_lock(event.episode_id):
            # Time-axis head (issue #92): computed from StepEvent timestamps at
            # the top of ingest(), under the episode lock so concurrent ingests
            # serialize exactly like detector state does.
            risks.extend(self._advance_clock(event))
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

    def _advance_clock(self, event: StepEvent) -> list[FailureRisk]:
        """Time-axis risks from StepEvent timestamps alone (issue #92).

        Fail-open like every other monitoring step: an internal error is logged
        fault-once and never propagates. Emits at most once per episode per
        threshold:

        * ``idle_gap`` (score 0.8) when a gap between consecutive ingests
          reaches ``idle_warn_seconds``;
        * ``wall_clock_budget`` warning (score 0.7, keeping it inside the
          "warning" severity band) at ``warn_fraction`` of
          ``max_episode_wall_seconds``, then a breach (score 1.0) at the limit;
          a single delta that jumps straight past the limit fires only the
          breach (mirrors TokenRunawayDetector's envelope ordering).
        """
        out: list[FailureRisk] = []
        # Inert unless one of the opt-in horizon knobs is set (issue #92). When
        # both are None the time axis claims to be inert, so do not retain
        # per-episode clocks at all (issue #184: _clocks grew even with the
        # feature off). No content retention beyond ids.
        if self._max_wall_seconds is None and self._idle_warn_seconds is None:
            return out
        try:
            clock = self._clocks.get(event.episode_id)
            if clock is None:
                # First event of the episode establishes the reference point;
                # there is no delta yet, so nothing can fire.
                self._clocks[event.episode_id] = _EpisodeClock(event.timestamp)
                return out
            delta = event.timestamp - clock.last_ts
            clock.last_ts = event.timestamp
            if (
                self._idle_warn_seconds is not None
                and not clock.idle_fired
                and delta >= self._idle_warn_seconds
            ):
                clock.idle_fired = True
                out.append(
                    FailureRisk(
                        event.episode_id,
                        event.step_id,
                        0.8,
                        "idle_gap",
                        f"no events for {delta:.1f}s "
                        f"(threshold {self._idle_warn_seconds:.1f}s)",
                        event.timestamp,
                    )
                )
            if delta > 0.0:
                # Negative deltas (out-of-order or skewed sources) must not
                # reduce consumed budget; clamp them out of the accumulation.
                clock.elapsed += delta
            budget = self._max_wall_seconds
            if budget is not None:
                if not clock.breached and clock.elapsed >= budget:
                    clock.breached = True
                    out.append(
                        FailureRisk(
                            event.episode_id,
                            event.step_id,
                            1.0,
                            "wall_clock_budget",
                            f"episode exceeded its {budget:.0f}s wall-clock "
                            f"budget ({clock.elapsed:.0f}s observed)",
                            event.timestamp,
                        )
                    )
                elif not clock.warned and clock.elapsed >= budget * self._warn_fraction:
                    clock.warned = True
                    out.append(
                        FailureRisk(
                            event.episode_id,
                            event.step_id,
                            # 0.7 keeps the pre-breach signal inside the
                            # "warning" severity band (>= 0.8 derives
                            # critical); the breach below is the critical.
                            0.7,
                            "wall_clock_budget",
                            f"episode at {clock.elapsed / budget:.0%} of its "
                            f"{budget:.0f}s wall-clock budget",
                            event.timestamp,
                        )
                    )
        except Exception as exc:
            self._log_fault_once(
                f"time-axis tracking raised ({exc}); ignoring (fail-open)"
            )
            if not self._fail_open:
                raise
        return out

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
        # Documented ordering (issue #93): detectors -> sinks -> policy. The
        # tail runs outside every episode lock; see _enforce.
        self._enforce(risk)

    # --- Enforcement policy tail (issue #93) ---------------------------------

    def _enforce(self, risk: FailureRisk) -> None:
        """Run the configured enforcement policy for one dispatched risk.

        Called strictly AFTER the sink loop, outside every episode lock, so a
        slow halt webhook delays only its own dispatch thread, never other
        episodes' ingest. With the default ``policy="observe"`` this is a
        single string comparison: no behavior change, no overhead.
        """
        if self._policy == "observe":
            return
        if self._policy == "callback":
            self._run_risk_callback(risk)
        else:
            self._run_halt_webhook(risk)

    def _run_risk_callback(self, risk: FailureRisk) -> None:
        """Invoke the host callback wrapped exactly like a sink (issue #93).

        Exceptions are counted under ``policy_errors``, logged fault-once, and
        swallowed unless ``fail_open=False`` -- identical semantics to the
        sink loop in ``_dispatch``, verified by parity tests against
        tests/test_monitor_fail_open.py.
        """
        callback = self._on_risk
        if callback is None:
            return
        try:
            callback(risk)
        except Exception:
            self._incr("policy_errors")
            self._log_fault_once("risk callback raised; ignoring (fail-open)")
            if not self._fail_open:
                raise

    def _run_halt_webhook(self, risk: FailureRisk) -> None:
        """Fire-and-collect POST of one risk to the halt endpoint (issue #93).

        Only risks scoring at or above ``min_severity_for_halt`` pay the
        round-trip cost. The response JSON ``{"action": ..., "reason": ...}``
        becomes ``monitor.last_directive``; anything unexpected -- timeout,
        connection error, malformed body, unknown action -- leaves the
        directive at continue (fail-open) and counts one ``policy_error``.
        Never raises into the host loop while ``fail_open=True``.
        """
        if risk.score < self._min_severity_for_halt:
            return
        payload = {
            "episode_id": risk.episode_id,
            "step_id": risk.step_id,
            "score": risk.score,
            "trigger": risk.trigger,
            "detail": risk.detail,
            "timestamp": risk.timestamp,
            "severity": risk.severity,
        }
        try:
            req = urllib.request.Request(
                self._halt_url or "",
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=self._halt_timeout_s) as resp:
                body = resp.read(_MAX_HALT_RESPONSE_BYTES)
            parsed = json.loads(body.decode("utf-8"))
            if not isinstance(parsed, dict):
                raise ValueError("halt response must be a JSON object")
            action = str(parsed.get("action", HALT_ACTION_CONTINUE)).strip().lower()
            if action not in VALID_HALT_ACTIONS:
                raise ValueError(f"unknown halt action {action!r}")
            reason = str(parsed.get("reason", ""))
            directive = HaltDirective(
                action=action, reason=reason, timestamp=time.time()
            )
        except Exception:
            self._incr("policy_errors")
            self._log_fault_once(
                f"halt webhook {self._halt_url} failed; continuing (fail-open)"
            )
            if not self._fail_open:
                raise
            return
        with self._directive_lock:
            self._last_directive = directive

    def _incr(self, name: str) -> None:
        with self._metrics_lock:
            setattr(self._metrics, name, getattr(self._metrics, name) + 1)

    def metrics(self) -> dict:
        """Return a snapshot of self-observability counters (P3, item 10)."""
        with self._metrics_lock:
            data = self._metrics.as_dict()
        # Issue #184: expose retained-episode count so the leak is observable
        # rather than inferred from RSS. Count is live LRU size (ids only).
        with self._live_lock:
            data["live_episodes"] = len(self._live_episodes)
            data["retained_episodes"] = len(self._live_episodes)
        return data

    @property
    def retained_episodes(self) -> int:
        """Number of live episode ids currently retained (issue #184)."""
        with self._live_lock:
            return len(self._live_episodes)

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

    def _evict_episode(self, episode_id: str) -> None:
        """Silent LRU eviction for the retention cap (issue #184).

        Same teardown as ``end_episode`` but without ``finalize`` risks: an
        episode that vanished because the host never called ``end_episode``
        should not suddenly emit a ``silent_abort`` risk long after the fact.
        Fail-open throughout, never propagates even with ``fail_open=False``
        for eviction (the cap is a safety net, not a judgment point).
        """
        # Race: pop selected this id as LRU, but another thread may have
        # re-added it before we acquire its episode lock. If it is back in
        # the live set, keep its fresh state.
        with self._live_lock:
            if episode_id in self._live_episodes:
                return
        try:
            with self._state.episode_lock(episode_id):
                self._clocks.pop(episode_id, None)
                for detector in self._detectors:
                    try:
                        detector.reset(episode_id)
                    except Exception:
                        self._log_fault_once(
                            f"detector {getattr(detector, 'name', repr(detector))} "
                            "reset raised during eviction; ignoring (fail-open)"
                        )
                release = getattr(self._state, "release", None)
                if callable(release):
                    try:
                        release(episode_id)
                    except Exception:
                        self._log_fault_once(
                            f"state backend {type(self._state).__name__} release "
                            "raised during eviction; ignoring (fail-open)"
                        )
        except Exception:
            self._log_fault_once(f"eviction of {episode_id!r} raised; ignoring")

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
        # Keep LRU honest: freeing here means the gauge and retained count
        # drop at once, not when the cap eventually evicts.
        with self._live_lock:
            self._live_episodes.pop(episode_id, None)
        finalized: list[FailureRisk] = []
        with self._state.episode_lock(episode_id):
            # Time-axis state (issue #92) is per-episode like detector state;
            # drop it here so a reused episode id starts with a clean clock.
            self._clocks.pop(episode_id, None)
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
        time_axis: dict[str, Any] = {}
        for episode_id, clock in self._clocks.items():
            time_axis[episode_id] = {
                "last_ts": clock.last_ts,
                "elapsed": clock.elapsed,
                "idle_fired": clock.idle_fired,
                "warned": clock.warned,
                "breached": clock.breached,
            }
        return {
            "format_version": SNAPSHOT_FORMAT_VERSION,
            "detectors": detectors,
            "sinks": sinks,
            "time_axis": time_axis,
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
        time_axis_data = data.get("time_axis")
        if isinstance(time_axis_data, dict):
            # Restore clocks in order of last_ts ascending so most recent
            # survives if we must evict under cap. Fall back to sorted keys
            # for deterministic behavior when last_ts is missing.
            try:
                ordered = sorted(
                    time_axis_data.items(),
                    key=lambda kv: (
                        float(kv[1].get("last_ts", 0.0))  # type: ignore[arg-type]
                        if isinstance(kv[1], dict)
                        else 0.0
                    ),
                )
            except Exception:
                ordered = sorted(time_axis_data.items())
            for episode_id, clock_data in ordered:
                if not isinstance(episode_id, str) or not isinstance(clock_data, dict):
                    continue
                try:
                    last_ts = float(clock_data.get("last_ts", 0.0))
                    elapsed = float(clock_data.get("elapsed", 0.0))
                    idle_fired = bool(clock_data.get("idle_fired", False))
                    warned = bool(clock_data.get("warned", False))
                    breached = bool(clock_data.get("breached", False))
                except Exception:
                    logger.warning(
                        "snagline: malformed time_axis entry for %r; ignored",
                        episode_id,
                    )
                    continue
                if elapsed < 0.0:
                    elapsed = 0.0
                clock = _EpisodeClock(last_ts)
                clock.elapsed = elapsed
                clock.idle_fired = idle_fired
                clock.warned = warned
                clock.breached = breached
                with self._live_lock:
                    if episode_id in self._live_episodes:
                        self._live_episodes.move_to_end(episode_id)
                    else:
                        self._live_episodes[episode_id] = None
                        if len(self._live_episodes) > self._max_live_episodes:
                            evicted, _ = self._live_episodes.popitem(last=False)
                            self._clocks.pop(evicted, None)
                    self._clocks[episode_id] = clock
            # If we evicted due to cap, live_snapshot entries not in time_axis
            # are already accounted for; no further action needed.

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
        unless ``sinks`` is given, the console sink. With
        ``config.log_format == "json"`` (env ``SNAGLINE_LOG_FORMAT=json``),
        LoggingSink is installed next to ConsoleSink so risk emission is also
        machine-readable JSON lines on the ``snagline`` logger (issues #99 and
        #119). Explicitly passed ``sinks`` lists are honored verbatim.

        With no ``config``, the effective ``Config`` is resolved through the
        12-factor layering (defaults < config file < ``SNAGLINE_*`` env vars,
        via ``Config.resolve()``), so ``SNAGLINE_LOG_FORMAT=json`` works with
        zero code changes. Pass an explicit ``Config`` to pin every knob.
        """
        from snagline.detectors.compaction_tripwire import CompactionTripwireDetector
        from snagline.detectors.error_cascade import ErrorCascadeDetector
        from snagline.detectors.goal_drift import GoalDriftDetector
        from snagline.detectors.latency_anomaly import LatencyAnomalyDetector
        from snagline.detectors.loop import LoopDetector
        from snagline.detectors.meltdown import MeltdownDetector
        from snagline.detectors.ml_ensemble import MLOrchestrator
        from snagline.detectors.side_effect_guard import SideEffectGuardDetector
        from snagline.detectors.silent_abort import SilentAbortDetector
        from snagline.detectors.stagnation import StagnationDetector
        from snagline.detectors.token_runaway import TokenRunawayDetector
        from snagline.sinks.console import ConsoleSink
        from snagline.sinks.logging_sink import LoggingSink

        # No explicit config: resolve the 12-factor layering so bare
        # Monitor.default() callers (examples, hooks) honor SNAGLINE_* env
        # vars exactly like the CLI does (issue #119 acceptance).
        cfg = config or Config.resolve()
        # Auto-calibration (issue #101): opt-in via calibration="auto" plus a
        # healthy BaselineProfile; None keeps every hand-tuned constant.
        cal = _auto_calibration_plan(cfg)
        if cal is not None:
            base: list[Any] = [
                LoopDetector(config=cfg),
                ErrorCascadeDetector(
                    error_threshold=cal.cascade_error_threshold,
                    consecutive_threshold=cal.cascade_consecutive_threshold,
                    config=cfg,
                ),
                # Seeded CUSUM: healthy mean/spread replace live warm-up.
                LatencyAnomalyDetector(baseline=cal.baseline, config=cfg),
            ]
        else:
            base = [
                LoopDetector(config=cfg),
                ErrorCascadeDetector(config=cfg),
                LatencyAnomalyDetector(config=cfg),
            ]
        # Stagnation is opt-in (issue #87): novelty-rate collapse detection,
        # default-off so the zero-dependency preset and the published bench
        # numbers are untouched. Appended to ``base`` so a concurrent
        # MLOrchestrator wraps it like any other signal.
        if cfg.stagnation_enabled:
            base.append(StagnationDetector(config=cfg))
        # Goal-drift is opt-in: only when explicitly enabled and a baseline is
        # supplied, so the zero-dependency default is unchanged (step 2).
        if cfg.goal_drift_enabled and cfg.goal_drift_baseline is not None:
            base.append(GoalDriftDetector(baseline=cfg.goal_drift_baseline, config=cfg))
        # Semantic goal-drift is opt-in (issue #81): needs the optional
        # ``drift`` extra AND a semantically fitted BaselineProfile (the
        # embedding_centroid must be set). Guarded import and the lazy model
        # load keep the zero-dependency preset byte-identical; any failure
        # degrades fail-open to an inert detector or, when the import itself
        # fails, to the common “drift extra missing” path inside the detector.
        if cfg.semantic_drift_enabled and cfg.goal_drift_baseline is not None:
            try:
                from snagline.drift.goal_drift import SemanticGoalDriftDetector

                base.append(
                    SemanticGoalDriftDetector(
                        baseline=cfg.goal_drift_baseline, config=cfg
                    )
                )
            except Exception:
                logger.debug(
                    "snagline: drift extra unavailable; semantic goal-drift stays off",
                    exc_info=True,
                )
        # Horizon detectors are opt-in too (issues #84/#85/#86): each needs
        # telemetry or validation the zero-config preset does not promise.
        if cfg.token_runaway_enabled:
            base.append(TokenRunawayDetector(config=cfg))
        if cfg.meltdown_enabled:
            base.append(MeltdownDetector(config=cfg))
        if cfg.silent_abort_enabled:
            base.append(SilentAbortDetector(config=cfg))
        # Side-effect guard is opt-in (issue #88): duplicate non-idempotent
        # action detection, default-off so the zero-dependency preset and the
        # published bench numbers are untouched.
        if cfg.side_effect_guard_enabled:
            base.append(SideEffectGuardDetector(config=cfg))
        # Compaction tripwire is opt-in (issue #90): governance-decay
        # detection across context compactions. Inert unless the adapter
        # emits the "compaction" / "constraint_present" action types.
        if cfg.compaction_tripwire_enabled:
            base.append(CompactionTripwireDetector(config=cfg))
        if cfg.ml_ensemble_enabled:
            # Optional ml extra: add the one-class ESN + CUSUM detector to the
            # ensemble when numpy is importable (issue #80). The import is
            # guarded so a missing or broken extra degrades fail-open to the
            # zero-dependency noisy-OR over the deterministic detectors.
            try:
                from snagline.ml.esn_ensemble import EsnCusumDetector

                base.append(EsnCusumDetector(baseline=cfg.goal_drift_baseline))
            except Exception:
                logger.debug(
                    "snagline: ml extra unavailable; noisy-OR runs over the "
                    "deterministic detectors only",
                    exc_info=True,
                )
            # Combine the base detectors into one orchestrated signal (step 3).
            detectors: list[Detector] = [MLOrchestrator(base, config=cfg)]
        else:
            detectors = list(base)
        if sinks is not None:
            chosen_sinks: list[AlertSink] = list(sinks)
        else:
            # Default composition (issues #99/#119): console stays the human
            # channel; log_format="json" adds LoggingSink NEXT TO it (the
            # agreed "alongside console" semantics) so SNAGLINE_LOG_FORMAT=json
            # makes risks machine-readable with zero code changes.
            chosen_sinks = [ConsoleSink()]
            if cfg.log_format == "json":
                chosen_sinks.append(LoggingSink())
        monitor = cls(detectors, chosen_sinks, fail_open=cfg.fail_open)
        # Set after construction so subclasses with a fixed __init__ signature
        # (e.g. test doubles) still work; base Monitor.__init__ also sets it.
        monitor._state = state_backend or default_state_backend()
        # Enforcement layer (issue #93): applied through _configure_policy for
        # the same subclass-compatibility reason as _state above. on_risk is
        # deliberately absent from Config (callables cannot arrive via env or
        # config files), so callback mode through default() runs inert unless
        # a callable is attached directly afterwards; _configure_policy warns
        # about exactly that case.
        monitor._configure_policy(
            policy=cfg.policy,
            on_risk=None,
            halt_url=cfg.halt_url,
            halt_timeout_s=cfg.halt_timeout_s,
            min_severity_for_halt=cfg.min_severity_for_halt,
        )
        # Horizon-scale time axis (issue #92): the same resolved config drives
        # idle-gap and wall-clock-budget tracking. Assigned post-construction
        # like _state so subclasses with a fixed __init__ signature keep working.
        monitor._max_wall_seconds = cfg.max_episode_wall_seconds
        monitor._warn_fraction = cfg.warn_fraction
        monitor._idle_warn_seconds = cfg.idle_warn_seconds
        # Per-episode retention cap (issue #184): ensure the resolved cap
        # wins over the default-constructed one from __init__.
        monitor._max_live_episodes = int(cfg.max_live_episodes)
        return monitor
