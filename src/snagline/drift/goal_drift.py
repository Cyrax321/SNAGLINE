"""Semantic goal-drift detector (the optional ``drift`` extra, issue #81).

Compares the running embedding centroid of a live episode against the
persisted ``BaselineProfile``'s ``embedding_centroid`` (see
``snagline.baseline``) and raises a ``goal_drift`` risk when the agent's
activity mix has diverged from the healthy reference in a *sustained* way.
Where the deterministic ``detectors/goal_drift.py`` compares error rates,
latencies, and tool-name sets, this detector adds meaning: two different
tool names that play similar roles land close together in embedding space,
so semantically equivalent-but-renamed behavior stays quiet while a real
change of goal (a healthy "search and summarize" mix sliding into bulk
destructive calls) moves the centroid away.

Privacy (project.md section 1.4): embeddings are computed over structural
labels only: ``action_type``, ``tool_name``, and ``error_type``. Prompt or
response content is never read (``metadata`` is never touched), nothing
textual is logged or persisted, and the only stored artifact is an averaged
vector of floats inside the baseline profile.

Dependency discipline (section 1.1): this module imports cleanly with no
third-party packages installed. ``sentence_transformers`` is imported only
inside :meth:`SemanticGoalDriftDetector._make_default_model_loader`, lazily,
on first use, and only when no explicit ``embedder`` was injected. A setup
failure on that path (extra missing, model download/load failing) or a
baseline/embedder dimension mismatch is caught, logged once, and leaves the
detector permanently inert. A *transient* inference exception (the loaded
embedder raising mid-run) is instead swallowed fail-open and the step is
skipped, so a backend that recovers resumes scoring -- but it too is logged
only once per fault, never once per step, so a persistently-broken embedder
cannot flood the host's logs with a traceback on every event. Either way,
monitoring must never crash or stall the host agent (section 1.2).

Performance (section 1.5): O(embedding dim) work per step for the running
sum plus one embedder call; per-episode memory is one vector and two
scalars. This path is opt-in and never runs in the zero-dependency preset.

Wiring: append to the Monitor's base list so, when
``Config.ml_ensemble_enabled`` is also set, the score joins the zero-dep
noisy-OR ``MLOrchestrator`` alongside the ESN ensemble from issue #80.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable, Iterable, Sequence
from typing import Any

from snagline.baseline import BaselineProfile
from snagline.config import Config
from snagline.detectors.base import snapshot_items
from snagline.events import StepEvent
from snagline.risk import FailureRisk

logger = logging.getLogger("snagline")

# Callable turning one step into an embedding vector. Injected by tests and
# by hosts with their own models; the default wraps sentence-transformers.
Embedder = Callable[[StepEvent], Sequence[float]]


def _event_label(event: StepEvent) -> str:
    """Text embedded for a step.

    Includes structural labels (action_type, tool_name, error_type) and any
    explicit host-supplied intent/purpose declared on the step (issue #603).
    Deliberately excludes ``action_signature`` (an opaque hash: meaningless
    in embedding space) and ``metadata`` (may carry raw content; detectors
    never read it, project.md section 4/11).
    """
    parts = [
        event.action_type,
        event.tool_name or "",
        event.error_type or "",
        getattr(event, "intent", None) or "",
    ]
    return " ".join(p for p in parts if p)


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine similarity with a degenerate-vector guard.

    If either vector has zero magnitude there is no directional evidence,
    and "no evidence of drift" must stay silent, so return 1.0 (identical).
    """
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b, strict=True):
        dot += x * y
        na += x * x
        nb += y * y
    if na <= 0.0 or nb <= 0.0:
        return 1.0
    return dot / math.sqrt(na * nb)


class _LiveState:
    """Per-episode accumulator: running embedding sum, sliding window, and CUSUM debt."""

    __slots__ = ("debt", "n", "sum", "window", "window_dev_sum", "window_sum")

    def __init__(self) -> None:
        self.sum: list[float] | None = None
        self.n = 0
        self.debt = 0.0
        self.window: list[list[float]] = []
        self.window_sum: list[float] | None = None
        self.window_dev_sum = 0.0


class SemanticGoalDriftDetector:
    """Embedding-centroid drift detector emitting the ``goal_drift`` trigger.

    Fail-open twice over: :meth:`observe` never raises, and a broken
    embedding backend never disrupts the host. A *setup* failure (missing
    extra, model load) latches the detector permanently inert after one
    attempt; a *transient* inference exception is swallowed and the step
    skipped so a recovering backend resumes, and in either case the fault is
    logged only once -- never a traceback per step (cf. the ConsoleSink
    warn-once latch, issue #327). Fires at most after
    ``semantic_drift_min_samples`` live steps, only when the cosine deviation
    from the healthy centroid persists (CUSUM gate with slack ``k`` and alarm
    ``h``), then re-arms so a still-drifting episode re-alarms later.
    """

    name = "semantic_goal_drift"

    def __init__(
        self,
        baseline: BaselineProfile,
        config: Config | None = None,
        embedder: Embedder | None = None,
        model_loader: Callable[[], Embedder | None] | None = None,
    ) -> None:
        self._cfg = config or Config()
        centroid = baseline.embedding_centroid
        # A profile fitted structurally only carries no reference vectors:
        # inert by design rather than guessing a semantics-free fallback.
        if centroid:
            converted = [float(v) for v in centroid]
            if not all(math.isfinite(v) for v in converted):
                logger.warning(
                    "snagline: semantic goal-drift inert; baseline "
                    "embedding_centroid contains non-finite values"
                )
                self._baseline_centroid = None
            else:
                self._baseline_centroid = converted
        else:
            self._baseline_centroid = None
            logger.info(
                "snagline: semantic goal-drift inert; baseline has no "
                "embedding_centroid (fit one via "
                "snagline.drift.fit_semantic_baseline)"
            )
        self._embedder: Embedder | None = embedder
        self._model_loader = model_loader or self._make_default_model_loader()
        # Explicit injection skips heavy setup entirely (also the test path);
        # otherwise resolve lazily exactly once on first use.
        self._resolved = embedder is not None
        self._episodes: dict[str, _LiveState] = {}
        # Warn-once latch for transient inference faults: without it a broken
        # embedder that keeps raising logs a full traceback on every step,
        # contradicting the "logged once" contract (issue #527; mirrors
        # ConsoleSink's _fault_logged, issue #327). Reset on the next clean
        # observe so a recovered backend re-arms the warning if it fails again.
        self._inference_fault_logged = False

    def observe(self, event: StepEvent) -> FailureRisk | None:
        """Score one step; fail-open wrapper around the semantic path.

        Never raises: any internal error is logged here and by the Monitor,
        then ignored (project.md section 1.2). A transient inference fault is
        logged once (not once per step) and the latch resets on the next clean
        step, so a recovering backend re-arms the warning without flooding.
        """
        try:
            result = self._observe(event)
        except Exception:
            if not self._inference_fault_logged:
                self._inference_fault_logged = True
                logger.exception(
                    "snagline: semantic_goal_drift raised; ignoring "
                    "(fail-open); further faults are silent until it recovers"
                )
            return None
        self._inference_fault_logged = False
        return result

    def reset(self, episode_id: str) -> None:
        """Drop all per-episode state (running centroid and CUSUM debt)."""
        self._episodes.pop(episode_id, None)

    def dump_state(self) -> dict[str, Any]:
        # snapshot_items: a concurrent ingest meeting a new episode must not
        # change the key set mid-comprehension (issue #231).
        return {
            "episodes": {
                ep: {
                    "sum": st.sum,
                    "n": st.n,
                    "debt": st.debt,
                    "window": st.window,
                    "window_sum": st.window_sum,
                    "window_dev_sum": st.window_dev_sum,
                }
                for ep, st in snapshot_items(self._episodes)
                if st.sum is not None
            }
        }

    def load_state(self, state: dict[str, Any]) -> None:
        episodes: dict[str, _LiveState] = {}
        for ep, raw in state.get("episodes", {}).items():
            st = _LiveState()
            vec = raw.get("sum")
            if isinstance(vec, list):
                st.sum = [float(v) for v in vec]
            # A restored running sum fitted under a different embedder than
            # the live baseline centroid is a wedge, not a head start
            # (issue #435): the live path's dimension check compares the
            # embedder against the centroid and passes when the operator
            # refit both together, then indexes the stale shorter sum out of
            # range on every subsequent step -- IndexError, swallowed by the
            # fail-open wrapper, zero risks and a traceback per step forever.
            # The live path handles a mismatch by latching inert; restore
            # instead drops only the affected episode, which re-accumulates
            # from scratch exactly as it would after a reset.
            if (
                st.sum is not None
                and self._baseline_centroid is not None
                and len(st.sum) != len(self._baseline_centroid)
            ):
                logger.warning(
                    "snagline: semantic goal-drift dropped restored state for "
                    "episode %r; snapshot dimension %d does not match baseline "
                    "centroid dimension %d (refit the baseline with the current "
                    "embedder)",
                    ep,
                    len(st.sum),
                    len(self._baseline_centroid),
                )
                episodes[str(ep)] = _LiveState()  # sum=None: re-accumulate
                continue
            st.n = int(raw.get("n", 0))
            st.debt = float(raw.get("debt", 0.0))
            w = raw.get("window")
            if isinstance(w, list):
                st.window = [
                    [float(v) for v in row] for row in w if isinstance(row, list)
                ]
            w_sum = raw.get("window_sum")
            if isinstance(w_sum, list):
                st.window_sum = [float(v) for v in w_sum]
            st.window_dev_sum = float(raw.get("window_dev_sum", 0.0))
            episodes[str(ep)] = st
        self._episodes = episodes

    # --- internals ---------------------------------------------------------

    def _observe(self, event: StepEvent) -> FailureRisk | None:
        if self._baseline_centroid is None:
            return None
        embed = self._resolve_embedder()
        if embed is None:
            return None
        vec = [float(v) for v in embed(event)]
        if not all(math.isfinite(v) for v in vec):
            logger.warning(
                "snagline: semantic goal-drift skipping non-finite "
                "embedding for step %r",
                event.step_id,
            )
            return None
        if len(vec) != len(self._baseline_centroid):
            logger.warning(
                "snagline: semantic goal-drift disabled; embedder dimension %d "
                "does not match baseline centroid dimension %d",
                len(vec),
                len(self._baseline_centroid),
            )
            self._baseline_centroid = None  # latch inert; do not re-warn
            return None
        st = self._episodes.get(event.episode_id)
        if st is None:
            st = _LiveState()
            self._episodes[event.episode_id] = st
        if st.sum is None:
            st.sum = [0.0] * len(vec)
        for i, v in enumerate(vec):
            st.sum[i] += v
        st.n += 1

        win_size = max(
            getattr(self._cfg, "semantic_drift_window_size", 20),
            self._cfg.semantic_drift_min_samples,
        )
        st.window.append(vec)
        if st.window_sum is None:
            st.window_sum = [0.0] * len(vec)
        for i, v in enumerate(vec):
            st.window_sum[i] += v
        dev_vec = max(0.0, 1.0 - _cosine(vec, self._baseline_centroid))
        st.window_dev_sum += dev_vec

        if len(st.window) > win_size:
            old_vec = st.window.pop(0)
            for i, v in enumerate(old_vec):
                st.window_sum[i] -= v
            old_dev = max(0.0, 1.0 - _cosine(old_vec, self._baseline_centroid))
            st.window_dev_sum = max(0.0, st.window_dev_sum - old_dev)

        if st.n < self._cfg.semantic_drift_min_samples:
            return None

        # Two complementary statistics:
        # 1. Centroid deviation over the active sliding window: captures coherent
        #    directional divergence or single-tool collapse without dilution from
        #    historical baselines (issue #603).
        # 2. Mean per-event deviation across the window: preserves distance when
        #    a multi-tool vocabulary shift would otherwise wash out under vector
        #    averaging (issue #603).
        live_window = [s / len(st.window) for s in st.window_sum]
        sim_window = _cosine(live_window, self._baseline_centroid)
        dev_window = max(0.0, 1.0 - sim_window)
        dev_events = st.window_dev_sum / len(st.window)
        dev = max(dev_window, dev_events)

        tol = self._cfg.semantic_drift_tolerance
        denom = max(0.001, 1.0 - tol if tol < 1.0 else 2.0 - tol)
        signal = 0.0 if dev <= tol else min(1.0, (dev - tol) / denom)
        st.debt = max(0.0, st.debt + signal - self._cfg.semantic_drift_cusum_k)
        if st.debt >= self._cfg.semantic_drift_cusum_h:
            h = self._cfg.semantic_drift_cusum_h
            score = min(1.0, 0.5 + 0.5 * (st.debt - h) / h)
            st.debt = 0.0  # re-arm so persistent drift re-alarms later
            sim = max(-1.0, min(1.0, 1.0 - dev))
            return FailureRisk(
                event.episode_id,
                event.step_id,
                score,
                "goal_drift",
                f"semantic activity centroid diverged from healthy baseline "
                f"(cosine {sim:.3f})",
                event.timestamp,
            )
        return None

    def _resolve_embedder(self) -> Embedder | None:
        """Resolve the embedding callable exactly once, fail-open.

        A failed setup latches permanently: retrying a broken download or a
        missing torch install on every step would stall precisely the hot
        path the extra promised not to touch.
        """
        if self._resolved:
            return self._embedder
        self._resolved = True
        try:
            embedder = self._model_loader()
        except Exception as exc:
            logger.warning(
                "snagline: semantic goal-drift disabled; embedding model "
                "unavailable (%s)",
                exc,
            )
            return None
        if embedder is None:
            # The loader already logged its specific reason.
            return None
        self._embedder = embedder
        return embedder

    def _make_default_model_loader(self) -> Callable[[], Embedder | None]:
        """Build the lazy sentence-transformers loader (the only heavy path).

        Returns a loader that yields an ``Embedder`` or ``None``. Both the
        import and the model construction are guarded; failures are logged
        with the pip hint and leave the detector inert.
        """
        model_name = self._cfg.semantic_drift_model

        def _load() -> Embedder | None:
            try:
                from sentence_transformers import SentenceTransformer
            except Exception as exc:
                logger.warning(
                    "snagline: drift extra unavailable (%s); run "
                    "`pip install snagline[drift]` for semantic "
                    "goal-drift",
                    exc,
                )
                return None
            try:
                st_model = SentenceTransformer(model_name)
            except Exception as exc:
                logger.warning(
                    "snagline: embedding model %r failed to load (%s); "
                    "semantic goal-drift stays inert",
                    model_name,
                    exc,
                )
                return None

            def _embed(event: StepEvent) -> Sequence[float]:
                return st_model.encode(
                    _event_label(event), show_progress_bar=False
                ).tolist()

            return _embed

        return _load


def fit_semantic_baseline(
    events: Iterable[StepEvent],
    embedder: Embedder,
    model: str | None = None,
) -> BaselineProfile:
    """Fit a ``BaselineProfile`` carrying both structure and semantics.

    Runs the standard structural fit (per-tool latency/error stats) and adds
    the mean embedding over the same healthy trajectory, so one persisted
    JSON serves the deterministic goal-drift detector and this semantic one.
    This is an explicit setup-time call: bad vectors raise ``ValueError``
    here instead of being swallowed, unlike the hot path which is fail-open.

    Only aggregated numbers are stored: the events' text never reaches the
    profile, matching the no-content-retention rule (project.md section 1.4).
    """
    profile = BaselineProfile()
    centroid: list[float] | None = None
    count = 0
    for ev in events:
        profile.add_event(ev)
        vec = [float(v) for v in embedder(ev)]
        if not all(math.isfinite(v) for v in vec):
            raise ValueError("embedding contains non-finite values")
        if centroid is None:
            centroid = [0.0] * len(vec)
        elif len(vec) != len(centroid):
            raise ValueError(
                f"embedding dimension changed during fit "
                f"({len(centroid)} -> {len(vec)})"
            )
        for i, v in enumerate(vec):
            centroid[i] += v
        count += 1
    if centroid is not None:
        profile.embedding_centroid = [c / count for c in centroid]
        profile.embedding_count = count
        profile.embedding_model = model
    return profile
