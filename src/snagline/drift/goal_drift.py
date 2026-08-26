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
inside :meth:`SemanticGoalDriftDetector._default_model_loader`, lazily, on
first use, and only when no explicit ``embedder`` was injected. Every
failure on that path (extra missing, model download/load failing) or during
inference is caught, logged once, and leaves the detector permanently inert:
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
from snagline.events import StepEvent
from snagline.risk import FailureRisk

logger = logging.getLogger("snagline")

# Callable turning one step into an embedding vector. Injected by tests and
# by hosts with their own models; the default wraps sentence-transformers.
Embedder = Callable[[StepEvent], Sequence[float]]


def _event_label(event: StepEvent) -> str:
    """Structural text embedded for a step. Content-free by construction.

    Deliberately excludes ``action_signature`` (an opaque hash: meaningless
    in embedding space) and ``metadata`` (may carry raw content; detectors
    never read it, project.md section 4/11).
    """
    parts = [event.action_type, event.tool_name or "", event.error_type or ""]
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
    """Per-episode accumulator: running embedding sum and CUSUM debt."""

    __slots__ = ("debt", "n", "sum")

    def __init__(self) -> None:
        self.sum: list[float] | None = None
        self.n = 0
        self.debt = 0.0


class SemanticGoalDriftDetector:
    """Embedding-centroid drift detector emitting the ``goal_drift`` trigger.

    Fail-open twice over: :meth:`observe` never raises, and a broken
    embedding backend degrades to permanent inertness (logged once), not to
    repeated retries or a crashed monitor. Fires at most after
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

    def observe(self, event: StepEvent) -> FailureRisk | None:
        """Score one step; fail-open wrapper around the semantic path.

        Never raises: any internal error is logged here and by the Monitor,
        then ignored (project.md section 1.2).
        """
        try:
            return self._observe(event)
        except Exception:
            logger.exception(
                "snagline: semantic_goal_drift raised; ignoring (fail-open)"
            )
            return None

    def reset(self, episode_id: str) -> None:
        """Drop all per-episode state (running centroid and CUSUM debt)."""
        self._episodes.pop(episode_id, None)

    def dump_state(self) -> dict[str, Any]:
        return {
            "episodes": {
                ep: {"sum": st.sum, "n": st.n, "debt": st.debt}
                for ep, st in self._episodes.items()
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
            st.n = int(raw.get("n", 0))
            st.debt = float(raw.get("debt", 0.0))
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
        if st.n < self._cfg.semantic_drift_min_samples:
            return None
        live = [s / st.n for s in st.sum]
        sim = _cosine(live, self._baseline_centroid)
        dev = max(0.0, 1.0 - sim)
        tol = self._cfg.semantic_drift_tolerance
        signal = 0.0 if dev <= tol else min(1.0, (dev - tol) / (2.0 - tol))
        st.debt = max(0.0, st.debt + signal - self._cfg.semantic_drift_cusum_k)
        if st.debt >= self._cfg.semantic_drift_cusum_h:
            h = self._cfg.semantic_drift_cusum_h
            score = min(1.0, 0.5 + 0.5 * (st.debt - h) / h)
            st.debt = 0.0  # re-arm so persistent drift re-alarms later
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
                    "`pip install snagline-agent[drift]` for semantic "
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
