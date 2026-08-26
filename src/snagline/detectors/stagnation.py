"""Stagnation detector: novelty-rate collapse ("stuck" is not "loop").

``LoopDetector`` asks whether the agent repeats identical actions; this
detector asks whether it discovers anything new at all (issue #87). An agent
can evade exact loop matching by varying its arguments slightly: near
duplicates produce distinct signatures, so exact-match windows never trip.
The share of *never-before-seen* signatures collapses anyway, and that is what
this detector measures. The two failure shapes need different interventions
(replan vs interrupt), so they stay independent detectors.

Per episode, every ``action_signature`` is checked against ``seen_all_time``,
the set of signatures observed so far in that episode. A step whose signature
was never seen before is "novel". The detector tracks how many of the last
``window_size`` steps were novel; when that share drops below ``min_novelty``
for ``patience`` consecutive full-window observations, it emits exactly one
``FailureRisk(score=0.6, trigger="stagnation")``. The risk fires once per
collapse and re-arms only after novelty recovers above the threshold, so a
long stagnant stretch alerts once instead of on every step (issue #4 alert
spam precedent).

Privacy: only the one-way ``action_signature`` hash is read. No prompt or
response content, no ``StepEvent.metadata`` (project.md §1.4 / §11).

Performance: O(1) amortized per step. One set membership test, one deque
append with an optional eviction of a single boolean, two integer updates.

Memory, stated honestly: ``seen_all_time`` grows monotonically per episode by
design, so worst-case memory is O(unique signatures in the episode) and
``reset(episode_id)`` releases all of it. Signatures are full 64-character
hex digests since issue #15 (the issue #87 estimate of single-digit MB
predates that change and assumed 16-character hashes). Measured on CPython
3.14: retaining one digest costs about 105 bytes plus roughly 70 bytes of
amortized set overhead, call it ~175 bytes per unique action, so a pathological
all-unique 100k-step episode holds on the order of 17 MB. Typical episodes
repeat actions heavily and cost far less. The sliding novelty counter itself
is bounded: ``window_size`` booleans plus three integers per episode.

The detector ships opt-in behind ``Config.stagnation_enabled`` (default off)
so the zero-dependency preset and the published bench numbers are untouched;
see :meth:`snagline.monitor.Monitor.default`.
"""

from __future__ import annotations

from collections import deque
from typing import Any

from snagline.config import Config
from snagline.detectors.windowing import effective_window_size
from snagline.events import StepEvent
from snagline.risk import FailureRisk


class _EpisodeWindow:
    """Per-episode state: sliding novelty flags plus the all-time seen set."""

    __slots__ = ("flags", "novel_in_window", "seen_all_time", "stale_windows")

    def __init__(self) -> None:
        # One bool per step in the current window: was this signature novel
        # (never seen before in this episode) at push time? Booleans keep the
        # sliding counter cheap; the signatures themselves live only in
        # ``seen_all_time``.
        self.flags: deque[bool] = deque()
        self.novel_in_window = 0
        self.seen_all_time: set[str] = set()
        self.stale_windows = 0


class StagnationDetector:
    """Flags episodes whose rate of new unique actions collapses to near zero.

    Fires once per stagnation period: ``patience`` consecutive full-window
    observations below ``min_novelty`` novelty emit one risk, then the
    detector stays quiet until novelty recovers (which resets the stale
    counter) and a later collapse can fire again.
    """

    name = "stagnation"

    def __init__(
        self,
        window_size: int | None = None,
        min_novelty: float | None = None,
        patience: int | None = None,
        config: Config | None = None,
    ) -> None:
        cfg = config or Config()
        self.window_size = (
            window_size if window_size is not None else cfg.stagnation_window_size
        )
        self.min_novelty = (
            min_novelty if min_novelty is not None else cfg.stagnation_min_novelty
        )
        self.patience = patience if patience is not None else cfg.stagnation_patience
        if self.window_size < 1:
            raise ValueError("window_size must be >= 1")
        if not 0.0 <= self.min_novelty <= 1.0:
            raise ValueError("min_novelty must be within [0.0, 1.0]")
        if self.patience < 1:
            raise ValueError("patience must be >= 1")
        self._windows: dict[str, _EpisodeWindow] = {}
        # Window auto-scaling (issue #92): inert unless cfg.window_scale_steps
        # > 0; the novelty window grows toward max_window on long episodes.
        self._scale_steps = cfg.window_scale_steps
        self._max_window = cfg.max_window
        self._counts: dict[str, int] = {}

    def observe(self, event: StepEvent) -> FailureRisk | None:
        w = self._windows.setdefault(event.episode_id, _EpisodeWindow())
        n = self._counts.get(event.episode_id, 0) + 1
        self._counts[event.episode_id] = n
        target = effective_window_size(
            self.window_size, n, self._scale_steps, self._max_window
        )
        sig = event.action_signature
        novel = sig not in w.seen_all_time
        w.seen_all_time.add(sig)
        if len(w.flags) >= target and w.flags.popleft():
            w.novel_in_window -= 1
        w.flags.append(novel)
        if novel:
            w.novel_in_window += 1

        if len(w.flags) >= target and (w.novel_in_window / target < self.min_novelty):
            w.stale_windows += 1
            # Escalate exactly when patience is first reached. Continuing past
            # it must not re-fire (one finding per collapse), and any fresh
            # observation resets ``stale_windows`` below, re-arming us.
            if w.stale_windows == self.patience:
                rate = w.novel_in_window / target
                return FailureRisk(
                    event.episode_id,
                    event.step_id,
                    0.6,
                    "stagnation",
                    f"{rate:.0%} new actions in last {target} steps, "
                    f"below {self.min_novelty:.0%} for {self.patience} "
                    "consecutive windows",
                    event.timestamp,
                )
        else:
            w.stale_windows = 0
        return None

    def reset(self, episode_id: str) -> None:
        """Drop the window, the stale counter, and the monotonic seen-set."""
        self._windows.pop(episode_id, None)
        self._counts.pop(episode_id, None)

    def dump_state(self) -> dict[str, Any]:
        """Serialize per-episode windows for ``Monitor.snapshot`` (#91/#149).

        JSON-compatible throughout: the novelty ``flags`` deque becomes a plain
        list of booleans and ``seen_all_time`` is sorted so snapshots are
        deterministic; ``load_state`` rebuilds the set from the sorted list
        (raw sets are not JSON-serializable). The auto-scaler position
        (``counts``, issue #92) rides along so a restored episode keeps its
        scaling cadence.
        """
        return {
            "windows": {
                ep: {
                    "flags": list(w.flags),
                    "novel_in_window": w.novel_in_window,
                    "stale_windows": w.stale_windows,
                    "seen_all_time": sorted(w.seen_all_time),
                }
                for ep, w in self._windows.items()
            },
            # Tolerant readers on load: pre-#92 payloads carry no counts.
            "counts": dict(self._counts),
        }

    def load_state(self, state: dict[str, Any]) -> None:
        self._windows = {}
        for ep, raw in state.get("windows", {}).items():
            w = _EpisodeWindow()
            # Clamp to the CURRENT window_size: restore is tolerant by default
            # (matches by name, ignores config), so a snapshot taken with a
            # larger window must not leave an overlong deque whose equality-
            # based eviction never fires again. Mirrors LoopDetector's
            # deque(sigs, maxlen=self.window_size) rebuild; the sliding
            # counter is recomputed from the truncated flags so the two stay
            # consistent. With auto-scaling enabled (issue #92) the deque is
            # deliberately UNCAPPED: observe() trims at the current effective
            # target every step, and a base-sized maxlen would silently drop
            # flags and blind the detector once the target grows past it.
            flags = [bool(b) for b in raw.get("flags", [])][-self.window_size :]
            w.flags = deque(
                flags,
                maxlen=self.window_size if self._scale_steps <= 0 else None,
            )
            w.novel_in_window = sum(flags)
            w.stale_windows = int(raw.get("stale_windows", 0))
            w.seen_all_time = set(raw.get("seen_all_time", []))
            self._windows[str(ep)] = w
        self._counts = {
            ep: int(n)
            for ep, n in state.get("counts", {}).items()
        }
