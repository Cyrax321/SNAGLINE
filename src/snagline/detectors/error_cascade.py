"""Error-cascade detector (tier-1, deterministic, O(1) amortized).

Same sliding-window shape as the loop detector but tracks the ``error``
boolean per episode instead of signatures. Fires on either of two conditions:

  * Consecutive: ``cascade_consecutive_threshold`` (default 3) errors in a row
    -- catches fast cascades immediately.
  * Windowed: ``cascade_error_threshold`` (default 3) errors anywhere in the
    last ``cascade_window_size`` (default 10) steps -- catches slow-burn
    degradations where errors are interleaved with occasional successes.

Only the boolean ``error`` flag is consulted; no content is read
(project.md §1.4).
"""

from __future__ import annotations

from collections import Counter, deque
from typing import Any

from snagline.config import Config
from snagline.detectors.base import snapshot_items
from snagline.detectors.windowing import (
    append_counted,
    effective_window_size,
    maintain_counter,
    next_window,
)
from snagline.events import StepEvent
from snagline.risk import FailureRisk


def _graded_cascade_score(observed: int, threshold: int) -> float:
    """Grade a cascade by how far it has run past ``threshold`` (issue #538).

    The alarm fires *at* the threshold, so a plain ``observed / threshold``
    ratio is always >= 1 and every alert lands on 1.0 / critical. Grading in
    bands keeps a marginal first crossing in the warning band while a cascade
    that keeps going still reaches critical, which is what the halt policy
    (``min_severity_for_halt``, default 0.8) is calibrated against.

    Bands are multiples of the threshold rather than a continuous ratio: the
    dedupe flag means the live path only ever fires at exactly
    ``observed == threshold``, so any continuous scaling would collapse to one
    value there and a real outage could never halt.
    """
    floor = max(int(threshold), 1)
    ratio = observed / floor
    if ratio >= 3.0:
        return 1.0
    if ratio >= 2.0:
        return 0.8
    return 0.5


class ErrorCascadeDetector:
    name = "error_cascade"

    def __init__(
        self,
        window_size: int | None = None,
        error_threshold: int | None = None,
        consecutive_threshold: int | None = None,
        config: Config | None = None,
    ) -> None:
        cfg = config or Config()
        self.window_size = (
            window_size if window_size is not None else cfg.cascade_window_size
        )
        self.error_threshold = (
            error_threshold
            if error_threshold is not None
            else cfg.cascade_error_threshold
        )
        self.consecutive_threshold = (
            consecutive_threshold
            if consecutive_threshold is not None
            else cfg.cascade_consecutive_threshold
        )
        # A tool failure and an LLM/chain error are different signals. By
        # default we only escalate *tool* failures as a cascade; flip
        # ``cascade_count_non_tool_errors`` to widen to every error step
        # (issue #16).
        self._count_non_tool = bool(
            getattr(cfg, "cascade_count_non_tool_errors", False)
        )
        self._windows: dict[str, deque] = {}
        self._consecutive: dict[str, int] = {}
        # Window auto-scaling (issue #92): inert unless cfg.window_scale_steps
        # > 0; see detectors/windowing.py for the growth rule.
        self._scale_steps = cfg.window_scale_steps
        self._max_window = cfg.max_window
        self._counts: dict[str, int] = {}
        # Dedupe: emit at most once per cascade, then stay quiet until the alarm
        # condition clears and re-arms (issue #4).
        # ``fired`` holds the band score the episode already alerted on (0.0
        # once cleared), not a bare bool: dedupe compares bands so a deepening
        # cascade can escalate (issue #538). Older snapshots carry booleans,
        # and ``bool`` is a valid float, so they restore as 1.0 -- "already
        # alerted at the top band", which only suppresses a repeat.
        self._fired: dict[str, float] = {}
        # Running count of True flags in each window (issue #298), maintained
        # only while scaling is on; see ``observe`` for why the default path
        # keeps using ``sum``.
        self._flags: dict[str, Counter[bool]] = {}
        self._sizes: dict[str, int | None] = {}

    def _is_error(self, event: StepEvent) -> bool:
        if not event.error:
            return False
        if event.action_type == "tool_call":
            return True
        return self._count_non_tool

    def observe(self, event: StepEvent) -> FailureRisk | None:
        counted = self._is_error(event)
        # Only *counted* errors feed the cascade window; an LLM/chain error
        # (when excluded) is treated as a clean step so it cannot inflate the
        # cascade signal.
        w = next_window(
            self._windows,
            self._counts,
            event.episode_id,
            self.window_size,
            self._scale_steps,
            self._max_window,
        )
        # Running count of True flags. ``sum(w)`` scans the whole deque, which
        # is O(window) per step -- the detector's entire cost once auto-scaling
        # (issue #92) grows the window past a few hundred steps. A Counter kept
        # in step with the deque makes it a single lookup (issue #298). Gated
        # on scaling: at the small fixed default window C-level ``sum`` is
        # still faster than the Python bookkeeping, and the published
        # default-path numbers must not move -- the same "defaults unchanged"
        # contract as issue #92.
        if self._scale_steps > 0:
            flags = maintain_counter(self._flags, self._sizes, event.episode_id, w)
            append_counted(w, flags, counted)
            total = flags[True]
        else:
            w.append(counted)
            total = sum(w)

        if counted:
            self._consecutive[event.episode_id] = (
                self._consecutive.get(event.episode_id, 0) + 1
            )
        else:
            self._consecutive[event.episode_id] = 0

        consecutive = self._consecutive[event.episode_id]
        consecutive_alarm = consecutive >= self.consecutive_threshold
        density_alarm = total >= self.error_threshold and len(w) >= self.error_threshold

        if not (consecutive_alarm or density_alarm):
            # The cascade cleared: re-arm so a later, independent cascade in the
            # same episode escalates again (mirrors ``LoopDetector``). Without
            # this the flag latches for the life of the episode, and a long-lived
            # episode -- a user session, or a sidecar episode that never calls
            # ``end_episode`` -- alerts exactly once, ever. A *sustained* cascade
            # still emits only once (issue #4): the flag clears only when neither
            # rule holds any more.
            self._fired[event.episode_id] = 0.0
            return None
        # A 0 threshold is rejected by Config validation (issue #322); guard
        # anyway, since Config is a plain mutable dataclass a host can
        # reconfigure after construction.
        if consecutive_alarm:
            graded = _graded_cascade_score(consecutive, self.consecutive_threshold)
        else:
            graded = _graded_cascade_score(total, self.error_threshold)
        # Dedupe on the *band*, not on a single boolean (issue #538, following
        # ``LoopDetector``): a cascade that deepens past the band it already
        # alerted on is new information, not a repeat. A plain flag latched at
        # the first crossing would keep a genuine outage pinned at 0.5/warning
        # for the rest of the episode, so a real failure could never reach
        # ``min_severity_for_halt``. Within one band the cascade still alerts
        # exactly once (issue #4); clearing the alarm below resets the band.
        already = self._fired.get(event.episode_id, 0.0)
        if graded <= already:
            return None

        self._fired[event.episode_id] = graded
        if consecutive_alarm:
            score = graded
            detail = f"{consecutive} consecutive errors"
        else:
            score = graded
            detail = f"{total} errors in last {len(w)} steps"
        return FailureRisk(
            event.episode_id,
            event.step_id,
            score,
            "error_cascade",
            detail,
            event.timestamp,
        )

    def reset(self, episode_id: str) -> None:
        self._windows.pop(episode_id, None)
        self._counts.pop(episode_id, None)
        self._consecutive.pop(episode_id, None)
        self._fired.pop(episode_id, None)
        self._flags.pop(episode_id, None)
        self._sizes.pop(episode_id, None)

    def dump_state(self) -> dict[str, Any]:
        # snapshot_items: a concurrent ingest meeting a new episode must not
        # change the key set mid-comprehension (issue #231).
        return {
            "windows": {ep: list(w) for ep, w in snapshot_items(self._windows)},
            "counts": dict(self._counts),
            "consecutive": dict(self._consecutive),
            "fired": dict(self._fired),
        }

    def load_state(self, state: dict[str, Any]) -> None:
        counts = state.get("counts", {})
        windows = state.get("windows", {})
        # Tolerant .get(): pre-#92 snapshots carry no scaler positions, so
        # each episode's position is inferred from the window it shipped.
        # The inferred value must seed _counts too, not merely size the
        # deque -- observe() reads _counts.get(ep, 0), so an episode left
        # absent restarts the scaler at the base and the first post-restore
        # observe refits the deque down, discarding the history that was
        # just restored (issue #403).
        # The windows and counts are built into locals and published only once
        # the whole snapshot has parsed: ``int()`` on a malformed count raises
        # partway through, and assigning live attribute-by-attribute would
        # leave the detector half-cleared -- some episodes restored, the rest
        # silently dropped -- with its live state destroyed and nothing
        # reporting the mismatch (review of #402).
        new_windows: dict[str, deque] = {}
        new_counts: dict[str, int] = {}
        for ep, flags in windows.items():
            n = int(counts.get(ep, len(flags)))
            new_windows[ep] = deque(
                flags,
                maxlen=effective_window_size(
                    self.window_size, n, self._scale_steps, self._max_window
                ),
            )
            new_counts[ep] = n
        for ep, n in counts.items():
            new_counts.setdefault(ep, int(n))
        # Parse the streaks and fired flags into locals too, before publishing
        # anything: ``int()`` on a malformed streak raises here, and publishing
        # windows/counts first (then letting the streak parse blow up) would
        # leave the detector half-restored -- snapshot windows/counts on top of
        # live streaks/fired -- the exact non-transactional state this guards
        # against (issue #402/#406).
        new_consecutive = {ep: int(v) for ep, v in state.get("consecutive", {}).items()}
        # ``fired`` carries a band score, not a bool, so a restored episode
        # keeps escalating from where it left off (issue #538). Snapshots
        # written before that change hold booleans; ``float(True)`` is 1.0,
        # which is exactly "already alerted at the top band", so they restore
        # without inventing a regression.
        new_fired = {ep: float(v) for ep, v in state.get("fired", {}).items()}
        # Every field has parsed; publish the whole snapshot at once.
        self._windows = new_windows
        self._counts = new_counts
        self._consecutive = new_consecutive
        self._fired = new_fired
        # The flag counts are derived from the windows above; a restored window
        # carries its own maxlen, so the cached sizes and counts are dropped and
        # recomputed on the first observe rather than trusted against a
        # snapshot whose scaling position differs.
        self._flags = {}
        self._sizes = {}
