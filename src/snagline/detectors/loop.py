"""Loop detector (tier-1, deterministic, O(1) amortized).

Per-episode sliding window (``collections.deque``) of recent
``action_signature`` values. If the same signature appears
``repeat_threshold`` times within ``window_size`` steps, emit a risk.

Each looping signature escalates once and then stays quiet until that loop
actually clears, so a long repetition alerts once rather than on every step.

Hardening modes (issue #89): three optional failure shapes beyond plain
repetition, each opt-in through ``Config`` and all disabled by default, so
with stock settings this detector behaves exactly as it did before #89:

- near-duplicate (``loop_near_duplicate_enabled``): retries whose signatures
  differ only by volatile identifiers (uuid-shaped substrings, digit runs)
  collapse onto one normalized key before hashing, then feed the same window
  and threshold logic as the plain path, emitting ``near_duplicate_loop``.
  The normalizer is a documented heuristic (see ``default_normalizer``) and
  replaceable per instance via the ``normalizer`` constructor hook.
- cycle (``loop_cycle_enabled``): A,B,A,B,... periodicity that never repeats
  one action often enough to trip ``repeat_threshold``. After each step an
  ascending scan finds the window content's minimal period p; it fires
  ``cycle`` once when p lies inside
  ``[loop_cycle_min_period, loop_cycle_max_period]`` (so configuring a band
  genuinely suppresses faster loops rather than re-flagging them through a
  multiple) and the window holds at least two full periods.
- stall (``loop_stall_enabled``): one action repeated N consecutive steps
  with no progress, firing ``stall`` after ``loop_stall_steps`` (default
  25). Wall-clock deltas never reset the streak: zero-delta steps count
  toward it (a frozen clock is itself evidence of a stall), and positive
  deltas do not either (tight retries burn real time while going nowhere).

No raw content is read -- only the one-way ``action_signature`` hash, so a
loop of identical retry attempts is caught without ever seeing the prompt or
response text (project.md §1.4). Hardening state follows the same rule:
normalized hashes, counts, timestamps, booleans.
"""

from __future__ import annotations

import hashlib
import re
from collections import deque
from collections.abc import Callable
from typing import Any, cast

from snagline.config import Config
from snagline.detectors.windowing import next_window
from snagline.events import StepEvent
from snagline.risk import FailureRisk, TriggerType

# Triggers added by the hardening modes (issue #89). These strings are API:
# downstream policy layers map them by name. They are declared here because
# widening the TriggerType literal in risk.py belongs to a change scoped to
# that module; the cast keeps mypy exact while the runtime value is a str.
TRIGGER_NEAR_DUPLICATE_LOOP = cast(TriggerType, "near_duplicate_loop")
TRIGGER_CYCLE = cast(TriggerType, "cycle")
TRIGGER_STALL = cast(TriggerType, "stall")

_UUID_LIKE_RE = re.compile(
    r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{12}"
)
_DIGIT_RUN_RE = re.compile(r"\d+")


def default_normalizer(signature: str) -> str:
    """Heuristic normalization applied before hashing in near-duplicate mode.

    Collapses uuid-like substrings to a single token and remaining digit runs
    to ``#``, so signatures differing only by volatile identifiers become
    equal. Deliberately blunt: against opaque hex digests collapsing digits
    raises collision odds, which is exactly why the mode is opt-in. Swap in a
    stricter function via the ``normalizer`` hook when needed.
    """
    text = _UUID_LIKE_RE.sub("<uuid>", signature)
    return _DIGIT_RUN_RE.sub("#", text)


def _normalized_key(normalize: Callable[[str], str], signature: str) -> str:
    """One-way short hash of the normalized signature (no content retained)."""
    return hashlib.sha256(normalize(signature).encode()).hexdigest()[:16]


class LoopDetector:
    name = "loop"

    def __init__(
        self,
        window_size: int | None = None,
        repeat_threshold: int | None = None,
        config: Config | None = None,
        normalizer: Callable[[str], str] | None = None,
    ) -> None:
        cfg = config or Config()
        self.window_size = (
            window_size if window_size is not None else cfg.loop_window_size
        )
        self.repeat_threshold = (
            repeat_threshold
            if repeat_threshold is not None
            else cfg.loop_repeat_threshold
        )
        self._windows: dict[str, deque] = {}
        # Dedupe: emit once per repetition episode (issue #4). Without this the
        # detector re-fires on every step while the same action keeps repeating,
        # which is alert spam.
        #
        # Keyed by signature, not by episode alone: two different actions looping
        # in one episode are two distinct findings, and one of them going quiet
        # must not re-arm the other.
        self._fired: dict[str, set[str]] = {}
        # --- hardening modes (issue #89). Everything below is inert unless its
        # enabling flag is set; with all flags off (the defaults) observe()
        # takes exactly the pre-#89 code path. ---
        self.loop_near_duplicate_enabled = cfg.loop_near_duplicate_enabled
        self.loop_cycle_enabled = cfg.loop_cycle_enabled
        self.loop_cycle_window_size = cfg.loop_cycle_window_size
        self.loop_cycle_min_period = max(1, cfg.loop_cycle_min_period)
        self.loop_cycle_max_period = max(
            cfg.loop_cycle_min_period, cfg.loop_cycle_max_period
        )
        self.loop_stall_enabled = cfg.loop_stall_enabled
        self.loop_stall_steps = cfg.loop_stall_steps
        self._normalize = normalizer or default_normalizer
        # Window auto-scaling (issue #92): inert unless cfg.window_scale_steps
        # > 0, in which case every window grows as base * ceil(len/steps),
        # capped at max_window. Per-episode event counts feed the scale factor.
        self._scale_steps = cfg.window_scale_steps
        self._max_window = cfg.max_window
        self._counts: dict[str, int] = {}
        self._cycle_counts: dict[str, int] = {}
        self._any_mode = (
            self.loop_near_duplicate_enabled
            or self.loop_cycle_enabled
            or self.loop_stall_enabled
        )
        self._near_windows: dict[str, deque] = {}
        self._near_fired: dict[str, set[str]] = {}
        self._cycle_windows: dict[str, deque] = {}
        self._cycle_fired: dict[str, bool] = {}
        self._stall_sig: dict[str, str] = {}
        self._stall_count: dict[str, int] = {}
        self._stall_start: dict[str, float] = {}
        self._stall_fired: dict[str, bool] = {}

    def observe(self, event: StepEvent) -> FailureRisk | None:
        hardened = self._observe_hardened(event) if self._any_mode else None
        plain = self._observe_plain(event)
        # The plain-loop trigger keeps precedence when both fire on one step;
        # every mode's state still advanced for this step either way.
        return plain if plain is not None else hardened

    def _observe_plain(self, event: StepEvent) -> FailureRisk | None:
        w = next_window(
            self._windows,
            self._counts,
            event.episode_id,
            self.window_size,
            self._scale_steps,
            self._max_window,
        )
        w.append(event.action_signature)
        count = w.count(event.action_signature)
        fired = self._fired.get(event.episode_id)
        if fired:
            # Re-arm any signature whose loop has ended -- it fell below the
            # threshold, or aged out of the window entirely -- so a later
            # repetition of it escalates again.
            #
            # Re-arming is decided here, from the window, and *not* from the
            # current step's own count. A step that is not part of the loop says
            # nothing about whether the loop is still running, so reading it as an
            # all-clear re-armed the flag and re-fired on the very next repeat:
            # one loop interleaved with distinct steps -- a retry between
            # reasoning turns -- then alerted on every repetition, which is the
            # spam this flag exists to prevent.
            #
            # Only the escalated signatures are re-checked, and the window holds
            # at most one per ``repeat_threshold`` slots. On the common path
            # nothing is looping, this dict has no entry, and the step costs
            # exactly what it did before.
            for sig in tuple(fired):
                seen = count if sig == event.action_signature else w.count(sig)
                if seen < self.repeat_threshold:
                    fired.discard(sig)
            if not fired:
                # Every escalated signature re-armed, so this episode is back to
                # "nothing looping". Drop the entry instead of leaving an empty
                # set behind: otherwise every episode that ever looped keeps
                # bookkeeping until ``reset()``, an unbounded per-episode leak of
                # exactly the kind the DedupSink sweep removes. Clearing the
                # local too makes the escalation path below recreate the entry
                # through ``setdefault`` -- mutating the detached set would
                # record the fire against a dict entry that no longer exists and
                # silently disable dedupe for the rest of the episode.
                del self._fired[event.episode_id]
                fired = None
        if count < self.repeat_threshold:
            return None
        if fired is None:
            fired = self._fired.setdefault(event.episode_id, set())
        elif event.action_signature in fired:
            return None
        fired.add(event.action_signature)
        score = min(1.0, count / self.repeat_threshold * 0.5)
        return FailureRisk(
            event.episode_id,
            event.step_id,
            score,
            "loop",
            f"action repeated {count}x in last {len(w)} steps",
            event.timestamp,
        )

    def _observe_hardened(self, event: StepEvent) -> FailureRisk | None:
        """Run every enabled mode; each advances its state on every step."""
        first: FailureRisk | None = None
        if self.loop_near_duplicate_enabled:
            risk = self._observe_near_duplicate(event)
            if risk is not None and first is None:
                first = risk
        if self.loop_cycle_enabled:
            risk = self._observe_cycle(event)
            if risk is not None and first is None:
                first = risk
        if self.loop_stall_enabled:
            risk = self._observe_stall(event)
            if risk is not None and first is None:
                first = risk
        return first

    def _observe_near_duplicate(self, event: StepEvent) -> FailureRisk | None:
        key = _normalized_key(self._normalize, event.action_signature)
        w = next_window(
            self._near_windows,
            self._counts,
            event.episode_id,
            self.window_size,
            self._scale_steps,
            self._max_window,
        )
        w.append(key)
        count = w.count(key)
        fired = self._near_fired.get(event.episode_id)
        if count < self.repeat_threshold:
            if fired is not None:
                # The normalized variant dropped below threshold or aged out:
                # re-arm so a later recurrence escalates again.
                fired.discard(key)
                if not fired:
                    del self._near_fired[event.episode_id]
            return None
        if fired is not None and key in fired:
            return None
        self._near_fired.setdefault(event.episode_id, set()).add(key)
        score = min(1.0, count / self.repeat_threshold * 0.5)
        return FailureRisk(
            event.episode_id,
            event.step_id,
            score,
            TRIGGER_NEAR_DUPLICATE_LOOP,
            f"signature variant repeated {count}x after id normalization",
            event.timestamp,
        )

    def _observe_cycle(self, event: StepEvent) -> FailureRisk | None:
        w = next_window(
            self._cycle_windows,
            self._cycle_counts,
            event.episode_id,
            self.loop_cycle_window_size,
            self._scale_steps,
            self._max_window,
        )
        w.append(event.action_signature)
        period = self._minimal_period(w)
        if period is None:
            # Periodicity broke (or never held): re-arm so a later cycle
            # escalates again instead of being silenced forever.
            self._cycle_fired.pop(event.episode_id, None)
            return None
        if self._cycle_fired.get(event.episode_id, False):
            return None
        self._cycle_fired[event.episode_id] = True
        repeats = len(w) // period
        score = min(1.0, repeats * 0.25)
        return FailureRisk(
            event.episode_id,
            event.step_id,
            score,
            TRIGGER_CYCLE,
            f"period-{period} cycle across last {len(w)} steps",
            event.timestamp,
        )

    def _minimal_period(self, w: deque) -> int | None:
        """True minimal period of the window content, if verifiable in band.

        Ascending scan from 1: the first candidate p whose repetition holds
        across the whole window is the content's minimal period, so a
        configured band filters on that true minimum. A period-2 loop with
        ``loop_cycle_min_period=3`` therefore stays silent instead of being
        re-flagged as a period-4 or period-6 multiple. Returns None when the
        minimal period sits outside the band (uniform windows have minimal
        period 1, below any useful band: single-action repetition belongs to
        the plain loop and stall modes), or when fewer than two full periods
        of the minimal period fit in the window yet.
        """
        n = len(w)
        for p in range(1, self.loop_cycle_max_period + 1):
            if n < 2 * p:
                return None  # larger candidates only need more history
            if all(w[i] == w[i + p] for i in range(n - p)):
                return p if p >= self.loop_cycle_min_period else None
        return None

    def _observe_stall(self, event: StepEvent) -> FailureRisk | None:
        sig = event.action_signature
        if self._stall_sig.get(event.episode_id) == sig:
            self._stall_count[event.episode_id] += 1
        else:
            # A different signature means something progressed: restart the
            # streak. Wall-clock deltas deliberately play no part in that
            # decision; see the module docstring for why zero-delta steps
            # must accumulate rather than reset.
            self._stall_sig[event.episode_id] = sig
            self._stall_count[event.episode_id] = 1
            self._stall_start[event.episode_id] = event.timestamp
        count = self._stall_count[event.episode_id]
        if count < self.loop_stall_steps:
            self._stall_fired.pop(event.episode_id, None)
            return None
        if self._stall_fired.get(event.episode_id, False):
            return None
        self._stall_fired[event.episode_id] = True
        elapsed = max(0.0, event.timestamp - self._stall_start[event.episode_id])
        score = min(1.0, count / self.loop_stall_steps * 0.5)
        return FailureRisk(
            event.episode_id,
            event.step_id,
            score,
            TRIGGER_STALL,
            f"identical action {count} steps in a row ({elapsed:.3f}s elapsed)",
            event.timestamp,
        )

    def reset(self, episode_id: str) -> None:
        self._windows.pop(episode_id, None)
        self._counts.pop(episode_id, None)
        self._fired.pop(episode_id, None)
        self._near_windows.pop(episode_id, None)
        self._near_fired.pop(episode_id, None)
        self._cycle_windows.pop(episode_id, None)
        self._cycle_counts.pop(episode_id, None)
        self._cycle_fired.pop(episode_id, None)
        self._stall_sig.pop(episode_id, None)
        self._stall_count.pop(episode_id, None)
        self._stall_start.pop(episode_id, None)
        self._stall_fired.pop(episode_id, None)

    def dump_state(self) -> dict[str, Any]:
        # _fired holds per-episode sets of escalated signatures (issue #94
        # re-arm semantics); sort so the JSON snapshot is deterministic.
        # Hardening modes (#89) persist their state too: a restart must not
        # silently reset an in-progress stall streak or cycle track.
        return {
            "windows": {ep: list(w) for ep, w in self._windows.items()},
            "counts": dict(self._counts),
            "fired": {ep: sorted(sigs) for ep, sigs in self._fired.items()},
            "near_windows": {ep: list(w) for ep, w in self._near_windows.items()},
            "near_fired": {ep: sorted(sigs) for ep, sigs in self._near_fired.items()},
            "cycle_windows": {ep: list(w) for ep, w in self._cycle_windows.items()},
            "cycle_counts": dict(self._cycle_counts),
            "cycle_fired": dict(self._cycle_fired),
            "stall_sig": dict(self._stall_sig),
            "stall_count": dict(self._stall_count),
            "stall_start": dict(self._stall_start),
            "stall_fired": dict(self._stall_fired),
        }

    def load_state(self, state: dict[str, Any]) -> None:
        self._windows = {
            ep: deque(sigs, maxlen=self.window_size)
            for ep, sigs in state.get("windows", {}).items()
        }
        # Tolerant .get() so pre-#92 snapshots restore cleanly; the counts only
        # position the auto-scaler and default to the window they imply.
        self._counts = {ep: int(n) for ep, n in state.get("counts", {}).items()}
        self._fired = {ep: set(sigs) for ep, sigs in state.get("fired", {}).items()}
        self._near_windows = {
            ep: deque(sigs, maxlen=self.window_size)
            for ep, sigs in state.get("near_windows", {}).items()
        }
        self._near_fired = {
            ep: set(sigs) for ep, sigs in state.get("near_fired", {}).items()
        }
        self._cycle_windows = {
            ep: deque(sigs, maxlen=self.loop_cycle_window_size)
            for ep, sigs in state.get("cycle_windows", {}).items()
        }
        self._cycle_counts = {
            ep: int(n) for ep, n in state.get("cycle_counts", {}).items()
        }
        self._cycle_fired = dict(state.get("cycle_fired", {}))
        self._stall_sig = dict(state.get("stall_sig", {}))
        self._stall_count = dict(state.get("stall_count", {}))
        self._stall_start = dict(state.get("stall_start", {}))
        self._stall_fired = dict(state.get("stall_fired", {}))
