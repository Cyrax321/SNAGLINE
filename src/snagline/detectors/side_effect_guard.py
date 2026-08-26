"""Side-effect guard: duplicate non-idempotent action detection (issue #88).

Adapters and hosts mark steps whose action is known to be non-idempotent (a
payment, a send, a deploy) with ``StepEvent.side_effect=True``. Within one
episode, the second occurrence of the same ``(tool_name, action_signature)``
pair emits a HIGH-severity risk with trigger ``"side_effect_duplicate"``:
duplicate sends/payments/deploys are the single most expensive failure class
in production agents, so by default the detector tolerates exactly one
occurrence (``allowed_repeats=1`` fires on the 2nd step).

Deliberately stricter than ``LoopDetector``, and different in three ways:

1. Scope: the loop detector watches a sliding window of arbitrary signatures
   to catch wasted-work repetition; this guard watches only host-declared
   side-effect steps, where repetition is itself the incident.
2. Threshold: the loop detector needs ``repeat_threshold`` (default 3) hits
   inside its window; here one repeat is already actionable.
3. Edge: the loop detector re-arms when its window drains; this guard fires
   once per ``(episode_id, tool_name, action_signature)`` key and stays quiet
   for the rest of the episode. A repeated payment must not alert-spam while
   the agent keeps making it worse; recovery is ``end_episode`` territory.
   Different-argument retries produce different signatures and never fire
   here (that shape belongs to the loop/stall detectors).

Privacy: only the boolean flag, tool name, and the one-way SHA-256
``action_signature`` digest are read. No prompt/response content, no
``StepEvent.metadata`` (project.md §1.4 / §11).

Performance: O(1) per step. Non-side-effect steps cost one attribute read
and a branch; marked steps do one dict upsert on a bounded inner dict.

Memory, stated honestly: state grows with *distinct* (tool_name, signature)
pairs per episode, not with repeats -- replaying the same charge a thousand
times still costs one entry. ``reset(episode_id)`` releases all of it.

The trigger string is API: CONTINUUM's policy table maps
``side_effect_duplicate`` to ABORT + immediate reconcile by name.
"""

from __future__ import annotations

from typing import Any, cast

from snagline.config import Config
from snagline.events import StepEvent
from snagline.risk import FailureRisk, TriggerType

# Declared here with the loop hardening modes' precedent (issue #89): widening
# the TriggerType literal in risk.py belongs to a change scoped to that module;
# the cast keeps mypy exact while the runtime value is a str.
TRIGGER_SIDE_EFFECT_DUPLICATE = cast(TriggerType, "side_effect_duplicate")


class SideEffectGuardDetector:
    """Fires once per repeated non-idempotent action within an episode."""

    name = "side_effect_guard"

    def __init__(
        self,
        allowed_repeats: int | None = None,
        score: float | None = None,
        config: Config | None = None,
    ) -> None:
        cfg = config or Config()
        self.allowed_repeats = (
            allowed_repeats
            if allowed_repeats is not None
            else cfg.side_effect_allowed_repeats
        )
        self.score = score if score is not None else cfg.side_effect_score
        if self.allowed_repeats < 1:
            raise ValueError("allowed_repeats must be >= 1")
        if not 0.0 <= self.score <= 1.0:
            raise ValueError("score must be within [0.0, 1.0]")
        # episode_id -> {(tool_name, action_signature): occurrence count}.
        # Inner dicts hold distinct actions only, so per-episode memory is
        # bounded by the episode's distinct marked actions, never by retries.
        self._counts: dict[str, dict[tuple[str | None, str], int]] = {}

    def observe(self, event: StepEvent) -> FailureRisk | None:
        if not event.side_effect:
            return None
        per_episode = self._counts.setdefault(event.episode_id, {})
        key = (event.tool_name, event.action_signature)
        count = per_episode.get(key, 0) + 1
        per_episode[key] = count
        # Edge-triggered: escalate exactly when the tolerated count is first
        # exceeded. Every later identical occurrence stays silent until
        # end_episode resets the episode (see module docstring).
        if count != self.allowed_repeats + 1:
            return None
        return FailureRisk(
            event.episode_id,
            event.step_id,
            self.score,
            TRIGGER_SIDE_EFFECT_DUPLICATE,
            f"non-idempotent {event.tool_name} fired {count}x in episode",
            event.timestamp,
        )

    def reset(self, episode_id: str) -> None:
        """Drop every counted action for ``episode_id`` (Monitor.end_episode)."""
        self._counts.pop(episode_id, None)

    def dump_state(self) -> dict[str, Any]:
        """Serialize occurrence counters for ``Monitor.snapshot`` (#91/#149).

        Plain nested dicts of ints by construction; only the inner
        ``(tool_name, action_signature)`` tuple keys need encoding, which
        become two-element JSON lists. Sorted so snapshots are deterministic.
        """
        return {
            "counts": {
                ep: [
                    [list(key), count]
                    for key, count in sorted(
                        per_episode.items(), key=lambda kv: repr(kv[0])
                    )
                ]
                for ep, per_episode in self._counts.items()
            }
        }

    def load_state(self, state: dict[str, Any]) -> None:
        self._counts = {
            str(ep): {(key[0], key[1]): int(count) for key, count in raw_counts}
            for ep, raw_counts in state.get("counts", {}).items()
        }
