"""Silent-abort detector (opt-in completion check, evaluated at episode end).

The run looked clean -- no loop, no cascade, no latency shift -- but the very
last ingested step was an error-free bare tool call instead of an output step.
The agent stopped mid-work without producing its result: a failure of
*omission*. No window or CUSUM statistic can see this coming, because there is
no behavioral buildup during the run; it only becomes decidable at the end.

This is the five-line "completion check" from *Real-Time Detection and Repair
of LLM Agent Failures* (arXiv:2608.02464), which caught 7/7 organic silent
aborts in that study -- the exact class the statistical monitors there scored
at or below chance on.

Implementation notes:

* ``Monitor.end_episode`` discovers detectors with a ``finalize`` method by
  duck typing (see ``detectors.base.EpisodeFinalizer``), so this detector adds
  zero overhead during ``ingest`` beyond storing a reference to the last event.
* A final step carrying ``error=True`` is NOT flagged: the error-cascade
  detector owns error signals, and double-counting one failure across two
  triggers would corrupt downstream alert policy.
* Only the action-type string and the error boolean are consulted; no content
  is read (project.md §1.4).
"""

from __future__ import annotations

from typing import Any

from snagline.config import Config
from snagline.detectors.base import snapshot_items
from snagline.events import StepEvent
from snagline.risk import FailureRisk

_DEFAULT_OUTPUT_TYPES = frozenset({"message", "plan_step"})


class SilentAbortDetector:
    name = "silent_abort"

    def __init__(
        self,
        output_action_types: frozenset[str] | set[str] | None = None,
        config: Config | None = None,
    ) -> None:
        del config  # thresholds live here only for symmetry with other detectors
        self.output_action_types = (
            frozenset(output_action_types)
            if output_action_types is not None
            else _DEFAULT_OUTPUT_TYPES
        )
        self._last: dict[str, StepEvent] = {}

    def observe(self, event: StepEvent) -> FailureRisk | None:
        self._last[event.episode_id] = event
        return None

    def finalize(self, episode_id: str) -> FailureRisk | None:
        last = self._last.pop(episode_id, None)
        if last is None or last.error:
            return None
        if last.action_type in self.output_action_types:
            return None
        return FailureRisk(
            episode_id,
            last.step_id,
            0.7,
            "silent_abort",
            f"episode ended on '{last.action_type}', not an output step",
            last.timestamp,
        )

    def reset(self, episode_id: str) -> None:
        self._last.pop(episode_id, None)

    def dump_state(self) -> dict[str, Any]:
        # snapshot_items: a concurrent ingest meeting a new episode must not
        # change the key set mid-comprehension (issue #231).
        return {
            "output_action_types": sorted(self.output_action_types),
            "last": {
                ep: {
                    "step_id": ev.step_id,
                    "timestamp": ev.timestamp,
                    "action_type": ev.action_type,
                    "error": ev.error,
                }
                for ep, ev in snapshot_items(self._last)
            },
        }

    def load_state(self, state: dict[str, Any]) -> None:
        types = state.get("output_action_types")
        if types:
            self.output_action_types = frozenset(types)
        self._last = {
            ep: StepEvent(
                step_id=raw["step_id"],
                episode_id=ep,
                timestamp=raw["timestamp"],
                action_type=raw["action_type"],
                action_signature="restored",
                error=raw["error"],
            )
            for ep, raw in state.get("last", {}).items()
        }
