"""CONTINUUM sink -- escalate ``FailureRisk`` into a CONTINUUM review (issue #79).

Closing the loop: when SNAGLINE detects a failure risk in an agent whose
actions live in a CONTINUUM run, this sink escalates it through CONTINUUM's
existing request-human mechanism so a person is pulled in exactly where
SNAGLINE would want a pause.

Verified against current CONTINUUM source: the writable escalation path is
``continuum.actions.ActionLedger.flag_for_review(key, reason)``, which sets the
action's status to ``ActionStatus.REQUIRES_REVIEW`` and appends an auditable
ledger event; CONTINUUM's recovery planner then treats that action as needing a
person and its engine escalates the run toward ``request_human``. Run-level
``request_human`` is itself only a derived recovery mode in current CONTINUUM
(no public append API), so this sink escalates through the action ledger and
asks the caller to say which action key a risk refers to::

    ContinuumSink(storage, run_id, key_from_risk=my_mapping)

Risks with no resolvable action key are dropped with a log line rather than
fabricated onto the ledger: an invented key would either raise inside
CONTINUUM or, worse, attach SNAGLINE's opinion to the wrong action.

Optional extra ``snagline-agent[continuum]``. The ``continuum`` import is
lazy and guarded: constructing the sink without CONTINUUM installed raises a
helpful ``ImportError`` naming the extra (host setup error), while runtime
``emit()`` failures are caught and logged -- monitoring must never break the
host agent (project.md principle 2). Pass ``ledger_factory`` to inject any
duck-typed replacement in tests or vendored setups.

Privacy: only ``FailureRisk`` fields travel (score, trigger, ids, timestamps,
its content-free ``detail``). No prompt, tool output or event metadata is ever
forwarded (project.md section 11).
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from snagline.risk import FailureRisk

logger = logging.getLogger("snagline")

__all__ = ["ContinuumSink"]


def _default_ledger_factory(storage: Any, run_id: str) -> Any:
    """Build a real ``continuum.actions.ActionLedger``, importing lazily."""
    try:
        from continuum.actions import ActionLedger
    except ImportError as exc:  # pragma: no cover - exercised via fake factory
        raise ImportError(
            "ContinuumSink needs CONTINUUM's ActionLedger; install the "
            "'snagline-agent[continuum]' extra or pass ledger_factory="
            "for a duck-typed replacement"
        ) from exc
    return ActionLedger(storage, run_id)


class ContinuumSink:
    """Forward detected risks into CONTINUUM as ``REQUIRES_REVIEW`` actions."""

    def __init__(
        self,
        storage: Any,
        run_id: str,
        *,
        key_from_risk: Callable[[FailureRisk], str | None] | None = None,
        ledger_factory: Callable[[Any, str], Any] | None = None,
    ) -> None:
        self._storage = storage
        self._run_id = run_id
        self._key_from_risk = key_from_risk
        self._ledger_factory = ledger_factory or _default_ledger_factory
        # Resolve the ledger eagerly so a missing extra fails loudly during
        # host setup instead of silently swallowing every future alert.
        self._ledger = self._ledger_factory(storage, run_id)

    def emit(self, risk: FailureRisk) -> None:
        """Escalate one risk. Fire-and-forget; never raises into the Monitor."""
        try:
            key = self._key_from_risk(risk) if self._key_from_risk else None
            if not key:
                logger.warning(
                    "snagline continuum sink: no action key for %s/%s (%s); "
                    "cannot escalate without one",
                    risk.episode_id,
                    risk.step_id,
                    risk.trigger,
                )
                return
            self.escalate_action(
                key,
                f"[snagline:{risk.trigger}] score={risk.score:.2f} "
                f"severity={risk.severity} step={risk.step_id} episode="
                f"{risk.episode_id}: {risk.detail}",
            )
        except Exception:
            logger.warning(
                "snagline continuum sink: escalation failed for %s/%s; dropped",
                risk.episode_id,
                risk.step_id,
                exc_info=True,
            )

    def escalate_action(self, action_key: str, reason: str) -> None:
        """Flag one CONTINUUM action for human review (fail-open wrapper)."""
        try:
            self._ledger.flag_for_review(action_key, reason)
        except Exception:
            logger.warning(
                "snagline continuum sink: flag_for_review failed for key %s",
                action_key,
                exc_info=True,
            )
