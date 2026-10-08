"""The silent-abort completion check's output types must be configurable.

``SilentAbortDetector.finalize`` asks whether an episode's last step was an
action type this host counts as "the agent produced its result". That set is
host-specific: of the seven shipped integrations only the LangChain adapter
ends a healthy episode on one of the built-in defaults (``message`` /
``plan_step``). The Claude Code bridge ends on a ``tool_call`` (it drops the
``Stop`` hook), the OpenAI/Anthropic auto-wrappers label every LLM call
``tool_call``, the LangGraph adapter emits ``node_run``, and CrewAI/AutoGen
emit ``agent_step`` -- so with the stock default the detector fires at
``end_episode`` on essentially every *successful* run, which is the exact
false-positive storm DETECTOR_GUIDE rule 5 says gets a detector uninstalled.

The detector's own contract (issue #347) calls ``output_action_types``
configuration; until this change ``Monitor.default`` hardcoded it and
``Config`` had no field for it, so the documented escape hatch did not exist.
"""

from __future__ import annotations

import json

import pytest

from snagline.config import Config
from snagline.events import StepEvent, make_signature
from snagline.monitor import Monitor


class _Sink:
    def __init__(self) -> None:
        self.risks: list = []

    def emit(self, risk) -> None:
        self.risks.append(risk)


def _tool_only_episode(n: int = 12, action_type: str = "tool_call") -> list[StepEvent]:
    """A clean run that never emits a ``message``/``plan_step`` step.

    This is the shape the Claude Code bridge, the OpenAI/Anthropic auto
    wrappers and the LangGraph adapter produce: only tool-shaped steps, the
    last of which is an error-free success.
    """
    return [
        StepEvent(
            step_id=str(i),
            episode_id="ep",
            timestamp=float(i),
            action_type=action_type,
            action_signature=make_signature(action_type, "tool", str(i)),
            tool_name="tool",
            latency_ms=10.0,
        )
        for i in range(n)
    ]


def _run(cfg: Config, events: list[StepEvent]) -> list[str]:
    sink = _Sink()
    monitor = Monitor.default(config=cfg, sinks=[sink])
    for event in events:
        monitor.ingest(event)
    monitor.end_episode(events[-1].episode_id)
    return [risk.trigger for risk in sink.risks]


def test_stock_default_fires_on_a_tool_only_episode():
    """Documents the stock default's limit, not a defect in itself.

    The default names types only the LangChain adapter emits; a host whose
    steps are all tool-shaped cannot avoid this without the new knob.
    """
    cfg = Config(silent_abort_enabled=True)
    assert _run(cfg, _tool_only_episode()) == ["silent_abort"]


def test_config_field_reaches_monitor_default():
    """The whole point of the fix: an operator can retune the check through
    ``Config`` and it reaches the detector ``Monitor.default`` builds."""
    cfg = Config(
        silent_abort_enabled=True,
        silent_abort_output_action_types=frozenset({"tool_call"}),
    )
    assert _run(cfg, _tool_only_episode()) == [], (
        "a host whose final step IS its output step must not be paged"
    )


def test_config_field_reaches_monitor_default_for_node_run():
    """The LangGraph adapter's steps are all ``node_run``."""
    cfg = Config(
        silent_abort_enabled=True,
        silent_abort_output_action_types=frozenset({"node_run"}),
    )
    assert _run(cfg, _tool_only_episode(action_type="node_run")) == []


def test_real_silent_abort_still_fires_under_a_retuned_host():
    """Retuning must not blind the check: a host that counts ``tool_call`` as
    output still sees an episode that ended on a non-output step."""
    cfg = Config(
        silent_abort_enabled=True,
        silent_abort_output_action_types=frozenset({"tool_call"}),
    )
    events = _tool_only_episode()
    events.append(
        StepEvent(
            step_id="final",
            episode_id="ep",
            timestamp=99.0,
            action_type="observation",
            action_signature=make_signature("observation", None, "final"),
        )
    )
    assert _run(cfg, events) == ["silent_abort"]


def test_default_is_the_documented_pair():
    """The shipped default is unchanged by this fix."""
    assert Config().silent_abort_output_action_types == frozenset(
        {"message", "plan_step"}
    )


# --- env layering --------------------------------------------------------


def test_from_env_parses_a_comma_separated_list():
    cfg = Config.from_env(
        environ={
            "SNAGLINE_SILENT_ABORT_ENABLED": "1",
            "SNAGLINE_SILENT_ABORT_OUTPUT_ACTION_TYPES": " tool_call ,node_run ",
        }
    )
    assert cfg.silent_abort_enabled is True
    assert cfg.silent_abort_output_action_types == frozenset({"tool_call", "node_run"})


def test_from_env_drops_a_valueless_list():
    """An empty list would disable the detector, so it is ignored (with a
    warning, like every other uncoercible env value) rather than applied."""
    cfg = Config.from_env(environ={"SNAGLINE_SILENT_ABORT_OUTPUT_ACTION_TYPES": ","})
    assert cfg.silent_abort_output_action_types == frozenset({"message", "plan_step"})


def test_resolve_rejects_a_non_string_element_from_a_file(tmp_path):
    """A JSON file can carry a native non-string element the env path cannot
    produce; it reaches the same rejection."""
    path = tmp_path / "snagline.json"
    path.write_text(json.dumps({"silent_abort_output_action_types": ["message", 42]}))
    with pytest.raises(ValueError, match="non-empty strings"):
        Config.resolve(str(path))


# --- file layering -------------------------------------------------------


def test_resolve_reads_a_native_list_from_a_file(tmp_path):
    path = tmp_path / "snagline.json"
    path.write_text(
        json.dumps(
            {
                "silent_abort_enabled": True,
                "silent_abort_output_action_types": ["tool_call", "agent_step"],
            }
        )
    )
    cfg = Config.resolve(str(path))
    assert cfg.silent_abort_output_action_types == frozenset(
        {"tool_call", "agent_step"}
    )


def test_resolve_reads_a_comma_separated_string_from_a_file(tmp_path):
    path = tmp_path / "snagline.json"
    path.write_text(
        json.dumps({"silent_abort_output_action_types": "tool_call, node_run"})
    )
    assert Config.resolve(str(path)).silent_abort_output_action_types == frozenset(
        {"tool_call", "node_run"}
    )


def test_resolve_rejects_an_empty_list_from_a_file(tmp_path):
    """The file path previously had no coercion at all; an empty list reaches
    the validator and aborts startup loudly instead of silently blinding the
    check."""
    path = tmp_path / "snagline.json"
    path.write_text(json.dumps({"silent_abort_output_action_types": []}))
    with pytest.raises(ValueError, match="silent_abort_output_action_types"):
        Config.resolve(str(path))


def test_resolve_rejects_an_empty_element_from_a_file(tmp_path):
    path = tmp_path / "snagline.json"
    path.write_text(json.dumps({"silent_abort_output_action_types": ["message", ""]}))
    with pytest.raises(ValueError, match="non-empty strings"):
        Config.resolve(str(path))


# --- validation ----------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "why"),
    [
        (frozenset(), "at least one action type"),
        (frozenset({"message", ""}), "non-empty stripped strings"),
        (frozenset({"  "}), "non-empty stripped strings"),
        (frozenset({" message"}), "non-empty stripped strings"),
    ],
)
def test_construction_rejects_a_set_that_cannot_match_anything(value, why):
    with pytest.raises(ValueError, match=why):
        Config(silent_abort_enabled=True, silent_abort_output_action_types=value)
