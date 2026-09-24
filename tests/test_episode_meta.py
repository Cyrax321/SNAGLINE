"""Behavioral tests for the public ``EpisodeMeta`` dataclass (issue #465).

``EpisodeMeta`` is re-exported in ``snagline.__all__`` and is a
``frozen=True, slots=True`` dataclass whose ``tags`` uses a
``default_factory=dict``. It had only a ``__name__`` smoke reference
(``test_issue_regressions.py``, from issue #7's dead-construction path), never
a test of its actual contract. These pin the three properties that would
silently break under a careless refactor: defaults, per-instance ``tags``
isolation, and immutability.
"""

from __future__ import annotations

import dataclasses

import pytest

from snagline import EpisodeMeta as ExportedEpisodeMeta
from snagline.events import EpisodeMeta


def test_episode_meta_is_the_top_level_export() -> None:
    # Guards the public surface: the name in __all__ is this very type.
    assert ExportedEpisodeMeta is EpisodeMeta


def test_defaults_are_none_and_an_empty_tags_dict() -> None:
    meta = EpisodeMeta(episode_id="ep-1")
    assert meta.episode_id == "ep-1"
    assert meta.agent_name is None
    assert meta.started_at is None
    assert meta.tags == {}


def test_each_instance_gets_an_independent_tags_dict() -> None:
    # The default_factory contract: mutating one instance's tags must not leak
    # into another. A bare ``tags: dict = {}`` default would share one dict.
    a = EpisodeMeta(episode_id="a")
    b = EpisodeMeta(episode_id="b")
    a.tags["k"] = "v"
    assert a.tags == {"k": "v"}
    assert b.tags == {}
    assert a.tags is not b.tags


def test_frozen_assignment_raises() -> None:
    meta = EpisodeMeta(episode_id="ep-1")
    with pytest.raises(dataclasses.FrozenInstanceError):
        meta.agent_name = "gpt"  # type: ignore[misc]


def test_slots_leaves_instances_without_a_dict() -> None:
    # slots=True: instances carry no per-instance __dict__, so arbitrary
    # attributes cannot be stashed on them. Asserting the absence of __dict__
    # is the robust way to pin this -- the exact exception raised by an
    # undeclared assignment on a frozen+slots dataclass is a CPython-internal
    # detail (it can surface as TypeError, not AttributeError).
    meta = EpisodeMeta(episode_id="ep-1")
    assert not hasattr(meta, "__dict__")


def test_supplied_fields_are_kept() -> None:
    meta = EpisodeMeta(
        episode_id="ep-2",
        agent_name="agent-x",
        started_at=1234.5,
        tags={"env": "prod"},
    )
    assert meta.agent_name == "agent-x"
    assert meta.started_at == 1234.5
    assert meta.tags == {"env": "prod"}
