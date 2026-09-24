"""Regression tests for the public-surface polish batch.

Each test corresponds to an upstream issue whose acceptance criterion is a
mechanical, re-checkable property: a flag exists, an export is present, a repr
leaks no credential, a dataclass contract holds, or the landing page does not
advertise a command that does not exist.

Covers issues #452 (``--version`` flag / no hardcoded version), #455 (site
terminal must not advertise a nonexistent ``snagline audit``), #458
(``--sink`` required-argument errors use the house-style prefix), #463
(user-facing sinks need a secret-free ``__repr__``), #464 (severity helpers
exported from the top-level package) and #465 (a behavioural unit test for the
public ``EpisodeMeta`` dataclass).
"""

from __future__ import annotations

import dataclasses
import re
from argparse import Namespace
from pathlib import Path

import pytest

from snagline import __version__
from snagline.cli import _build_parser, _build_sinks, main
from snagline.config import Config
from snagline.events import EpisodeMeta

REPO_ROOT = Path(__file__).resolve().parent.parent
PYPROJECT = REPO_ROOT / "pyproject.toml"
SITE_INDEX = REPO_ROOT / "site" / "index.html"

# ---------------------------------------------------------------- #452 -----
# ``snagline --version`` must print the installed version, and ``--help`` must
# not advertise a hardcoded one that can drift from pyproject.toml.


def test_version_flag_prints_installed_version(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert out.strip() == f"snagline {__version__}", out


def test_version_flag_prints_the_declared_pyproject_version():
    """The flag reads distribution metadata, not a literal (issue #452), so it
    must agree with the one place a release actually bumps the version.
    """
    tomllib = pytest.importorskip("tomllib")  # 3.11+; CI's oldest leg is 3.10
    with PYPROJECT.open("rb") as fh:
        declared = tomllib.load(fh)["project"]["version"]
    # A declared ``.dev0`` suffix is the one sanctioned mismatch (a local
    # editable install resolving to the released number it prefixes).
    assert __version__ == declared or declared.endswith(".dev0"), (
        f"pyproject.toml declares {declared!r} but the installed distribution "
        f"reports {__version__!r}: --version prints the latter"
    )


def test_help_no_longer_hardcodes_a_version(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    # The stale literal this replaced (issue #452).
    assert "v0.1" not in out, out
    assert "--version" in out, "the --version flag must appear in --help"


# ---------------------------------------------------------------- #458 -----
# The three ``--sink`` required-argument errors are the only CLI errors that
# were printed without the ``snagline ...:`` house style prefix.


@pytest.mark.parametrize(
    ("sink", "missing_flag"),
    [
        ("webhook", "--webhook-url"),
        ("slack", "--slack-url"),
        ("pagerduty", "--pagerduty-key"),
    ],
)
def test_sink_required_argument_errors_are_prefixed(capsys, sink, missing_flag):
    with pytest.raises(SystemExit) as exc:
        _build_sinks(
            Namespace(
                sink=sink,
                webhook_url=None,
                slack_url=None,
                pagerduty_key=None,
                pagerduty_source="snagline",
                min_severity=None,
                cooldown_seconds=0.0,
            ),
            Config(),
        )
    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert err.startswith("snagline: "), err
    assert missing_flag in err


# ---------------------------------------------------------------- #464 -----
# The score->severity mapping and the severity constants are part of the
# documented public surface and must import from the top-level package.


def test_severity_helpers_are_exported_from_the_top_level_package():
    from snagline import (
        SEVERITY_CRITICAL,
        SEVERITY_INFO,
        SEVERITY_WARNING,
        severity_from_score,
    )

    for name in (
        "severity_from_score",
        "SEVERITY_CRITICAL",
        "SEVERITY_WARNING",
        "SEVERITY_INFO",
    ):
        assert name in __import__("snagline").__all__, name
    assert severity_from_score(0.9) == SEVERITY_CRITICAL
    assert severity_from_score(0.6) == SEVERITY_WARNING
    assert severity_from_score(0.1) == SEVERITY_INFO


def test_severity_mapping_matches_the_risk_module():
    from snagline import severity_from_score as top_level
    from snagline.risk import severity_from_score as in_module

    assert top_level is in_module, "the export must be the same object"


# ---------------------------------------------------------------- #463 -----
# Sinks print as opaque ``<... object at 0x...>`` without a __repr__, but the
# repr is a log surface, so credential-bearing fields must never appear in it.


def test_every_shipped_sink_has_a_readable_repr():
    from snagline.sinks.batching import BatchingSink
    from snagline.sinks.console import ConsoleSink
    from snagline.sinks.dedup import DedupSink
    from snagline.sinks.heartbeat import HeartbeatSink
    from snagline.sinks.logging_sink import LoggingSink

    console = ConsoleSink()
    cases = [
        (console, "ConsoleSink"),
        (DedupSink(console, cooldown_seconds=30.0), "DedupSink"),
        (BatchingSink(console, max_batch=10, flush_interval=1.0), "BatchingSink"),
        (HeartbeatSink("/tmp/snagline-heartbeat"), "HeartbeatSink"),
        (LoggingSink(), "LoggingSink"),
    ]
    for sink, type_name in cases:
        text = repr(sink)
        # No sink may fall back to object.__repr__'s opaque address form.
        assert " object at 0x" not in text, (type_name, text)
        assert text.startswith(f"{type_name}("), text
        # AlertSink is a structural protocol (not runtime_checkable), so check
        # the shape directly: a repr is only useful on something that emits.
        assert callable(getattr(sink, "emit", None)), type_name


@pytest.mark.parametrize(
    ("module_name", "class_name", "secret"),
    [
        ("snagline.sinks.webhook", "WebhookSink", "https://hooks.example/SECRETKEY"),
        (
            "snagline.sinks.slack",
            "SlackSink",
            "https://hooks.slack.com/services/SECRETKEY",
        ),
        ("snagline.sinks.pagerduty", "PagerDutySink", "SECRET_ROUTING_KEY"),
    ],
)
def test_network_sink_reprs_leak_no_credential(module_name, class_name, secret):
    module = __import__(module_name, fromlist=[class_name])
    cls = getattr(module, class_name)
    if class_name == "PagerDutySink":
        sink = cls(secret, source="prod", min_severity="warning")
    else:
        sink = cls(secret, min_severity="warning")
    text = repr(sink)
    assert secret not in text, f"{class_name} repr leaks the credential: {text}"
    assert text.startswith(f"{class_name}("), text


# ---------------------------------------------------------------- #465 -----
# EpisodeMeta is a public, frozen, slotted dataclass whose ``tags`` default
# must be a fresh dict per instance (the ``default_factory`` contract).


def test_episode_meta_field_defaults():
    meta = EpisodeMeta(episode_id="run-1")
    assert meta.agent_name is None
    assert meta.started_at is None
    assert meta.tags == {}


def test_episode_meta_tags_default_factory_is_per_instance():
    a = EpisodeMeta(episode_id="run-1")
    b = EpisodeMeta(episode_id="run-2")
    a.tags["env"] = "prod"
    assert b.tags == {}, "two instances must not share one tags dict"
    assert dataclasses.asdict(a)["tags"] == {"env": "prod"}


def test_episode_meta_is_frozen():
    meta = EpisodeMeta(episode_id="run-1")
    with pytest.raises(dataclasses.FrozenInstanceError):
        meta.episode_id = "run-2"  # type: ignore[misc]


# ---------------------------------------------------------------- #455 -----
# The landing-page terminal is a simulation, but its ``--help`` output is what
# a newcomer copies. It must not advertise a ``snagline audit`` that does not
# exist, and must not omit the real subcommands a reader would otherwise never
# discover.


def test_site_terminal_help_lists_only_real_commands():
    if not SITE_INDEX.is_file():
        pytest.skip(f"{SITE_INDEX} not present in this checkout")

    text = SITE_INDEX.read_text(encoding="utf-8")
    # The simulated help block, from its echo line to the next command.
    block = text.split("function runHelp(")[1].split("function runClear(")[0]

    # Derive the real subcommands from the parser's own usage line rather than
    # duplicating the list here, so a new subcommand cannot silently go
    # unadvertised.
    usage = _build_parser().format_help()
    choices = re.search(r"\{([^}]*)\}", usage)
    assert choices, usage
    for real_subcommand in choices.group(1).split(","):
        assert f'<span class="ac">{real_subcommand}</span>' in block, (
            f"the simulated --help omits the real `snagline {real_subcommand}` "
            "subcommand"
        )
    # audit must not be presented as a real available command...
    audit_lines = [
        ln for ln in block.splitlines() if "audit" in ln and "demo" not in ln
    ]
    assert not audit_lines, (
        "the simulated --help presents `audit` as a real command; it is a "
        f"demo-only visualization: {audit_lines[:2]}"
    )
    # ...but the demo preset itself stays, honestly labelled.
    assert "not a real snagline command" in block
