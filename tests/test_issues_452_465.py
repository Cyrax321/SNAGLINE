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


def test_site_terminal_dispatches_every_command_it_advertises():
    """The simulated ``--help`` lists the six real subcommands (issue #455),
    and the dispatcher must route each one -- help text and the "command not
    recognized" fallback both suggest them, so a listed-but-unrouted command
    sends a newcomer straight into an error. #455 guarded the listing; this
    guards the wiring behind it.
    """
    if not SITE_INDEX.is_file():
        pytest.skip(f"{SITE_INDEX} not present in this checkout")

    text = SITE_INDEX.read_text(encoding="utf-8")
    js = "\n".join(re.findall(r"<script[^>]*>(.*?)</script>", text, re.S))

    # The commands runHelp advertises as "Available commands".
    advertised = set(re.findall(r"addLine\('  <span class=\"ac\">([a-z]+)</span>", js))
    # The dispatcher's own routed names (aliases included).
    routed = set(re.findall(r"main === '([a-z]+)'", js))
    # ``clear`` is routed through the real runClear, not the main dispatch
    # table's name list, so it is exempt from the routing check.
    unrouted = {c for c in advertised - routed if c != "clear"}

    assert advertised, "could not read the simulated --help command list"
    assert not unrouted, (
        "the site terminal advertises commands its own dispatcher does not "
        f"route, so typing one errors out: {sorted(unrouted)}"
    )

    # The "not recognized" fallback is the second place a stale suggestion can
    # live; it must not name a command that also fails to dispatch.
    fallback = text.split("command not recognized")[1].split("setBadge")[0]
    suggested = set(re.findall(r'<span class="ac">([a-z]+)</span>', fallback))
    bad = {c for c in suggested - routed if c != "clear"}
    assert not bad, (
        "the not-recognized error suggests commands that also fail to "
        f"dispatch: {sorted(bad)}"
    )


# ---------------------------------------------------------------- #565 -----
# The demo ``audit`` panel presents itself as "detector suite verification",
# so a reader takes its class names and numbers as the shipped suite. Every
# name it prints must be a real detector class, and every default must be the
# real one -- the previous text invented ``CascadeDetector``/``EnsembleVoter``
# and parameters (``ngram=3``, ``threshold=0.75``, ``h=3.0σ``) that do not
# exist anywhere in the package.


def _audit_panel_lines(text: str) -> list[str]:
    """The AUDIT_LINES array, one cleaned string per entry."""
    block = text.split("var AUDIT_LINES = [")[1].split("];")[0]
    return re.findall(r"text:\s*'([^']*)'", block)


def test_audit_panel_names_only_real_default_detectors():
    """The panel's ``[ONLINE]`` rows must be exactly the detectors
    ``Monitor.default()`` actually wires -- derived, not duplicated, so a
    newly promoted detector cannot leave the panel stale.
    """
    if not SITE_INDEX.is_file():
        pytest.skip(f"{SITE_INDEX} not present in this checkout")

    from snagline import Monitor
    from snagline.config import Config

    shipped = {type(d).__name__ for d in Monitor.default(Config())._detectors}

    lines = _audit_panel_lines(SITE_INDEX.read_text(encoding="utf-8"))
    online = {
        name
        for ln in lines
        for name in re.findall(r'<span class="ac">([A-Za-z]+Detector)</span>', ln)
        if "[ONLINE]" in ln
    }
    assert online, "could not read the audit panel's [ONLINE] rows"
    assert online == shipped, (
        "the audit panel's [ONLINE] rows are not the detectors "
        f"Monitor.default() ships: panel={sorted(online)} "
        f"actual={sorted(shipped)}"
    )


def test_audit_panel_quotes_real_config_defaults():
    """Each default the panel prints must match the Config dataclass, so the
    numbers a reader copies are the numbers the code uses.
    """
    if not SITE_INDEX.is_file():
        pytest.skip(f"{SITE_INDEX} not present in this checkout")

    cfg = Config()
    lines = _audit_panel_lines(SITE_INDEX.read_text(encoding="utf-8"))
    online = "\n".join(ln for ln in lines if "[ONLINE]" in ln)

    # Every "<attr>=<value>" the panel asserts, checked against the dataclass.
    assertions = {
        "window": cfg.loop_window_size,
        "repeat_threshold": cfg.loop_repeat_threshold,
        "error_threshold": cfg.cascade_error_threshold,
        "consecutive": cfg.cascade_consecutive_threshold,
        "min_samples": cfg.cusum_min_samples,
    }
    for attr, real in assertions.items():
        m = re.search(rf"{attr}=(\d+)", online)
        if m is None:
            continue  # the panel may omit a knob; it may not misstate one
        assert int(m.group(1)) == real, (
            f"the audit panel prints {attr}={m.group(1)} but Config defaults "
            f"to {attr}={real}"
        )

    # The CUSUM detector's k/h are floats; the panel renders them with a sigma
    # suffix, so check the leading digit only.
    k = re.search(r"k=([0-9.]+)σ", online)
    assert k and float(k.group(1)) == cfg.cusum_k, (k, cfg.cusum_k)
    h = re.search(r"h=([0-9.]+)σ", online)
    assert h and float(h.group(1)) == cfg.cusum_h, (h, cfg.cusum_h)


def test_audit_panel_optin_trigger_list_matches_the_registry():
    """The panel claims a count of opt-in triggers; both the count and the
    names must match ``TriggerType`` minus what the zero-config preset emits.
    """
    if not SITE_INDEX.is_file():
        pytest.skip(f"{SITE_INDEX} not present in this checkout")

    import typing

    from snagline.risk import TriggerType

    # The always-on surface: the three tier-1 detectors plus the two
    # Monitor-level guards (idle_gap, wall_clock_budget) and the loop
    # detector's extra trigger aliases.
    always_on = {
        "loop",
        "cycle",
        "stall",
        "near_duplicate_loop",
        "error_cascade",
        "latency_anomaly",
        "idle_gap",
        "wall_clock_budget",
    }
    opt_in = sorted(set(typing.get_args(TriggerType)) - always_on)

    block = SITE_INDEX.read_text(encoding="utf-8")
    lines = _audit_panel_lines(block)
    listed = sorted(
        name
        for ln in lines
        for name in re.findall(
            r"(stagnation|goal_drift|token_runaway|meltdown|silent_abort|"
            r"side_effect_duplicate|governance_decay|budget_breach|"
            r"ml_ensemble)\b",
            ln,
        )
    )
    assert listed, "could not read the opt-in trigger list from the panel"
    assert listed == opt_in, (
        "the audit panel's opt-in trigger list does not match TriggerType "
        f"minus the always-on set: panel={listed} actual={opt_in}"
    )

    # The headline count must agree with the list it sits above.
    count = re.search(r"(\d+) further triggers are opt-in", block)
    assert count, "the panel no longer states an opt-in trigger count"
    assert int(count.group(1)) == len(opt_in), (
        f"the panel says {int(count.group(1))} triggers are opt-in but "
        f"{len(opt_in)} are: {opt_in}"
    )


# The ``bench`` demo is the same class of problem one panel over: it printed a
# per-detector breakdown (``[1/4] RingBuffer``, ``[2/4] LoopDetector (FNV-1a
# ngram hash sequence)``) that the real ``snagline bench`` never produces and
# whose named mechanisms do not exist in the package at all.

# Mechanism names the panel invented; none appear anywhere in the package.
_FABRICATED_MECHANISMS = (
    "RingBuffer",
    "FNV-1a",
    "ngram",
    "bitmask",
    "recursive tabular",
    "CascadeDetector",
    "LatencyCUSUM",
    "EnsembleVoter",
    "FAIL_EARLY",
)


SRC_ROOT = SITE_INDEX.resolve().parents[1] / "src"


def _package_source() -> str:
    """Every ``.py`` under ``src/``, or an empty string in an installed
    checkout where the sources are not next to the site file.
    """
    if not SRC_ROOT.is_dir():
        return ""
    return "\n".join(
        p.read_text(encoding="utf-8", errors="ignore") for p in SRC_ROOT.rglob("*.py")
    )


@pytest.mark.parametrize("term", _FABRICATED_MECHANISMS)
def test_site_panels_invent_no_mechanism_the_package_lacks(term):
    """A reader may ``grep`` the source for any mechanism the site asserts;
    every one of these must resolve to nothing in the site *or* the package.
    """
    if not SITE_INDEX.is_file():
        pytest.skip(f"{SITE_INDEX} not present in this checkout")

    text = SITE_INDEX.read_text(encoding="utf-8")
    # Word-boundary match so ``ErrorCascadeDetector`` (a real class) is not
    # mistaken for the invented ``CascadeDetector``.
    pattern = rf"\b{re.escape(term)}\b"
    assert not re.search(pattern, text), (
        f"the site asserts a `{term}` mechanism that does not exist anywhere "
        "in the package"
    )
    # The site half above only proves the demo stopped naming it. This half
    # keeps the claim honest: the term must be absent from the sources too, so
    # the docstring's "a reader may grep the source" stays true.
    source = _package_source()
    if source:
        assert not re.search(pattern, source), (
            f"`{term}` does exist in the package sources, so this test's "
            "premise is wrong -- re-check the site text for this term"
        )


def test_bench_panel_mirrors_the_real_command_output():
    """The real ``snagline bench`` prints percentiles for the whole per-step
    path, not a per-detector breakdown; the demo must show that shape.
    """
    if not SITE_INDEX.is_file():
        pytest.skip(f"{SITE_INDEX} not present in this checkout")

    import inspect

    from snagline.cli import _cmd_bench

    text = SITE_INDEX.read_text(encoding="utf-8")
    bench = text.split("var BENCH_LINES = [")[1].split("];")[0]

    # Every label the real command prints must appear in the demo, so a
    # reader who runs `snagline bench` recognizes the output.
    source = inspect.getsource(_cmd_bench)
    labels = [
        label.strip()
        for label in re.findall(r'f?"  (\w[\w ]*) :', source)
        if label.strip()
    ]
    # The regex above is pinned to the literal spacing in ``_cmd_bench``'s
    # format strings. If that formatting changes it silently matches nothing
    # and the loop below asserts nothing at all, so require that it matched.
    assert labels, (
        "could not extract any bench labels from _cmd_bench; the demo/"
        "command label regex has drifted from the real command's formatting"
    )
    for label in labels:
        # The real code prints "median 2.43 us/step"; the demo spans
        # '<span>' tags around the value, so match the label alone.
        assert re.search(rf"{re.escape(label)}\b", bench) or re.search(
            rf">{re.escape(label)}\s*<", bench
        ), f"the bench demo omits the real `{label}` line"


def test_bench_panel_reports_live_numbers_not_hardcoded_results():
    """The demo measures in-browser, so its median must be a computed value,
    not the hardcoded 2.08 the previous version printed as its result.
    """
    if not SITE_INDEX.is_file():
        pytest.skip(f"{SITE_INDEX} not present in this checkout")

    text = SITE_INDEX.read_text(encoding="utf-8")
    bench = text.split("var BENCH_LINES = [")[1].split("];")[0]
    # The result line must interpolate the measured value...
    assert "MEDIAN PIPELINE OVERHEAD" in bench
    assert "' + medUs + '" in bench, (
        "the bench result line is not derived from the in-browser measurement"
    )
    # ...and the badge it sets must be the same computed string.
    tail = text.split("var BENCH_LINES = [")[1]
    assert "'2.08µs/step'" not in tail, (
        "the bench badge is still hardcoded to a published figure rather than "
        "the measurement it just took"
    )


# ---------------------------------------------------------------- #538 -----
# The replay demo prints the cascade's score where it intercepts. The detector
# grades that score now (issue #538), so the panel must print what the detector
# actually emits for that streak, not the flat 1.00 the graded formula
# replaced.


def _replay_steps_block(text: str) -> str:
    """The REPLAY_STEPS array body the terminal types out."""
    return text.split("var REPLAY_STEPS = [")[1].split("];")[0]


def _replay_intercept_scores(block: str) -> list[tuple[int, float]]:
    """``[(streak, score)]`` for every cascade intercept in the block."""
    return [
        (int(m.group(1)), float(m.group(2)))
        for m in re.finditer(r"streak=(\d+), score=([0-9.]+)", block)
    ]


def _detector_score_for_streak(streak: int) -> float:
    """The score the shipped detector emits once a cascade reaches ``streak``
    consecutive tool errors under the Config defaults -- derived by running the
    detector, not by restating its grading formula.
    """
    from snagline.detectors.error_cascade import ErrorCascadeDetector
    from snagline.events import StepEvent

    detector = ErrorCascadeDetector()
    last = None
    for i in range(1, streak + 1):
        risk = detector.observe(
            StepEvent(str(i), "ep", float(i), "tool_call", "tool:fetch_url", error=True)
        )
        if risk is not None:
            last = risk
    assert last is not None, f"no cascade fired after {streak} consecutive errors"
    return last.score


def test_replay_panel_prints_the_graded_cascade_score():
    """Each intercept's score must be the one the detector emits for that
    streak (issue #538). The panel printed a flat 1.00 at the first crossing,
    which is a 0.50 warning now -- and which could never have halted.
    """
    if not SITE_INDEX.is_file():
        pytest.skip(f"{SITE_INDEX} not present in this checkout")

    scores = _replay_intercept_scores(
        _replay_steps_block(SITE_INDEX.read_text(encoding="utf-8"))
    )
    assert scores, "the replay panel no longer prints a cascade intercept"

    for streak, printed in scores:
        emitted = _detector_score_for_streak(streak)
        assert printed == emitted, (
            f"the replay panel prints score={printed} at streak={streak} but "
            f"the detector emits {emitted}"
        )


def test_replay_panel_halts_only_once_the_cascade_reaches_the_halt_band():
    """The AGENT_HALT the panel dispatches must follow an intercept inside the
    halt band: ``min_severity_for_halt`` is what gates the webhook, so a panel
    that halts on a 0.50 warning demonstrates a path the package never takes.
    """
    if not SITE_INDEX.is_file():
        pytest.skip(f"{SITE_INDEX} not present in this checkout")

    block = _replay_steps_block(SITE_INDEX.read_text(encoding="utf-8"))
    halt = block.find("AGENT_HALT")
    if halt < 0:
        pytest.skip("the replay panel no longer dispatches a halt")
    before_halt = _replay_intercept_scores(block[:halt])
    assert before_halt, "no cascade intercept precedes the halt"
    best = max(score for _, score in before_halt)
    cfg = Config()
    assert best >= cfg.min_severity_for_halt, (
        f"the replay panel dispatches AGENT_HALT but the cascade in view tops "
        f"out at {best}, below min_severity_for_halt={cfg.min_severity_for_halt}"
    )


# ------------------------------------------- site: watch/serve/baseline ---
# The remaining scripted panels type out detector output and CLI output line
# by line. Each line must be something the package actually emits -- the same
# contract #567 established for audit/bench and #580 for replay. These three
# panels still invented a detector count (5, the default monitor wires 3), a
# cascade score (0.80 for three consecutive errors; the graded formula emits
# 0.50, #538), a "cusum 4.12 > h=3.0" line the latency detector never prints
# (h defaults to 5.0), an ENSEMBLE risk from an opt-in detector, a p95
# breakdown the baseline command never produces, and a `snagline load_baseline`
# command that does not exist.


def _panel_block(text: str, marker: str) -> str:
    """The body of one scripted panel array, e.g. ``var WATCH_STREAM = [...]``."""
    return text.split(f"var {marker} = [")[1].split("];")[0]


class _RecordingSink:
    """Collects what the monitor dispatches, so a scripted panel can be
    replayed and its claims compared against real emissions.
    """

    def __init__(self) -> None:
        self.risks: list = []

    def emit(self, risk) -> None:
        self.risks.append(risk)

    def reset(self, episode_id: str) -> None:
        self.risks.clear()


# The uppercase risk labels the watch stream prints -> the wire trigger names.
_WATCH_LABELS = {
    "LOOP": "loop",
    "CASCADE": "error_cascade",
    "LATENCY": "latency_anomaly",
}

_RISK_LABEL_RE = r'!</span> <span class="\w+">([A-Z][A-Z_]+)</span>'


def test_watch_panel_reports_the_real_default_detector_count():
    """The panel states "N detectors online"; N must be the count
    ``Monitor.default()`` actually wires, not a larger marketing number.
    """
    if not SITE_INDEX.is_file():
        pytest.skip(f"{SITE_INDEX} not present in this checkout")

    from snagline.monitor import Monitor

    block = _panel_block(SITE_INDEX.read_text(encoding="utf-8"), "WATCH_STREAM")
    m = re.search(r"\((\d+) detectors online\)", block)
    assert m, "the watch panel no longer states a detector count"
    monitor = Monitor.default()
    assert int(m.group(1)) == len(monitor._detectors), (
        f"the watch panel claims {m.group(1)} detectors online but "
        f"Monitor.default() wires {len(monitor._detectors)}: "
        f"{[d.name for d in monitor._detectors]}"
    )


def test_watch_panel_only_names_detectors_a_plain_watch_runs():
    """Every risk label in the default stream must be one of the detectors a
    plain ``snagline watch`` runs. ``ml_ensemble`` is opt-in, so an ENSEMBLE
    line in the default stream is machinery the watcher does not have.
    """
    if not SITE_INDEX.is_file():
        pytest.skip(f"{SITE_INDEX} not present in this checkout")

    from snagline.monitor import Monitor

    block = _panel_block(SITE_INDEX.read_text(encoding="utf-8"), "WATCH_STREAM")
    default = {d.name for d in Monitor.default()._detectors}
    labels = re.findall(_RISK_LABEL_RE, block)
    assert labels, "could not read any risk label from the watch panel"
    for label in labels:
        trigger = _WATCH_LABELS.get(label)
        assert trigger is not None, (
            f"the watch panel labels a risk {label!r} that this test does not "
            "map to a trigger; add the mapping and confirm it is a default "
            "detector before allowing it"
        )
        assert trigger in default, (
            f"the watch panel presents {label} in the default stream, but "
            f"{trigger} is not among the default detectors {sorted(default)}"
        )


def test_watch_panel_scores_are_what_the_detectors_actually_emit():
    """Replay the panel's own typed-out step sequence through the real monitor
    and assert every (trigger, score) the panel claims really fires for it.
    """
    if not SITE_INDEX.is_file():
        pytest.skip(f"{SITE_INDEX} not present in this checkout")

    from snagline.events import StepEvent, make_signature
    from snagline.monitor import Monitor

    block = _panel_block(SITE_INDEX.read_text(encoding="utf-8"), "WATCH_STREAM")

    # Rebuild the episode the panel types out: "tool_call:<name> ok|err [Nms]".
    # The spans are colour markup only, so strip them before parsing the text.
    visible = re.sub(r"<[^>]+>", "", block)
    sequence = re.findall(r"tool_call:(\w+)\s+(ok|err)(?:\s+(\d+)ms)?", visible)
    assert sequence, "could not parse any steps from the watch panel"

    sink = _RecordingSink()
    monitor = Monitor.default(Config(), sinks=[sink])
    for i, (tool, outcome, ms) in enumerate(sequence, start=1):
        monitor.ingest(
            StepEvent(
                step_id=str(i),
                episode_id="ep",
                timestamp=float(i),
                action_type="tool_call",
                action_signature=make_signature("tool_call", tool),
                tool_name=tool,
                latency_ms=float(ms) if ms else 50.0,
                error=(outcome == "err"),
            )
        )
    fired = {(r.trigger, r.score) for r in sink.risks}

    for label, score in re.findall(_RISK_LABEL_RE + r".*?score ([0-9.]+)", block):
        claimed = (_WATCH_LABELS[label], float(score))
        assert claimed in fired, (
            f"the watch panel claims {label} at score {score} for the exact "
            f"sequence it types out, but the monitor emits {sorted(fired)}"
        )


def test_serve_panel_posts_bodies_the_server_accepts():
    """The panel is labelled "demo of the real server", so every body it POSTs
    must construct a ``StepEvent`` the way the server does (``StepEvent(**obj)``)
    -- it used to omit ``timestamp`` and ``action_signature``, which have no
    defaults, so the real server answers 400 to every one of them.
    """
    if not SITE_INDEX.is_file():
        pytest.skip(f"{SITE_INDEX} not present in this checkout")

    import json

    from snagline.events import StepEvent

    block = _panel_block(SITE_INDEX.read_text(encoding="utf-8"), "SERVE_LINES")
    bodies = re.findall(
        r"POST /events</span> &lt; <span class=\"ac\">(.*?)</span>", block
    )
    assert bodies, "could not parse any POST body from the serve panel"
    for body in bodies:
        StepEvent(**json.loads(body))  # must not raise


def test_serve_panel_risk_is_what_the_detectors_actually_emit():
    """Replay the panel's POSTed bodies and assert the risk it shows really
    fires, with the score the panel prints. It claimed a latency_anomaly at
    0.84 off a single 4.0s spike with an invented "(k=3.0 sigma)" -- one spike
    after two fast steps fires nothing, and ``cusum_k`` defaults to 0.5.
    """
    if not SITE_INDEX.is_file():
        pytest.skip(f"{SITE_INDEX} not present in this checkout")

    import json

    from snagline.events import StepEvent
    from snagline.monitor import Monitor

    block = _panel_block(SITE_INDEX.read_text(encoding="utf-8"), "SERVE_LINES")
    bodies = re.findall(
        r"POST /events</span> &lt; <span class=\"ac\">(.*?)</span>", block
    )
    assert bodies, "could not parse any POST body from the serve panel"

    sink = _RecordingSink()
    monitor = Monitor.default(Config(), sinks=[sink])
    for body in bodies:
        monitor.ingest(StepEvent(**json.loads(body)))
    fired = {(r.trigger, r.score): r.detail for r in sink.risks}

    visible = re.sub(r"<[^>]+>", "", block)
    m = re.search(r"latency_anomaly risk ([0-9.]+)", visible)
    if m is None:
        pytest.skip("the serve panel no longer claims a latency risk")
    claimed = ("latency_anomaly", float(m.group(1)))
    assert claimed in fired, (
        f"the serve panel claims {claimed[0]} at score {claimed[1]} for the "
        f"events it POSTs, but the monitor emits {sorted(fired)}"
    )
    # The detail the panel prints must also be the detector's real detail.
    # [^'] stops at the JS string terminator, not the end of the block.
    detail = re.search(r"risk [0-9.]+ &middot; ([^']+)", visible)
    assert detail is not None, "could not read the panel's risk detail text"
    assert detail.group(1).strip() == fired[claimed], (
        f"the serve panel prints {detail.group(1)!r} but the detector emits "
        f"{fired[claimed]!r}"
    )


def test_baseline_panel_mirrors_the_real_command(capsys, tmp_path):
    """Every per-tool row the panel prints must appear in the real command's
    output on the real fixture. The panel showed an invented p95 breakdown for
    tools the fixture never calls, at a step count it does not have.
    """
    if not SITE_INDEX.is_file():
        pytest.skip(f"{SITE_INDEX} not present in this checkout")

    fixture = REPO_ROOT / "tests" / "fixtures" / "trajectories" / "healthy_run.jsonl"
    if not fixture.is_file():
        pytest.skip(f"{fixture} not present in this checkout")

    out = tmp_path / "baseline.json"
    rc = main(["baseline", str(fixture), "--output", str(out)])
    assert rc == 0, f"`snagline baseline` exited {rc}"
    real = capsys.readouterr().out

    block = _panel_block(SITE_INDEX.read_text(encoding="utf-8"), "BASELINE_LINES")
    visible = re.sub(r"<[^>]+>", "", block)
    rows = re.findall(r"\|\s*(\S+: n=\d+ [^']+)", visible)
    assert rows, "could not parse any per-tool rows from the baseline panel"
    for row in rows:
        assert row in real, (
            f"the baseline panel prints {row!r}, which the real command never "
            f"emits on this fixture:\n{real}"
        )


def test_baseline_panel_advertises_only_real_subcommands():
    """The panel used to send the reader to ``snagline load_baseline``, which is
    not a subcommand. Any ``snagline <word>`` the panel names must be real.
    """
    if not SITE_INDEX.is_file():
        pytest.skip(f"{SITE_INDEX} not present in this checkout")

    block = _panel_block(SITE_INDEX.read_text(encoding="utf-8"), "BASELINE_LINES")
    choices = re.search(r"\{([^}]*)\}", _build_parser().format_help())
    assert choices, "could not read the parser's subcommands"
    real = set(choices.group(1).split(","))
    for word in re.findall(r"snagline ([a-z_]+)", block):
        assert word in real, (
            f"the baseline panel sends the reader to `snagline {word}`, which "
            f"is not one of the real subcommands {sorted(real)}"
        )
