"""Mechanical guards for the README's metrics documentation (issue #616).

The README documents ``Monitor.metrics()``'s keys and ``GET /metrics``'s
Prometheus series. Both surfaces are code -- ``Monitor.metrics()`` builds its
dict in ``monitor.py``, and the exposition body is rendered by
``SidecarMetricsCollector.render_prometheus`` -- so prose quoting them can
drift the way any count in prose drifts (the same reason issue #310 bans
hardcoded test counts). These tests parse the prose back out of the README
and compare it against the live surfaces, so a series or key added to the
code either gets documented here or fails the suite.
"""

from __future__ import annotations

import re
from pathlib import Path

from snagline import Monitor
from snagline.server.http_server import SidecarMetricsCollector

README = Path(__file__).resolve().parent.parent / "README.md"

# A family rendered as `snagline_ingest_seconds_count` / `_sum` is one
# series family in the exposition; the README cell names both samples.
_SAMPLE_SUFFIXES = ("_count", "_sum")


def _readme_section(heading: str, end: str) -> str:
    text = README.read_text(encoding="utf-8")
    start = text.index(heading)
    return text[start : text.index(end, start)]


def _family(name: str) -> str:
    """Reduce a sample-line name (`..._count`/`..._sum`) to its family."""
    for suffix in _SAMPLE_SUFFIXES:
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def _readme_series_families() -> set[str]:
    """Series families named in the README's GET /metrics table."""
    section = _readme_section("### HTTP sidecar", "### Claude Code integration")
    # A cell like `snagline_risks_total{trigger,severity}` carries the label
    # block inside the backticks; the optional group keeps that row matching
    # while the capture stays the bare family name.
    names = set(re.findall(r"`(snagline_[a-z0-9_]+)(?:\{[^`}]*\})?`", section))
    families = {_family(name) for name in names}
    # `snagline_risks_total{trigger,severity}`'s label block is not part of
    # the family name; the table renders it inside one backticked run, so
    # strip it back to the name the exposition uses.
    return {f.split("{")[0] for f in families}


def _rendered_series_families() -> set[str]:
    """Family names the sidecar actually exposes, from a live render."""
    body = SidecarMetricsCollector().render_prometheus(Monitor.default().metrics())
    return set(re.findall(r"^# HELP (\S+)", body, re.M))


def test_readme_metrics_table_matches_the_rendered_exposition() -> None:
    """Every documented series exists in the render, and vice versa.

    A series added to the code without a README row is an undocumented
    surface (the original gap); a README row for a series the render no
    longer emits is a lie an operator would build a dashboard on.
    """
    rendered = _rendered_series_families()
    documented = _readme_series_families()
    assert documented == rendered, (
        "README GET /metrics table drifted from the rendered exposition:\n"
        f"  documented but not rendered: {sorted(documented - rendered)}\n"
        f"  rendered but not documented: {sorted(rendered - documented)}"
    )


def test_readme_names_every_monitor_metrics_key() -> None:
    """The `Monitor.metrics()` row names every key the dict returns.

    The row exists to make all the counters inspectable; a key added to
    ``metrics()`` that the row never names is the original issue in miniature.
    """
    # Scope to the table under "### Python API": a prose mention in the
    # sidecar section also backticks the name, and it is the row that must
    # name the keys.
    table = _readme_section("### Python API", "### CLI")
    row = table[table.index("`Monitor.metrics()`") :]
    row = row[: row.index("|\n")]
    missing = sorted(
        set(Monitor.default().metrics()) - set(re.findall(r"`(\w+)`", row))
    )
    assert not missing, (
        f"Monitor.metrics() keys missing from README's Monitor.metrics() row: {missing}"
    )
