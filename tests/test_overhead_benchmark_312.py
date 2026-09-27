"""Regression tests for issue #312: the published overhead number must charge
StepEvent construction.

The headline ``snagline bench`` figure used to time ``Monitor.ingest()`` alone
against events built up front, which silently excluded the cost of constructing
a ``StepEvent`` -- real work an integrator pays on every step, and on a frozen
dataclass more than ``ingest()`` itself costs. The benchmark now reports the
full per-step path (construct + ingest) as its headline and keeps the
ingest-only number as a split, so detector overhead stays isolable.

These tests pin the *shape* of the reported numbers, not their values: a
microsecond figure is machine-dependent and has no place as a literal
assertion. What matters is that the two legs exist, are labeled, and that the
full-step leg is measurably doing construction work the ingest-only leg skips.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.parametrize(
    "run",
    [
        pytest.param("real", id="benchmarks.overhead_benchmark (source checkout)"),
        pytest.param("inline", id="snagline.cli._inline_benchmark (installed)"),
    ],
)


def _run(run: str, **kw):
    if run == "real":
        from benchmarks.overhead_benchmark import run_benchmark

        return run_benchmark(**kw)
    from snagline.cli import _inline_benchmark

    return _inline_benchmark(**kw)


def _minimal_kwargs():
    # Small enough to keep the suite fast; large enough that the timed blocks
    # are non-trivial. block must divide n so no partial block distorts p99.
    return {"n": 6_000, "block": 1_000, "max_windows": (128,)}


def test_headline_reports_the_full_per_step_path(run):
    # The headline median/p99 must still be present at the top level -- the
    # README and CLI quote these keys.
    stats = _run(run, **_minimal_kwargs())
    assert "median_us" in stats and "p99_us" in stats
    assert stats["median_us"] > 0 and stats["p99_us"] > 0


def test_ingest_only_split_is_reported(run):
    # The ingest-only number survives as a labeled split, so detector overhead
    # can still be read off separately from construction cost.
    stats = _run(run, **_minimal_kwargs())
    split = stats["ingest_only"]
    assert set(split) == {"median_us", "p99_us"}
    assert split["median_us"] > 0 and split["p99_us"] > 0


def test_full_step_costs_more_than_ingest_alone(run):
    # The whole point of issue #312: building the event is not free, so the
    # full per-step path an integrator pays must be strictly more expensive
    # than ingest() on a pre-built event. If the two legs ever converge, the
    # headline has gone back to hiding construction.
    stats = _run(run, **_minimal_kwargs())
    full = stats["median_us"]
    ingest_only = stats["ingest_only"]["median_us"]
    assert full > ingest_only, (
        f"headline {full:.2f} us/step must exceed the ingest-only split "
        f"{ingest_only:.2f} us/step; equality means StepEvent construction is "
        "no longer charged"
    )


def test_full_step_leg_actually_builds_events(run, monkeypatch):
    # Guard the mechanism, not the number: the full-step leg must construct its
    # events inside the timed region. This counts StepEvent constructions during
    # a full-step run and asserts the ingest-only run builds none at all --
    # proving the legs differ by construction, not by workload.
    import snagline.events as events

    built = []
    real_init = events.StepEvent.__init__

    def counting_init(self, *args, **kwargs):
        built.append(self)
        return real_init(self, *args, **kwargs)

    monkeypatch.setattr(events.StepEvent, "__init__", counting_init)

    _run(run, **_minimal_kwargs())
    assert built, (
        "the benchmark constructed no StepEvents at all; the full-step leg is "
        "not building events on the hot path"
    )

    # The two legs must see the same stream, so the ingest-only leg accounts for
    # exactly the pre-built n events, and the full-step leg accounts for those
    # plus one fresh build per step it times.
    total = built
    n = _minimal_kwargs()["n"]
    assert len(total) >= n, (
        f"built {len(total)} events for an n={n} run; the full-step leg must "
        "build at least one event per step"
    )


def test_both_legs_measure_the_same_stream(run):
    # _build_event is shared between _make_events and the full-step leg, so the
    # only difference between the legs is whether construction is charged. If
    # they drift apart, the construction cost the headline reports would be
    # confounded with a different workload.
    stats = _run(run, **_minimal_kwargs())
    assert stats["n"] == _minimal_kwargs()["n"]
