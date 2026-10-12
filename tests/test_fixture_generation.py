"""Determinism and coverage tests for the fixture corpus generator (issue #82).

The corpus is the ground truth every accuracy claim rests on, so this pins the
three properties that make the numbers reproducible:

* regeneration is byte-identical (no wall clock, pid, global RNG, or hash
  randomization leaks in),
* every episode's seed is derived from its id and persisted into the JSONL
  record, so any fixture can be reconstructed from its seed alone,
* the healthy controls cover the eight difficult-but-legitimate behaviour
  classes the corpus is supposed to stress (stage 3).
"""

from __future__ import annotations

import importlib.util
import json
import random
import subprocess
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
_GENERATOR_PATH = _REPO_ROOT / "benchmarks" / "fixtures" / "generate_fixtures.py"
_FIXTURES = _REPO_ROOT / "benchmarks" / "fixtures"


def _load_generator():
    spec = importlib.util.spec_from_file_location("generate_fixtures", _GENERATOR_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


gen = _load_generator()

# Measured distinct structural shapes per trigger over the committed corpus.
# Pinning the counts (rather than only a floor) makes a change in the corpus's
# variation visible as a test failure instead of silently passing.
PINNED_DISTINCT_SHAPES: dict[str, int] = {
    "budget_breach": 7,
    "cycle": 4,
    "error_cascade": 7,
    "goal_drift": 8,
    "governance_decay": 6,
    "idle_gap": 5,
    "latency_anomaly": 7,
    "loop": 7,
    "meltdown_high": 4,
    "meltdown_low": 6,
    "ml_ensemble": 6,
    "side_effect_duplicate": 6,
    "silent_abort": 7,
    "stagnation": 10,
    "stall": 6,
    "token_runaway": 7,
    "wall_clock_budget": 7,
}


def _read(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _shape_digest(rec: dict) -> str:
    """A structural fingerprint over telemetry, ignoring arguments and indices."""

    import hashlib

    fields = (
        "action_type",
        "tool_name",
        "error",
        "error_type",
        "latency_ms",
        "tokens_in",
        "tokens_out",
        "side_effect",
    )
    body = [{k: e.get(k) for k in fields} for e in rec["events"]]
    return hashlib.sha256(json.dumps(body, sort_keys=True).encode("utf-8")).hexdigest()


@pytest.fixture(scope="module")
def regenerated(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Generate the whole corpus into a scratch directory once."""
    out = tmp_path_factory.mktemp("fixtures")
    proc = subprocess.run(
        [sys.executable, str(_GENERATOR_PATH), "--out", str(out)],
        capture_output=True,
        text=True,
        cwd=str(_REPO_ROOT),
        timeout=600,
    )
    assert proc.returncode == 0, proc.stderr
    return out


# --------------------------------------------------------------------------
# Determinism
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "filename",
    ["labeled_episodes.jsonl", "healthy_controls.jsonl", "goal_drift_baseline.json"],
)
def test_regeneration_is_byte_identical_to_the_committed_corpus(
    regenerated: Path, filename: str
) -> None:
    """Two independent runs reproduce the committed files byte for byte."""
    got = (regenerated / filename).read_bytes()
    want = (_FIXTURES / filename).read_bytes()
    assert got == want, filename


def test_regeneration_is_byte_identical_to_a_second_run(
    tmp_path: Path,
) -> None:
    """A third run into a fresh directory is also identical (no order or
    environment leak between invocations)."""
    out = tmp_path / "second"
    proc = subprocess.run(
        [sys.executable, str(_GENERATOR_PATH), "--out", str(out)],
        capture_output=True,
        text=True,
        cwd=str(_REPO_ROOT),
        timeout=600,
    )
    assert proc.returncode == 0, proc.stderr
    for name in ("labeled_episodes.jsonl", "healthy_controls.jsonl"):
        assert (out / name).read_bytes() == (_FIXTURES / name).read_bytes(), name


def test_generator_creates_a_missing_output_directory(tmp_path: Path) -> None:
    """--out may point at a directory that does not exist yet."""
    out = tmp_path / "nested" / "fresh"
    proc = subprocess.run(
        [sys.executable, str(_GENERATOR_PATH), "--out", str(out)],
        capture_output=True,
        text=True,
        cwd=str(_REPO_ROOT),
        timeout=600,
    )
    assert proc.returncode == 0, proc.stderr
    assert (out / "labeled_episodes.jsonl").is_file()


def test_seeds_derive_from_the_episode_id_only() -> None:
    """The seed is a pure function of the episode id: same id, same stream,
    forever, with no dependence on process state."""
    assert gen._episode_seed("loop-04") == "loop-04/seed"
    a = random.Random(gen._episode_seed("stagnation-07"))
    b = random.Random(gen._episode_seed("stagnation-07"))
    assert [a.random() for _ in range(50)] == [b.random() for _ in range(50)]
    # A different episode yields a different stream.
    c = random.Random(gen._episode_seed("stagnation-08"))
    assert [a.random() for _ in range(5)] != [c.random() for _ in range(5)]


def test_builder_output_is_stable_across_invocations() -> None:
    """Each builder is a pure function of (episode id, rng): calling it twice
    with the same seeded rng reproduces the identical events and onset.

    Episode ids carry a numeric variant suffix because two builders
    (goal_drift, ml_ensemble) dispatch on it."""
    for name, builder in gen.LABELED_BUILDERS.items():
        ep = f"{name}-04"
        seed = gen._episode_seed(ep)
        events1, onset1 = builder(ep, random.Random(seed))
        events2, onset2 = builder(ep, random.Random(seed))
        assert events1 == events2, name
        assert onset1 == onset2, name


def test_variants_differ_from_the_legacy_shape() -> None:
    """The stage-3 variation axes are real: variant 04+ episodes must not be
    byte-copies of the legacy 00 shape for the same trigger."""
    for name in gen.LABELED_BUILDERS:
        variation = gen.LABELED_VARIATION_BUILDERS[name]
        ep = f"{name}-04"
        seed = gen._episode_seed(ep)
        legacy, _ = gen.LABELED_BUILDERS[name](ep, random.Random(seed))
        varied, _ = variation(ep, random.Random(seed), 4)
        assert json.dumps(legacy, sort_keys=True) != json.dumps(
            varied, sort_keys=True
        ), name


def test_variants_are_individually_deterministic() -> None:
    """Same id and variant reproduce the same episode. 14 of the 17 variation
    builders never read their ``v`` parameter and derive every axis from the
    episode-seeded rng alone (test_variation_axes pins which three do read
    it), so variation must be verified across distinct committed episode ids,
    not across variant numbers."""
    for name, variation in gen.LABELED_VARIATION_BUILDERS.items():
        ep = f"{name}-04"
        seed = gen._episode_seed(ep)
        a = variation(ep, random.Random(seed), 4)
        b = variation(ep, random.Random(seed), 4)
        assert a == b, name


def test_committed_variants_are_pairwise_distinct() -> None:
    """The 10 committed episodes per trigger are genuinely different traces,
    not one trace duplicated.

    The digest deliberately drops arguments (``action_signature``) and step
    indices, so it measures structural shape: which tool, which action type,
    whether it errored, how long it took, how many tokens it moved. Episodes
    that differ only in argument text or in the number of repeated steps can
    share a shape, so 10/10 distinctness is too strong to demand -- but no
    single shape may describe a majority of a trigger's episodes, and every
    trigger must present at least four distinct shapes.
    """
    labeled = _read(_FIXTURES / "labeled_episodes.jsonl")
    by_trigger: dict[str, list[dict]] = {}
    for rec in labeled:
        by_trigger.setdefault(rec["label"]["trigger"], []).append(rec)
    assert set(by_trigger) == set(gen.LABELED_BUILDERS)
    for trig, recs in sorted(by_trigger.items()):
        assert len(recs) == gen.N_LABELED_VARIANTS, trig
        counts: dict[str, int] = {}
        for rec in recs:
            counts[_shape_digest(rec)] = counts.get(_shape_digest(rec), 0) + 1
        assert len(counts) == PINNED_DISTINCT_SHAPES[trig], (
            f"{trig}: distinct-shape count moved ({len(counts)}); the corpus's "
            f"variation changed underneath this gate"
        )
        # And no single shape may describe a majority of a trigger's episodes.
        most_common = max(counts.values())
        assert most_common <= len(recs) // 2, (
            f"{trig}: one shape covers {most_common} of {len(recs)} episodes"
        )
        lengths = {len(r["events"]) for r in recs}
        onsets = {r["label"]["onset_step"] for r in recs}
        assert len(lengths) > 1 or len(onsets) > 1, trig


def test_variation_axes_documented_and_used() -> None:
    """Pin which variation builders read their variant number.

    ``v`` is the only structural handle on which axis a variant targets.
    Recording the set keeps the corpus's axis list honest: 14 of 17 builders
    vary purely through the episode-seeded rng.
    """
    import ast

    tree = ast.parse(_GENERATOR_PATH.read_text(encoding="utf-8"))
    uses_v = set()
    for node in tree.body:
        if (
            isinstance(node, ast.FunctionDef)
            and node.name.startswith("vary_")
            and any(a.arg == "v" for a in node.args.args)
        ):
            if any(isinstance(n, ast.Name) and n.id == "v" for n in ast.walk(node)):
                uses_v.add(node.name)
    assert uses_v == {
        "vary_governance_decay",
        "vary_goal_drift",
        "vary_ml_ensemble",
    }


def test_each_fixture_reproduces_from_its_persisted_seed() -> None:
    """Rebuilding an episode from the seed in its record reproduces the
    committed events exactly -- the seed field is load-bearing, not
    decorative."""
    labeled = _read(_FIXTURES / "labeled_episodes.jsonl")
    seen = 0
    for rec in labeled:
        name = rec["label"]["trigger"]
        ep = rec["episode_id"]
        variant = int(ep.rsplit("-", 1)[1])
        if variant < gen.LEGACY_VARIANT_COUNT:
            events, _ = gen.LABELED_BUILDERS[name](ep, random.Random(rec["seed"]))
        else:
            events, _ = gen.LABELED_VARIATION_BUILDERS[name](
                ep, random.Random(rec["seed"]), variant
            )
        assert events == rec["events"], ep
        seen += 1
    assert seen > 0


# --------------------------------------------------------------------------
# Healthy-control coverage of the eight difficult-legitimate classes
# --------------------------------------------------------------------------


def _tools(rec: dict) -> set[str]:
    return {e["tool_name"] for e in rec["events"] if e.get("tool_name")}


def test_healthy_controls_cover_the_eight_required_classes() -> None:
    """Every class the corpus set out to stress must be present by name."""
    by_id = {
        rec["episode_id"]: rec for rec in _read(_FIXTURES / "healthy_controls.jsonl")
    }
    names = {
        eid: {
            "errors": sum(1 for e in r["events"] if e.get("error")),
            "tools": len(_tools(r)),
            "sigs": len({e["action_signature"] for e in r["events"]}),
            "steps": len(r["events"]),
            "reasoning": sum(
                1 for e in r["events"] if e.get("action_type") == "reasoning"
            ),
            "timed": sum(1 for e in r["events"] if e.get("latency_ms") is not None),
            "tokens": sum(
                (e.get("tokens_in") or 0) + (e.get("tokens_out") or 0)
                for e in r["events"]
            ),
        }
        for eid, r in by_id.items()
    }

    # (a) Legitimate retries: fail once, retry, succeed.
    retry = [eid for eid, s in names.items() if s["errors"] == 2 and s["steps"] <= 20]
    # (b) Bursty latency without a true anomaly.
    bursty = [eid for eid, s in names.items() if s["timed"] >= 7 and s["errors"] == 0]
    # (c) A long-running task, many steps, no stagnation.
    long_task = [eid for eid, s in names.items() if s["steps"] >= 55]
    # (d) A single recovered error.
    one_err = [eid for eid, s in names.items() if s["errors"] == 1]
    # (e) Repeated legitimate tool calls: many steps on one or two tools.
    repeated = [
        eid
        for eid, s in names.items()
        if s["steps"] >= 18 and s["tools"] <= 3 and s["errors"] == 0
    ]
    # (f) High but legitimate token usage.
    high_tokens = [eid for eid, s in names.items() if s["tokens"] >= 20_000]
    # (g) A long planning phase with reasoning steps.
    planning = [eid for eid, s in names.items() if s["reasoning"] >= 10]
    # (h) Similar actions with legitimately changing inputs: many distinct
    # signatures across few tools.
    similar = [
        eid
        for eid, s in names.items()
        if s["sigs"] >= 14 and s["steps"] <= 40 and s["errors"] == 0
    ]

    for label, found in (
        ("legitimate retries", retry),
        ("bursty latency", bursty),
        ("long-running task", long_task),
        ("one-off recovered error", one_err),
        ("repeated legitimate calls", repeated),
        ("high token usage", high_tokens),
        ("long planning phase", planning),
        ("similar actions with changing inputs", similar),
    ):
        assert found, f"no healthy control exercises {label}"
