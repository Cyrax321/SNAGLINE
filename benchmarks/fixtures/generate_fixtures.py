"""Generate the labeled detection-accuracy fixture corpus (issue #82).

Writes two JSONL files next to this script:

* ``labeled_episodes.jsonl``  -- four episodes per shipped trigger, each line's
  ``label.trigger`` naming the failure shape injected into that trajectory and
  ``label.onset_step`` naming the zero-based index of the first event that
  intentionally introduces it (issue #82 stage 2: detection latency). Since
  issue #118 the ``goal_drift`` and ``ml_ensemble`` episodes additionally
  carry a ``config`` envelope field naming the harness config variant they
  replay under (see benchmarks/detection_accuracy.py).
* ``healthy_controls.jsonl``  -- ``label: null`` trajectories, including
  deliberately tricky near-threshold cases (two repeats then recovery,
  warmup-window latency jitter, tool-choice entropy near 2.3 bits). The last
  few controls replay under the ``ml_ensemble`` / ``goal_drift`` variants.
  Nothing is injected, so they carry no onset step.
* ``goal_drift_baseline.json`` -- the committed healthy BaselineProfile the
  ``goal_drift`` variant replays against (issue #118).

Onset semantics: ``onset_step`` is the FIRST STEP AT WHICH THE FAILURE
CONDITION IS INTRODUCED, not the first step at which a detector could notice
it. For repeat/threshold-driven shapes it is the first of the injected
repeats, even though the detector only fires once the threshold is crossed;
for cumulative shapes (meltdown, stagnation, cycle, wall_clock_budget,
goal-drift latency blowout) the whole trajectory is the failure and the onset
is step 0; for silent_abort it is the deliberately-unclosed final step. Latency
against this value therefore includes the detector's own threshold delay,
which is exactly what stage 2 set out to measure.

Every episode replays through ``Monitor.default()`` plus the opt-in flags the
labels require (token_runaway, meltdown, silent_abort, and a fixed episode
token budget) so that each labeled episode fires EXACTLY its labeled trigger
and every healthy control fires nothing. The numeric patterns below are
hand-derived against the detector implementations (window arithmetic, CUSUM
sigma floors, entropy bands, budget-envelope ordering); see the comments on
each builder for the arithmetic. Generation is fully deterministic: fixed
constants plus seeded ``random.Random`` instances, no wall-clock time, no
entropy from the environment. Re-running this script reproduces the committed
JSONL byte-for-byte.

Privacy: fixtures carry structure only -- SHA-256 action signatures computed
over synthetic identifier strings ("q=1", "page_7"), boolean error flags,
latency/token counts, and synthetic tool names. No prompt or response text
exists anywhere in the corpus, matching project.md section 1.4.

Usage::

    python benchmarks/fixtures/generate_fixtures.py [--out DIR]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import random
import subprocess
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from snagline.baseline import BaselineProfile, save_baseline
from snagline.events import StepEvent, make_signature

# Generator version, persisted in generation_manifest.json so a replay can be
# attributed to the exact code that produced it (stage 3, issue #82).
GENERATOR_VERSION = "3.0"

# Corpus size. Stage 1/2 shipped four episodes per trigger; stage 3 extends to
# ten. Variants 0..3 stay on the ORIGINAL builders so the committed Stage 1/2
# episode bytes are reproduced exactly (byte-identity is asserted by the
# generator twice-run check in tests/test_eval_corpus.py); variants
# LEGACY_VARIANT_COUNT..N_LABELED_VARIANTS-1 use the *_variation_* builders
# below, which vary onset position, episode length, jitter, retry counts, error
# placement, tool ordering, latency and token volumes per episode.
N_LABELED_VARIANTS = 10
LEGACY_VARIANT_COUNT = 4

# Fixed epoch so committed timestamps never change between regenerations.
BASE_TS = 1_700_000_000.0
STEP_GAP_S = 1.5

# Five-tool pool used where healthy near-uniform entropy is wanted: a
# strict cycle over five identities yields log2(5) ~= 2.32 bits per full
# 20-call meltdown window, comfortably inside the silent band (0.4, 3.4).
# The eight-tool round-robin used in the regression corpus gives ~=2.97
# bits, also below the 3.4 high threshold (~10.5 tools uniform).
TOOL_POOL = ("search_web", "read_file", "run_sql", "call_api", "summarize")

# Twelve-tool pool for meltdown_high churn: any full 20-window of a strict
# 12-cycle holds eight identities twice and four once, giving H ~= 3.52
# bits, safely above the 3.4-bit thrash threshold.
CHURN_POOL = (
    "tool_a",
    "tool_b",
    "tool_c",
    "tool_d",
    "tool_e",
    "tool_f",
    "tool_g",
    "tool_h",
    "tool_i",
    "tool_j",
    "tool_k",
    "tool_l",
)

# Episode token budget the harness configures; see detection_accuracy.py.
# Warn fires at >= 40,000 cumulative tokens, breach at >= 50,000.
EPISODE_TOKEN_BUDGET = 50_000

# --- Goal-drift corpus (issue #118) ---------------------------------------
# Healthy reference the goal_drift variant replays against: three tools at
# exactly constant latencies and zero errors, so every spread the detector
# computes collapses onto its documented relative floor (5% of the mean).
# Constant values keep every hand arithmetic in comments/test bodies exact.
GOAL_DRIFT_BASELINE_TOOLS: tuple[tuple[str, float], ...] = (
    ("search_web", 100.0),
    ("read_file", 50.0),
    ("run_sql", 80.0),
)
GOAL_DRIFT_BASELINE_CALLS_PER_TOOL = 40

# Near-k healthy control: live means one k multiple above the reference but
# still inside goal_drift_latency_k = 3 sigmas.
GOAL_DRIFT_NEAR_K_LATENCIES: dict[str, float] = {
    "search_web": 110.0,
    "read_file": 55.0,
    "run_sql": 88.0,
}


def _tc(
    ep: str,
    i: int,
    tool: str,
    args: str,
    *,
    error: bool = False,
    error_type: str | None = None,
    latency_ms: float | None = None,
    tokens_in: int | None = None,
    tokens_out: int | None = None,
    action_type: str = "tool_call",
    side_effect: bool = False,
    metadata: dict | None = None,
    timestamp: float | None = None,
) -> dict:
    """Build one StepEvent-compatible dict (structure only, no content)."""
    d: dict = {
        "step_id": f"{ep}-s{i}",
        "episode_id": ep,
        "timestamp": BASE_TS + i * STEP_GAP_S if timestamp is None else timestamp,
        "action_type": action_type,
        "action_signature": make_signature(
            action_type, tool if action_type == "tool_call" else None, args
        ),
        "tool_name": tool if action_type == "tool_call" else None,
        "latency_ms": latency_ms,
        "error": error,
        "error_type": error_type,
        "tokens_in": tokens_in,
        "tokens_out": tokens_out,
    }
    if side_effect:
        d["side_effect"] = True
    if metadata is not None:
        d["metadata"] = metadata
    return d


class _Builder:
    """Per-episode event accumulator with deterministic arg uniquing.

    ``onset`` is the failure-injection onset (stage 2): the zero-based index of
    the FIRST event that intentionally introduces the labeled failure
    condition. Builders call :meth:`mark_onset` immediately before adding that
    event, so the value is derived from the construction sequence rather than
    hand-written per episode. Builders whose whole trajectory IS the failure
    mark step 0 explicitly, so an unmarked labeled builder fails loudly in
    main() instead of silently reporting onset ``None``.
    """

    def __init__(self, ep: str, rng: random.Random) -> None:
        self.ep = ep
        self.rng = rng
        self.events: list[dict] = []
        self._n = 0
        self.onset: int | None = None

    def add(self, tool: str, args: str, **kw) -> None:
        self.events.append(_tc(self.ep, self._n, tool, args, **kw))
        self._n += 1

    def mark_onset(self) -> None:
        """Record that the NEXT event added is the failure-injection onset."""
        self.onset = self._n

    def result(self) -> tuple[list[dict], int | None]:
        """Return ``(events, onset)``.

        Every builder -- labeled and healthy -- returns this pair so the onset
        is derived from the construction sequence rather than maintained in a
        separate table. Healthy builders never call :meth:`mark_onset`, so
        their onset is ``None``.
        """
        return self.events, self.onset

    def finish_with_output(self) -> list[dict]:
        """Close the episode on a message step so silent_abort stays silent."""
        self.add("__output__", f"final-answer-{self.ep}", action_type="message")
        return self.events


def _cycle_tool(i: int) -> str:
    return TOOL_POOL[i % len(TOOL_POOL)]


# --------------------------------------------------------------------------
# Labeled failure shapes
# --------------------------------------------------------------------------


def build_loop(ep: str, rng: random.Random) -> tuple[list[dict], int | None]:
    """Three identical retry calls inside the 12-signature loop window.

    Total tool calls stays below the 20-slot meltdown window so the entropy
    statistic never evaluates; args are otherwise unique so only the deliberate
    triple matches the repeat threshold.
    """
    b = _Builder(ep, rng)
    for i in range(4):
        b.add(_cycle_tool(i), f"q={rng.randrange(10**6)}")
    # Onset: the first of the three identical retries (step 4). Three copies
    # must exist before the repeat threshold is crossed, but the failure is
    # introduced by the first deliberate repeat.
    b.mark_onset()
    # The loop: same tool, same args, hence one identical signature x3.
    for _ in range(3):
        b.add("fetch_record", "id=42")
    b.add(_cycle_tool(1), f"q={rng.randrange(10**6)}")
    b.finish_with_output()
    return b.result()


def build_error_cascade(ep: str, rng: random.Random) -> tuple[list[dict], int | None]:
    """Three consecutive erroring tool calls trip the consecutive rule (3)."""
    b = _Builder(ep, rng)
    for i in range(3):
        b.add(_cycle_tool(i), f"req={rng.randrange(10**6)}")
    # Onset: the first of the three erroring calls (step 3).
    b.mark_onset()
    for i in range(3):
        b.add("deploy_service", f"svc-{i}", error=True, error_type="TimeoutError")
    b.add(_cycle_tool(4), f"req={rng.randrange(10**6)}")
    b.finish_with_output()
    return b.result()


def build_latency_anomaly(ep: str, rng: random.Random) -> tuple[list[dict], int | None]:
    """Stable 5-call baseline freezes mu0~=400ms/sigma0=20ms, then spikes.

    Baseline [400, 395, 405, 398, 402] has std ~= 3.5, so the relative floor
    (5% of 400 = 20ms) sets sigma0. A 2500ms spike yields z = (2500-400)/20 =
    105, cusum ~= 104.5 >> h=5: instant alarm on the first post-warmup step.
    """
    b = _Builder(ep, rng)
    for ms in (400.0, 395.0, 405.0, 398.0, 402.0):
        b.add("search_web", f"q={rng.randrange(10**6)}", latency_ms=ms)
    # Onset: the first spike (step 5), immediately after the frozen baseline.
    b.mark_onset()
    for ms in (2500.0, 2400.0):
        b.add("search_web", f"q={rng.randrange(10**6)}", latency_ms=ms)
    b.finish_with_output()
    return b.result()


def build_token_runaway(ep: str, rng: random.Random) -> tuple[list[dict], int | None]:
    """Sustained-burn shift after the 20-sample warmup; envelope never fires.

    Steps 1..20 carry 900 tokens each (freeze at n=20: mu0=900, std=0, floor
    gives sigma0=max(1.0, 0.05*900)=45). Step 21 carries 2600 tokens:
    z=(2600-900)/45~=37.8, cusum~=37.3 >> h=5, alarm. Cumulative peaks at
    18,000 + 6*2,600 = 33,600 < 40,000, below the warn fraction, so the
    budget envelope stays out of the way and only token_runaway fires.
    Tools cycle through all five identities so meltdown entropy stays ~2.32
    bits (in band); latency fields are absent so the CUSUM latency detector
    is inert.
    """
    b = _Builder(ep, rng)
    for i in range(20):
        b.add(
            _cycle_tool(i),
            f"item={rng.randrange(10**6)}",
            tokens_in=700,
            tokens_out=200,
        )
    # Onset: the first above-baseline burn (step 20), right after warmup.
    b.mark_onset()
    for i in range(6):
        b.add(
            _cycle_tool(20 + i),
            f"item={rng.randrange(10**6)}",
            tokens_in=2300,
            tokens_out=300,
        )
    b.finish_with_output()
    return b.result()


def build_budget_breach(ep: str, rng: random.Random) -> tuple[list[dict], int | None]:
    """Jump straight across the warn fraction so only budget_breach fires.

    Thirty baseline steps at 1,000 tokens accumulate 30,000 (< 40,000 warn).
    One 21,000-token step pushes the total to 51,000 >= 50,000; the envelope
    checks breach before warn, so the warning branch is never taken. The two
    follow-up steps return to baseline volume (z = -0.5 each against the
    frozen mu0=1000/sigma0=50 CUSUM) and cannot alarm. Five-tool cycling
    keeps meltdown in band throughout.
    """
    b = _Builder(ep, rng)
    for i in range(30):
        b.add(
            _cycle_tool(i), f"row={rng.randrange(10**6)}", tokens_in=800, tokens_out=200
        )
    # Onset: the single oversized step (step 30) that crosses the budget.
    b.mark_onset()
    b.add(_cycle_tool(30), "bulk_export=huge", tokens_in=21_000, tokens_out=0)
    for i in range(2):
        b.add(
            _cycle_tool(31 + i),
            f"row={rng.randrange(10**6)}",
            tokens_in=800,
            tokens_out=200,
        )
    b.finish_with_output()
    return b.result()


def build_meltdown_low(ep: str, rng: random.Random) -> tuple[list[dict], int | None]:
    """Rote collapse: one tool identity, unique args, entropy -> 0 bits.

    The meltdown window reasons about tool_name, the loop detector about
    action_signature; unique args keep signatures distinct so only the
    collapse fires once the 20-call window fills (H = 0 < 0.4 bits).
    """
    b = _Builder(ep, rng)
    # Onset: step 0 -- the rote collapse is the entire trajectory, so the
    # first event is where the failure condition begins.
    b.mark_onset()
    for i in range(22):
        b.add("scrape_page", f"page_{i}")
    b.finish_with_output()
    return b.result()


def build_meltdown_high(ep: str, rng: random.Random) -> tuple[list[dict], int | None]:
    """Churn: strict 12-tool cycle puts every full 20-window at ~3.52 bits."""
    b = _Builder(ep, rng)
    # Onset: step 0 -- the churn shape is the whole trajectory.
    b.mark_onset()
    for i in range(24):
        b.add(CHURN_POOL[i % len(CHURN_POOL)], f"call-{i}-{rng.randrange(10**6)}")
    b.finish_with_output()
    return b.result()


def build_silent_abort(ep: str, rng: random.Random) -> tuple[list[dict], int | None]:
    """Clean run whose LAST step is an error-free bare tool call.

    finalize() at end_episode sees a non-output final action and fires; during
    the run nothing else can fire (fewer than 20 tool calls, no errors, no
    latency/token telemetry).
    """
    b = _Builder(ep, rng)
    for i in range(5):
        b.add(_cycle_tool(i), f"k={rng.randrange(10**6)}")
    # Onset: the final non-output step (step 5). The injected failure is the
    # ABSENCE of a closing output step after it, so this is the event from
    # which the trajectory diverges from a healthy run.
    b.mark_onset()
    b.add("write_report", "draft=v1")  # ends mid-work, never emits output
    return b.result()  # deliberately NOT closed with a message step


# --------------------------------------------------------------------------
# New shipped triggers for #181 (stagnation, side_effect, governance, cycle,
# stall, idle_gap, wall_clock_budget)
# --------------------------------------------------------------------------


def build_stagnation(ep: str, rng: random.Random) -> tuple[list[dict], int | None]:
    """Novelty collapse: 6 signatures reused randomly for 100 steps, 0.5s gaps.

    Window 50, min_novelty 0.05, patience 2: need two consecutive windows with
    <2.5 novel. Six distinct signatures keep loop quiet (each appears ~2 per
    12 window) and meltdown entropy at log2(6)=2.58 inside (0.4, 2.8). Random
    order breaks strict periodicity so cycle does not fire. Gaps 0.5s keep
    total wall 50s <120s so wall_clock_budget stays silent while windows
    complete.
    """
    b = _Builder(ep, rng)
    # Onset: step 0 -- the low-novelty sequence is the whole trajectory.
    b.mark_onset()
    sigs = [(f"st_tool_{i}", f"fixed_{i}") for i in range(6)]
    for i in range(100):
        # Random pick among 6 keeps novelty low but not periodic
        tool, args = sigs[rng.randrange(6)]
        ts = BASE_TS + i * 0.5
        b.events.append(_tc(b.ep, b._n, tool, args, timestamp=ts))
        b._n += 1
    # Final message step after the run, keep wall monotonic
    b.events.append(
        _tc(
            b.ep,
            b._n,
            "__output__",
            f"final-answer-{ep}",
            action_type="message",
            timestamp=BASE_TS + 100 * 0.5 + 0.5,
        )
    )
    b._n += 1
    return b.result()


def build_side_effect_duplicate(
    ep: str, rng: random.Random
) -> tuple[list[dict], int | None]:
    """Second identical side_effect=True fires side_effect_duplicate.

    Two identical (tool_name, signature) with side_effect=True. First is
    tolerated, second exceeds allowed_repeats=1. Keep total tool calls low
    and args distinct elsewhere so loop/meltdown stay quiet.
    """
    b = _Builder(ep, rng)
    for i in range(3):
        b.add(_cycle_tool(i), f"init={rng.randrange(10**6)}")
    # Onset: the first of the duplicated charges (step 3). Only the second
    # identical side-effect crosses allowed_repeats, but the injected failure
    # is the duplicated pair, which starts here.
    b.mark_onset()
    # First charge
    b.add("charge_card", "amt=10", side_effect=True)
    b.add(_cycle_tool(3), f"mid={rng.randrange(10**6)}")
    # Duplicate charge, same signature
    b.add("charge_card", "amt=10", side_effect=True)
    b.add(_cycle_tool(4), f"tail={rng.randrange(10**6)}")
    b.finish_with_output()
    return b.result()


def build_governance_decay(
    ep: str, rng: random.Random
) -> tuple[list[dict], int | None]:
    """Compaction with pinned hashes, 3 grace steps without confirmation.

    Grace 3: compaction at ordinal n, deadline n+3. After 3 steps with pins
    still unconfirmed, fires governance_decay once. Use fixed pin hashes so
    regeneration is deterministic.
    """
    import hashlib

    b = _Builder(ep, rng)
    for i in range(2):
        b.add(_cycle_tool(i), f"pre={rng.randrange(10**6)}")
    # Onset: the compaction step (step 2) that pins constraints the run never
    # confirms. The decay is only DECIDABLE once grace expires, but it is
    # introduced by the pinning.
    b.mark_onset()
    pin_a = hashlib.sha256(b"constraint-a").hexdigest()
    pin_b = hashlib.sha256(b"constraint-b").hexdigest()
    # Compaction pins two constraints
    b.add(
        "__compaction__",
        "compaction",
        action_type="compaction",
        metadata={"pinned": [pin_a, pin_b]},
    )
    # Three grace steps, none confirm the pins
    for i in range(3):
        b.add(_cycle_tool(i), f"grace={rng.randrange(10**6)}")
    b.finish_with_output()
    return b.result()


def build_cycle(ep: str, rng: random.Random) -> tuple[list[dict], int | None]:
    """Period-6 cycle across 24 steps fires LoopDetector cycle but not loop.

    Window 12, min_period 2, max_period 6: period 6 is inside band, and 12
    holds 2 periods. Six distinct signatures cycling keeps each count at 2 per
    12 window (threshold 3), so plain loop stays silent while cycle fires.
    Entropy log2(6)=2.58 inside (0.4, 2.8) so meltdown silent. Total wall
    36s <120s, gap 1.5s, no idle.
    """
    b = _Builder(ep, rng)
    # Onset: step 0 -- the period-6 cycle is the whole trajectory.
    b.mark_onset()
    cyc = [(f"c_tool_{i}", f"c-arg-{i}") for i in range(6)]
    for i in range(24):
        tool, args = cyc[i % 6]
        b.add(tool, args)
    b.finish_with_output()
    return b.result()


def build_stall(ep: str, rng: random.Random) -> tuple[list[dict], int | None]:
    """25 consecutive identical signatures fires LoopDetector stall.

    Stall threshold 25. Use single tool/args repeated. Keep preceding steps
    varied and short to avoid early loop: 2 distinct warmup steps, then 25
    repeats, then output. Meltdown will see single tool for 25 steps and
    fire meltdown_low as well, but stall labeled episode will count stall TP
    and meltdown FP; the gate still sees stall covered. To keep healthy
    controls silent we avoid long stalls there.
    """
    b = _Builder(ep, rng)
    for i in range(2):
        b.add(_cycle_tool(i), f"warm={rng.randrange(10**6)}")
    # Onset: the first of the 25 identical calls (step 2).
    b.mark_onset()
    for _ in range(25):
        b.add("stalled_tool", "repeat-id-99")
    b.finish_with_output()
    return b.result()


def build_idle_gap(ep: str, rng: random.Random) -> tuple[list[dict], int | None]:
    """One gap >=10s between consecutive ingests fires idle_gap.

    Five steps at 1.5s gaps, then a 15s jump, then 3 more at 1.5s. Idle
    fires once per episode at the gap. Wall budget 120s not reached (total
    ~22.5s before the final message). Keep tool entropy inside band and
    signatures unique so loop/meltdown/stagnation stay silent.
    """
    b = _Builder(ep, rng)
    for i in range(5):
        b.add(_cycle_tool(i), f"q={rng.randrange(10**6)}")
    # Onset: the step whose timestamp jumps 15s (step 5) -- the injected gap.
    b.mark_onset()
    # Create a gap by bumping the next event's timestamp by 15s.
    # _Builder uses i*1.5, so we manually adjust the timestamp of the next
    # event after building it.
    b.add(_cycle_tool(5), f"gap={rng.randrange(10**6)}")
    # Adjust last event's timestamp to be 15s after previous
    prev_ts = b.events[-2]["timestamp"]
    b.events[-1]["timestamp"] = prev_ts + 15.0
    # Shift subsequent events to keep monotonic gaps
    gap_base = b.events[-1]["timestamp"]
    for i in range(3):
        b.add(_cycle_tool(6 + i), f"q={rng.randrange(10**6)}")
        b.events[-1]["timestamp"] = gap_base + (i + 1) * STEP_GAP_S
    # Final output message must stay after the gap, at +1.5s
    b.add("__output__", f"final-answer-{ep}", action_type="message")
    b.events[-1]["timestamp"] = gap_base + 4 * STEP_GAP_S
    return b.result()


def build_wall_clock_budget(
    ep: str, rng: random.Random
) -> tuple[list[dict], int | None]:
    """Total wall >=120s fires wall_clock_budget (warning at 96s, breach at 120s).

    85 steps at 1.5s gaps = 126s wall, triggers breach once. No idle gaps.
    Keep 5-tool cycle for meltdown silence, no side_effect, no compaction.
    """
    b = _Builder(ep, rng)
    # Onset: step 0 -- the budget burn accumulates across the whole run, so
    # the clock starts at the first event.
    b.mark_onset()
    for i in range(85):
        b.add(_cycle_tool(i), f"w={rng.randrange(10**6)}")
    b.finish_with_output()
    return b.result()


# --------------------------------------------------------------------------
# Goal-drift failure shapes (issue #118)
# --------------------------------------------------------------------------


def build_goal_drift_baseline() -> BaselineProfile:
    """Fit the committed healthy BaselineProfile from synthetic events.

    GOAL_DRIFT_BASELINE_CALLS_PER_TOOL constant-latency calls per tool with
    zero errors freeze each reference on its exact mean (std 0), so every
    spread the detector later derives comes from its documented floors:
    max(std, 1ms, 5% of mean). Deterministic; no RNG.
    """
    profile = BaselineProfile()
    i = 0
    for tool, ms in GOAL_DRIFT_BASELINE_TOOLS:
        for _ in range(GOAL_DRIFT_BASELINE_CALLS_PER_TOOL):
            profile.add_event(
                StepEvent(
                    step_id=f"baseline-s{i}",
                    episode_id="baseline-fit",
                    timestamp=BASE_TS + i * STEP_GAP_S,
                    action_type="tool_call",
                    action_signature=make_signature("tool_call", tool, f"q={i}"),
                    tool_name=tool,
                    latency_ms=ms,
                )
            )
            i += 1
    return profile


def _goal_drift_variant(ep: str) -> int:
    return int(ep.rsplit("-", 1)[1])


def _build_goal_drift_latency_shape(
    ep: str, rng: random.Random
) -> tuple[list[dict], int | None]:
    """Latency blowout: live per-tool mean doubles the healthy reference.

    Twelve search_web calls at exactly 200ms against a baseline frozen at
    100.0ms (std 0): the detector's floored spread is max(0, 5% of 100) =
    5ms, so once goal_drift_min_samples = 10 is reached the z-score is
    (200 - 100) / 5 = 20 > k = 3 and the contribution is
    min(1, (20 - 3) / 10) = 1.0 >= score threshold 0.5: fires exactly once
    (the emission latches). The latency CUSUM sees constant 200ms samples:
    warmup freezes mu0 = 200 with sigma0 = max(1, 5% of 200) = 10, so every
    post-freeze increment is max(0, 0 - 0.5) = 0 and it stays silent. Unique
    args keep loop signatures distinct, no errors keep cascade quiet, and 12
    tool calls never fill the 20-slot meltdown window.
    """
    b = _Builder(ep, rng)
    # Onset: step 0 -- every call sits above the healthy reference, so the
    # drift begins with the first event.
    b.mark_onset()
    for _ in range(12):
        b.add("search_web", f"q={rng.randrange(10**6)}", latency_ms=200.0)
    b.finish_with_output()
    return b.result()


def _build_goal_drift_unseen_shape(
    ep: str, rng: random.Random
) -> tuple[list[dict], int | None]:
    """Unseen-capability drift: a tool absent from the healthy baseline.

    Nine calls cycle the three baseline tools at their exact healthy
    latencies (100/50/80ms; each collects only 3 CUSUM samples, below the
    5-sample warmup, so the latency detector stays inert), then one call to
    deploy_canary, which never appears in the committed baseline. Absent
    tools contribute a flat 0.6 >= threshold 0.5, and that step is also the
    tenth live sample, so goal_drift fires there and only there. Known tools
    match their references exactly (z = 0); unique args, zero errors.

    An error-rate drift shape was considered and rejected: keeping every
    10-step cascade window at <= 2 errors caps the live error rate near 0.2,
    whose contribution min(1, (0.2 - 0.1) * 2) = 0.2 can never cross 0.5
    alone, so such an episode could not fire cleanly by itself.
    """
    b = _Builder(ep, rng)
    for i in range(9):
        tool, ms = GOAL_DRIFT_BASELINE_TOOLS[i % len(GOAL_DRIFT_BASELINE_TOOLS)]
        b.add(tool, f"req={rng.randrange(10**6)}", latency_ms=ms)
    # Onset: the first call to a tool absent from the healthy baseline (step
    # 9); the preceding nine calls match the reference exactly.
    b.mark_onset()
    b.add("deploy_canary", f"env={rng.randrange(10**6)}", latency_ms=70.0)
    b.finish_with_output()
    return b.result()


def build_goal_drift(ep: str, rng: random.Random) -> tuple[list[dict], int | None]:
    """Dispatch labeled goal_drift episodes across both deviation shapes."""
    shape = _goal_drift_variant(ep)
    if shape in (0, 1):
        return _build_goal_drift_latency_shape(ep, rng)
    return _build_goal_drift_unseen_shape(ep, rng)


# --------------------------------------------------------------------------
# ML-ensemble failure shapes (issue #118)
# --------------------------------------------------------------------------


def _ml_loop_signal(b: _Builder) -> None:
    """One signature triple: LoopDetector emits count/threshold*0.5 = 0.5."""
    b.add("fetch_record", "id=7")
    b.add("fetch_record", "id=7")
    b.add("fetch_record", "id=7")


def _ml_latency_spike(b: _Builder) -> None:
    """Post-warmup spike: cusum (2500-400)/20 - 0.5 = 104.5 > h = 5.

    The five warmup samples only freeze the CUSUM baseline; the spike is the
    injected failure, so the onset is marked here, immediately before it.
    """
    for ms in (400.0, 395.0, 405.0, 398.0, 402.0):
        b.add("search_web", f"q={b.rng.randrange(10**6)}", latency_ms=ms)
    b.mark_onset()
    b.add("search_web", f"q={b.rng.randrange(10**6)}", latency_ms=2500.0)


def _ml_cascade_signal(b: _Builder) -> None:
    """Three consecutive errors: consecutive score min(1, 3/3) = 1.0."""
    b.add("deploy_service", "svc-a", error=True, error_type="TimeoutError")
    b.add("deploy_service", "svc-b", error=True, error_type="TimeoutError")
    b.add("deploy_service", "svc-c", error=True, error_type="TimeoutError")


def build_ml_ensemble(ep: str, rng: random.Random) -> tuple[list[dict], int | None]:
    """Labeled ml_ensemble episodes; the variant picks the base signals.

    Under the ``ml_ensemble`` harness variant Monitor.default wraps every
    base detector in the noisy-OR MLOrchestrator, so individual triggers are
    swallowed into scores and only the combined ``ml_ensemble`` risk reaches
    sinks. Variant arithmetic:

    * 00 (loop signal): third repeat emits loop score 3/3*0.5 = 0.5;
      noisy-OR over the lone score is 1-(1-0.5) = 0.5, which is not below
      the 0.5 ensemble threshold, so exactly one ml_ensemble risk emits.
    * 01 (latency signal): warmup [400,395,405,398,402] freezes mu0 ~= 400
      with floored sigma0 = 20ms; the 2500ms spike gives cusum 104.5 > 5 and
      score min(1, 0.6 + 0.1*(104.5/5 - 1)) = 1.0 -> combined 1.0.
    * 02 (cascade signal): three consecutive tool errors give consecutive
      score min(1, 3/3) = 1.0 -> combined 1.0.
    * 03 (combined): the same event carries BOTH the third id=9 repeat AND
      a 2500ms spike after a 400ms-class five-sample warmup, so the loop
      score 0.5 and latency score 1.0 land on ONE step:
      noisy-OR = 1 - (1 - 0.5)*(1 - 1.0) = 1.0 >= 0.5, one emission.

    Every variant keeps <= 13 tool calls (meltdown window never fills), no
    token fields (token CUSUM inert), and closes on a message step (silent
    abort has nothing to judge).
    """
    b = _Builder(ep, rng)
    shape = _goal_drift_variant(ep)
    if shape == 0:
        for i in range(2):
            b.add(_cycle_tool(i), f"q={rng.randrange(10**6)}")
        # Onset: the first of the three identical calls (step 2).
        b.mark_onset()
        _ml_loop_signal(b)
        b.add(_cycle_tool(2), f"q={rng.randrange(10**6)}")
    elif shape == 1:
        _ml_latency_spike(b)
    elif shape == 2:
        b.add(_cycle_tool(0), f"q={rng.randrange(10**6)}")
        # Onset: the first of the three erroring calls (step 1).
        b.mark_onset()
        _ml_cascade_signal(b)
        b.add(_cycle_tool(1), f"q={rng.randrange(10**6)}")
    else:
        # Warmup samples 400-class on distinct args except two id=9 repeats;
        # the spike step is ALSO the third id=9 occurrence.
        for args, ms in (("a=1", 400.0), ("a=2", 395.0), ("a=3", 405.0)):
            b.add("search_web", args, latency_ms=ms)
        # Onset: the first id=9 repeat (step 3). The three id=9 calls ARE the
        # injected loop signal, and the third also carries the latency spike.
        b.mark_onset()
        b.add("search_web", "id=9", latency_ms=398.0)
        b.add("search_web", "id=9", latency_ms=402.0)  # warmup sample n=5
        b.add("search_web", "id=9", latency_ms=2500.0)  # loop x3 + spike
    b.finish_with_output()
    return b.result()


LABELED_BUILDERS = {
    "loop": build_loop,
    "error_cascade": build_error_cascade,
    "latency_anomaly": build_latency_anomaly,
    "token_runaway": build_token_runaway,
    "budget_breach": build_budget_breach,
    "meltdown_low": build_meltdown_low,
    "meltdown_high": build_meltdown_high,
    "silent_abort": build_silent_abort,
    # Issue #118 additions: opt-in detectors, replayed under their named
    # harness config variants (see LABELED_TRIGGER_VARIANTS in main()).
    "goal_drift": build_goal_drift,
    "ml_ensemble": build_ml_ensemble,
    # Issue #181 additions: the 7 shipped triggers that were missing from
    # SHIPPED_TRIGGERS and from harness_config.
    "stagnation": build_stagnation,
    "side_effect_duplicate": build_side_effect_duplicate,
    "governance_decay": build_governance_decay,
    "cycle": build_cycle,
    "stall": build_stall,
    "idle_gap": build_idle_gap,
    "wall_clock_budget": build_wall_clock_budget,
}

# Which harness config variant each labeled trigger replays under; anything
# absent from this map uses the standard flags.
LABELED_TRIGGER_VARIANTS = {
    "goal_drift": "goal_drift",
    "ml_ensemble": "ml_ensemble",
}


# --------------------------------------------------------------------------
# Stage 3 variation corpus (variants 4..9)
# --------------------------------------------------------------------------
# Every builder below takes (ep, rng, variant) and produces ONE episode whose
# labeled failure is unambiguous but whose surface trajectory differs from the
# legacy four: onset position, episode length, timing jitter, retry counts,
# error placement, tool-call ordering, latency and token volumes all move.
# Parameters are drawn from the episode's OWN seeded rng, so a fixture's seed
# alone reproduces it. The arithmetic envelopes are documented per builder;
# each keeps the labeled trigger firing while leaving the episode free to
# cross-fire other detectors (that is data for the evaluation, not a bug).


def _apply_step_gaps(events: list[dict], step_gaps: Sequence[float]) -> list[dict]:
    """Rewrite timestamps from an explicit per-event gap schedule.

    ``step_gaps[i]`` is the wall-clock seconds between event i-1 and event i.
    Used wherever an episode needs non-default spacing (idle gaps, long
    running tasks, stagnation's compressed clock) instead of ad-hoc math on
    BASE_TS.
    """
    ts = BASE_TS
    # strict: every caller passes one gap per event, and a mismatched schedule
    # would silently leave some events on the default spacing.
    for event, gap in zip(events, step_gaps, strict=True):
        ts += gap
        event["timestamp"] = ts
    return events


def vary_loop(ep: str, rng: random.Random, v: int) -> tuple[list[dict], int | None]:
    """Retry loop with varied warmup, repeat count, tool and tail.

    repeats 3..6 identical calls; warmup 2..8 so the 12-signature loop window
    still holds all three copies when the third lands (warmup + 3 <= 11).
    Total tool calls <= 18 so the 20-slot meltdown window never fills.
    """
    warm = 2 + rng.randrange(7)
    reps = 3 + rng.randrange(4)
    tail = 1 + rng.randrange(4)
    tool = rng.choice(("fetch_record", "sync_calendar", "send_email", "run_query"))
    args = rng.choice(("id=42", "batch=7", "mode=retry", "cursor=abc"))
    b = _Builder(ep, rng)
    for i in range(warm):
        b.add(_cycle_tool(i), f"q={rng.randrange(10**6)}")
    b.mark_onset()
    for _ in range(reps):
        b.add(tool, args)
    for i in range(tail):
        b.add(_cycle_tool(warm + i), f"t={rng.randrange(10**6)}")
    b.finish_with_output()
    return b.result()


def vary_error_cascade(
    ep: str, rng: random.Random, v: int
) -> tuple[list[dict], int | None]:
    """Consecutive failures of one operation, varied root cause and length.

    3..5 consecutive errors sharing one error_type (a single root cause);
    warmup 1..6 keeps warmup + 3 <= 9 inside the 10-slot cascade window when
    the third error lands. Tools vary so entropy stays high and meltdown has
    no single-tool collapse to see; no latency/token telemetry.
    """
    warm = 1 + rng.randrange(6)
    nerr = 3 + rng.randrange(3)
    tail = 1 + rng.randrange(4)
    etype = rng.choice(
        ("TimeoutError", "ConnectionError", "HTTP503", "PermissionError")
    )
    tools = ("deploy_service", "sync_contacts", "push_build", "send_invoice")
    b = _Builder(ep, rng)
    for i in range(warm):
        b.add(_cycle_tool(i), f"q={rng.randrange(10**6)}")
    b.mark_onset()
    for i in range(nerr):
        b.add(
            rng.choice(tools),
            f"svc-{i}-{rng.randrange(10**3)}",
            error=True,
            error_type=etype,
        )
    for i in range(tail):
        b.add(_cycle_tool(warm + i), f"t={rng.randrange(10**6)}")
    b.finish_with_output()
    return b.result()


def vary_latency_anomaly(
    ep: str, rng: random.Random, v: int
) -> tuple[list[dict], int | None]:
    """Latency blowout over a realistically jittery baseline.

    5..8 baseline samples at +-12% of a 300..600ms centre: sample std ~= 7% of
    the mean, which beats the 5% relative floor, so sigma0 ~= 0.07*centre.
    Spikes at 4..8x the centre give z >= 3*centre/(0.07*centre) ~= 43, cusum
    ~= 42 >> h=5 on the first post-warmup step. Spike count 1..3; the
    sustained shift keeps the accumulator high but only the first emission
    matters for scoring.
    """
    centre = rng.choice((300.0, 400.0, 500.0, 600.0))
    nbase = 5 + rng.randrange(4)
    nspike = 1 + rng.randrange(3)
    spike = centre * rng.choice((4.0, 5.0, 6.0, 8.0))
    jitter = centre * 0.12
    b = _Builder(ep, rng)
    for _ in range(nbase):
        ms = centre + (rng.random() - 0.5) * 2 * jitter
        b.add("search_web", f"q={rng.randrange(10**6)}", latency_ms=round(ms, 1))
    b.mark_onset()
    for _ in range(nspike):
        b.add("search_web", f"s={rng.randrange(10**6)}", latency_ms=round(spike, 1))
    for i in range(2):
        b.add(_cycle_tool(i + 1), f"t={rng.randrange(10**6)}")
    b.finish_with_output()
    return b.result()


def vary_token_runaway(
    ep: str, rng: random.Random, v: int
) -> tuple[list[dict], int | None]:
    """Sustained token burn after a 20..21-sample warmup, under the warn line.

    Warmup volume 700..1000/step (+-60 jitter) freezes mu0 ~= base and
    sigma0 = max(std ~= 35, 5% of base ~= 50) = 50. Burns 4..6 steps at
    2200..2700/step give z >= (2200-700)/50 = 30, cusum ~= 29 >> h=5. The
    cumulative total stays <= 21*1000 + 6*2700 + 1000 = 38,200 < 40,000, so
    the budget envelope's warn branch never fires ahead of the CUSUM. Tools
    cycle all five identities so meltdown holds 2.32 bits as windows fill.
    """
    nbase = rng.choice((20, 21))
    base_vol = rng.choice((700, 800, 900, 1000))
    nburn = rng.choice((4, 5, 6))
    burn_vol = rng.choice((2200, 2400, 2600, 2700))
    ntail = rng.randrange(2)
    b = _Builder(ep, rng)
    for i in range(nbase):
        vol = base_vol + rng.randrange(-60, 61)
        b.add(
            _cycle_tool(i),
            f"item={rng.randrange(10**6)}",
            tokens_in=max(50, vol - 200),
            tokens_out=200,
        )
    b.mark_onset()
    for i in range(nburn):
        b.add(
            _cycle_tool(nbase + i),
            f"burn={rng.randrange(10**6)}",
            tokens_in=burn_vol - 200,
            tokens_out=200,
        )
    for i in range(ntail):
        b.add(
            _cycle_tool(nbase + nburn + i),
            f"tail={rng.randrange(10**6)}",
            tokens_in=base_vol - 200,
            tokens_out=200,
        )
    b.finish_with_output()
    return b.result()


def vary_budget_breach(
    ep: str, rng: random.Random, v: int
) -> tuple[list[dict], int | None]:
    """One oversized step that crosses the 50,000 budget from below the warn.

    20..32 baseline steps at 700..1000/step accumulate <= 32,000 < 40,000, so
    the warn branch is never taken first. The single oversized step carries
    exactly the shortfall to 51,000..55,000 total, so the envelope's breach
    branch fires and returns BEFORE the token CUSUM ever sees the outlier --
    that early return is what keeps token_runaway quiet here. Tail steps
    return to baseline volume (increment ~= -0.5). Exactly one oversized
    step: a second would fall through to the CUSUM and alarm.
    """
    nbase = rng.choice((20, 24, 28, 32))
    base_vol = rng.choice((700, 800, 900, 1000))
    ntail = 1 + rng.randrange(3)
    target = rng.choice((51_000, 53_000, 55_000))
    shortfall = target - nbase * base_vol
    assert shortfall > 0, f"{ep}: baseline already reaches the budget"
    b = _Builder(ep, rng)
    for i in range(nbase):
        b.add(
            _cycle_tool(i),
            f"row={rng.randrange(10**6)}",
            tokens_in=base_vol - 200,
            tokens_out=200,
        )
    b.mark_onset()
    b.add(_cycle_tool(nbase), "bulk_export=huge", tokens_in=shortfall, tokens_out=0)
    for i in range(ntail):
        b.add(
            _cycle_tool(nbase + 1 + i),
            f"row={rng.randrange(10**6)}",
            tokens_in=base_vol - 200,
            tokens_out=200,
        )
    b.finish_with_output()
    return b.result()


def vary_meltdown_low(
    ep: str, rng: random.Random, v: int
) -> tuple[list[dict], int | None]:
    """Rote single-tool collapse of varied length.

    One tool identity for 20..32 calls; entropy over tool_name collapses to
    0 bits and the first full 20-window fires. Unique args keep signatures
    distinct (loop needs a repeated SIGNATURE, stall needs 25 identical) and
    length stays under the 50-step stagnation window.
    """
    tool = rng.choice(("scrape_page", "crawl_sitemap", "fetch_log", "poll_queue"))
    n = rng.choice((20, 24, 28, 32))
    b = _Builder(ep, rng)
    b.mark_onset()
    for i in range(n):
        b.add(tool, f"row_{i}_{rng.randrange(10**3)}")
    b.finish_with_output()
    return b.result()


_CHURN_POOLS: tuple[tuple[str, ...], ...] = (
    CHURN_POOL,
    tuple(f"churn_{i}" for i in range(13)),
    tuple(f"churn_{i}" for i in range(14)),
)


def vary_meltdown_high(
    ep: str, rng: random.Random, v: int
) -> tuple[list[dict], int | None]:
    """Tool churn over a 12/13/14-tool strict cycle, varied length.

    Any full 20-window of a strict P-cycle (P >= 12) holds most identities
    twice and the rest once: H(P=12) ~= 3.52, H(13) ~= 3.54, H(14) ~= 3.61
    bits, all above the 3.4 thrash threshold. Each signature appears at most
    once per 12 signatures, so loop stays quiet, and the period exceeds the
    cycle scanner's max_period of 6.
    """
    pool = rng.choice(_CHURN_POOLS)
    n = rng.choice((20, 24, 28))
    b = _Builder(ep, rng)
    b.mark_onset()
    for i in range(n):
        b.add(pool[i % len(pool)], f"call-{i}-{rng.randrange(10**6)}")
    b.finish_with_output()
    return b.result()


def vary_silent_abort(
    ep: str, rng: random.Random, v: int
) -> tuple[list[dict], int | None]:
    """A clean run that stops mid-work, varied length and final action.

    3..10 varied warmup calls then one error-free bare tool call: finalize()
    sees a non-output final action and fires. <= 11 tool calls keeps the
    meltdown window unfilled; no errors, latency or token telemetry.
    """
    n = 3 + rng.randrange(8)
    final = rng.choice(("write_report", "commit_changes", "post_results", "save_draft"))
    b = _Builder(ep, rng)
    for i in range(n):
        b.add(_cycle_tool(i), f"k={rng.randrange(10**6)}")
    b.mark_onset()
    b.add(final, f"draft=v{rng.randrange(10**3)}")  # ends mid-work
    return b.result()  # deliberately NOT closed with a message step


def vary_stagnation(
    ep: str, rng: random.Random, v: int
) -> tuple[list[dict], int | None]:
    """Novelty collapse over a varied signature pool, length and step gap.

    5..8 signatures reused at random over 100..120 steps. After the first
    window every signature has been seen, so the novelty share of each
    50-window is 0 < min_novelty 0.05; two consecutive stale windows plus
    patience 2 fires. Random order breaks strict periodicity (cycle silent).
    log2(5..8) = 2.32..3.00 bits keeps meltdown in band. Compressed 0.4..0.7s
    gaps hold total wall <= 84s < 96s so wall_clock_budget stays silent while
    windows complete.
    """
    pool_n = rng.choice((5, 6, 7, 8))
    n = rng.choice((100, 110, 120))
    gap = rng.choice((0.4, 0.5, 0.6, 0.7))
    sigs = [(f"g_tool_{i}", f"g_arg_{i}") for i in range(pool_n)]
    b = _Builder(ep, rng)
    b.mark_onset()
    for _ in range(n):
        tool, args = sigs[rng.randrange(pool_n)]
        b.add(tool, args)
    b.finish_with_output()
    return _apply_step_gaps(b.events, [gap] * len(b.events)), b.onset


def vary_side_effect_duplicate(
    ep: str, rng: random.Random, v: int
) -> tuple[list[dict], int | None]:
    """Duplicated side-effecting call with varied separation.

    Two identical side_effect=True calls separated by 1..4 unrelated steps;
    the second crosses allowed_repeats=1. Warmup 1..4, tail 1..4; total <= 14
    tool calls keeps every window-based detector quiet.
    """
    warm = 1 + rng.randrange(4)
    sep = 1 + rng.randrange(4)
    tail = 1 + rng.randrange(4)
    tool = rng.choice(("charge_card", "send_invoice", "post_payment", "ship_order"))
    args = rng.choice(("amt=10", "amt=25", "ref=x9"))
    b = _Builder(ep, rng)
    for i in range(warm):
        b.add(_cycle_tool(i), f"init={rng.randrange(10**6)}")
    b.mark_onset()
    b.add(tool, args, side_effect=True)
    for i in range(sep):
        b.add(_cycle_tool(warm + i), f"mid={rng.randrange(10**6)}")
    b.add(tool, args, side_effect=True)
    for i in range(tail):
        b.add(_cycle_tool(warm + sep + i), f"tail={rng.randrange(10**6)}")
    b.finish_with_output()
    return b.result()


def vary_governance_decay(
    ep: str, rng: random.Random, v: int
) -> tuple[list[dict], int | None]:
    """Compaction pinning constraints the run never confirms, varied pins.

    A compaction pins 1..3 constraint hashes; 3..5 follow-up steps none of
    which re-confirm them, so grace 3 expires and exactly one risk fires.
    Pin hashes are derived from the variant index so they differ per episode
    but stay deterministic.
    """
    warm = 1 + rng.randrange(5)
    npins = 1 + rng.randrange(3)
    grace = rng.choice((3, 4, 5))
    b = _Builder(ep, rng)
    for i in range(warm):
        b.add(_cycle_tool(i), f"pre={rng.randrange(10**6)}")
    b.mark_onset()
    pins = [
        hashlib.sha256(f"constraint-{v}-{k}".encode()).hexdigest() for k in range(npins)
    ]
    b.add(
        "__compaction__",
        "compaction",
        action_type="compaction",
        metadata={"pinned": pins},
    )
    for i in range(grace):
        b.add(_cycle_tool(i), f"grace={rng.randrange(10**6)}")
    b.finish_with_output()
    return b.result()


def vary_cycle(ep: str, rng: random.Random, v: int) -> tuple[list[dict], int | None]:
    """Strict period-P cycle for P in 2..6, at least 12 steps long.

    The cycle scanner needs its 12-window filled, so the episode holds
    max(3, ceil(12/P)) full periods. Entropy log2(P) = 1.00..2.58 bits, all
    inside the (0.4, 3.4) meltdown band. NOTE the deliberate overlap: for
    P <= 4 each phase repeats at least 3 times inside 12 consecutive
    signatures, so the plain loop rule ALSO fires -- that overlap is measured
    by the evaluation, not avoided here.
    """
    period = rng.choice((2, 3, 4, 5, 6))
    periods = max(3, math.ceil(12 / period))
    n = period * periods
    sigs = [(f"p_tool_{period}_{i}", f"p_arg_{i}") for i in range(period)]
    b = _Builder(ep, rng)
    b.mark_onset()
    for i in range(n):
        tool, args = sigs[i % period]
        b.add(tool, args)
    b.finish_with_output()
    return b.result()


def vary_stall(ep: str, rng: random.Random, v: int) -> tuple[list[dict], int | None]:
    """25+ consecutive identical calls with varied warmup and length.

    Stall fires at 25 consecutive identical signatures; warmup 1..6 keeps the
    repeat sequence contiguous. The single tool also collapses meltdown's
    entropy (fires at 20 tool calls) and the repeats satisfy the loop rule at
    3 -- both are real properties of the shape and are measured.
    """
    warm = 1 + rng.randrange(6)
    reps = rng.choice((25, 30, 35))
    tool = rng.choice(("stalled_tool", "hung_parser", "blocked_worker"))
    args = rng.choice(("repeat-id-99", "job=7", "lock=abc"))
    b = _Builder(ep, rng)
    for i in range(warm):
        b.add(_cycle_tool(i), f"warm={rng.randrange(10**6)}")
    b.mark_onset()
    for _ in range(reps):
        b.add(tool, args)
    b.finish_with_output()
    return b.result()


def vary_idle_gap(ep: str, rng: random.Random, v: int) -> tuple[list[dict], int | None]:
    """One >= 10s ingest gap at a varied position and duration.

    2..6 calls, a gap of 10..25s, then 2..5 more; total wall <= 41.5s < 96s so
    the wall-clock envelope stays silent. Total <= 12 tool calls keeps
    window-based detectors unfilled.
    """
    before = 2 + rng.randrange(5)
    gap = rng.choice((10.0, 12.0, 15.0, 20.0, 25.0))
    after = 2 + rng.randrange(4)
    b = _Builder(ep, rng)
    for i in range(before):
        b.add(_cycle_tool(i), f"q={rng.randrange(10**6)}")
    b.mark_onset()
    b.add(_cycle_tool(before), f"gap={rng.randrange(10**6)}")
    for i in range(after):
        b.add(_cycle_tool(before + 1 + i), f"q={rng.randrange(10**6)}")
    b.finish_with_output()
    gaps = [STEP_GAP_S] * before + [gap] + [STEP_GAP_S] * (len(b.events) - before - 1)
    return _apply_step_gaps(b.events, gaps), b.onset


def vary_wall_clock_budget(
    ep: str, rng: random.Random, v: int
) -> tuple[list[dict], int | None]:
    """Wall-clock burn past 120s at varied step spacing.

    Uniform 1.0..2.0s gaps for ceil(125/gap)+0..5 steps gives total wall in
    [125, 135]s: the 96s warning fires first, then the 120s breach, both under
    the wall_clock_budget trigger. Unique args keep novelty at 1.0 (stagnation
    silent) and the five-tool cycle holds 2.32 bits as windows fill. No gap
    reaches the 10s idle threshold.
    """
    gap = rng.choice((1.0, 1.2, 1.5, 1.8, 2.0))
    n = math.ceil(125.0 / gap) + rng.randrange(6)
    b = _Builder(ep, rng)
    b.mark_onset()
    for i in range(n):
        b.add(_cycle_tool(i), f"w={rng.randrange(10**6)}")
    b.finish_with_output()
    return _apply_step_gaps(b.events, [gap] * len(b.events)), b.onset


def vary_goal_drift(
    ep: str, rng: random.Random, v: int
) -> tuple[list[dict], int | None]:
    """Goal-drift deviation shapes, cycled by variant parity.

    Parity 0 -- latency blowout on ONE baseline capability: n=10..12 calls of
    a single baseline tool at ref + z*spread with z in 8..20. z=8 gives
    contribution min(1, (8-3)/10) = 0.5 >= the 0.5 score threshold; larger z
    saturates. Constant latency freezes the CUSUM onto itself (inert).
    Parity 1 -- unseen capability: nine baseline-matching calls, then one call
    to a tool absent from the committed baseline; absent tools contribute a
    flat 0.6, and that step is the tenth live sample, so goal_drift fires
    there and only there.
    """
    if v % 2 == 0:
        n = rng.choice((10, 11, 12))
        idx = rng.randrange(len(GOAL_DRIFT_BASELINE_TOOLS))
        tool, ref = GOAL_DRIFT_BASELINE_TOOLS[idx]
        spread = max(1.0, 0.05 * ref)
        z = rng.choice((8, 10, 14, 20))
        b = _Builder(ep, rng)
        b.mark_onset()
        for _ in range(n):
            b.add(tool, f"q={rng.randrange(10**6)}", latency_ms=ref + z * spread)
        b.finish_with_output()
        return b.result()
    unseen = rng.choice(("deploy_canary", "wipe_cache", "pivot_db", "send_wire"))
    b = _Builder(ep, rng)
    for i in range(9):
        tool, ms = GOAL_DRIFT_BASELINE_TOOLS[i % len(GOAL_DRIFT_BASELINE_TOOLS)]
        b.add(tool, f"req={rng.randrange(10**6)}", latency_ms=ms)
    b.mark_onset()
    b.add(
        unseen, f"env={rng.randrange(10**6)}", latency_ms=rng.choice((60.0, 70.0, 90.0))
    )
    b.finish_with_output()
    return b.result()


def vary_ml_ensemble(
    ep: str, rng: random.Random, v: int
) -> tuple[list[dict], int | None]:
    """ML-ensemble base signals cycled by variant mod 4, varied parameters.

    Under the ml_ensemble variant every base detector is wrapped in the
    noisy-OR orchestrator, so only the combined ml_ensemble risk reaches
    sinks. Each shape must leave at least one base score >= 0.5 on one step.
    """
    shape = v % 4
    b = _Builder(ep, rng)
    if shape == 0:
        # Loop signal: three identical calls give the loop detector
        # count/threshold*0.5 = 3/3*0.5 = 0.5; noisy-OR over it is 0.5.
        warm = rng.randrange(5)
        tool = rng.choice(("fetch_record", "pull_ticket", "get_order"))
        args = rng.choice(("id=7", "ref=abc", "sku=99"))
        for i in range(warm):
            b.add(_cycle_tool(i), f"q={rng.randrange(10**6)}")
        b.mark_onset()
        for _ in range(3):
            b.add(tool, args)
        b.add(_cycle_tool(2), f"q={rng.randrange(10**6)}")
    elif shape == 1:
        # Latency signal: 400-class warmup freezes sigma0 = 20ms; a 2500ms
        # spike gives cusum 104.5 > h=5, score min(1, 0.6 + 0.1*(104.5/5-1)).
        for ms in (400.0, 395.0, 405.0, 398.0, 402.0):
            b.add("search_web", f"q={rng.randrange(10**6)}", latency_ms=ms)
        b.mark_onset()
        b.add("search_web", f"q={rng.randrange(10**6)}", latency_ms=2500.0)
    elif shape == 2:
        # Cascade signal: three consecutive errors give consecutive score 1.0.
        etype = rng.choice(("TimeoutError", "ConnectionError", "HTTP503"))
        b.add(_cycle_tool(0), f"q={rng.randrange(10**6)}")
        b.mark_onset()
        for i in range(3):
            b.add("deploy_service", f"svc-{i}", error=True, error_type=etype)
        b.add(_cycle_tool(1), f"q={rng.randrange(10**6)}")
    else:
        # Combined: the third identical id=9 call also carries the spike.
        for args, ms in (("a=1", 400.0), ("a=2", 395.0), ("a=3", 405.0)):
            b.add("search_web", args, latency_ms=ms)
        b.mark_onset()
        b.add("search_web", "id=9", latency_ms=398.0)
        b.add("search_web", "id=9", latency_ms=402.0)
        b.add("search_web", "id=9", latency_ms=2500.0)
    b.finish_with_output()
    return b.result()


LABELED_VARIATION_BUILDERS = {
    "loop": vary_loop,
    "error_cascade": vary_error_cascade,
    "latency_anomaly": vary_latency_anomaly,
    "token_runaway": vary_token_runaway,
    "budget_breach": vary_budget_breach,
    "meltdown_low": vary_meltdown_low,
    "meltdown_high": vary_meltdown_high,
    "silent_abort": vary_silent_abort,
    "goal_drift": vary_goal_drift,
    "ml_ensemble": vary_ml_ensemble,
    "stagnation": vary_stagnation,
    "side_effect_duplicate": vary_side_effect_duplicate,
    "governance_decay": vary_governance_decay,
    "cycle": vary_cycle,
    "stall": vary_stall,
    "idle_gap": vary_idle_gap,
    "wall_clock_budget": vary_wall_clock_budget,
}


# --------------------------------------------------------------------------
# Healthy controls (label: null)
# --------------------------------------------------------------------------


def healthy_plain(
    ep: str, rng: random.Random, n_calls: int = 12
) -> tuple[list[dict], int | None]:
    b = _Builder(ep, rng)
    for i in range(n_calls):
        b.add(_cycle_tool(i), f"q={rng.randrange(10**6)}")
    b.finish_with_output()
    return b.result()


def healthy_repeat_pair_recovery(
    ep: str, rng: random.Random
) -> tuple[list[dict], int | None]:
    """Two repeats of one signature then recovery: under the threshold of 3.

    A second pair appears later, still never a third copy. Fewer than 20 tool
    calls keeps the meltdown window unfilled.
    """
    b = _Builder(ep, rng)
    b.add("search_web", "q=pair")
    b.add("search_web", "q=pair")  # second repeat: count 2 < 3, no alarm
    for i in range(3):
        b.add(_cycle_tool(i + 1), f"q={rng.randrange(10**6)}")  # recovery
    b.add("search_web", "q=pair2")
    b.add("search_web", "q=pair2")  # another pair, still only 2
    b.add(_cycle_tool(4), f"q={rng.randrange(10**6)}")
    b.finish_with_output()
    return b.result()


def healthy_warmup_jitter(ep: str, rng: random.Random) -> tuple[list[dict], int | None]:
    """Jittery samples absorbed during warmup cannot alarm after freeze.

    Variant 1: warmup [380,430,395,460,370] freezes mu0=407, sigma0~=37.4;
    post-freeze [440,405] give increments <= 0.39, cusum stays under 0.4
    against an alarm threshold of 5.
    """
    b = _Builder(ep, rng)
    for ms in (380.0, 430.0, 395.0, 460.0, 370.0):
        b.add("search_web", f"q={rng.randrange(10**6)}", latency_ms=ms)
    for ms in (440.0, 405.0):
        b.add("search_web", f"q={rng.randrange(10**6)}", latency_ms=ms)
    for i in range(2):
        b.add(_cycle_tool(i + 1), f"q={rng.randrange(10**6)}")
    b.finish_with_output()
    return b.result()


def healthy_warmout_wide_jitter(
    ep: str, rng: random.Random
) -> tuple[list[dict], int | None]:
    """Wide warmup spread inflates sigma0 so ordinary variation stays quiet.

    Warmup [350,480,390,460,370]: mu0=410, sample std ~= 57.0 (> floors);
    post-freeze [420,400,430] all produce negative increments
    (-0.33, -0.68, -0.15).
    """
    b = _Builder(ep, rng)
    for ms in (350.0, 480.0, 390.0, 460.0, 370.0):
        b.add("search_web", f"j={rng.randrange(10**6)}", latency_ms=ms)
    for ms in (420.0, 400.0, 430.0):
        b.add("search_web", f"j={rng.randrange(10**6)}", latency_ms=ms)
    b.add(_cycle_tool(2), f"j={rng.randrange(10**6)}")
    b.finish_with_output()
    return b.result()


def healthy_near_threshold_entropy(
    ep: str, rng: random.Random
) -> tuple[list[dict], int | None]:
    """Strict five-tool cycle: every full 20-window sits at 2.32 bits.

    That is 0.48 bits below the 2.8 thrash threshold and far above the 0.4
    collapse threshold: the documented near-threshold healthy case.
    """
    b = _Builder(ep, rng)
    for i in range(24):
        b.add(_cycle_tool(i), f"c{i}-{rng.randrange(10**6)}")
    b.finish_with_output()
    return b.result()


def healthy_isolated_errors(
    ep: str, rng: random.Random
) -> tuple[list[dict], int | None]:
    """Two spaced single errors: consecutive count never passes 1 and any
    10-step window holds at most 2 counted errors (threshold 3)."""
    b = _Builder(ep, rng)
    b.add(_cycle_tool(0), f"r={rng.randrange(10**6)}")
    b.add("send_email", "to=team", error=True, error_type="SMTPError")
    for i in range(5):
        b.add(_cycle_tool(i + 1), f"r={rng.randrange(10**6)}")
    b.add("send_email", "to=ops", error=True, error_type="SMTPError")
    for i in range(3):
        b.add(_cycle_tool(i + 2), f"r={rng.randrange(10**6)}")
    b.finish_with_output()
    return b.result()


def healthy_steady_tokens(ep: str, rng: random.Random) -> tuple[list[dict], int | None]:
    """Steady ~1,000 tokens/step on the five-tool cycle.

    Freeze at n=20 gives mu0=1000 and sigma0=max(std~=51, 50)=~51;
    post-freeze alternation of 950/1050 yields increments of +0.48 / -1.47,
    so the CUSUM never accumulates. Cumulative 24,000 < 40,000 warn.
    Entropy stays at 2.32 bits.
    """
    b = _Builder(ep, rng)
    for i in range(24):
        vol = 950 if i % 2 == 0 else 1050
        b.add(
            _cycle_tool(i),
            f"t={rng.randrange(10**6)}",
            tokens_in=vol - 200,
            tokens_out=200,
        )
    b.finish_with_output()
    return b.result()


def healthy_jitter_plus_error(
    ep: str, rng: random.Random
) -> tuple[list[dict], int | None]:
    """Warmup jitter AND one isolated error: neither signal reaches a rule."""
    b = _Builder(ep, rng)
    for ms in (380.0, 430.0, 395.0, 460.0, 370.0):
        b.add("search_web", f"m={rng.randrange(10**6)}", latency_ms=ms)
    b.add("notify_hook", "dst=ci", error=True, error_type="ConnReset")
    for ms in (440.0, 405.0):
        b.add("search_web", f"m={rng.randrange(10**6)}", latency_ms=ms)
    for i in range(3):
        b.add(_cycle_tool(i + 1), f"m={rng.randrange(10**6)}")
    b.finish_with_output()
    return b.result()


def healthy_moderate_jitter(
    ep: str, rng: random.Random
) -> tuple[list[dict], int | None]:
    """Moderate warmup spread freezes sigma0~=31.9; post-freeze values give
    increments <= -0.12 against h=5. Warmup [380,420,360,440,390]:
    mu0=398, sample std ~= 31.9 (> both floors)."""
    b = _Builder(ep, rng)
    for ms in (380.0, 420.0, 360.0, 440.0, 390.0):
        b.add("search_web", f"w={rng.randrange(10**6)}", latency_ms=ms)
    for ms in (410.0, 380.0):
        b.add("search_web", f"w={rng.randrange(10**6)}", latency_ms=ms)
    b.add(_cycle_tool(3), f"w={rng.randrange(10**6)}")
    b.finish_with_output()
    return b.result()


def healthy_long_cycle(ep: str, rng: random.Random) -> tuple[list[dict], int | None]:
    """Thirty calls on the five-tool cycle: the meltdown window fills and
    slides repeatedly while entropy holds at 2.32 bits the whole way."""
    b = _Builder(ep, rng)
    for i in range(30):
        b.add(_cycle_tool(i), f"L={rng.randrange(10**6)}")
    b.finish_with_output()
    return b.result()


def healthy_eight_tool_long(
    ep: str, rng: random.Random
) -> tuple[list[dict], int | None]:
    """Regression for #180: 30 calls cycling over eight tools, entropy ~2.97
    bits, comfortably below the 3.4 high threshold. A healthy ReAct agent
    with a broad toolbelt must stay quiet; the old 2.8 threshold fired here."""
    eight_pool = tuple(f"tool_{i}" for i in range(8))
    b = _Builder(ep, rng)
    for i in range(30):
        b.add(eight_pool[i % len(eight_pool)], f"q={rng.randrange(10**6)}")
    b.finish_with_output()
    return b.result()


def healthy_spaced_errors_wide(
    ep: str, rng: random.Random
) -> tuple[list[dict], int | None]:
    """Two errors nine steps apart: never adjacent, at most 2 in any window."""
    b = _Builder(ep, rng)
    for i in range(2):
        b.add(_cycle_tool(i), f"s={rng.randrange(10**6)}")
    b.add("sync_calendar", "scope=day", error=True, error_type="HTTP401")
    for i in range(8):
        b.add(_cycle_tool(i + 2), f"s={rng.randrange(10**6)}")
    b.add("sync_contacts", "scope=all", error=True, error_type="HTTP401")
    for i in range(3):
        b.add(_cycle_tool(i), f"s={rng.randrange(10**6)}")
    b.finish_with_output()
    return b.result()


def healthy_jitter_and_tokens(
    ep: str, rng: random.Random
) -> tuple[list[dict], int | None]:
    """Latency jitter and steady token burn together, each signal quiet."""
    b = _Builder(ep, rng)
    for ms in (380.0, 430.0, 395.0, 460.0, 370.0):
        b.add(
            "search_web",
            f"c={rng.randrange(10**6)}",
            latency_ms=ms,
            tokens_in=750,
            tokens_out=200,
        )
    for ms in (440.0, 405.0):
        b.add(
            "search_web",
            f"c={rng.randrange(10**6)}",
            latency_ms=ms,
            tokens_in=750,
            tokens_out=200,
        )
    b.finish_with_output()
    return b.result()


def healthy_pair_plus_tokens(
    ep: str, rng: random.Random
) -> tuple[list[dict], int | None]:
    """A repeat pair plus steady token volume: both signals stay sub-threshold."""
    b = _Builder(ep, rng)
    for i in range(6):
        vol = 950 if i % 2 == 0 else 1050
        b.add(
            _cycle_tool(i),
            f"p={rng.randrange(10**6)}",
            tokens_in=vol - 200,
            tokens_out=200,
        )
    b.add("translate_text", "lang=fr")
    b.add("translate_text", "lang=fr")  # pair, count 2
    for i in range(4):
        b.add(_cycle_tool(i + 1), f"p={rng.randrange(10**6)}")
    b.finish_with_output()
    return b.result()


def _healthy_long(ep: str, rng: random.Random, n: int) -> tuple[list[dict], int | None]:
    """Long healthy run with high novelty and 5-tool entropy, 50-100 range.

    Uses random args so every signature is novel (novelty 1.0, stagnation
    silent) and 5-tool cycle keeps meltdown at 2.32 bits (silent). No idle
    gaps (1.5s), total wall under 120s for n<=75, no side_effect or
    compaction, so all new #181 detectors stay silent while windows complete.
    """
    return healthy_plain(ep, rng, n)


HEALTHY_BUILDERS = [
    # plain varied runs at several lengths
    lambda ep, rng: healthy_plain(ep, rng, 10),
    lambda ep, rng: healthy_plain(ep, rng, 12),
    lambda ep, rng: healthy_plain(ep, rng, 14),
    lambda ep, rng: healthy_plain(ep, rng, 16),
    lambda ep, rng: healthy_plain(ep, rng, 11),
    lambda ep, rng: healthy_plain(ep, rng, 13),
    lambda ep, rng: healthy_plain(ep, rng, 15),
    lambda ep, rng: healthy_plain(ep, rng, 9),
    healthy_repeat_pair_recovery,
    healthy_repeat_pair_recovery,
    healthy_repeat_pair_recovery,
    healthy_warmup_jitter,
    healthy_warmup_jitter,
    healthy_warmout_wide_jitter,
    healthy_moderate_jitter,
    healthy_near_threshold_entropy,
    healthy_near_threshold_entropy,
    healthy_near_threshold_entropy,
    healthy_isolated_errors,
    healthy_spaced_errors_wide,
    healthy_steady_tokens,
    healthy_long_cycle,
    healthy_jitter_plus_error,
    healthy_pair_plus_tokens,
    healthy_jitter_and_tokens,
    lambda ep, rng: healthy_plain(ep, rng, 18),
    healthy_repeat_pair_recovery,
    healthy_near_threshold_entropy,
    healthy_steady_tokens,
    healthy_isolated_errors,
    # Appended at the end so existing episode ids stay stable: ids derive
    # from list position.
    lambda ep, rng: healthy_plain(ep, rng, 17),
    lambda ep, rng: healthy_plain(ep, rng, 8),
    healthy_eight_tool_long,
    # Long controls (50-100) so window-based detectors complete at least one
    # window while staying healthy (issue #181). Total wall 82.5-93s <96s warn.
    lambda ep, rng: _healthy_long(ep, rng, 55),
    lambda ep, rng: _healthy_long(ep, rng, 58),
    lambda ep, rng: _healthy_long(ep, rng, 60),
    lambda ep, rng: _healthy_long(ep, rng, 62),
]


# --------------------------------------------------------------------------
# Variant healthy controls (issue #118)
# --------------------------------------------------------------------------


def healthy_goal_drift_steady(
    ep: str, rng: random.Random
) -> tuple[list[dict], int | None]:
    """Baseline-indistinguishable traffic under the goal_drift variant.

    Twelve calls cycle ONLY baseline tools at their exact healthy latencies:
    every live per-tool mean equals its reference, so each z-score is 0 and
    the drift score is exactly 0.0 < threshold 0.5. Four CUSUM samples per
    tool stay under the 5-sample warmup; unique args keep loop quiet; 12
    tool calls never fill the meltdown window; closed with a message step.
    """
    b = _Builder(ep, rng)
    for i in range(12):
        tool, ms = GOAL_DRIFT_BASELINE_TOOLS[i % len(GOAL_DRIFT_BASELINE_TOOLS)]
        b.add(tool, f"ok={rng.randrange(10**6)}", latency_ms=ms)
    b.finish_with_output()
    return b.result()


def healthy_goal_drift_near_k(
    ep: str, rng: random.Random
) -> tuple[list[dict], int | None]:
    """Latency above baseline but inside goal_drift_latency_k stays silent.

    Live means sit exactly two sigmas above their references (below k = 3),
    so no contribution accumulates and the score is 0.0. Spreads are the
    documented floors: search_web max(1, 5% of 100) = 5 -> (110-100)/5 = 2;
    read_file max(1, 5% of 50) = 2.5 -> (55-50)/2.5 = 2; run_sql
    max(1, 5% of 80) = 4 -> (88-80)/4 = 2. All other detectors quiet for the
    same reasons as healthy_goal_drift_steady.
    """
    b = _Builder(ep, rng)
    for i in range(12):
        tool = GOAL_DRIFT_BASELINE_TOOLS[i % len(GOAL_DRIFT_BASELINE_TOOLS)][0]
        b.add(
            tool,
            f"nk={rng.randrange(10**6)}",
            latency_ms=GOAL_DRIFT_NEAR_K_LATENCIES[tool],
        )
    b.finish_with_output()
    return b.result()


# Reused under the ml_ensemble variant: both builders already keep every
# base detector sub-threshold (a pair never reaches repeat count 3; warmup
# jitter freezes into a CUSUM that cannot accumulate), so the orchestrator
# collects no scores at all and must stay silent.
HEALTHY_ML_BUILDERS = [healthy_repeat_pair_recovery, healthy_warmup_jitter]

HEALTHY_GOAL_DRIFT_BUILDERS = [healthy_goal_drift_steady, healthy_goal_drift_near_k]


# --------------------------------------------------------------------------
# Stage 3 hard healthy controls (issue #82 stage 3)
# --------------------------------------------------------------------------
# Difficult-but-legitimate traffic: each family stresses one detector's
# decision boundary from the healthy side. Every builder documents the exact
# reason it must stay silent. These are appended AFTER the legacy 41 controls
# so existing healthy ids (00..40) stay byte-identical.


def healthy_retry_then_success(
    ep: str, rng: random.Random
) -> tuple[list[dict], int | None]:
    """Legitimate retries: each failing call is retried once and succeeds.

    Error count never reaches the consecutive rule (max 1 in a row) and any
    10-step cascade window holds at most 2 counted errors, below the density
    threshold of 3. Two fail/retry pairs at different points in the run make
    the recovery pattern explicit rather than a single lucky error.
    """
    b = _Builder(ep, rng)
    for i in range(3):
        b.add(_cycle_tool(i), f"q={rng.randrange(10**6)}")
    b.add("deploy_service", "svc-a", error=True, error_type="TimeoutError")
    b.add("deploy_service", "svc-a-retry")  # retry succeeds
    for i in range(5):
        b.add(_cycle_tool(i + 3), f"q={rng.randrange(10**6)}")
    b.add("send_email", "to=ops", error=True, error_type="SMTPError")
    b.add("send_email", "to=ops-retry")  # retry succeeds
    for i in range(3):
        b.add(_cycle_tool(i + 6), f"q={rng.randrange(10**6)}")
    b.finish_with_output()
    return b.result()


_HEARTBEAT_JITTER = (320.0, 480.0, 350.0, 460.0, 390.0)


def healthy_bursty_latency(
    ep: str, rng: random.Random
) -> tuple[list[dict], int | None]:
    """Bursty latency inside the baseline's own spread: no true anomaly.

    Warmup _HEARTBEAT_JITTER has mean 400 and sample std ~= 68.9, which beats
    both sigma floors (1ms, 5% of mean = 20ms), so sigma0 ~= 68.9. Two bursts
    at 560ms each give increments (560-400)/68.9 - 0.5 ~= 1.82; the pair
    accumulates 3.64 < h = 5 and the return to 400ms decrements it. The burst
    is ~1.8 sigma -- a slow page inside the run's own variability, not a
    regime change.
    """
    b = _Builder(ep, rng)
    for ms in _HEARTBEAT_JITTER:
        b.add("search_web", f"q={rng.randrange(10**6)}", latency_ms=ms)
    for _ in range(2):
        b.add("search_web", f"slow={rng.randrange(10**6)}", latency_ms=560.0)
    for ms in (405.0, 395.0):
        b.add("search_web", f"q={rng.randrange(10**6)}", latency_ms=ms)
    for i in range(2):
        b.add(_cycle_tool(i + 1), f"q={rng.randrange(10**6)}")
    b.finish_with_output()
    return b.result()


def healthy_bursty_latency_in_warmup(
    ep: str, rng: random.Random
) -> tuple[list[dict], int | None]:
    """Bursts that fall inside the CUSUM warmup are absorbed by the freeze.

    The five warmup samples are [400, 560, 560, 400, 90]: mean 402, std ~=
    193, so the burst RAISES sigma0 rather than alarming. Post-freeze values
    at 400ms give (400-402)/193 - 0.5 < 0. Exercises the detector's
    learn-only window with genuinely bursty traffic.
    """
    b = _Builder(ep, rng)
    for ms in (400.0, 560.0, 560.0, 400.0, 90.0):
        b.add("search_web", f"w={rng.randrange(10**6)}", latency_ms=ms)
    for _ in range(3):
        b.add("search_web", f"q={rng.randrange(10**6)}", latency_ms=400.0)
    b.finish_with_output()
    return b.result()


def healthy_long_task(
    ep: str, rng: random.Random, n: int, gap: float
) -> tuple[list[dict], int | None]:
    """A long legitimate task: many steps, high novelty, no stagnation.

    n unique-arg calls on the five-tool cycle at `gap` second spacing. Unique
    args hold the novelty share at 1.0 while the 50-step stagnation window
    fills and slides repeatedly; the five-tool cycle holds entropy at 2.32
    bits as the 20-slot meltdown window fills; n*gap is capped under 96s so
    the wall-clock warning never fires, and uniform spacing never reaches the
    10s idle threshold. This is the control that proves long != stuck.
    """
    assert n * gap < 96.0, f"{ep}: a long healthy task must not cross the 96s warn"
    b = _Builder(ep, rng)
    for i in range(n):
        b.add(_cycle_tool(i), f"L={rng.randrange(10**6)}")
    b.finish_with_output()
    return _apply_step_gaps(b.events, [gap] * len(b.events)), None


def healthy_single_error_recovered(
    ep: str, rng: random.Random, at: int
) -> tuple[list[dict], int | None]:
    """Exactly one error, placed at `at`, recovered and moved past.

    One error in the whole run: the consecutive count peaks at 1 and every
    10-step window holds 1 error, both far under the threshold of 3. The
    placement varies (early / mid / late) so the recovery is not always
    followed by enough steps to matter.
    """
    b = _Builder(ep, rng)
    n = 12
    for i in range(n):
        if i == at:
            b.add("sync_calendar", "scope=day", error=True, error_type="HTTP501")
            continue
        b.add(_cycle_tool(i), f"r={rng.randrange(10**6)}")
    b.finish_with_output()
    return b.result()


def healthy_repeated_legit_calls(
    ep: str, rng: random.Random
) -> tuple[list[dict], int | None]:
    """Repeated calls the task genuinely requires, with changing inputs.

    Eighteen pagination calls to one tool, each with a distinct page argument.
    Signatures are exact-match and every one is distinct, so the loop repeat
    count never passes 1. Eighteen stays UNDER the 20-slot meltdown window, so
    the single-tool entropy collapse never gets a window to evaluate. The
    near-identical args ("page=01".."page=18") are the point: the shape looks
    repetitive to a human but is not a loop.
    """
    b = _Builder(ep, rng)
    for i in range(18):
        b.add("read_file", f"page={i:02d}")
    b.finish_with_output()
    return b.result()


def healthy_repeated_legit_two_tools(
    ep: str, rng: random.Random
) -> tuple[list[dict], int | None]:
    """A long alternating two-tool task: entropy 1.00 bit, above the floor.

    Twenty-four calls alternating read_file/write_file at unique args. Any
    20-window holds 10 of each: H = 1.00 bit, above the 0.4 collapse floor.
    Each signature appears once per 12 signatures at most, so loop is quiet.
    Deliberately near the collapse band: a one-tool version of this same task
    would fire meltdown_low, which is probed separately (see the stress probe
    in reports/) rather than asserted healthy here.
    """
    b = _Builder(ep, rng)
    for i in range(24):
        b.add("read_file" if i % 2 == 0 else "write_file", f"row={i:03d}")
    b.finish_with_output()
    return b.result()


def healthy_high_token_usage(
    ep: str, rng: random.Random
) -> tuple[list[dict], int | None]:
    """Large but bounded context: near the warn line, no runaway.

    Thirty-four steps at a constant 1150 tokens (33,800 + 300 in/out split =
    39,100 cumulative), just under the 40,000 warn threshold of the 50,000
    budget. A constant burn freezes mu0 onto itself and every post-freeze
    increment is max(0, 0 - 0.5) = 0, so the CUSUM cannot alarm. Large here
    means legitimately large, not accelerating.
    """
    b = _Builder(ep, rng)
    for i in range(34):
        b.add(
            _cycle_tool(i), f"ctx={rng.randrange(10**6)}", tokens_in=900, tokens_out=250
        )
    b.finish_with_output()
    return b.result()


def healthy_long_planning(ep: str, rng: random.Random) -> tuple[list[dict], int | None]:
    """A long reasoning phase interleaved with on-goal work (goal_drift).

    Replayed under the goal_drift variant. Reasoning steps (action_type
    "reasoning") alternate with search_web calls at exactly the baseline's
    100ms, so the live per-tool mean equals its reference and every z-score is
    0: the drift score is 0.0. The twelve search_web calls clear
    goal_drift_min_samples = 10, so the detector IS scoring during the
    planning phase and still stays silent. Constant latency also freezes the
    CUSUM onto itself.
    """
    b = _Builder(ep, rng)
    for i in range(24):
        if i % 2 == 0:
            b.add(
                "__plan__", f"plan-step-{i}", action_type="reasoning", latency_ms=100.0
            )
        else:
            b.add("search_web", f"q={rng.randrange(10**6)}", latency_ms=100.0)
    b.finish_with_output()
    return b.result()


def healthy_similar_actions(
    ep: str, rng: random.Random
) -> tuple[list[dict], int | None]:
    """Near-identical actions with legitimately varying inputs.

    Fourteen translate_text calls, one per language code, plus four
    summarise_text calls over distinct inputs. Distinct signatures keep the
    loop repeat count at 1; 18 total calls stays under the 20-slot meltdown
    window. Args differ in two characters each, so a fuzzy/approximate
    signature match would misfire -- the exact-match signature keeps it
    honest.
    """
    langs = (
        "fr",
        "es",
        "de",
        "it",
        "pt",
        "nl",
        "ru",
        "ja",
        "ko",
        "zh",
        "ar",
        "hi",
        "tr",
        "sv",
    )
    b = _Builder(ep, rng)
    for lang in langs:
        b.add("translate_text", f"lang={lang}")
    for i in range(4):
        b.add("summarize_text", f"doc={i}")
    b.finish_with_output()
    return b.result()


# (builder, config variant, reason it must stay silent) -- the variant is
# recorded so the harness replays each control under the detectors it stresses.
HEALTHY_HARD_BUILDERS: list[tuple[Any, str]] = [
    (healthy_retry_then_success, "standard"),
    (healthy_retry_then_success, "standard"),
    (healthy_bursty_latency, "standard"),
    (healthy_bursty_latency, "standard"),
    (healthy_bursty_latency_in_warmup, "standard"),
    (lambda ep, rng: healthy_long_task(ep, rng, 90, 1.0), "standard"),
    (lambda ep, rng: healthy_long_task(ep, rng, 110, 0.8), "standard"),
    (lambda ep, rng: healthy_long_task(ep, rng, 130, 0.7), "standard"),
    (lambda ep, rng: healthy_single_error_recovered(ep, rng, 1), "standard"),
    (lambda ep, rng: healthy_single_error_recovered(ep, rng, 6), "standard"),
    (lambda ep, rng: healthy_single_error_recovered(ep, rng, 10), "standard"),
    (healthy_repeated_legit_calls, "standard"),
    (healthy_repeated_legit_two_tools, "standard"),
    (healthy_high_token_usage, "standard"),
    (healthy_long_planning, "goal_drift"),
    (healthy_long_planning, "goal_drift"),
    (healthy_similar_actions, "standard"),
    # The two ml_ensemble controls already keep every base detector
    # sub-threshold, so reusing them under the orchestrator is the hardest
    # available check that noisy-OR amplification of clean traffic stays at 0.
    (healthy_repeat_pair_recovery, "ml_ensemble"),
    (healthy_bursty_latency, "ml_ensemble"),
]


def _episode_seed(episode_id: str) -> str:
    """The per-episode seed string the generator derives its rng from.

    Persisted into every fixture line (stage 3) so a single JSONL record is
    sufficient to reproduce that episode: the seed names the exact rng stream
    and the generator version in the manifest names the code that consumed it.
    """
    return f"{episode_id}/seed"


def _generator_manifest(labeled: int, healthy: int, out: Path) -> dict:
    """Record everything a replay needs to attribute output to this code.

    Best-effort commit SHA: the generator can be run from a tarball, and a
    missing SHA must not break regeneration -- it is reported as empty.
    """
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        sha = proc.stdout.strip() if proc.returncode == 0 else ""
    except (OSError, subprocess.SubprocessError):
        sha = ""
    return {
        "generator_version": GENERATOR_VERSION,
        "labeled_episodes": labeled,
        "healthy_controls": healthy,
        "total_episodes": labeled + healthy,
        "labeled_variants_per_trigger": N_LABELED_VARIANTS,
        "legacy_variants_per_trigger": LEGACY_VARIANT_COUNT,
        "trigger_count": len(LABELED_BUILDERS),
        "seed_scheme": (
            'random.Random(seed) per episode, seed = f"{episode_id}/seed"; '
            "the seed string is persisted as the record's `seed` field"
        ),
        "commit_sha": sha,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "step_index_convention": "zero-based index into the episode's events list",
        "onset_semantics": (
            "the first event that intentionally introduces the labeled failure "
            "condition; null on healthy controls"
        ),
        "files": {
            "labeled": "labeled_episodes.jsonl",
            "healthy": "healthy_controls.jsonl",
            "baseline": "goal_drift_baseline.json",
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    here = Path(__file__).resolve().parent
    parser.add_argument("--out", type=Path, default=here)
    ns = parser.parse_args(argv)

    labeled_lines: list[str] = []
    for trigger, builder in LABELED_BUILDERS.items():
        variant_name = LABELED_TRIGGER_VARIANTS.get(trigger, "standard")
        variation = LABELED_VARIATION_BUILDERS.get(trigger)
        for variant_no in range(N_LABELED_VARIANTS):
            ep = f"{trigger}-{variant_no:02d}"
            seed = _episode_seed(ep)
            rng = random.Random(seed)
            if variant_no < LEGACY_VARIANT_COUNT:
                # Variants 0..3 stay on the original builders so the Stage 1/2
                # episode bytes are reproduced exactly.
                events, onset = builder(ep, rng)
            else:
                assert variation is not None, f"no variation builder for {trigger}"
                events, onset = variation(ep, rng, variant_no)
            # mark_onset() is part of every labeled builder's contract; a
            # None here means a builder was written without one, which would
            # silently drop latency coverage for that trigger.
            assert onset is not None, f"{ep} has no marked failure onset"
            record: dict = {
                "episode_id": ep,
                "label": {"trigger": trigger, "onset_step": onset},
                "events": events,
                # Explicit persisted seed (stage 3).
                "seed": seed,
            }
            # Only variant episodes carry the field so pre-#118 lines stay
            # byte-identical across regenerations.
            if variant_name != "standard":
                record["config"] = variant_name
            labeled_lines.append(json.dumps(record, sort_keys=True))

    healthy_lines: list[str] = []
    idx = 0
    # Groups in id order: the legacy controls first so existing ids (00..40)
    # stay byte-identical across regenerations, then the stage-3 hard controls
    # grouped by the config variant they stress. carry_seed is False for the
    # legacy groups (they were written without a seed field) and True for the
    # stage-3 additions.
    healthy_groups: list[tuple[str, list, bool]] = [
        ("standard", HEALTHY_BUILDERS, False),
        ("ml_ensemble", HEALTHY_ML_BUILDERS, False),
        ("goal_drift", HEALTHY_GOAL_DRIFT_BUILDERS, False),
    ]
    hard_by_variant: dict[str, list] = {}
    for builder, variant in HEALTHY_HARD_BUILDERS:
        hard_by_variant.setdefault(variant, []).append(builder)
    for vname in ("standard", "ml_ensemble", "goal_drift"):
        if vname in hard_by_variant:
            healthy_groups.append((vname, hard_by_variant[vname], True))

    for variant, builders, carry_seed in healthy_groups:
        for builder in builders:
            ep = f"healthy-{idx:02d}"
            seed = _episode_seed(ep)
            rng = random.Random(seed)
            # Healthy controls carry no onset: no failure is injected.
            events, _onset = builder(ep, rng)
            healthy_record: dict = {"episode_id": ep, "label": None, "events": events}
            if carry_seed:
                healthy_record["seed"] = seed
            if variant != "standard":
                healthy_record["config"] = variant
            healthy_lines.append(json.dumps(healthy_record, sort_keys=True))
            idx += 1

    # --out defaults to this directory but can point anywhere; create it
    # before the first write so a fresh directory does not crash mid-run.
    ns.out.mkdir(parents=True, exist_ok=True)
    save_baseline(build_goal_drift_baseline(), str(ns.out / "goal_drift_baseline.json"))
    (ns.out / "labeled_episodes.jsonl").write_text(
        "\n".join(labeled_lines) + "\n", encoding="utf-8"
    )
    (ns.out / "healthy_controls.jsonl").write_text(
        "\n".join(healthy_lines) + "\n", encoding="utf-8"
    )
    manifest = _generator_manifest(len(labeled_lines), len(healthy_lines), ns.out)
    (ns.out / "generation_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        f"wrote {len(labeled_lines)} labeled episodes, "
        f"{len(healthy_lines)} healthy controls, the goal-drift baseline "
        f"fixture, and generation_manifest.json to {ns.out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
