"""Generate the labeled detection-accuracy fixture corpus (issue #82).

Writes two JSONL files next to this script:

* ``labeled_episodes.jsonl``  -- four episodes per shipped trigger, each line's
  ``label.trigger`` naming the failure shape injected into that trajectory.
  Since issue #118 the ``goal_drift`` and ``ml_ensemble`` episodes additionally
  carry a ``config`` envelope field naming the harness config variant they
  replay under (see benchmarks/detection_accuracy.py).
* ``healthy_controls.jsonl``  -- ``label: null`` trajectories, including
  deliberately tricky near-threshold cases (two repeats then recovery,
  warmup-window latency jitter, tool-choice entropy near 2.3 bits). The last
  few controls replay under the ``ml_ensemble`` / ``goal_drift`` variants.
* ``goal_drift_baseline.json`` -- the committed healthy BaselineProfile the
  ``goal_drift`` variant replays against (issue #118).

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
import json
import random
from pathlib import Path

from snagline.baseline import BaselineProfile, save_baseline
from snagline.events import StepEvent, make_signature

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
    """Per-episode event accumulator with deterministic arg uniquing."""

    def __init__(self, ep: str, rng: random.Random) -> None:
        self.ep = ep
        self.rng = rng
        self.events: list[dict] = []
        self._n = 0

    def add(self, tool: str, args: str, **kw) -> None:
        self.events.append(_tc(self.ep, self._n, tool, args, **kw))
        self._n += 1

    def finish_with_output(self) -> list[dict]:
        """Close the episode on a message step so silent_abort stays silent."""
        self.add("__output__", f"final-answer-{self.ep}", action_type="message")
        return self.events


def _cycle_tool(i: int) -> str:
    return TOOL_POOL[i % len(TOOL_POOL)]


# --------------------------------------------------------------------------
# Labeled failure shapes
# --------------------------------------------------------------------------


def build_loop(ep: str, rng: random.Random) -> list[dict]:
    """Three identical retry calls inside the 12-signature loop window.

    Total tool calls stays below the 20-slot meltdown window so the entropy
    statistic never evaluates; args are otherwise unique so only the deliberate
    triple matches the repeat threshold.
    """
    b = _Builder(ep, rng)
    for i in range(4):
        b.add(_cycle_tool(i), f"q={rng.randrange(10**6)}")
    # The loop: same tool, same args, hence one identical signature x3.
    for _ in range(3):
        b.add("fetch_record", "id=42")
    b.add(_cycle_tool(1), f"q={rng.randrange(10**6)}")
    return b.finish_with_output()


def build_error_cascade(ep: str, rng: random.Random) -> list[dict]:
    """Three consecutive erroring tool calls trip the consecutive rule (3)."""
    b = _Builder(ep, rng)
    for i in range(3):
        b.add(_cycle_tool(i), f"req={rng.randrange(10**6)}")
    for i in range(3):
        b.add("deploy_service", f"svc-{i}", error=True, error_type="TimeoutError")
    b.add(_cycle_tool(4), f"req={rng.randrange(10**6)}")
    return b.finish_with_output()


def build_latency_anomaly(ep: str, rng: random.Random) -> list[dict]:
    """Stable 5-call baseline freezes mu0~=400ms/sigma0=20ms, then spikes.

    Baseline [400, 395, 405, 398, 402] has std ~= 3.5, so the relative floor
    (5% of 400 = 20ms) sets sigma0. A 2500ms spike yields z = (2500-400)/20 =
    105, cusum ~= 104.5 >> h=5: instant alarm on the first post-warmup step.
    """
    b = _Builder(ep, rng)
    for ms in (400.0, 395.0, 405.0, 398.0, 402.0):
        b.add("search_web", f"q={rng.randrange(10**6)}", latency_ms=ms)
    for ms in (2500.0, 2400.0):
        b.add("search_web", f"q={rng.randrange(10**6)}", latency_ms=ms)
    return b.finish_with_output()


def build_token_runaway(ep: str, rng: random.Random) -> list[dict]:
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
    for i in range(6):
        b.add(
            _cycle_tool(20 + i),
            f"item={rng.randrange(10**6)}",
            tokens_in=2300,
            tokens_out=300,
        )
    return b.finish_with_output()


def build_budget_breach(ep: str, rng: random.Random) -> list[dict]:
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
    b.add(_cycle_tool(30), "bulk_export=huge", tokens_in=21_000, tokens_out=0)
    for i in range(2):
        b.add(
            _cycle_tool(31 + i),
            f"row={rng.randrange(10**6)}",
            tokens_in=800,
            tokens_out=200,
        )
    return b.finish_with_output()


def build_meltdown_low(ep: str, rng: random.Random) -> list[dict]:
    """Rote collapse: one tool identity, unique args, entropy -> 0 bits.

    The meltdown window reasons about tool_name, the loop detector about
    action_signature; unique args keep signatures distinct so only the
    collapse fires once the 20-call window fills (H = 0 < 0.4 bits).
    """
    b = _Builder(ep, rng)
    for i in range(22):
        b.add("scrape_page", f"page_{i}")
    return b.finish_with_output()


def build_meltdown_high(ep: str, rng: random.Random) -> list[dict]:
    """Churn: strict 12-tool cycle puts every full 20-window at ~3.52 bits."""
    b = _Builder(ep, rng)
    for i in range(24):
        b.add(CHURN_POOL[i % len(CHURN_POOL)], f"call-{i}-{rng.randrange(10**6)}")
    return b.finish_with_output()


def build_silent_abort(ep: str, rng: random.Random) -> list[dict]:
    """Clean run whose LAST step is an error-free bare tool call.

    finalize() at end_episode sees a non-output final action and fires; during
    the run nothing else can fire (fewer than 20 tool calls, no errors, no
    latency/token telemetry).
    """
    b = _Builder(ep, rng)
    for i in range(5):
        b.add(_cycle_tool(i), f"k={rng.randrange(10**6)}")
    b.add("write_report", "draft=v1")  # ends mid-work, never emits output
    return b.events  # deliberately NOT closed with a message step


# --------------------------------------------------------------------------
# New shipped triggers for #181 (stagnation, side_effect, governance, cycle,
# stall, idle_gap, wall_clock_budget)
# --------------------------------------------------------------------------


def build_stagnation(ep: str, rng: random.Random) -> list[dict]:
    """Novelty collapse: 6 signatures reused randomly for 100 steps, 0.5s gaps.

    Window 50, min_novelty 0.05, patience 2: need two consecutive windows with
    <2.5 novel. Six distinct signatures keep loop quiet (each appears ~2 per
    12 window) and meltdown entropy at log2(6)=2.58 inside (0.4, 2.8). Random
    order breaks strict periodicity so cycle does not fire. Gaps 0.5s keep
    total wall 50s <120s so wall_clock_budget stays silent while windows
    complete.
    """
    b = _Builder(ep, rng)
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
    return b.events


def build_side_effect_duplicate(ep: str, rng: random.Random) -> list[dict]:
    """Second identical side_effect=True fires side_effect_duplicate.

    Two identical (tool_name, signature) with side_effect=True. First is
    tolerated, second exceeds allowed_repeats=1. Keep total tool calls low
    and args distinct elsewhere so loop/meltdown stay quiet.
    """
    b = _Builder(ep, rng)
    for i in range(3):
        b.add(_cycle_tool(i), f"init={rng.randrange(10**6)}")
    # First charge
    b.add("charge_card", "amt=10", side_effect=True)
    b.add(_cycle_tool(3), f"mid={rng.randrange(10**6)}")
    # Duplicate charge, same signature
    b.add("charge_card", "amt=10", side_effect=True)
    b.add(_cycle_tool(4), f"tail={rng.randrange(10**6)}")
    return b.finish_with_output()


def build_governance_decay(ep: str, rng: random.Random) -> list[dict]:
    """Compaction with pinned hashes, 3 grace steps without confirmation.

    Grace 3: compaction at ordinal n, deadline n+3. After 3 steps with pins
    still unconfirmed, fires governance_decay once. Use fixed pin hashes so
    regeneration is deterministic.
    """
    import hashlib

    b = _Builder(ep, rng)
    for i in range(2):
        b.add(_cycle_tool(i), f"pre={rng.randrange(10**6)}")
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
    return b.finish_with_output()


def build_cycle(ep: str, rng: random.Random) -> list[dict]:
    """Period-6 cycle across 24 steps fires LoopDetector cycle but not loop.

    Window 12, min_period 2, max_period 6: period 6 is inside band, and 12
    holds 2 periods. Six distinct signatures cycling keeps each count at 2 per
    12 window (threshold 3), so plain loop stays silent while cycle fires.
    Entropy log2(6)=2.58 inside (0.4, 2.8) so meltdown silent. Total wall
    36s <120s, gap 1.5s, no idle.
    """
    b = _Builder(ep, rng)
    cyc = [(f"c_tool_{i}", f"c-arg-{i}") for i in range(6)]
    for i in range(24):
        tool, args = cyc[i % 6]
        b.add(tool, args)
    return b.finish_with_output()


def build_stall(ep: str, rng: random.Random) -> list[dict]:
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
    for _ in range(25):
        b.add("stalled_tool", "repeat-id-99")
    return b.finish_with_output()


def build_idle_gap(ep: str, rng: random.Random) -> list[dict]:
    """One gap >=10s between consecutive ingests fires idle_gap.

    Five steps at 1.5s gaps, then a 15s jump, then 3 more at 1.5s. Idle
    fires once per episode at the gap. Wall budget 120s not reached (total
    ~22.5s before the final message). Keep tool entropy inside band and
    signatures unique so loop/meltdown/stagnation stay silent.
    """
    b = _Builder(ep, rng)
    for i in range(5):
        b.add(_cycle_tool(i), f"q={rng.randrange(10**6)}")
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
    return b.events


def build_wall_clock_budget(ep: str, rng: random.Random) -> list[dict]:
    """Total wall >=120s fires wall_clock_budget (warning at 96s, breach at 120s).

    85 steps at 1.5s gaps = 126s wall, triggers breach once. No idle gaps.
    Keep 5-tool cycle for meltdown silence, no side_effect, no compaction.
    """
    b = _Builder(ep, rng)
    for i in range(85):
        b.add(_cycle_tool(i), f"w={rng.randrange(10**6)}")
    return b.finish_with_output()


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


def _build_goal_drift_latency_shape(ep: str, rng: random.Random) -> list[dict]:
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
    for _ in range(12):
        b.add("search_web", f"q={rng.randrange(10**6)}", latency_ms=200.0)
    return b.finish_with_output()


def _build_goal_drift_unseen_shape(ep: str, rng: random.Random) -> list[dict]:
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
    b.add("deploy_canary", f"env={rng.randrange(10**6)}", latency_ms=70.0)
    return b.finish_with_output()


def build_goal_drift(ep: str, rng: random.Random) -> list[dict]:
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
    """Post-warmup spike: cusum (2500-400)/20 - 0.5 = 104.5 > h = 5."""
    for ms in (400.0, 395.0, 405.0, 398.0, 402.0):
        b.add("search_web", f"q={b.rng.randrange(10**6)}", latency_ms=ms)
    b.add("search_web", f"q={b.rng.randrange(10**6)}", latency_ms=2500.0)


def _ml_cascade_signal(b: _Builder) -> None:
    """Three consecutive errors: consecutive score min(1, 3/3) = 1.0."""
    b.add("deploy_service", "svc-a", error=True, error_type="TimeoutError")
    b.add("deploy_service", "svc-b", error=True, error_type="TimeoutError")
    b.add("deploy_service", "svc-c", error=True, error_type="TimeoutError")


def build_ml_ensemble(ep: str, rng: random.Random) -> list[dict]:
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
        _ml_loop_signal(b)
        b.add(_cycle_tool(2), f"q={rng.randrange(10**6)}")
    elif shape == 1:
        _ml_latency_spike(b)
    elif shape == 2:
        b.add(_cycle_tool(0), f"q={rng.randrange(10**6)}")
        _ml_cascade_signal(b)
        b.add(_cycle_tool(1), f"q={rng.randrange(10**6)}")
    else:
        # Warmup samples 400-class on distinct args except two id=9 repeats;
        # the spike step is ALSO the third id=9 occurrence.
        for args, ms in (("a=1", 400.0), ("a=2", 395.0), ("a=3", 405.0)):
            b.add("search_web", args, latency_ms=ms)
        b.add("search_web", "id=9", latency_ms=398.0)
        b.add("search_web", "id=9", latency_ms=402.0)  # warmup sample n=5
        b.add("search_web", "id=9", latency_ms=2500.0)  # loop x3 + spike
    return b.finish_with_output()


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
# Healthy controls (label: null)
# --------------------------------------------------------------------------


def healthy_plain(ep: str, rng: random.Random, n_calls: int = 12) -> list[dict]:
    b = _Builder(ep, rng)
    for i in range(n_calls):
        b.add(_cycle_tool(i), f"q={rng.randrange(10**6)}")
    return b.finish_with_output()


def healthy_repeat_pair_recovery(ep: str, rng: random.Random) -> list[dict]:
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
    return b.finish_with_output()


def healthy_warmup_jitter(ep: str, rng: random.Random) -> list[dict]:
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
    return b.finish_with_output()


def healthy_warmout_wide_jitter(ep: str, rng: random.Random) -> list[dict]:
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
    return b.finish_with_output()


def healthy_near_threshold_entropy(ep: str, rng: random.Random) -> list[dict]:
    """Strict five-tool cycle: every full 20-window sits at 2.32 bits.

    That is 0.48 bits below the 2.8 thrash threshold and far above the 0.4
    collapse threshold: the documented near-threshold healthy case.
    """
    b = _Builder(ep, rng)
    for i in range(24):
        b.add(_cycle_tool(i), f"c{i}-{rng.randrange(10**6)}")
    return b.finish_with_output()


def healthy_isolated_errors(ep: str, rng: random.Random) -> list[dict]:
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
    return b.finish_with_output()


def healthy_steady_tokens(ep: str, rng: random.Random) -> list[dict]:
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
    return b.finish_with_output()


def healthy_jitter_plus_error(ep: str, rng: random.Random) -> list[dict]:
    """Warmup jitter AND one isolated error: neither signal reaches a rule."""
    b = _Builder(ep, rng)
    for ms in (380.0, 430.0, 395.0, 460.0, 370.0):
        b.add("search_web", f"m={rng.randrange(10**6)}", latency_ms=ms)
    b.add("notify_hook", "dst=ci", error=True, error_type="ConnReset")
    for ms in (440.0, 405.0):
        b.add("search_web", f"m={rng.randrange(10**6)}", latency_ms=ms)
    for i in range(3):
        b.add(_cycle_tool(i + 1), f"m={rng.randrange(10**6)}")
    return b.finish_with_output()


def healthy_moderate_jitter(ep: str, rng: random.Random) -> list[dict]:
    """Moderate warmup spread freezes sigma0~=31.9; post-freeze values give
    increments <= -0.12 against h=5. Warmup [380,420,360,440,390]:
    mu0=398, sample std ~= 31.9 (> both floors)."""
    b = _Builder(ep, rng)
    for ms in (380.0, 420.0, 360.0, 440.0, 390.0):
        b.add("search_web", f"w={rng.randrange(10**6)}", latency_ms=ms)
    for ms in (410.0, 380.0):
        b.add("search_web", f"w={rng.randrange(10**6)}", latency_ms=ms)
    b.add(_cycle_tool(3), f"w={rng.randrange(10**6)}")
    return b.finish_with_output()


def healthy_long_cycle(ep: str, rng: random.Random) -> list[dict]:
    """Thirty calls on the five-tool cycle: the meltdown window fills and
    slides repeatedly while entropy holds at 2.32 bits the whole way."""
    b = _Builder(ep, rng)
    for i in range(30):
        b.add(_cycle_tool(i), f"L={rng.randrange(10**6)}")
    return b.finish_with_output()


def healthy_eight_tool_long(ep: str, rng: random.Random) -> list[dict]:
    """Regression for #180: 30 calls cycling over eight tools, entropy ~2.97
    bits, comfortably below the 3.4 high threshold. A healthy ReAct agent
    with a broad toolbelt must stay quiet; the old 2.8 threshold fired here."""
    eight_pool = tuple(f"tool_{i}" for i in range(8))
    b = _Builder(ep, rng)
    for i in range(30):
        b.add(eight_pool[i % len(eight_pool)], f"q={rng.randrange(10**6)}")
    return b.finish_with_output()


def healthy_spaced_errors_wide(ep: str, rng: random.Random) -> list[dict]:
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
    return b.finish_with_output()


def healthy_jitter_and_tokens(ep: str, rng: random.Random) -> list[dict]:
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
    return b.finish_with_output()


def healthy_pair_plus_tokens(ep: str, rng: random.Random) -> list[dict]:
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
    return b.finish_with_output()


def _healthy_long(ep: str, rng: random.Random, n: int) -> list[dict]:
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


def healthy_goal_drift_steady(ep: str, rng: random.Random) -> list[dict]:
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
    return b.finish_with_output()


def healthy_goal_drift_near_k(ep: str, rng: random.Random) -> list[dict]:
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
    return b.finish_with_output()


# Reused under the ml_ensemble variant: both builders already keep every
# base detector sub-threshold (a pair never reaches repeat count 3; warmup
# jitter freezes into a CUSUM that cannot accumulate), so the orchestrator
# collects no scores at all and must stay silent.
HEALTHY_ML_BUILDERS = [healthy_repeat_pair_recovery, healthy_warmup_jitter]

HEALTHY_GOAL_DRIFT_BUILDERS = [healthy_goal_drift_steady, healthy_goal_drift_near_k]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    here = Path(__file__).resolve().parent
    parser.add_argument("--out", type=Path, default=here)
    ns = parser.parse_args(argv)

    labeled_lines: list[str] = []
    for trigger, builder in LABELED_BUILDERS.items():
        variant = LABELED_TRIGGER_VARIANTS.get(trigger, "standard")
        for variant_no in range(4):
            ep = f"{trigger}-{variant_no:02d}"
            rng = random.Random(f"{ep}/seed")
            events = builder(ep, rng)
            record: dict = {
                "episode_id": ep,
                "label": {"trigger": trigger},
                "events": events,
            }
            # Only variant episodes carry the field so pre-#118 lines stay
            # byte-identical across regenerations.
            if variant != "standard":
                record["config"] = variant
            labeled_lines.append(json.dumps(record, sort_keys=True))

    healthy_lines: list[str] = []
    idx = 0
    # Groups in id order: standard controls first so existing ids stay
    # stable, then the issue-#118 variant groups appended after them.
    healthy_groups: list[tuple[str, list]] = [
        ("standard", HEALTHY_BUILDERS),
        ("ml_ensemble", HEALTHY_ML_BUILDERS),
        ("goal_drift", HEALTHY_GOAL_DRIFT_BUILDERS),
    ]
    for variant, builders in healthy_groups:
        for builder in builders:
            ep = f"healthy-{idx:02d}"
            rng = random.Random(f"{ep}/seed")
            events = builder(ep, rng)
            record: dict = {"episode_id": ep, "label": None, "events": events}
            if variant != "standard":
                record["config"] = variant
            healthy_lines.append(json.dumps(record, sort_keys=True))
            idx += 1

    save_baseline(build_goal_drift_baseline(), str(ns.out / "goal_drift_baseline.json"))
    (ns.out / "labeled_episodes.jsonl").write_text(
        "\n".join(labeled_lines) + "\n", encoding="utf-8"
    )
    (ns.out / "healthy_controls.jsonl").write_text(
        "\n".join(healthy_lines) + "\n", encoding="utf-8"
    )
    print(
        f"wrote {len(labeled_lines)} labeled episodes, "
        f"{len(healthy_lines)} healthy controls, and the goal-drift "
        f"baseline fixture to {ns.out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
