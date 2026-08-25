"""Generate the labeled detection-accuracy fixture corpus (issue #82).

Writes two JSONL files next to this script:

* ``labeled_episodes.jsonl``  -- four episodes per shipped trigger, each line's
  ``label.trigger`` naming the failure shape injected into that trajectory.
* ``healthy_controls.jsonl``  -- 30 ``label: null`` trajectories, including
  deliberately tricky near-threshold cases (two repeats then recovery,
  warmup-window latency jitter, tool-choice entropy near 2.3 bits).

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

from snagline.events import make_signature

# Fixed epoch so committed timestamps never change between regenerations.
BASE_TS = 1_700_000_000.0
STEP_GAP_S = 1.5

# Five-tool pool used wherever healthy near-uniform entropy is wanted: a
# strict cycle over five identities yields log2(5) ~= 2.32 bits per full
# 20-call meltdown window, comfortably inside the silent band (0.4, 2.8).
TOOL_POOL = ("search_web", "read_file", "run_sql", "call_api", "summarize")

# Twelve-tool pool for meltdown_high churn: any full 20-window of a strict
# 12-cycle holds eight identities twice and four once, giving H ~= 3.52
# bits, safely above the 2.8-bit thrash threshold.
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
) -> dict:
    """Build one StepEvent-compatible dict (structure only, no content)."""
    return {
        "step_id": f"{ep}-s{i}",
        "episode_id": ep,
        "timestamp": BASE_TS + i * STEP_GAP_S,
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


LABELED_BUILDERS = {
    "loop": build_loop,
    "error_cascade": build_error_cascade,
    "latency_anomaly": build_latency_anomaly,
    "token_runaway": build_token_runaway,
    "budget_breach": build_budget_breach,
    "meltdown_low": build_meltdown_low,
    "meltdown_high": build_meltdown_high,
    "silent_abort": build_silent_abort,
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

    Warmup [350,480,390,460,370]: mu0=410, sample std ~= 93.5 (> floors);
    post-freeze [420,400,430] all produce negative increments.
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
]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    here = Path(__file__).resolve().parent
    parser.add_argument("--out", type=Path, default=here)
    ns = parser.parse_args(argv)

    labeled_lines: list[str] = []
    for trigger, builder in LABELED_BUILDERS.items():
        for variant in range(4):
            ep = f"{trigger}-{variant:02d}"
            rng = random.Random(f"{ep}/seed")
            events = builder(ep, rng)
            labeled_lines.append(
                json.dumps(
                    {
                        "episode_id": ep,
                        "label": {"trigger": trigger},
                        "events": events,
                    },
                    sort_keys=True,
                )
            )

    healthy_lines: list[str] = []
    for idx, builder in enumerate(HEALTHY_BUILDERS):
        ep = f"healthy-{idx:02d}"
        rng = random.Random(f"{ep}/seed")
        events = builder(ep, rng)
        healthy_lines.append(
            json.dumps(
                {"episode_id": ep, "label": None, "events": events},
                sort_keys=True,
            )
        )

    (ns.out / "labeled_episodes.jsonl").write_text(
        "\n".join(labeled_lines) + "\n", encoding="utf-8"
    )
    (ns.out / "healthy_controls.jsonl").write_text(
        "\n".join(healthy_lines) + "\n", encoding="utf-8"
    )
    print(
        f"wrote {len(labeled_lines)} labeled episodes and "
        f"{len(healthy_lines)} healthy controls to {ns.out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
