"""Read-only audit probes for the loop/stagnation/meltdown family."""

import sys

sys.path.insert(0, "src")

from snagline.config import Config
from snagline.detectors.loop import LoopDetector
from snagline.detectors.meltdown import MeltdownDetector
from snagline.detectors.stagnation import StagnationDetector
from snagline.events import StepEvent

T0 = 1000.0


def ev(ep, i, sig, tool="tool", atype="tool_call", ts=None):
    return StepEvent(
        episode_id=ep,
        step_id=f"{ep}-{i}",
        timestamp=T0 + i if ts is None else ts,
        action_type=atype,
        action_signature=sig,
        tool_name=tool,
        latency_ms=10,
        error=False,
        error_type=None,
        tokens_in=1,
        tokens_out=1,
        side_effect=None,
        metadata={},
    )


def drain(det, ep, sigs, **kw):
    out = []
    for i, s in enumerate(sigs):
        r = det.observe(ev(ep, i, s, **kw))
        if r is not None:
            out.append((i, r.trigger, r.score, r.detail))
    return out


def section(t):
    print("\n" + "=" * 72 + "\n" + t + "\n" + "=" * 72)


# ---------------------------------------------------------------- 1. stagnation
section("1. STAGNATION EPISODE: novelty collapse with all loop modes ON")
cfg = Config(
    stagnation_enabled=True,
    loop_stall_enabled=True,
    loop_cycle_enabled=True,
    loop_near_duplicate_enabled=True,
    meltdown_enabled=True,
)
st = StagnationDetector(config=cfg)
lp = LoopDetector(config=cfg)
# 60 steps: 3 distinct signatures cycling, then 60 repeats of one signature.
sigs = [f"a{i % 3}" for i in range(60)] + ["zzz"] * 60
r_st = drain(st, "ep1", sigs)
r_lp = drain(lp, "ep1", sigs)
print("stagnation fires:", [(i, tr, sc) for i, tr, sc, _ in r_st])
print("loop-family fires:", [(i, tr, sc) for i, tr, sc, _ in r_lp])

# ---------------------------------------------------------------- 2. stall
section("2. STALL EPISODE: 40 identical tool_call steps")
cfg2 = Config(
    loop_stall_enabled=True,
    loop_cycle_enabled=True,
    loop_near_duplicate_enabled=True,
    meltdown_enabled=True,
)
lp2 = LoopDetector(config=cfg2)
md2 = MeltdownDetector(config=cfg2)
sigs2 = ["same"] * 40
r_lp2 = drain(lp2, "ep2", sigs2, tool="search")
r_md2 = drain(md2, "ep2", sigs2, tool="search")
print("loop-family fires:", [(i, tr, sc) for i, tr, sc, _ in r_lp2])
print("meltdown fires    :", [(i, tr, sc, d) for i, tr, sc, d in r_md2])

# ---------------------------------------------------------------- 3. cycle
section("3. CYCLE EPISODE: A,B,A,B,... period 2, 30 steps")
cfg3 = Config(
    loop_stall_enabled=True,
    loop_cycle_enabled=True,
    loop_near_duplicate_enabled=True,
)
lp3 = LoopDetector(config=cfg3)
sigs3 = ["A", "B"] * 30
r_lp3 = drain(lp3, "ep3", sigs3)
print("loop-family fires:", [(i, tr, sc) for i, tr, sc, _ in r_lp3])
print("max count of A in 12-window = 6; repeat_threshold default =",
      Config().loop_repeat_threshold)

# ---------------------------------------------------------------- 3b. cycle period 6
section("3b. CYCLE EPISODE: period-6 cycle, 30 steps (A..F)")
lp3b = LoopDetector(config=cfg3)
sigs3b = ["A", "B", "C", "D", "E", "F"] * 30
print("loop-family fires:", [(i, tr, sc) for i, tr, sc, _ in drain(lp3b, "ep3b", sigs3b)])

# ---------------------------------------------------------------- 4. invalid config
section("4. DIRECT CONSTRUCTION WITH INVALID CONFIG")


def try_con(label, fn):
    try:
        fn()
        print(f"  {label}: ACCEPTED (no error)")
    except Exception as e:
        print(f"  {label}: rejected -> {type(e).__name__}: {e}")


try_con("stagnation min_novelty=0.0",
        lambda: StagnationDetector(min_novelty=0.0))
try_con("stagnation min_novelty=1.5",
        lambda: StagnationDetector(min_novelty=1.5))
try_con("stagnation window_size=0",
        lambda: StagnationDetector(window_size=0))
try_con("stagnation patience=0",
        lambda: StagnationDetector(patience=0))
try_con("loop window_size=0",
        lambda: LoopDetector(window_size=0))
try_con("loop repeat_threshold=0",
        lambda: LoopDetector(repeat_threshold=0))
try_con("loop repeat_threshold=-5",
        lambda: LoopDetector(repeat_threshold=-5))
try_con("loop stall_steps=0",
        lambda: LoopDetector(config=Config(loop_stall_enabled=True,
                                           loop_stall_steps=0)))
try_con("loop stall_steps=-1",
        lambda: LoopDetector(config=Config(loop_stall_enabled=True,
                                           loop_stall_steps=-1)))
try_con("loop cycle window < 2*min_period",
        lambda: LoopDetector(config=Config(loop_cycle_enabled=True,
                                           loop_cycle_window_size=3,
                                           loop_cycle_min_period=2)))
try_con("loop cycle max_period < min_period",
        lambda: LoopDetector(config=Config(loop_cycle_enabled=True,
                                           loop_cycle_min_period=5,
                                           loop_cycle_max_period=2)))
try_con("loop cycle window_size=0",
        lambda: LoopDetector(config=Config(loop_cycle_enabled=True,
                                           loop_cycle_window_size=0)))
try_con("meltdown low>=high",
        lambda: MeltdownDetector(low_entropy=3.0, high_entropy=1.0))
try_con("meltdown window_size=2",
        lambda: MeltdownDetector(window_size=2))
try_con("meltdown low_entropy=-1.0",
        lambda: MeltdownDetector(low_entropy=-1.0, high_entropy=1.0))
try_con("meltdown low_entropy=inf",
        lambda: MeltdownDetector(low_entropy=float("inf"), high_entropy=9.0))
try_con("meltdown rearm_steps=0",
        lambda: MeltdownDetector(rearm_steps=0))

# ---------------------------------------------------------------- 5. reset
section("5. STATE RESET BETWEEN EPISODES (same episode id reused)")
lp5 = LoopDetector(config=cfg2)
r = drain(lp5, "ep5", ["x"] * 40)
print("episode A fires:", [(i, tr) for i, tr, _, _ in r])
lp5.reset("ep5")
r = drain(lp5, "ep5", ["x"] * 40)
print("after reset, same sigs fires:", [(i, tr) for i, tr, _, _ in r])

md5 = MeltdownDetector(config=cfg2)
r = drain(md5, "ep5", ["y"] * 30, tool="t")
print("meltdown ep A fires:", [(i, tr) for i, tr, _, _ in r])
md5.reset("ep5")
r = drain(md5, "ep5", ["y"] * 30, tool="t")
print("after reset, meltdown fires:", [(i, tr) for i, tr, _, _ in r])

st5 = StagnationDetector(config=cfg)
drain(st5, "ep5", ["q"] * 60)
st5.reset("ep5")
r = drain(st5, "ep5", ["q"] * 60)
print("after reset, stagnation fires:", [(i, tr) for i, tr, _, _ in r])
print("residual state keys after reset:",
      {k: v for k, v in vars(st5).items() if k.startswith("_") and v})

# ---------------------------------------------------------------- 6. stall=0 behaviour
section("6. loop_stall_steps=0 : what does observe do?")
lp6 = LoopDetector(config=Config(loop_stall_enabled=True, loop_stall_steps=0))
r = drain(lp6, "ep6", ["a", "b", "c"])
print("fires on 3 distinct steps:", [(i, tr, sc) for i, tr, sc, _ in r])

# ---------------------------------------------------------------- 7. scaling
section("7. STAGNATION with window_size > window_scale_steps")
for base, steps in ((50, 10), (50, 50), (12, 100), (50, 100)):
    c = Config(stagnation_enabled=True, window_scale_steps=steps, max_window=512)
    d = StagnationDetector(window_size=base, min_novelty=0.5, patience=1, config=c)
    rr = drain(d, "ep7", ["uniq-%d" % i for i in range(4)] + ["rep"] * 400)
    print(f"  base={base:3d} scale_steps={steps:3d}: fires={len(rr)}",
          rr[:2] if rr else "")
