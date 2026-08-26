# Writing a SNAGLINE detector

A detector is any object satisfying the `Detector` protocol
(`src/snagline/detectors/base.py`). It sees every `StepEvent`, keeps whatever
per-episode state it needs, and returns a `FailureRisk` (or `None`) from each
`observe()` call. That's the whole contract.

```python
from snagline.events import StepEvent
from snagline.risk import FailureRisk


class MyDetector:
    name = "my_detector"

    def __init__(self, config=None):
        self._state = {}  # keyed by episode_id -- a Monitor is shared across episodes

    def observe(self, event: StepEvent) -> FailureRisk | None:
        ...  # update state; return a FailureRisk when tripped, else None

    def reset(self, episode_id: str) -> None:
        self._state.pop(episode_id, None)  # called on Monitor.end_episode()
```

Register it by constructing the Monitor yourself:

```python
from snagline import Monitor
monitor = Monitor.default()
monitor._detectors.append(MyDetector())   # or: Monitor(default_detectors + [mine], [sink])
```

(For a first-class API, pass a full detector list to `Monitor(...)`  -
`Monitor.default()` is just a convenience preset.)

## Rules (from project.md §1 - treat as constraints)

1. **O(1) amortized per step, no network, no LLM calls.** `ingest()` runs on
   every step of a week-long run; anything slower than ~100 µs is a bug.
2. **Never read `StepEvent.metadata`.** Detectors reason on hashes, timings,
   counts, and booleans only. This is the privacy property adopters rely on.
3. **Never raise on bad input.** The Monitor is fail-open *by default* and will
   swallow the exception, but a detector that throws on malformed fields is
   effectively disabled from then on. Validate defensively.

   Note: fail-open is the default but not unconditional. `Monitor(...,
   fail_open=False)` makes detector and sink exceptions propagate (this is how
   the test suite asserts the fire-and-forget contract). Treat fail-open as the
   production-safe default; flip it off only in tests or in callers that want
   strict error reporting.
4. **Key all state by `episode_id`** and clear it in `reset()`. A single
   Monitor can watch several episodes concurrently.
5. **Prefer no false positives on healthy traffic.** A noisy detector gets
   uninstalled; look at the healthy fixtures in `tests/fixtures/trajectories/`
   as the bar (your detector must stay silent on them - see how the existing
   detector tests assert both detection *and* silence).

## Worked reference implementations

- `detectors/loop.py` - sliding-window repetition count on `action_signature`
- `detectors/error_cascade.py` - consecutive and windowed error counts
- `detectors/latency_anomaly.py` - Welford baseline + CUSUM deviation per tool
- `detectors/stagnation.py` - all-time novelty set plus a sliding novelty share
- `detectors/side_effect_guard.py` - per-episode duplicate count of
  host-declared non-idempotent actions
- `detectors/goal_drift.py` - compares a live run to a persisted `BaselineProfile` (structural error rate, latency, tool set)
- `drift/goal_drift.py` - semantic embedding centroid drift vs the same profile (`snagline[drift]`, issue #81)
- `detectors/ml_ensemble.py` - `MLOrchestrator` combining base detector scores
- `detectors/token_runaway.py` - token-volume CUSUM + per-episode budget envelope
- `detectors/meltdown.py` - sliding-window entropy collapse/thrash detection
- `detectors/silent_abort.py` - end-of-episode completion check via `finalize`

Each has a matching test in `tests/detectors/` showing the synthetic-sequence
pattern (injected failure fires; healthy sequence stays silent). Copy that
test shape for your own detector.

### Loop hardening modes (issue #89)

`LoopDetector` has three opt-in extensions for failure shapes that plain
repetition counting misses. All are off by default: with stock config the
detector's behavior is exactly the plain path above.

- **Near-duplicate** (`Config(loop_near_duplicate_enabled=True)`): retries
  whose signatures differ only by volatile identifiers (uuid-shaped
  substrings, digit runs) collapse onto one normalized key before hashing,
  then feed the same window and threshold logic as the plain path. Trigger:
  `near_duplicate_loop`. The default normalizer is a documented heuristic
  (uuid-like substrings become one token, remaining digit runs become `#`);
  it is deliberately blunt, and against opaque hex digests collapsing digits
  raises collision odds, so treat enabling it as a deliberate choice. Swap in
  your own strategy with `LoopDetector(config=cfg, normalizer=fn)`, where
  `fn` maps a signature string to its normalized form.
- **Cycle** (`Config(loop_cycle_enabled=True)`): A,B,A,B,... periodicity that
  never repeats one action often enough to trip `repeat_threshold`. After
  each step an ascending scan (O(window)) finds the window content's minimal
  period p; it fires once when p lies inside the configured band
  (`loop_cycle_min_period` to `loop_cycle_max_period`, default 2..6) and the
  recent `loop_cycle_window_size` window holds at least two full periods,
  then re-arms when periodicity breaks. Filtering on the true minimum means a
  custom band genuinely suppresses faster loops instead of re-flagging them
  through a multiple, and uniform repetition (minimal period 1) is always
  ignored: single-action loops belong to the plain loop/stall modes.
  Trigger: `cycle`.
- **Stall** (`Config(loop_stall_enabled=True)`): N consecutive identical
  signatures with no progress fires after `loop_stall_steps` (default 25).
  Wall-clock deltas never reset the streak: zero-delta steps count toward it
  (a frozen clock is itself evidence of a stall), and positive deltas do not
  reset it either (tight retries burn real time while going nowhere); only a
  different signature restarts the count. Trigger: `stall`.

The trigger strings `near_duplicate_loop`, `cycle`, and `stall` are API:
downstream policy layers map them by name. When several shapes fire on the
same step the plain-loop risk keeps precedence; each mode still advances its
state every step, so nothing is lost.

## Optional detectors: baseline, goal-drift, ensemble, and horizon set

`LoopDetector`, `ErrorCascadeDetector`, and `LatencyAnomalyDetector` ship in
`Monitor.default()`. The detectors below are opt-in behind config flags so the
zero-dependency preset (and its published bench numbers) are unchanged.

### `BaselineProfile` and the `baseline` command

`src/snagline/baseline.py` fits a `BaselineProfile` (per-tool latency
mean/std/min/max and error rate) from a JSONL trajectory. The `snagline
baseline <trajectory> [--output baseline.json]` CLI command persists one, and
`load_baseline` / `save_baseline` round-trip it.

### `GoalDriftDetector`

`GoalDriftDetector(baseline=profile, config=cfg)` flags a live run that
diverges from that healthy reference: rising error rate (beyond
`goal_drift_error_tolerance`), latency blowing past the healthy mean by
`goal_drift_latency_k` sigmas, or tools that never appeared in the baseline. A
zero-variance baseline uses a floored spread (5% of mean, min 1 ms) so tiny
deviations are not treated as infinite z. Enable it with
`Config(goal_drift_enabled=True, goal_drift_baseline=profile)`.

### `SemanticGoalDriftDetector` (`semantic_drift_enabled=True`, `snagline[drift]`, issue #81)

`SemanticGoalDriftDetector(baseline=profile, config=cfg, embedder=None)` is
the optional embedding counterpart to `GoalDriftDetector`. It compares the
running centroid of embedded structural labels against the persisted
`BaselineProfile.embedding_centroid` built by `fit_semantic_baseline`
(`drift/goal_drift.py`).

* **What is embedded:** a short label built from `action_type`,
  `tool_name`, and `error_type` only. Prompt or response content and
  `metadata` are never read, never logged, and never persisted. The only
  stored artifact is an averaged vector of floats inside the baseline JSON.
* **How it decides:** cosine similarity between the live running centroid
  and the healthy reference gives a deviation `1 - cos`. Values within
  `semantic_drift_tolerance` (default 0.3) are treated as noise. Above it,
  a CUSUM accumulates `signal - cusum_k` and fires a `goal_drift` risk
  when the debt reaches `semantic_drift_cusum_h` (defaults `k=0.05`,
  `h=0.5`). After firing the accumulator re-arms, so persistent drift
  re-alarms later. Fewer than `semantic_drift_min_samples` (default 10)
  live steps never fire.
* **Extra and laziness:** `pip install snagline-agent[drift]`
  (`sentence-transformers`). The package imports without the extra; the
  transformer is loaded lazily, exactly once, on first use. An injectable
  `embedder=callable(StepEvent) -> Sequence[float]` bypasses the heavy
  dependency entirely and is the path the test suite uses, so no torch is
  needed to run the tests. Pass `model_loader` to control loading if you
  embed elsewhere.
* **Fail-open and privacy:** model load failures, inference exceptions,
  degenerate or dimension-mismatched embeddings, and a baseline without an
  `embedding_centroid` all leave the detector permanently inert, logged at
  most once, never crashing or stalling the host. Dimension is checked on
  the first scored step; the detector latches off after a mismatch.
* **Wiring:** append to the Monitor's base list. Under `Monitor.default()`
  a semantically fitted profile plus `Config(semantic_drift_enabled=True,
  goal_drift_baseline=profile)` adds it to the base set. With
  `ml_ensemble_enabled` as well it joins the same noisy-OR `MLOrchestrator`
  that already wraps the ESN ensemble (issue #80), so its `goal_drift`
  trigger becomes `ml_ensemble` through the ensemble.
* **Snapshot:** implements `dump_state` / `load_state` (per-episode running
  sum, count, and CUSUM debt) and participates in `Monitor.snapshot` /
  `restore` like any stateful detector.

Fit a baseline that carries semantics from the same healthy trajectory:

```python
from snagline.drift.goal_drift import fit_semantic_baseline
profile = fit_semantic_baseline(healthy_events, embedder, model="all-MiniLM-L6-v2")
save_baseline(profile, "baseline.json")  # JSON now carries embedding_centroid
```

Or build one structurally with `snagline baseline` and without semantics;
the detector then stays inert (the intended default) until you give it a
reference it can compare against.

### Auto-calibrated thresholds (`calibration="auto"`, issue #101)

`Config(calibration="auto", calibration_baseline=profile)` (or
`calibration_baseline_path="baseline.json"`) retunes the tier-1 detectors from
the healthy profile instead of using worst-case hand-tuned constants:

* Error-cascade thresholds become the smallest windowed/consecutive error
  counts whose exceedance probability under the deployment's observed error
  rate stays within `calibration_alpha` (default 0.001). The planning rate is
  max(pooled rate, p99 of per-tool rates); tools with fewer than 20 samples
  are excluded from the percentile.
* The latency/CUSUM detector starts frozen at each tool's healthy mean and
  floored spread when the profile holds at least `cusum_min_samples` samples
  for it: episodes shorter than the old warm-up are monitorable from their
  first step.

Safety rails: derived counts clamp into `[2, hand-tuned default]` so auto can
only ever be more sensitive than shipped behavior, and with no usable baseline
every default stays exactly as today. Baseline loading and derivation are
fail-open: any problem logs one warning and falls back. The loop detector
keeps hand-tuned thresholds because profiles deliberately store no
action-repetition evidence (no content, project.md §1.4).

### `MLOrchestrator`

`MLOrchestrator(detectors, config, model=None)` combines base detector scores
into one stronger risk. The default combiner is a transparent noisy-OR
(`1 - prod(1 - score_i)`), which boosts confidence when multiple independent
detectors agree. Pass `model=callable(scores) -> float` (e.g. a fitted
scikit-learn pipeline from the `ml` extra) to replace the combiner. Enable it
with `Config(ml_ensemble_enabled=True)`; `Monitor.default()` then wraps the
base detectors in one orchestrator so there is no double counting. The
orchestrator forwards `finalize`, `dump_state`, and `load_state` to its bases,
so orchestrated completion checks and snapshots keep working.

### `StagnationDetector` (issue #87)

`StagnationDetector(config=cfg)` asks a question the loop detector cannot:
not "is it repeating?" but "is it discovering anything?". It tracks the share
of never-before-seen `action_signature` values in the last
`stagnation_window_size` steps (default 50). When that novelty share stays
below `stagnation_min_novelty` (default 0.05) for `stagnation_patience`
(default 2) consecutive full-window observations, it emits exactly one risk
(score 0.6, trigger `stagnation`). Novelty recovering resets the stale
counter, so a later collapse fires again: one alert per stagnation period,
never one per step.

The two detectors are complementary by construction. Near-duplicate actions
with slightly varied arguments produce fresh signatures each time and evade
exact-match loop windows entirely; the all-time novelty set still collapses
because the agent keeps drawing from the same small template space. The tests
in `tests/detectors/test_stagnation.py` prove a sequence that trips
stagnation while the loop detector stays silent.

Memory, stated honestly: the per-episode `seen_all_time` set grows
monotonically by design and only `reset(episode_id)` (called by
`Monitor.end_episode()`) releases it. Signatures are full 64-character hex
digests since issue #15; measured on CPython 3.14, retaining one costs about
105 bytes plus roughly 70 bytes of amortized set overhead, so budget around
175 bytes per unique action: a pathological all-unique 100k-step episode
holds on the order of 17 MB. Typical episodes repeat heavily and cost far
less. The sliding counter itself is bounded (`window_size` booleans per
episode).

Enable it with `Config(stagnation_enabled=True)`. It ships opt-in so the
zero-dependency preset and the published bench numbers are untouched.

### `SideEffectGuardDetector` (`side_effect_guard_enabled=True`, issue #88)

`SideEffectGuardDetector(config=cfg)` watches steps the *host* marked as
non-idempotent (`StepEvent.side_effect=True`: a payment, a send, a deploy)
and fires on the second identical occurrence within one episode: same
`(tool_name, action_signature)` pair counted per `episode_id`. With the
default `side_effect_allowed_repeats=1` a repeated charge escalates
immediately at score 0.9, which routes as critical severity. Trigger:
`side_effect_duplicate`. That string is API: CONTINUUM's policy table maps
it to ABORT plus immediate reconcile.

Deliberately stricter than `LoopDetector`, and different in three ways:

1. **Scope:** the loop detector watches a sliding window of arbitrary
   signatures for wasted-work repetition; this guard watches only
   host-declared side-effect steps, where repetition is itself the incident.
2. **Threshold:** the loop detector needs `repeat_threshold` hits (default 3)
   inside its window before saying anything; here one repeat is already the
   finding.
3. **Edge:** the loop detector re-arms when its window drains; this guard
   fires exactly once per `(episode_id, tool_name, action_signature)` key and
   stays quiet for the rest of the episode. A repeated payment must not
   alert-spam while the agent keeps making it worse; recovery is
   `end_episode()` territory. Different-argument retries produce different
   signatures and never fire here (that shape belongs to the loop/stall
   modes).

How hosts mark steps: adapters forward the flag mechanically and never
invent it. Today you can pass it to the raw adapter's `step(...,
side_effect=True)`, `observe_openai_call(...)`, `observe_anthropic_call(...)`,
or `observe_crewai_step(...)`. Framework callback paths (LangChain hooks,
Autogen events, LangGraph nodes, Claude Code hook payloads) have no
caller-supplied flag source, so their events carry the schema default
(`False`) and legacy payloads written before #88 load unchanged.

Privacy and cost, as everywhere: the detector reads the boolean, the tool
name, and the one-way signature digest, nothing else, and never touches
`StepEvent.metadata`. Non-marked steps cost one attribute check; marked
steps do one dict upsert. Per-episode memory grows with distinct marked
actions, never with repeats: replaying one charge a thousand times still
costs a single counter entry, and `reset(episode_id)` releases all of it.

## The horizon detectors (long-run set, issues #84/#85/#86)

Three opt-in detectors aimed at multi-day episodes. All are stdlib-only and
O(1) amortized per step.

### `TokenRunawayDetector` (`token_runaway_enabled=True`)

Sustained-burn CUSUM over per-step token volume (`tokens_in + tokens_out`),
plus an optional hard envelope: one warning at
`token_budget_warn_fraction` (default 80%) of `episode_token_budget` and a
single critical `budget_breach` risk at 100%. Trigger names (`token_runaway`,
`budget_breach`) are API: downstream policy layers map them by string.

### `MeltdownDetector` (`meltdown_enabled=True`)

Sliding-window Shannon entropy over tool-call identities, flagging both
collapse shapes documented for long-horizon agents (arXiv:2603.29231):
rote collapse below `meltdown_low_entropy` bits and churn above
`meltdown_high_entropy` bits. Thresholds were tuned against fixtures so
healthy five-tool alternation (~2.32 bits) stays silent.

### `SilentAbortDetector` (`silent_abort_enabled=True`)

The completion check from arXiv:2608.02464: evaluated once at
`Monitor.end_episode()`, it fires when an episode's last step was an
error-free bare tool call instead of an output step. To judge at episode end,
a detector implements the duck-typed `finalize(episode_id)` method
(`detectors.base.EpisodeFinalizer`); `end_episode` discovers it by attribute,
so ordinary detectors are unaffected and fail-open applies as always.

### `CompactionTripwireDetector` (`compaction_tripwire_enabled=True`, issue #90)

Governance-decay detection across context compactions. Motivation comes from
arXiv:2606.22528, which studies how compaction (summarization, truncation,
eviction) can silently drop governance constraints from an agent's context
and measurably raises policy-violation rates afterwards. Their finding is
cited as motivation only; SNAGLINE claims no benchmark numbers of its own
for this detector.

The contract uses two adapter-defined action types that core otherwise treats
opaquely:

    step("compaction", metadata={"pinned": ["<sha256>", ...]})  # just compacted
    step("constraint_present", metadata={"pin": "<sha256>"})     # pin re-seen

After a `compaction` event the host has `compaction_tripwire_grace_steps`
(default 3) subsequent events to re-confirm every pinned hash. Pins still
unconfirmed at the deadline produce exactly one `FailureRisk(score=0.9,
trigger="governance_decay")` naming 16-hex prefixes of the missing pins; a
later compaction replaces the pending set and restarts the grace window.

Privacy: hashes only. The adapter hashes canonical constraint text itself,
so constraint text never reaches snagline. This detector is also the ONE
documented exception to "detectors never read metadata": it reads exactly
two keys (`pinned` on compaction events, `pin` on confirmations) and nothing
else.

Honest limits: the tripwire is inert by design wherever the host offers no
compaction hooks or no way to observe constraint presence. See ADAPTER_GUIDE
and FRAMEWORK_BRIDGES for what adapters must do, and what they honestly
cannot.

## Restart-survivable state: snapshot/restore (issue #91)

Detectors that implement the duck-typed `dump_state()` / `load_state(state)`
pair (`detectors.base.StatefulDetector`; JSON-compatible data only, never
pickle) participate in `Monitor.snapshot(path)` / `Monitor.restore(path)`.
Snapshots are written atomically (tmp + `os.replace`). Restore is a
setup-time operation: a version or strict-composition mismatch raises rather
than failing open -- a monitor that silently starts blank is exactly the quiet
misbehavior this exists to prevent. All shipped detectors implement it, and
`DedupSink` persists cooldowns when used with its default key function.
