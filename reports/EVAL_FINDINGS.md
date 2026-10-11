# SNAGLINE Detection Quality — Evaluation Findings

Stage 3 evaluation. Stage 2 baseline commit: `f91b1fd934f20f776778b3ce1b8422d2c2a28bc0`.

Every claim below points at a committed fixture, a test, a benchmark artifact, a source
path, or a measured metric. Nothing in this stage changed detector behaviour, thresholds,
or alerting (see section 9.4 on what *was* changed, which is evaluation surface only).

How to reproduce the headline numbers:

```bash
.venv/Scripts/python.exe benchmarks/fixtures/generate_fixtures.py --out /tmp/gen
.venv/Scripts/python.exe benchmarks/detection_accuracy.py \
    --fixtures benchmarks/fixtures --format table --report-dir reports/results
.venv/Scripts/python.exe benchmarks/semantic_drift_eval.py
.venv/Scripts/python.exe -m pytest
```

---

## 1. Dataset

| | Stage 2 | Stage 3 |
|---|---|---|
| total episodes | 109 | **230** |
| labeled | 68 | **170** |
| healthy controls | 41 | **60** |
| episodes per trigger | 4 | **10** (4 legacy + 6 varied) |
| triggers covered | 17 | 17 (unchanged) |

Generator: `benchmarks/fixtures/generate_fixtures.py`, `GENERATOR_VERSION = "3.0"` (line 69).

Determinism is a hard property, and it is now verified by a test rather than assumed:

- Seeds are derived purely from the episode id — `_episode_seed()` (`generate_fixtures.py:1938`)
  returns `f"{episode_id}/seed"` — and every episode is built with a fresh
  `random.Random(seed)` (`generate_fixtures.py:2005-2006`, `2053-2054`). There is no
  `import time`, `import os`, or `import uuid` in the generator; timestamps come from a
  fixed epoch (`BASE_TS = 1_700_000_000.0`, line 82). No `set` iteration, no
  `sorted()` instability, no `random.*` global draws.
- The seed is **persisted into the fixture**: all 170 labeled records carry a `seed`
  field, and so do the stage-3 hard controls (`healthy-41`..`healthy-59`); the 41 legacy
  controls deliberately omit it so their bytes stay identical.
  `tests/test_fixture_generation.py::test_every_labeled_episode_persists_its_seed_and_onset`.
- Two independent runs are **byte-identical**, and identical to the committed corpus:
  `tests/test_fixture_generation.py::test_regeneration_is_byte_identical_to_the_committed_corpus`
  and `..._to_a_second_run`. Verified also under `PYTHONHASHSEED=7` (string seeding goes
  through `sha512`, so it is hash-randomization-free).
- Each builder is a pure function of `(episode_id, rng)`:
  `test_builder_output_is_stable_across_invocations`.

One caveat, stated plainly: `generation_manifest.json` is **not** byte-reproducible. It
records `commit_sha`, `python_version` and `platform` (`generate_fixtures.py:1977-1979`),
so it changes across machines, Python versions and commits. The three data files
(`labeled_episodes.jsonl`, `healthy_controls.jsonl`, `goal_drift_baseline.json`) are
byte-stable; the manifest is provenance, not data. Recorded in
`test_manifest_records_the_corpus_provenance`.

A bug found while verifying this: `--out` crashed on a directory that did not yet exist,
because the first write preceded any `mkdir`. Fixed at `generate_fixtures.py:2065`
(`ns.out.mkdir(parents=True, exist_ok=True)`).

Variation across the 10 episodes per trigger is real but shallower than the parameter
list suggests. 14 of the 17 `vary_*` builders accept a variant number `v` and **never
read it** — an AST check pins the set to exactly
`{vary_governance_decay, vary_goal_drift, vary_ml_ensemble}`
(`tests/test_fixture_generation.py::test_variation_axes_documented_and_used`). All
variation therefore comes from the episode-seeded rng, not from a swept axis. The
committed episodes are nonetheless pairwise distinct in structure:
`test_committed_variants_are_pairwise_distinct` pins the measured distinct-shape counts
per trigger (`stagnation` 10, `goal_drift` 8, `budget_breach`/`error_cascade`/
`latency_anomaly`/`loop`/`silent_abort`/`token_runaway`/`wall_clock_budget` 7, …,
`cycle`/`meltdown_high` 4).

Measured variation per trigger (committed corpus):

| trigger | distinct lengths | distinct onsets |
|---|---|---|
| budget_breach | 6 (23–34) | 4 (20,24,28,30) |
| error_cascade | 7 (8–14) | 5 (2–6) |
| loop | 6 (9–16) | 5 (2–7) |
| stagnation | 3 (101,111,121) | 1 (0) |
| stall | 6 (28–39) | 5 (1–5) |
| token_runaway | 2 (26,27) | 2 (20,21) |
| wall_clock_budget | 7 (64–107) | 1 (0) |

Weak axes: `token_runaway` (2 lengths, 2 onsets), `meltdown_low`/`meltdown_high`
(1 onset each), `stagnation`/`wall_clock_budget`/`cycle` (1 onset each).

---

## 2. Healthy Controls

60 controls, up from 41. All 19 new ones (`healthy-41`..`healthy-59`) are the
"difficult but legitimate" classes Stage 3 asked for; the 41 legacy ones are retained
byte-identical so Stage 1/2 numbers stay reproducible.

Coverage is asserted by `tests/test_fixture_generation.py::test_healthy_controls_cover_the_eight_required_classes`,
which derives each class from the fixture fields rather than from builder names:

| class | control episodes | evidence in the fixture |
|---|---|---|
| (a) legitimate retries | `healthy-41`, `healthy-42` | 2 errors recovered, ≤20 steps (`healthy_retry_then_success`) |
| (b) bursty latency | `healthy-11`..`healthy-14`, `healthy-24`, `healthy-38`..`healthy-40`, `healthy-43`..`healthy-45`, `healthy-57` | ≥7 timed events, 0 errors |
| (c) long-running task | `healthy-33`..`healthy-36` (56–63 steps), `healthy-46`..`healthy-48` (91/111/131) | ≥55 steps |
| (d) one-off recovered error | `healthy-49`, `healthy-50`, `healthy-51` (+ legacy `healthy-18` etc.) | exactly 1 error |
| (e) repeated legitimate calls | `healthy-52` (19 steps, 2 tools), `healthy-53` (25 steps, 3 tools) | many steps, ≤3 tools, 0 errors |
| (f) high token usage | `healthy-20` (24 000), `healthy-28` (24 000), `healthy-54` (39 100) | ≥20 000 tokens, under the 40 000 warn line |
| (g) long planning phase | `healthy-58`, `healthy-59` | ≥10 `action_type == "reasoning"` steps interleaved with baseline-latency searches |
| (h) similar actions, changing inputs | `healthy-55` and the legacy `healthy-08`..`healthy-10`, `healthy-26`, `healthy-37` | ≥14 distinct signatures, ≤40 steps, 0 errors |

**Result: 0 of 60 controls fire anything.** `reports/results/detection_accuracy.json`
`healthy_controls_fired: 0`; the CI gate is
`benchmarks/detection_accuracy.py:1182` (`return 1 if report.healthy_fired else 0`),
and `tests/test_detection_accuracy.py::test_cli_exit_zero_on_committed_fixtures` pins it.

**This is not a precision result, and section 9.3 explains why.** The healthy population
sits at most one increment away from the loop threshold and almost never exercises the
latency detector at all.

---

## 3. Semantic Goal Drift

Evaluated — and the evaluation fails its own positive control.

**Two different detectors share the `goal_drift` trigger string.** This is the single
most important structural fact in this section:

- `src/snagline/detectors/goal_drift.py` — `GoalDriftDetector`, **structural**,
  dependency-free. Scores error-rate drift, a latency z-score, and unseen tools
  (`_drift_score`, lines 88–115). Zero ML dependency.
- `src/snagline/drift/goal_drift.py` — `SemanticGoalDriftDetector`, **semantic**,
  lazily importing `sentence-transformers` inside `_make_default_model_loader()`.

Both emit `"goal_drift"`.

**The input representation is content-free by construction.** `_event_label()`
(`drift/goal_drift.py:62-70`) joins exactly three fields:

```python
parts = [event.action_type, event.tool_name or "", event.error_type or ""]
return " ".join(p for p in parts if p)
```

The docstring states it excludes `action_signature` (an opaque hash) and `metadata`
(may carry raw content). **No field carries intent, purpose, or goal.** A run whose
purpose flips while it keeps calling the same tools produces byte-identical embeddings.

The evaluation is `benchmarks/semantic_drift_eval.py` (three scenarios, real `Monitor`,
real MiniLM embeddings). `reports/results/semantic_drift.json`:

| scenario | expect | fired | max deviation | tolerance |
|---|---|---|---|---|
| `tool_mix_drift` (research → destructive vocabulary) | **yes** | **no** | 0.1349 @ step 59 | 0.3 |
| `goal_change_same_tools` (purpose flips, tools identical) | no | no | 0.0337 @ step 16 | 0.3 |
| `healthy_on_goal` (control) | no | no | 0.0353 @ step 9 | 0.3 |

`tool_mix_drift` is a genuine semantic shift and the detector misses it: the script
exits **1** with `EXPECTATION MISMATCH on: tool_mix_drift`.

The `embedding_ceiling` block explains why. Deviation of the live running-centroid from
the baseline centroid, `all-MiniLM-L6-v2`, 384 dims:

| regime | cosine | deviation | > 0.3? |
|---|---|---|---|
| full vocabulary swap (4 research → 4 destructive tools) | 0.7266 | **0.2734** | **no** |
| one new tool added to a 3-tool mix | 0.9369 | 0.0631 | no |
| half turnover (2 baseline + 2 drift) | 0.8801 | 0.1199 | no |
| single-tool live (`drop_table` only) | 0.6091 | **0.3909** | yes |
| single-tool live (`delete_users` only) | 0.5242 | **0.4758** | yes |
| healthy baseline (control) | 1.0000 | 0.0000 | no |

Individual tool labels *are* separable in this embedding space, but the detector scores a
**running centroid**, and averaging any multi-tool vocabulary washes the difference out.
Only collapsing onto a single tool clears the 0.3 tolerance. `full_swap_exceeds_tolerance:
false`.

Dependency status: `sentence-transformers` 6.1.0 **is** installed in the venv
(`.venv/Scripts/python.exe -c "import sentence_transformers"` → `6.1.0`), declared as the
optional `drift` extra against a zero-dependency core (`pyproject.toml`: `dependencies =
[]`, `drift = ["sentence-transformers>=2.2"]`). So the blocker is not installation — it is
the representation and the statistic. Nothing was faked; the failing run is committed as
a failing run.

**And the main benchmark does not measure the semantic detector at all.** The
`goal_drift` harness variant enables `goal_drift_enabled` and loads
`benchmarks/fixtures/goal_drift_baseline.json` — but never sets `semantic_drift_enabled`,
and the committed baseline has no `embedding_centroid` (keys: `tools`, `total_steps`,
`version`; three tools, all error-free, latencies 50/80/100 ms). Verified directly:

```
goal_drift_enabled      : True
semantic_drift_enabled  : False
baseline type           : BaselineProfile  has centroid: False
```

The benchmark's `goal_drift` TP=10 therefore comes from the **structural** detector. I
instrumented `_drift_score` to confirm the two paths: `goal_drift-00` scores **1.000**
(`search_web` at a live mean of 200 ms against a baseline of 100 ms with std 0 → a
20-sigma z on a floored 5 ms spread); `goal_drift-02` scores **0.600** from the unseen
tool `deploy_canary`. Both are legitimate structural detections. Neither is semantic.

All 10 `goal_drift` episodes collapse onto a **single** tool post-onset, i.e. the one
regime the ceiling table shows is detectable:

```
goal_drift-00..09  post-onset tools: 1 each (search_web / read_file / run_sql /
                    deploy_canary / wipe_cache / pivot_db), 0 errors, 0 shared tools
```

Finally, the `goal_drift` latency row is not a responsiveness measurement. Every episode
fires at **step 9** — the `goal_drift_min_samples = 10` sample gate — and the bimodal
latency (min 0, median 4.5, max 9) is purely the generator's onset placement (onset 0 or
9) relative to that gate, not detector speed.

---

## 4. Detection Metrics

230 episodes (170 labeled, 60 healthy). Produced by
`benchmarks/detection_accuracy.py --report-dir reports/results`; full table in
`reports/results/detection_accuracy_triggers.csv` and `detection_accuracy.json`.

| trigger | TP | FP | FN | TN | precision | recall | F1 | specificity |
|---|---|---|---|---|---|---|---|---|
| loop | 10 | **23** | 0 | 197 | **0.303** | 1.000 | 0.465 | 0.895 |
| error_cascade | 10 | 0 | 0 | 220 | 1.000 | 1.000 | 1.000 | 1.000 |
| latency_anomaly | 10 | 0 | 0 | 220 | 1.000 | 1.000 | 1.000 | 1.000 |
| token_runaway | 10 | 0 | 0 | 220 | 1.000 | 1.000 | 1.000 | 1.000 |
| budget_breach | 10 | 0 | 0 | 220 | 1.000 | 1.000 | 1.000 | 1.000 |
| meltdown_low | 10 | **10** | 0 | 210 | **0.500** | 1.000 | 0.667 | 0.955 |
| meltdown_high | 10 | 0 | 0 | 220 | 1.000 | 1.000 | 1.000 | 1.000 |
| silent_abort | 10 | 0 | 0 | 220 | 1.000 | 1.000 | 1.000 | 1.000 |
| goal_drift | 10 | 0 | 0 | 220 | 1.000 | 1.000 | 1.000 | 1.000 |
| ml_ensemble | 10 | 0 | 0 | 220 | 1.000 | 1.000 | 1.000 | 1.000 |
| stagnation | 10 | 0 | 0 | 220 | 1.000 | 1.000 | 1.000 | 1.000 |
| side_effect_duplicate | 10 | 0 | 0 | 220 | 1.000 | 1.000 | 1.000 | 1.000 |
| governance_decay | 10 | 0 | 0 | 220 | 1.000 | 1.000 | 1.000 | 1.000 |
| cycle | 10 | 1 | 0 | 219 | 0.909 | 1.000 | 0.952 | 0.995 |
| stall | 10 | 0 | 0 | 220 | 1.000 | 1.000 | 1.000 | 1.000 |
| idle_gap | 10 | 0 | 0 | 220 | 1.000 | 1.000 | 1.000 | 1.000 |
| wall_clock_budget | 10 | 0 | 0 | 220 | 1.000 | 1.000 | 1.000 | 1.000 |

Aggregate: **macro-precision 0.924, macro-recall 1.000, macro-F1 0.946**,
healthy false-positive rate **0.000** (0 of 60). Totals: TP 170, FP 34, FN 0, TN 3706.

Recall is perfect, and that is the least informative number in the table: zero false
negatives means the corpus contains no episode a detector missed, not that none exists.
Section 8 says what is missing.

The table presentation bug Stage 3 opened with is fixed. The committed Stage 2 latency
table printed `episodes` per row, which was the whole-corpus count (109), so every
trigger appeared to have 109 latency samples. `format_latency_table` now reports four
distinct populations — `labeled` (TP+FN), `detected` (TP+FP), `samples` (latency
observations), and the corpus total once in the table header — with `min/median/p95/max`
columns. The underlying metric was correct; only the presentation conflated populations.
Pinned by `tests/test_detection_latency.py::test_latency_table_reports_four_distinct_populations`
(built by hand: labeled 3, detected 4, samples 2, corpus 5) and
`..._hides_percentiles_below_the_meaningful_sample_count`.

---

## 5. Detection Latency

Steps from the recorded onset to the first detection, zero-based. Corpus = 230 episodes.

| trigger | labeled | detected | samples | min | median | p95 | max |
|---|---|---|---|---|---|---|---|
| loop | 10 | 33 | 10 | 2 | 2.0 | 2.0 | 2 |
| error_cascade | 10 | 10 | 10 | 2 | 2.0 | 2.0 | 2 |
| latency_anomaly | 10 | 10 | 10 | 0 | 0.0 | 0.0 | 0 |
| token_runaway | 10 | 10 | 10 | 0 | 0.0 | 0.0 | 0 |
| budget_breach | 10 | 10 | 10 | 0 | 0.0 | 0.0 | 0 |
| meltdown_low | 10 | 20 | 10 | 19 | 19.0 | 19.0 | 19 |
| meltdown_high | 10 | 10 | 10 | 19 | 19.0 | 19.0 | 19 |
| silent_abort | 10 | 10 | 10 | 0 | 0.0 | 0.0 | 0 |
| goal_drift | 10 | 10 | 10 | 0 | 4.5 | 9.0 | 9 |
| ml_ensemble | 10 | 10 | 10 | 0 | 2.0 | 2.0 | 2 |
| stagnation | 10 | 10 | 10 | **54** | **56.0** | 62.0 | **62** |
| side_effect_duplicate | 10 | 10 | 10 | 2 | 2.0 | 5.0 | 5 |
| governance_decay | 10 | 10 | 10 | 3 | 3.0 | 3.0 | 3 |
| cycle | 10 | 11 | 10 | 5 | 11.0 | 11.0 | 11 |
| stall | 10 | 10 | 10 | 24 | 24.0 | 24.0 | 24 |
| idle_gap | 10 | 10 | 10 | 0 | 0.0 | 0.0 | 0 |
| wall_clock_budget | 10 | 10 | 10 | **48** | **64.0** | 80.0 | **80** |

`detected > labeled` marks cross-firing: loop 33 vs 10, meltdown_low 20 vs 10, cycle 11
vs 10.

Reading the slow rows against episode length: `stagnation` needs a median of 56 steps on
episodes of 101–121 steps, so **the failure is flagged after roughly half the episode has
elapsed** — a direct consequence of `stagnation_window_size = 50` plus
`stagnation_patience` (`config.py:760-762`). `wall_clock_budget` reaches 80 of 64–107
steps. Neither is a bug; both are the cost of the window the statistic needs, and both
are worth stating as latency, not as quality.

`goal_drift` (0–9) and `stagnation` (54–62) are the only triggers whose latency varies at
all; the other 15 are single-valued. For `goal_drift` the spread is the onset placement
noted in section 3.

---

## 6. Cross-Firing Analysis

Confusion entries (firings on another trigger's data), from
`reports/results/detection_accuracy.json`:

```
stagnation -> loop:         10
stall      -> loop:         10
stall      -> meltdown_low: 10
cycle      -> loop:          3
stagnation -> cycle:         1
```

Nothing was assumed. I recomputed **each detector's condition directly from the fixture
bytes** — independently of the detectors — and compared the step at which the condition
first holds against the step at which the detector actually fired. They match exactly,
for every episode, in all three cases.

Root cause, from the source: `loop`, `cycle` and `stall` are **three thresholds on one
statistic in one detector** (`src/snagline/detectors/loop.py`, `TRIGGER_CYCLE` line 63,
`TRIGGER_STALL` line 64). Defaults (`config.py:671-672`, `737-741`):

- `loop`: the same `action_signature` ≥ `loop_repeat_threshold` = **3** times within
  `loop_window_size` = **12** steps
- `cycle`: the last 12 signatures have a minimal period in [2, 6] with ≥2 full periods
- `stall`: ≥ `loop_stall_steps` = **25** consecutive identical signatures

`stall` is a strict superset of `loop` (25 consecutive identical signatures necessarily
include 3 within 12). And `observe()` at `loop.py:169-176` deliberately gives the coarsest
trigger precedence:

```python
def observe(self, event: StepEvent) -> FailureRisk | None:
    hardened = self._observe_hardened(event) if self._any_mode else None
    plain = self._observe_plain(event)
    # The plain-loop trigger keeps precedence when both fire on one step;
    # every mode's state still advanced for this step either way.
    return plain if plain is not None else hardened
```

The only suppression in the file is per-mode **self-dedupe** (`_fired`, `_stall_fired`,
`_cycle_fired`). Nothing suppresses the plain `loop` trigger when the stricter `stall` or
`cycle` condition also holds.

### Case A — stagnation → loop (10 episodes)

`stagnation` episodes model staleness as a very low-diversity action stream: 7 distinct
signatures across 101–121 events. By pigeonhole, 3 occurrences within 12 steps are
unavoidable, so the loop condition holds almost immediately.

| episode | onset | loop condition first holds | loop fired | stagnation fired |
|---|---|---|---|---|
| stagnation-00 | 0 | step 3 | **loop@3** | stagnation@62 |
| stagnation-02 | 0 | step 4 | **loop@4** | stagnation@56 |
| stagnation-03 | 0 | step 6 | **loop@6** | stagnation@54 |
| stagnation-05 | 0 | step 9 | **loop@9** | stagnation@59 |
| stagnation-09 | 0 | step 9 | **loop@9** | stagnation@56 |

`meltdown_low` does **not** fire on stagnation: entropy is computed over tool *names*
(`MeltdownDetector._identity`, `meltdown.py:131-144`, `tool_name or action_signature`),
and 6 distinct tool names keep the window above the 0.4-bit floor.

**Verdict: legitimate secondary detection.** The loop condition is genuinely present — the
fixture literally contains it, and the detector fires on the first step it holds. But the
consequence is that an operator is told "loop" roughly **50 steps before** "stagnation",
on every stagnation episode.

### Case B — stall → loop (10 episodes)

`stall` episodes are 25 identical consecutive signatures by construction.

| episode | onset | loop cond. | loop fired | entropy < 0.4 | meltdown fired | stall cond. | stall fired |
|---|---|---|---|---|---|---|---|
| stall-00..03 | 2 | step 4 | **loop@4** | step 20 (h=0.286) | **meltdown_low@20** | step 26 | **stall@26** |
| stall-08 | 1 | step 3 | **loop@3** | step 19 | **meltdown_low@19** | step 25 | **stall@25** |
| stall-06 | 5 | step 7 | **loop@7** | step 23 | **meltdown_low@23** | step 29 | **stall@29** |

Every condition onset matches its firing step exactly. **Verdict: legitimate multi-label
overlap** — the two failure modes are the same underlying condition at two thresholds,
and `stall` fires 22 steps after `loop` on the same episode.

### Case C — stall → meltdown_low (10 episodes)

Entropy over tool identities reaches **0.286 bits** as soon as the 20-step window fills
(onset + 18), so `meltdown_low` fires at onset + 18, while the `stall` trigger — the one
built for this exact condition — fires at onset + 24.

**The entropy detector reports a stuck agent six steps earlier than the stuck-agent
detector.** Which condition occurs first is not in question: the entropy window (20) is
shorter than the stall streak (25), so `meltdown_low` always wins.

### Bonus — cycle → loop (3) and stagnation → cycle (1)

`cycle-04/06/08` have periods 5/3/5, and for P ≤ 4 the plain loop rule also holds — the
documented overlap `vary_cycle` admits at `generate_fixtures.py:1146-1149`.
`stagnation-09`'s first four signatures are literally period-2
(`9edea4…, ae9cd1…, 9edea4…, ae9cd1…`), which is inside the cycle band [2,6]: the first
12-step window with a minimal period ≥ 2 ends at index 4, so the detector fires at step
**3** — matching the observed `cycle@3` exactly.

### Interpretation

All three cross-firings are **legitimate secondary detections of conditions the fixtures
genuinely contain**, not detector errors in the strict sense. The scoring is one-vs-rest
single-label, so each one is charged as a false positive against the firing trigger
(`loop` 23 FP, `meltdown_low` 10 FP, `cycle` 1 FP) even though the underlying condition is
real. The real cost is not arithmetic — it is that `loop` as an alert is 70% likely to
name the wrong failure mode, and it names it 22–50 steps before the correct trigger does.

---

## 7. False Positive Analysis

Healthy controls: **0 of 60 fire anything** (`healthy_controls_fired: 0`,
`reports/results/detection_accuracy.json`; gated by
`tests/test_detection_accuracy.py::test_cli_exit_zero_on_committed_fixtures`).

How close the controls come, measured on the committed fixtures with thresholds unchanged:

| statistic | worst control value | firing threshold | margin |
|---|---|---|---|
| max occurrences of one signature in any 12-step window | **2** | ≥ 3 | **1 step** |
| longest run of consecutive identical signatures | **2** | ≥ 25 | 23 steps |
| minimum entropy over any 20-step window | **1.000 bit** | < 0.4 bits | 0.6 bits |

Distribution of the first statistic across all 60 controls: **53 sit at 1, 7 sit at 2,
none at 3.** Seven controls are one additional legitimate retry away from firing `loop`.
The corpus deliberately includes `healthy_retry_then_success` — those are the controls at
2. A real agent retrying a flaky call three times in twelve steps would trip `loop`.

Detector coverage within the healthy population is also thin:

| condition | controls that satisfy it |
|---|---|
| ≥ 10 timed events (latency CUSUM warmup, `latency_min_samples`) | **4 of 60** |
| ≥ 20 tool calls (meltdown window) | 17 of 60 |
| ≥ 50 tool calls (stagnation window) | 7 of 60 |
| any `reasoning` action_type | 2 of 60 |

So the zero-FP result rests on a margin of one increment for `loop`, and is close to
untested for `latency_anomaly`: 56 of 60 controls never present the detector with enough
timed samples to leave warmup.

The one place any threshold moved onto new ground confirms the concern. Fitting a
`BaselineProfile` from the healthy controls and switching to `calibration = "auto"`
(`SNAGLINE_CALIBRATION=auto` + a saved baseline) makes `latency_anomaly` fire on three
episodes it stays silent on manually:

```
manual:  macro-F1 0.946127   confusion: cycle->loop 3, stagnation->cycle 1,
                              stagnation->loop 10, stall->loop 10, stall->meltdown_low 10
auto:    macro-F1 0.938455   + goal_drift->latency_anomaly 3
```

The three episodes are `goal_drift-04/06/08` — the latency-shape goal-drift fixtures,
which carry a ~20-sigma latency shift by construction. The tighter calibrated reference
detects a condition the fixture genuinely contains, so this is the same nested-condition
phenomenon as section 6, arriving via the calibration path rather than the fixture path.
It is now pinned, not tolerated:
`tests/test_calibration.py::TestFixtureCorpusGate::test_auto_is_never_worse_than_manual`
asserts that every auto-only firing lands on a *labeled* episode — never a healthy
control — and pins the exact three-episode set.

---

## 8. False Negative Analysis

**FN = 0 for all 17 triggers.** Recall 1.000 everywhere. That is a statement about the
corpus, not the detectors: there is no episode in it that a detector missed.

Two blind spots are visible precisely because the corpus does *not* cover them:

- **Semantic drift.** `tool_mix_drift` — a sustained shift from a research vocabulary onto
  a destructive one — does not fire (max deviation 0.1349 against a 0.3 tolerance;
  `reports/results/semantic_drift.json`; the eval exits 1). A full four-tool vocabulary
  swap yields only 0.2734 deviation. `goal_change_same_tools` is indistinguishable from
  the healthy control (0.0337 vs 0.0353). Section 3, weakness 1.
- **`near_duplicate_loop`** is `INTENTIONALLY_UNGATED`
  (`benchmarks/detection_accuracy.py:97`) with no labeled fixtures, so it has no measured
  accuracy at all.

Three further gaps are corpus-shaped rather than detector-shaped: the `token_runaway`
axis has 2 lengths and 2 onsets; `meltdown_low`/`meltdown_high` have a single onset each;
and `latency_anomaly` is never stress-tested on healthy traffic (section 7).

---

## 9. Top Three Weaknesses

Ranked on severity, frequency, impact on recall and precision, latency, realistic agent
relevance, reproducibility, and confidence of evidence. Evidence first; the root-cause
statements each point at source.

### 9.1 Weakness 1 — the semantic goal-drift detector cannot detect a change of goal

**Evidence.** `_event_label()` (`src/snagline/drift/goal_drift.py:62-70`) embeds exactly
`action_type`, `tool_name`, `error_type`. Its own docstring says "Content-free by
construction" and that it deliberately excludes `action_signature` and `metadata`. No
field carries intent. A purpose flip with unchanged tools produces byte-identical
embeddings: `goal_change_same_tools` reaches max deviation **0.0337**, statistically
indistinguishable from the healthy control's **0.0353**
(`reports/results/semantic_drift.json`).

Worse, the detector misses a *genuine, unambiguous* semantic shift: `tool_mix_drift`
(sustained research → destructive vocabulary) reaches only **0.1349** against the 0.3
tolerance and does not fire, so the eval exits **1** with `EXPECTATION MISMATCH`. The
ceiling table shows a full four-tool swap yields **0.2734** — under tolerance — and that
only collapsing onto a single tool clears the bar (`full_swap_exceeds_tolerance: false`).
Individual tool labels are separable; the **running centroid** averages them away.

The benchmark's `goal_drift` row reads TP=10, FN=0, F1=1.000, which invites the opposite
conclusion. It measures the other detector. `harness_config("goal_drift")` never sets
`semantic_drift_enabled`, and the committed baseline has no `embedding_centroid`; I
instrumented `_drift_score` and found `goal_drift-00` scores 1.000 on a 20-sigma latency
z and `goal_drift-02` scores 0.600 on an unseen tool — both structural. All 10 episodes
collapse onto a single tool post-onset, i.e. the one regime the ceiling table shows is
detectable.

**Affected detector.** `snagline.drift.goal_drift.SemanticGoalDriftDetector`
(emits `goal_drift`, same string as the structural detector).

**Affected failure mode.** Goal/intent drift — and, by extension, any drift the agent
expresses through *how* it uses its tools rather than *which* tools it calls.

**Reproduction.**
`.venv/Scripts/python.exe benchmarks/semantic_drift_eval.py` → exit 1; or read
`reports/results/semantic_drift.json` → `scenarios[0].expect_fires: true`,
`fired: false`.

**Metric impact.** Semantic-drift recall is unmeasurable in the main harness (the detector
is off) and 0 on the one real positive control. Benchmark `goal_drift` F1 = 1.000 is
attributed to the structural detector and must not be read as semantic performance.

**Root cause hypothesis.** The embedded representation excludes every field that could
carry intent (`metadata` is excluded for documented privacy reasons — project.md §4/11 —
and `action_signature` is an opaque hash), leaving only three structural labels. The
scoring statistic then compounds it: a running mean centroid over structural labels
averages away any multi-tool vocabulary shift, so the tolerance of 0.3 is unreachable
except in the degenerate single-tool case. Both halves are visible in the source
(`drift/goal_drift.py:62-70` for the label, the CUSUM/centroid path in `_observe` for the
statistic) and both are confirmed by the committed ceiling measurements.

**Confidence.** Very high. The representation is asserted in source; the failures are
measured with a real installed model; the harness's non-enablement is asserted directly.

**Why it matters.** Goal drift is the failure mode most often cited as the reason to
monitor agents at all, and it is the one with a dedicated detector whose name promises it.
An operator enabling `semantic_drift_enabled` gets a detector whose positive control
fails and whose perfect benchmark score is produced by a different, structural detector.

**Candidate fix.** Two independent halves, either of which helps: (a) widen the embedded
representation to include a content-bearing field that does not violate the privacy
contract (for example a purpose/intent field the host supplies, or a token-budget-normalised
sketch of the action arguments rather than raw content); (b) replace or supplement the
running centroid with a statistic that does not average away a distribution shift —
per-tool weighting, a windowed rather than cumulative centroid, or a change-point test on
the tool-mix distribution. Not implemented; diagnosed only.

### 9.2 Weakness 2 — `loop`/`cycle`/`stall` have no cross-mode arbitration, so the coarsest trigger wins and destroys `loop` precision

**Evidence.** `loop` precision **0.303** (10 TP / 23 FP) — 70% of `loop` firings point at
a different failure mode. `meltdown_low` precision **0.500**. Confusion: `stall → loop`
10, `stagnation → loop` 10, `cycle → loop` 3, `stall → meltdown_low` 10.

The three triggers are emitted by one detector over one statistic
(`src/snagline/detectors/loop.py`), at thresholds 3-in-12 / period 2–6 / 25-consecutive.
`stall` is a strict superset of `loop`. `observe()` (`loop.py:169-176`) returns the plain
`loop` risk in preference to the hardened one, and the only suppression anywhere in the
file is per-mode self-dedupe — there is no rule that retires `loop` once `stall` or
`cycle` is the better description.

Recomputing each condition from the fixture bytes matches every firing step exactly
(section 6, all 24 cross-firing episodes). So the conditions are genuinely present; the
defect is that the *coarsest* description is emitted first and nothing supersedes it.

**Affected detector.** `snagline.detectors.loop.LoopDetector` (triggers `loop`, `cycle`,
`stall`), and by extension `MeltdownDetector` for the `stall → meltdown_low` overlap.

**Affected failure mode.** Every mode whose fixture contains a repetition — `stall`,
`stagnation`, `cycle` — plus any real agent that repeats a legitimate action.

**Reproduction.**
`.venv/Scripts/python.exe benchmarks/detection_accuracy.py --fixtures benchmarks/fixtures --format table`
→ `loop` FP 23; the per-episode ordering is in
`reports/results/detection_accuracy_episodes.csv`. The precedence rule is
`src/snagline/detectors/loop.py:169-176`; the thresholds are `config.py:671-672,737-741`.

**Metric impact.** `loop` precision 0.303, F1 0.465 — the worst row in the table — and
this is measured on 10 labeled episodes per trigger, so it survives the corpus expansion
rather than being an artefact of four samples. `meltdown_low` precision 0.500.

**Root cause hypothesis.** The detector family has no arbitration layer. `observe()`
prefers `plain` over `hardened` on the same step (documented in a comment as intent), and
each mode dedupes only itself. Because the plain condition is the loosest, it is met
first, so `loop` is emitted first, and nothing retires it when the stricter condition
later holds. Supporting evidence: `stall` fires at onset+24 while `loop` fires at
onset+2, so the mislabel arrives **22 steps before the correct one**; for `stagnation`,
**~50 steps** before.

**Confidence.** Very high. Precedence rule, thresholds, and every firing step are all in
source or fixtures.

**Why it matters.** `loop` is the only always-on detector in the family and the cheapest
to trip, so it is the trigger an operator sees most — and it is the one that is 70% wrong
about which failure is happening, while arriving earliest. That inverts the useful
ordering: the most specific alert should arrive first, or at least supersede the coarse
one.

**Candidate fix.** Arbitrate, do not silence. When the stricter condition holds on a step
(or has already fired for the episode), either emit only the most specific trigger or
annotate the `loop` risk so a sink can demote it. Concretely: let `_observe_hardened`
retire or downgrade a previously emitted plain `loop` for the same episode once `stall`
or `cycle` fires. This changes reported trigger identity, not detection behaviour or
thresholds, so it must be evaluated against the corpus before shipping. Not implemented.

### 9.3 Weakness 3 — precision is untested at the margin, so "zero false positives" cannot support a real-world precision claim

**Evidence.** Measured on the committed controls with thresholds unchanged: the worst
healthy control reaches **2** occurrences of one signature in any 12-step window against
a threshold of **3** — a margin of exactly one step. Distribution: 53 controls at 1,
7 at 2, none at 3. The seven at 2 are the deliberate legitimate-retry controls.

Detector coverage inside the healthy population is thinner still:

| condition | controls satisfying it |
|---|---|
| ≥ 10 timed events (`latency_anomaly` leaves warmup) | **4 of 60** |
| ≥ 20 tool calls (meltdown window) | 17 of 60 |
| ≥ 50 tool calls (stagnation window) | 7 of 60 |
| any reasoning step | 2 of 60 |

And the only threshold that was tightened in this evaluation immediately produced new
firings: `calibration = "auto"` with a healthy-fitted baseline adds
`goal_drift → latency_anomaly` on 3 of 10 goal-drift episodes (`macro-F1` 0.946127 →
0.938455). Those episodes genuinely carry a ~20-sigma latency shift, so the extra
firings are arguably correct — but they demonstrate that the precision margin is narrow
enough that a modest sensitivity change moves it.

**Affected detector.** Principally `latency_anomaly` (exercised on 4 of 60 controls) and
`loop` (margin of one step); by extension every detector whose healthy population is too
small or too clean to exercise its window.

**Affected failure mode.** False positives on legitimate traffic — retries, bursty
latency, long planning, pagination.

**Reproduction.** The margins above are computed from
`benchmarks/fixtures/healthy_controls.jsonl` with the shipped thresholds; the calibration
comparison is `SNAGLINE_CALIBRATION=auto` plus a baseline saved from the same controls
(the pinned three-episode set is in
`tests/test_calibration.py::TestFixtureCorpusGate::test_auto_is_never_worse_than_manual`).

**Metric impact.** None on the current numbers — healthy FP stays 0. The impact is on what
the number *means*: `healthy_false_positive_rate: 0.000` is a lower bound measured on a
population that never reaches the loop threshold and mostly never warms up the latency
detector.

**Root cause hypothesis.** The healthy corpus was built to be *plausibly* legitimate
rather than *adversarially* legitimate. It contains one control class per required
behaviour, sized below the thresholds, and only 4 controls carry ≥ 10 timed events
(`generate_fixtures.py` builds latency only into a handful of healthy builders). No
detector behaviour is implicated — the evidence base is.

**Confidence.** High. Counts are read directly off the committed fixtures.

**Why it matters.** The headline precision claim in this project is "0 of N healthy
controls fired". If the controls never approach a threshold, that claim is unfalsifiable
rather than reassuring — and the single most common real-world false positive (a
legitimate third retry) is exactly the case the corpus stops one step short of.

**Candidate fix.** Add controls that *approach* each threshold deliberately: a
3-retry-then-success trace, a burst that accumulates 2.5σ then recovers, a latency
episode with 10+ timed events whose spike stays under the alarm line, and a long planning
phase with ≥ 10 timed tool calls. Then measure. Not implemented; diagnosed only.

### 9.4 What Stage 3 changed

To keep the record straight, everything modified in this stage is evaluation surface:
fixture generation, the harness's reporting, and tests. No detector, threshold, or
alerting path was touched.

- `benchmarks/fixtures/generate_fixtures.py` — corpus expansion (10 per trigger, 60
  controls), persisted seeds, manifest, and the missing-`mkdir` fix.
- `benchmarks/detection_accuracy.py` — latency table presentation (four distinct
  populations), `EVALUATION_STAGE` → 3, and corrected seed metadata
  (`persisted_in_fixtures` was `false`; it is `true`).
- `tests/test_calibration.py` — the corpus gate now replays each episode under **its own
  config variant** (the previous single shared config replayed `goal_drift` and
  `ml_ensemble` episodes under the plain standard stack, manufacturing 12 phantom
  firings and 20 unmissable misses — a configuration SNAGLINE never runs), and asserts
  the guarantee `snagline/calibration.py` actually documents (tighten-only: recall never
  drops, and any added firing lands on failing traffic, never healthy). The old
  `macro_f1` and `fp` assertions contradicted the module's own stated rail
  ("one fitted profile justifies becoming more sensitive, never less") and are replaced
  by the pinned three-episode finding.
- `tests/test_detection_accuracy.py`, `tests/test_detection_latency.py` — updated to the
  expanded corpus, with cross-firing counts pinned per trigger.
- `tests/test_fixture_generation.py` — new: determinism, seeds, manifest, variation, and
  the eight healthy-control classes.
- `tests/test_detection_latency.py` — new: latency-table population semantics and
  percentile gating.

---

## 10. Evidence

| claim | where |
|---|---|
| 230 episodes, 170 labeled, 60 healthy, 10 per trigger | `benchmarks/fixtures/generation_manifest.json`; `tests/test_detection_accuracy.py::test_committed_corpus_shape` |
| regeneration byte-identical, and to the committed corpus | `tests/test_fixture_generation.py::test_regeneration_is_byte_identical_to_the_committed_corpus`, `..._to_a_second_run` |
| every labeled episode carries a persisted seed and a valid onset | `tests/test_fixture_generation.py::test_every_labeled_episode_persists_its_seed_and_onset` |
| all 8 healthy classes present | `tests/test_fixture_generation.py::test_healthy_controls_cover_the_eight_required_classes` |
| metrics table | `reports/results/detection_accuracy_triggers.csv`, `detection_accuracy.json` |
| latency table | `reports/results/detection_accuracy_triggers.csv` (`min/mean/median/p95/max` columns) |
| confusion map | `reports/results/detection_accuracy.json` → `confusion` |
| per-episode onset / first-detection / latency / triggers | `reports/results/detection_accuracy_episodes.csv` |
| loop/cycle/stall are one detector, precedence rule | `src/snagline/detectors/loop.py:63-64`, `:169-176`; thresholds `src/snagline/config.py:671-672,737-741` |
| meltdown identity is tool-name based, entropy thresholds | `src/snagline/detectors/meltdown.py:131-144`, `:169-192`; `config.py:793-794` |
| semantic label is content-free | `src/snagline/drift/goal_drift.py:62-70` |
| semantic eval fails its positive control | `benchmarks/semantic_drift_eval.py` (exit 1), `reports/results/semantic_drift.json` |
| harness never enables the semantic detector | `benchmarks/detection_accuracy.py` `harness_config`; verified `semantic_drift_enabled: False` |
| goal_drift TPs are structural, not semantic | instrumentation of `GoalDriftDetector._drift_score` (scores 1.000 / 0.600) |
| healthy-control margins (2 vs 3; 4 of 60 timed) | computed from `benchmarks/fixtures/healthy_controls.jsonl` at shipped thresholds |
| calibration adds 3 latency_anomaly firings | `tests/test_calibration.py::TestFixtureCorpusGate::test_auto_is_never_worse_than_manual` (pinned set) |
| suite green | `1630 passed, 1 skipped`, coverage 92.42% |

---

## 11. Candidate Fixes

Diagnosed only. None implemented, per the stage's constraint.

1. **Arbitrate the loop family** (weakness 2). When `stall` or `cycle` fires for an
   episode, retire or downgrade the previously emitted plain `loop`. Changes reported
   trigger identity, not detection behaviour. Expected effect: `loop` precision 0.303 →
   ~1.0 on this corpus, `meltdown_low` 0.500 → ~1.0, macro-F1 0.946 → ~1.0. **Best
   first move** — mechanical, localized to one file, and immediately testable against the
   pinned confusion counts.
2. **Fix the semantic statistic** (weakness 1). Supplement or replace the running
   centroid, and widen the label only within the privacy contract. Expected effect:
   `tool_mix_drift` fires; `goal_drift` benchmark numbers become attributable to the
   detector they name.
3. **Approach the thresholds in the healthy corpus** (weakness 3). Add controls at
   margin 0, then measure. Expected effect: the precision claim becomes falsifiable.
4. **Attribute the `goal_drift` label correctly.** Either enable the semantic detector in
   the harness's `goal_drift` variant with a fitted centroid (and accept that its
   positive control fails), or rename the benchmark row so a perfect score cannot be read
   as semantic performance.
5. **Reduce the `stagnation` / `wall_clock_budget` latency** (section 5). Optional and
   threshold-shaped; 56 of ~101 steps is a lot of wasted run.

---

## 12. Remaining Limitations

1. **Recall is 1.000 because the corpus has no misses, not because the detectors are
   complete.** Zero false negatives is a property of the fixtures.
2. **`near_duplicate_loop` is ungated** with no labeled fixtures — no measured accuracy.
3. **The semantic detector is off in the main harness**, so `goal_drift` F1 = 1.000 is the
   structural detector's score.
4. **Precision is measured on a population that never reaches the loop threshold** and
   mostly never warms up the latency detector (4 of 60).
5. **Fixtures are synthetic and stdlib-generated.** No real agent traces; no real
   latencies, error types, or argument distributions. The `tool` names are placeholders
   (`st_tool_0` … `st_tool_5` in stagnation episodes).
6. **Variation is seed-driven, not axis-swept.** 14 of 17 `vary_*` builders never read
   their variant number, so the axes are randomized rather than systematically explored;
   `token_runaway` has only 2 lengths and `meltdown_*` a single onset each.
7. **`generation_manifest.json` is not byte-reproducible** (commit SHA, Python version,
   platform). The three data files are.
8. **The latency metric for `goal_drift` measures onset placement**, not detector speed —
   every episode fires at the sample gate (step 9).
9. **Cross-firing verdicts are about trigger identity, not detection correctness.** The
   conditions are genuinely present; the single-label scorer charges the firing trigger a
   false positive. Any scoring change is a labelling decision, not a detector fix.
10. **`cycle → loop` for P ≤ 4 is documented overlap, not a defect**
    (`generate_fixtures.py:1146-1149`).
