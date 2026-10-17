# Stage 2 — Detection latency and confusion metrics

Stage 2 extends **only the evaluation infrastructure**. No detector algorithm,
threshold, or production monitoring path was changed; the entire detector stack
was left untouched and the Stage 1 numbers are reproduced byte-for-byte.

Reproduce with the **Stage 2 fixture revision** — this report measures the
109-episode corpus (68 labeled, 41 healthy controls), which the Stage 3 corpus
expansion later replaced with the committed 230-episode set. Running the
commands against `benchmarks/fixtures` as it stands today regenerates the
current corpus, not these tables. Check out the revision first:

```bash
git checkout 10178e0 -- benchmarks/fixtures
python benchmarks/detection_accuracy.py --fixtures benchmarks/fixtures --format table
python benchmarks/detection_accuracy.py --fixtures benchmarks/fixtures --format json
python benchmarks/detection_accuracy.py --fixtures benchmarks/fixtures --format csv
python benchmarks/detection_accuracy.py --fixtures benchmarks/fixtures --report-dir reports/results
git checkout HEAD -- benchmarks/fixtures   # restore the committed corpus
```

The commands above are also the current-corpus entry point: drop the
`git checkout` lines to reproduce the Stage 4 numbers from
`reports/EVAL_FINDINGS.md` against the 230-episode set.

---

## 1. Implementation changes

### `benchmarks/detection_accuracy.py`

| Change | What was added |
|---|---|
| `Episode.onset_step` | New optional fixture field: the zero-based index of the first event at which the failure condition is introduced. `None` for healthy controls and for any labeled fixture whose generator cannot derive one. Backward compatible: fixtures without the key parse as before and latency is simply not reported. |
| `EpisodeOutcome.first_detection` | `dict[str, int]` mapping each fired label-space trigger to the **earliest** event index that emitted it. Repeated firings never replace it. `EpisodeOutcome.predicted` (the trigger *set*) is unchanged, so every existing scorer path is untouched. |
| `EpisodeOutcome.onset_step` | Carried through from the fixture so the scorer does not re-read the corpus. |
| `detection_latency(onset, first)` | Pure helper: `first - onset`, or `None`. Never a sentinel number. Negative is reported as-is (see §3.3). |
| `percentile(values, p)` | Nearest-rank percentile (`ceil(p/100·n)`, clipped, 1-based). Returns `None` for an empty sample. |
| `TriggerStats.tn`, `.latencies`, `.n_onset` | The TN cell completes each trigger's one-vs-rest confusion matrix; `latencies` holds per-episode expected-trigger detection delays; `n_onset` counts expected episodes with a usable onset. |
| `TriggerStats.specificity / mean_latency / median_latency / p95_latency / n_episodes / n_detected` | Derived views. `precision`/`recall`/`f1` are computed from TP/FP/FN only, exactly as before. |
| `format_table` | Two new columns: `TN` and `specificity`. |
| `format_latency_table` | New per-trigger latency block (printed after the confusion table). |
| `--format csv` | One row per (episode, trigger) for every expected **or** fired trigger. |
| `--report-dir DIR` | Writes `detection_accuracy.json`, `detection_accuracy_episodes.csv`, `detection_accuracy_triggers.csv` for archival. |
| `report_to_json` | Now carries `metadata` (schema, stage, commit SHA, Python/platform, fixture counts, seed scheme, step convention), the full per-trigger block, per-episode records, and a `definitions` block that documents every term. The stage-1 summary keys (`per_trigger`, `macro_f1`, `confusion`, `healthy_controls_fired`, `episodes`) are unchanged; per-episode records live under the new `episode_records` key so the existing `episodes` summary contract is preserved. |

### `benchmarks/fixtures/generate_fixtures.py`

* `_Builder.mark_onset()` records that the *next* event added is the
  failure-injection onset; `_Builder.result()` returns `(events, onset)` so the
  onset is **derived from the construction sequence** rather than maintained in
  a hand-written table.
* Every labeled builder marks its onset immediately before the first injected
  event. `main()` asserts no labeled builder returns `None`, so a future
  builder that forgets the mark fails loudly instead of silently dropping
  latency coverage.
* Labeled fixtures now carry `label: {"trigger": ..., "onset_step": N}`.
  Healthy controls carry no onset (nothing is injected) and stay
  **byte-identical** to Stage 1.

### Fixture regeneration — hash audit

| File | Stage 1 (git blob) | Stage 2 (git blob) | Reason |
|---|---|---|---|
| `labeled_episodes.jsonl` | `cd4c71b714cdb947ef9dc6ffbba465ad14d41e3f` | `809383c8cbba007c2c21451ca9930f98bb69cefd` | `onset_step` added to all 68 labels. **Event payloads are byte-identical** — verified by diffing pre/post regeneration: only the `label` object gained one key. |
| `healthy_controls.jsonl` | `3d3355c7bf72d7fead47a632d7bccbddd76539f2` | `3d3355c7bf72d7fead47a632d7bccbddd76539f2` | unchanged — no onset on healthy controls |
| `goal_drift_baseline.json` | `4cc7d226c22c39fff2cebe9bf44e1944882c6af3` | `4cc7d226c22c39fff2cebe9bf44e1944882c6af3` | unchanged — baseline generator untouched |

Regeneration remains fully deterministic: a second run reproduces every file
byte-for-byte (verified by md5 diff).

### `tests/test_detection_latency.py` (new, 24 tests)

Written before the metric logic existed. Covers the task's Tests A–J plus
invariants the corpus depends on: first detection is the earliest firing (A),
repeated alerts never move it (I), per-trigger independence, latency arithmetic
(B/C/D) including negative latency, one-vs-rest TN/FP/TP/FN (E–H), the
confusion matrix summing to the episode count, specificity, Stage 1 arithmetic
unchanged (J), the committed corpus reproducing Stage 1 aggregates, every
emitted risk resolving to a fixture event index, and every committed latency
being non-negative.

---

## 2. Metric definitions

**Step index convention: ZERO-BASED**, i.e. the position of the event in the
episode's `events` list. This is the repository's own convention — the generator
writes that index into each `step_id` as `f"{episode_id}-s{i}"`, so an emitted
risk's `step_id` maps back to an index by lookup. All onset, first-detection and
latency numbers use it consistently.

**Onset step** — the first step at which the failure condition is *introduced*,
not the first step at which a detector could notice it:

| Shape | Onset |
|---|---|
| repeat / threshold shapes (`loop`, `error_cascade`, `stall`, `side_effect_duplicate`, `ml_ensemble` loop/cascade) | the first of the injected repeats |
| spike shapes (`latency_anomaly`, `token_runaway`, `budget_breach`, `idle_gap`) | the injected step itself |
| cumulative shapes (`meltdown_low/high`, `stagnation`, `cycle`, `wall_clock_budget`, goal-drift latency blowout) | step 0 — the whole trajectory is the failure |
| `silent_abort` | the deliberately-unclosed final tool call (the divergence point) |
| `governance_decay` | the compaction step that pins the never-confirmed constraints |
| goal-drift unseen-tool shape | the first call to a tool absent from the healthy baseline |
| healthy controls | none |

Latency measured against this value therefore **includes the detector's own
threshold delay**, which is the point of stage 2.

**First detection step** — the earliest event index whose `ingest` (or the
`end_episode` finalize pass) emitted that trigger. Repeat emissions count once
and never replace it. The `silent_abort` finalize risk is attributed by the
detector to the last ingested event, which is in the fixture, so it resolves
like any other.

**Detection latency** — `first_detection_step − onset_step`. `null` when the
expected detector never fired, when no onset was recorded, or when a detection
could not be pinned to a fixture event. Negative values are reported, not
clamped (see §6).

**TP / FP / FN** — unchanged since Stage 1: exact trigger-label match against
the label-space trigger set; healthy controls can only ever produce FP.

**TN (new)** — one-vs-rest: for trigger X, an episode is a TN when X was *not*
expected (it expects another trigger, or it is an unlabeled healthy control)
*and* X did not fire. Consequences made explicit:

* `TP + FP + TN + FN == 109` for every trigger — the full corpus is the
  denominator.
* An expecting-but-silent episode is an FN, **not** a TN.
* TN is dominated by irrelevancy here (41 healthy controls + 16 other failure
  modes), so it exists to make specificity computable, not as a quality claim.

**Specificity** — `TN / (TN + FP)`; 0.0 (not NaN) when a trigger never had a
negative episode.

**Latency scope** — per-trigger latency statistics cover **only expected-trigger
hits on episodes with a recorded onset**. Cross-firing detections (e.g. `loop`
firing on a `stall` fixture) have no onset *for that trigger* — the episode's
onset belongs to a different failure — so they are excluded from the aggregate
but retained in the per-episode `first_detection` map and the episode CSV for
later cross-firing analysis.

**p95** — nearest-rank, and reported as `n/a` in the table below 5 samples
(`MEANINGFUL_PERCENTILE_N`), because with 4 samples the 95th percentile is
definitionally the sample maximum. The raw value and the sample count are always
carried in JSON with a `p95_meaningful` flag.

---

## 3. Benchmark results (committed corpus, 109 episodes)

### 3.1 Confusion + metrics

```
trigger                 TP   FP   FN   TN  precision   recall      f1 specificity
---------------------------------------------------------------------------------
loop                     4    8    0   97      0.333    1.000   0.500       0.924
error_cascade            4    0    0  105      1.000    1.000   1.000       1.000
latency_anomaly          4    0    0  105      1.000    1.000   1.000       1.000
token_runaway            4    0    0  105      1.000    1.000   1.000       1.000
budget_breach            4    0    0  105      1.000    1.000   1.000       1.000
meltdown_low             4    4    0  101      0.500    1.000   0.667       0.962
meltdown_high            4    0    0  105      1.000    1.000   1.000       1.000
silent_abort             4    0    0  105      1.000    1.000   1.000       1.000
goal_drift               4    0    0  105      1.000    1.000   1.000       1.000
ml_ensemble              4    0    0  105      1.000    1.000   1.000       1.000
stagnation               4    0    0  105      1.000    1.000   1.000       1.000
side_effect_duplicate    4    0    0  105      1.000    1.000   1.000       1.000
governance_decay         4    0    0  105      1.000    1.000   1.000       1.000
cycle                    4    0    0  105      1.000    1.000   1.000       1.000
stall                    4    0    0  105      1.000    1.000   1.000       1.000
idle_gap                 4    0    0  105      1.000    1.000   1.000       1.000
wall_clock_budget        4    0    0  105      1.000    1.000   1.000       1.000
---------------------------------------------------------------------------------
macro-F1: 0.951
episodes: 109 (68 labeled, 41 healthy controls)
confusion (firings on other data):
  stagnation -> loop: 4
  stall -> loop: 4
  stall -> meltdown_low: 4
healthy controls that fired: 0
```

Totals: **TP 68, FP 12, FN 0, TN 1773**, macro-F1 **0.951**.

### 3.2 Detection latency

```
trigger               episodes detected  onset   median      p95
----------------------------------------------------------------
loop                       109       12      4      2.0      n/a
error_cascade              109        4      4      2.0      n/a
latency_anomaly            109        4      4      0.0      n/a
token_runaway              109        4      4      0.0      n/a
budget_breach              109        4      4      0.0      n/a
meltdown_low               109        8      4     19.0      n/a
meltdown_high              109        4      4     19.0      n/a
silent_abort               109        4      4      0.0      n/a
goal_drift                 109        4      4      4.5      n/a
ml_ensemble                109        4      4      2.0      n/a
stagnation                 109        4      4     55.5      n/a
side_effect_duplicate      109        4      4      2.0      n/a
governance_decay           109        4      4      3.0      n/a
cycle                      109        4      4     11.0      n/a
stall                      109        4      4     24.0      n/a
idle_gap                   109        4      4      0.0      n/a
wall_clock_budget          109        4      4     64.0      n/a
```

`onset` = episodes with a recorded onset (4 per trigger — every labeled
episode has one). `p95` is `n/a` everywhere because no trigger has more than 4
latency samples; per the documented rule it needs ≥5. The raw nearest-rank p95
is still in the JSON (it equals the maximum at this sample size).

Reading the numbers against the detector mechanics (the arithmetic is the
generator's, not a claim about real traffic): spike-shaped failures
(`latency_anomaly`, `token_runaway`, `budget_breach`, `idle_gap`,
`silent_abort`) fire on the injected step itself (latency 0) because they alarm
on the current event; threshold-window shapes pay their window cost
(`meltdown` 19 steps = the 20-call window filling; `cycle` 11 = a 12-slot
window; `stall` 24 = the 25-repeat threshold); the patience-based detectors pay
theirs (`stagnation` ~55.5 across two 50-step windows plus patience 2;
`governance_decay` 3 grace steps; `wall_clock_budget` 64 steps of 1.5s before
the 120s breach).

### 3.3 Negative latency check

None. Every expected trigger fired at or after its recorded onset
(24 corpus assertions in `tests/test_detection_latency.py`), which independently
validates the onset derivation against the real detector stack.

---

## 4. Stage 1 vs Stage 2 comparison

| | Stage 1 | Stage 2 | Change |
|---|---|---|---|
| Episodes | 109 | 109 | none |
| Labeled / healthy | 68 / 41 | 68 / 41 | none |
| TP | 68 | 68 | **none** |
| FP | 12 | 12 | **none** |
| FN | 0 | 0 | **none** |
| Macro-F1 | 0.951 | 0.951 | **none** |
| Healthy controls firing | 0 | 0 | **none** |
| Cross-firings | stagnation→loop 4, stall→loop 4, stall→meltdown_low 4 | identical | **none** |
| TN | not calculated | 1773 | new (by design) |
| Specificity | n/a | 0.924–1.000 | new (by design) |
| Detection step / latency | not recorded | recorded, 68/68 labeled episodes | new |

The Stage 1 metrics are reproduced exactly. This is expected and verified rather
than assumed: `precision`/`recall`/`f1` read only TP/FP/FN, the TN cell is
accumulated in a separate pass over the same outcomes, `EpisodeOutcome.predicted`
still drives all scoring, and the fixture regeneration added a label key without
touching a single event payload. No label, threshold, or detector changed.

---

## 5. Test results

* **Focused tests** (`tests/test_detection_latency.py`): 24 passed.
* **Full suite** (`python -m pytest`): see `stage2_tests.txt`.
* **Coverage**: unchanged at 92.40% (the harness is benchmark code; the corpus
  gate still requires ≥80% on `snagline`).
* **Lint/type**: `ruff check` and `ruff format --check` clean across
  `benchmarks/` and `tests/`; `mypy` clean on the three changed files.

---

## 6. Known limitations

1. **Semantic goal drift is not evaluated.** The `goal_drift` trigger is
   covered by two *statistical* shapes (latency blowout, unseen tool). The
   semantic-drift path (a trajectory that drifts from its stated goal without
   any statistical signature) has no fixture, and `semantic_drift` is not a
   distinct trigger string — the drift extra emits the same `goal_drift` label.
   Measuring it needs a goal-representation fixture format that does not exist
   yet. Deliberately out of scope per the stage rules.
2. **Small sample size.** 4 episodes per trigger. Medians are honest but
   coarse, and p95 is not meaningful anywhere (reported as `n/a`). No dataset
   growth was permitted this stage.
3. **Synthetic fixtures.** Every trajectory is hand-derived against the
   detector arithmetic by the generator. Latency therefore measures the
   detectors' own threshold mechanics on idealized signals, not behavior on
   real agent traffic. The corpus is an anchor for regression, not a field
   estimate.
4. **Cross-firing semantics are unresolved.** The 12 FPs are exactly the three
   known overlaps (`stagnation→loop`, `stall→loop`, `stall→meltdown_low`) on
   *other failure fixtures*, never on healthy controls. No acceptable-trigger
   mapping was added, per the rules. Stage 2 does now expose the raw timing
   that analysis needs — e.g. on `stagnation` episodes `loop` fires at step 4–6
   while `stagnation` itself fires at ~55.5, and on `stall` episodes `loop`
   fires at step 4 versus `stall` at 26 — available in the episode CSV's
   `first_detection_step` column without any scoring interpretation.
5. **TN is dominated by irrelevancy.** With 41 healthy controls and 16 other
   failure modes, specificity (0.924–1.000) mostly measures "stayed quiet when
   irrelevant". It is reported for completeness; precision/recall/F1 remain the
   comparable metrics.
6. **Seeds are not persisted in the fixture files** (Stage 1 finding). The
   generator's scheme (`random.Random(f"{episode_id}/seed")` per episode) is
   recorded in the JSON metadata instead; reproduction requires the generator,
   not the JSONL alone.
7. **Negative latencies are reported, not clamped.** None occur on this corpus
   (asserted), but the harness reports `first - onset` as-is so a future onset
   bug surfaces rather than being hidden at zero.
8. **`p95_meaningful` is a sample-size flag, not a statistical guarantee.** The
   ≥5 threshold is a documented convention for "distinct from the maximum".
9. **One detector-code path was observed, not changed:** the `silent_abort`
   finalize emission is attributed to the last ingested event, giving latency 0
   against an onset on that same event. That is the detector's own attribution
   and was left exactly as-is.
