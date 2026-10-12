# Stage 1 — Baseline Verification Audit

**Date:** 2026-10-08
**Branch:** `eval/detection-quality`
**Commit:** `356e8df943d1834992b3ca31de44484c5b3a4807`
**Working tree:** clean (only untracked `reports/` — this evidence directory)

Stage 1 scope: baseline verification only. No detector logic, thresholds, fixtures,
labels, scoring, or production behavior were modified. Nothing was implemented.

---

## 1. Repository state

Working tree was clean before branching; no uncommitted work existed to preserve.
`eval/detection-quality` was created from `docs/site-professional-restyle`
(commit `356e8df`, "style(site): restrained dark restyle, strip the landing-page
showmanship"). No reset/stash/checkout-overwrite was performed.

Relevant layout:

```
benchmarks/detection_accuracy.py          # TP/FP/F1 harness (issue #82)
benchmarks/enforcement_benchmark.py
benchmarks/overhead_benchmark.py
benchmarks/fixtures/generate_fixtures.py  # deterministic corpus generator
benchmarks/fixtures/labeled_episodes.jsonl   # 68 labeled episodes
benchmarks/fixtures/healthy_controls.jsonl   # 41 healthy controls
benchmarks/fixtures/goal_drift_baseline.json # baseline for goal_drift variant
src/snagline/detectors/                   # loop, error_cascade, latency_anomaly,
                                           # token_runaway, meltdown, silent_abort,
                                           # stagnation, side_effect_guard,
                                           # compaction_tripwire, goal_drift, ml_ensemble
src/snagline/drift/goal_drift.py          # semantic goal-drift (opt-in `drift` extra)
src/snagline/monitor.py                   # detector registration + episode replay
README.md                                 # claims under comparison
```

## 2. Test suite

**Command:** `python -m pytest` (per `pyproject.toml`: `testpaths = ["tests"]`,
`addopts = "--cov=snagline --cov-fail-under=80"`). Run from repo root with
`.venv/Scripts/python.exe`.

**Result:** `1550 passed, 2 skipped in 237.14s` — exit code 0.
Coverage: **92.40%** (gate requires 80%). 0 failed, 0 errors.

The 2 skips are dependency-gated, not test failures:

| Skipped at | Reason |
|---|---|
| `tests/test_ml_esn_ensemble.py:15` | `ml extra (numpy) not installed` |
| `tests/sinks/test_continuum_sink.py:125` | `could not import 'continuum'` |

Note: `numpy`, `sklearn`, `sentence_transformers`, and `torch` are **not installed**
in this venv. The 1550/2-skipped result is the correct full-suite outcome for this
environment. No failures, so no code was changed.

## 3. Detection benchmark

**Command (exactly as documented in README):**

```
python benchmarks/detection_accuracy.py --fixtures benchmarks/fixtures --format table
```

**Result — reproduced verbatim:**

```
trigger                 TP   FP   FN  precision   recall      f1
----------------------------------------------------------------
loop                     4    8    0      0.333    1.000   0.500
error_cascade            4    0    0      1.000    1.000   1.000
latency_anomaly          4    0    0      1.000    1.000   1.000
token_runaway            4    0    0      1.000    1.000   1.000
budget_breach            4    0    0      1.000    1.000   1.000
meltdown_low             4    4    0      0.500    1.000   0.667
meltdown_high            4    0    0      1.000    1.000   1.000
silent_abort             4    0    0      1.000    1.000   1.000
goal_drift               4    0    0      1.000    1.000   1.000
ml_ensemble              4    0    0      1.000    1.000   1.000
stagnation               4    0    0      1.000    1.000   1.000
side_effect_duplicate    4    0    0      1.000    1.000   1.000
governance_decay         4    0    0      1.000    1.000   1.000
cycle                    4    0    0      1.000    1.000   1.000
stall                    4    0    0      1.000    1.000   1.000
idle_gap                 4    0    0      1.000    1.000   1.000
wall_clock_budget        4    0    0      1.000    1.000   1.000
----------------------------------------------------------------
macro-F1: 0.951
episodes: 109 (68 labeled, 41 healthy controls)
confusion (firings on other data):
  stagnation -> loop: 4
  stall -> loop: 4
  stall -> meltdown_low: 4
healthy controls that fired: 0
```

**Exit code: 0** (the false-positive gate — any firing healthy control returns 1).

Aggregates: **macro-F1 0.951** (exact 0.95098 from the `--format json` run).
TP = 68, FP = 12, FN = 0 across the 17 gated triggers. Recall is 1.000 on every
trigger: all 68 labeled episodes were caught. All 12 FPs are cross-firings on
*other failure* fixtures (never on healthy traffic):

- `stagnation -> loop` (4): stagnation episodes genuinely contain loops
- `stall -> loop` (4): stall episodes genuinely contain loops
- `stall -> meltdown_low` (4): stall episodes also collapse entropy

**Healthy controls: 41 episodes, 0 fired.** This is the gate the harness exists for,
and it holds.

## 4. README comparison

README (lines 438–465) pins these numbers to commit `22faeae`. Actual run below.
Every line matched — no discrepancies, no manipulation.

| Metric | README | Actual Run | Match |
|---|---:|---:|---|
| Macro-F1 | 0.951 | 0.951 (0.95098) | ✅ |
| Loop precision | 0.333 | 0.333 | ✅ |
| Meltdown_low precision | 0.500 | 0.500 | ✅ |
| Meltdown_low F1 | 0.667 | 0.667 | ✅ |
| Healthy controls fired | 0 | 0 | ✅ |
| Episodes | 109 (68 labeled, 41 healthy) | 109 (68 labeled, 41 healthy) | ✅ |
| Loop TP/FP/FN | 4 / 8 / 0 | 4 / 8 / 0 | ✅ |
| Meltdown_low TP/FP/FN | 4 / 4 / 0 | 4 / 4 / 0 | ✅ |
| All other triggers | TP 4, FP 0, FN 0, P/R/F1 = 1.000 | identical | ✅ |
| Confusion entries | stagnation->loop 4, stall->loop 4, stall->meltdown_low 4 | identical | ✅ |

Note the README's pinned commit (`22faeae`) is **not** this branch's HEAD
(`356e8df`); the numbers still reproduce exactly at HEAD, so the intervening
commits did not perturb detection behavior.

## 5. `snagline bench` (overhead)

**Command:** `python -m snagline bench` (200,000 synthetic steps)

| Metric | Value |
|---|---:|
| median | 9.17–10.27 us/step (two runs) |
| p99 | 19.69–26.64 us/step |
| ingest only | median 6.34–6.81 us/step, p99 8.93–9.49 us/step |
| scaled max_window=512 | median 10.31–10.79 us/step |
| scaled max_window=2048 | median 10.94–12.37 us/step |

Run-to-run spread is normal benchmark noise on this machine. The README quotes
median 1.53 us/step, p99 1.66 us/step on **Apple M1 arm64, CPython 3.13.5** —
different hardware and OS, so those numbers are **not directly comparable** and
are not treated as a mismatch. This environment is Windows 11 / AMD64 /
CPython 3.13.1, roughly 6–7x slower per step, which is consistent with a
non-M1 machine running under a Windows + OneDrive-synced working tree.

## 6. Fixture analysis

109 episodes across 2 JSONL files (1 JSONL per file, 1 line per episode).
Top-level fields present across **all** episodes: `episode_id`, `events`, `label`,
plus `config` on the 12 variant episodes. Nothing else exists.

**Answers to the schema questions, from the actual files:**

1. **How is the expected trigger stored?** As `label: {"trigger": <name>}` — a
   single-trigger object. `label: null` (absent `label` value) marks a healthy
   control. The harness validates every label against `SHIPPED_TRIGGERS`
   (`parse_episode` rejects unknown triggers).
2. **How many steps/events per episode?** Ranges 6–101 events; median 13. Breakdown
   by length: 6–19 (short episodes), 23–34, and 56–101 (the long token-runaway /
   wall-clock-budget style episodes).
3. **Is failure onset explicitly represented?** **No.** No `onset`, `onset_step`,
   `inject_at`, `failure_start`, or equivalent field exists at the episode or event
   level (verified programmatically across all 109 episodes and every event).
4. **Can the first detector firing step currently be recovered?** **No.** The
   harness's `RiskCollector`/`EpisodeOutcome` records only *which* triggers fired
   (`outcome.predicted` is a `set[str]`), discarding the event/step at which each
   fired. No per-event index is retained.
5. **Are timestamps present?** Yes — `timestamp` (float epoch seconds) is required
   on every event (`_EVENT_FIELDS`, enforced by `_parse_event`). Example:
   `1700000000.0`.
6. **Are seeds present?** **No** — no `seed` field exists in any fixture. Seeding is
   internal to the generator (`random.Random(f"{ep}/seed")` in
   `generate_fixtures.py`), which makes generation reproducible, but the seed is
   not persisted into the emitted files.
7. **How are healthy episodes represented?** As `label: null` episodes living in
   `healthy_controls.jsonl` (41 of them). They replay through the same monitors;
   any emission counts as FP, and they can never produce an FN.

Event-level fields (all episodes): `step_id`, `episode_id`, `timestamp`,
`action_type`, `action_signature`, `tool_name`, `latency_ms`, `error`,
`error_type`, `tokens_in`, `tokens_out`, `side_effect`. `metadata` is permitted by
the parser but absent from the shipped fixtures. Unknown event fields are rejected.

## 7. Requested failure-mode coverage

Corpus has exactly 4 labeled episodes per trigger (17 triggers = 68 labeled).
Config variants: `standard` 97, `goal_drift` 6, `ml_ensemble` 6.

| Failure mode | Fixture exists? | Detector exists? | Currently evaluated? |
|---|---|---|---|
| loop | Yes (4) | Yes — `loop` | Yes (TP 4, FP 8, F1 0.500) |
| error cascade | Yes (4) | Yes — `error_cascade` | Yes (F1 1.000) |
| latency anomaly | Yes (4) | Yes — `latency_anomaly` | Yes (F1 1.000) |
| goal drift | Yes (4, `goal_drift` variant) | Yes — `goal_drift` (statistical) | Yes (F1 1.000) |
| **semantic goal drift** | **No** | **Yes** — `semantic_goal_drift` (inert) | **No** |
| stagnation | Yes (4) | Yes — `stagnation` | Yes (F1 1.000) |
| token runaway | Yes (4) | Yes — `token_runaway` | Yes (F1 1.000) |
| meltdown | Yes (8: 4 `meltdown_low` + 4 `meltdown_high`) | Yes — `meltdown` (one trigger, split by detail wording) | Yes (F1 0.667 / 1.000) |
| silent abort | Yes (4) | Yes — `silent_abort` | Yes (F1 1.000) |
| side-effect anomalies | Yes (4) | Yes — `side_effect_duplicate` | Yes (F1 1.000) |
| healthy behavior | Yes (41 controls) | (target class, not a detector) | Yes — FP gate, 0 fired |

Also present in the corpus but not in the request list: `budget_breach`,
`governance_decay`, `cycle`, `stall`, `idle_gap`, `wall_clock_budget`,
`ml_ensemble` — all 4 fixtures each, all F1 1.000.

### Semantic goal drift — dependency status

**Requires an optional dependency, which is not installed.** The detector
(`SemanticGoalDriftDetector`, `name = "semantic_goal_drift"`) lives in
`src/snagline/drift/goal_drift.py` and needs the `drift` extra
(`sentence-transformers>=2.2`). In this venv:

```
sentence_transformers: NOT INSTALLED (ModuleNotFoundError)
numpy:                 NOT INSTALLED (ModuleNotFoundError)
torch:                 NOT INSTALLED (ModuleNotFoundError)
```

Per Stage 1 constraints, **nothing was installed.** The detector is designed to be
*inert by default* rather than guess a semantics-free fallback: it logs
`"semantic goal-drift inert; baseline has no embeddings"` and emits nothing. It is
registered only when `cfg.semantic_drift_enabled and cfg.goal_drift_baseline is not
None` (`monitor.py:1348`), and **no harness config variant sets
`semantic_drift_enabled`** — so it never runs in the benchmark today, regardless of
whether the extra is installed. It also emits the `goal_drift` trigger rather than a
distinct one, which is why `INTENTIONALLY_UNGATED` documents it as a separate
label that would double-count.

## 8. Current evaluation gaps

From the actual harness code (`detection_accuracy.py`) — inspection only,
nothing implemented:

| Capability | Supported? | Evidence |
|---|---|---|
| TP | **Yes** | `TriggerStats.tp` |
| FP | **Yes** | `TriggerStats.fp` |
| FN | **Yes** | `TriggerStats.fn` |
| **TN** | **No** | `TriggerStats` has no `tn` field; true negatives are never counted anywhere |
| Precision | **Yes** | `TriggerStats.precision` |
| Recall | **Yes** | `TriggerStats.recall` |
| F1 | **Yes** | `TriggerStats.f1` |
| Macro-F1 | **Yes** | `ScoreReport.macro_f1` (mean F1 over triggers with support) |
| **Detection step** | **No** | `EpisodeOutcome.predicted` is a `set[str]`; the firing step index is discarded by `RiskCollector.emit` |
| **Detection latency** | **No** | no onset field in fixtures (gap 6.3) and no firing step recorded, so latency is not computable on either end |
| Per-trigger metrics | **Yes** | per-trigger TP/FP/FN/P/R/F1 rows + confusion map |
| Aggregate metrics | **Partially** | macro-F1 and confusion only; no micro/weighted accuracy, no TN, no healthy-vs-labeled ROC-style summary |
| Per-episode detail | **No** | `EpisodeOutcome` carries only `episode_id`, `label_triggers`, `predicted` — no per-episode firings, steps, or timing |
| FP gate | **Yes** | exit code 1 if any healthy control fires |

The three hard gaps for later stages are **TN**, **detection step**, and
**detection latency** — all three trace to two missing pieces: `TriggerStats` lacks
a TN counter, and neither the fixtures nor the replay path retain onset or firing-step
information.

## 9. Evidence files

All written by the commands above; none overwrote pre-existing content
(`reports/baseline/` did not exist before this run).

| File | Contents |
|---|---|
| `reports/baseline/output.txt` | detection benchmark table + JSON + `snagline bench`, with timestamps and exit codes |
| `reports/baseline/tests.txt` | full `pytest` run (1550 passed, 2 skipped, 92.40% coverage) |
| `reports/baseline/environment.txt` | OS, Python, interpreter, package versions, optional-extra status |
| `reports/baseline/git.txt` | branch, HEAD, commit metadata, status, fixture blob hashes |
| `reports/baseline/AUDIT.md` | this file |

## 10. Integrity notes

- The prior report's claim of "Linux x86_64, Python 3.12.3" is **wrong for this
  machine** — this is Windows 11 (10.0.26300) AMD64, CPython 3.13.1.
- The prior claim of "1546 passed" is also stale — the suite now reports
  **1550 passed, 2 skipped**.
- The prior claims of "macro-F1 0.951", "0 healthy controls fired", and "exit 0"
  are **confirmed accurate** by this independent run.
- No prior claimed commit hash (`4fffb1b`) exists in this branch's history;
  HEAD is `356e8df`.
- The benchmark was run unmodified; `SNAGLINE_CALIBRATION` was unset, so the
  harness used its default hand-tuned thresholds.
