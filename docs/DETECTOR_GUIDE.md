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
- `detectors/goal_drift.py` - compares a live run to a persisted `BaselineProfile`
- `detectors/ml_ensemble.py` - `MLOrchestrator` combining base detector scores
- `detectors/token_runaway.py` - token-volume CUSUM + per-episode budget envelope
- `detectors/meltdown.py` - sliding-window entropy collapse/thrash detection
- `detectors/silent_abort.py` - end-of-episode completion check via `finalize`

Each has a matching test in `tests/detectors/` showing the synthetic-sequence
pattern (injected failure fires; healthy sequence stays silent). Copy that
test shape for your own detector.

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

## Restart-survivable state: snapshot/restore (issue #91)

Detectors that implement the duck-typed `dump_state()` / `load_state(state)`
pair (`detectors.base.StatefulDetector`; JSON-compatible data only, never
pickle) participate in `Monitor.snapshot(path)` / `Monitor.restore(path)`.
Snapshots are written atomically (tmp + `os.replace`). Restore is a
setup-time operation: a version or strict-composition mismatch raises rather
than failing open -- a monitor that silently starts blank is exactly the quiet
misbehavior this exists to prevent. All shipped detectors implement it, and
`DedupSink` persists cooldowns when used with its default key function.
