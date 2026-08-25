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

## Optional detectors: baseline, goal-drift, and ensemble

`LoopDetector`, `ErrorCascadeDetector`, and `LatencyAnomalyDetector` ship in
`Monitor.default()`. Two further detectors are opt-in behind config flags so the
zero-dependency preset is unchanged.

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
base detectors in one orchestrator so there is no double counting.
