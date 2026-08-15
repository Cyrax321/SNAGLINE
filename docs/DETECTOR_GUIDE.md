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
3. **Never raise on bad input.** The Monitor is fail-open and will swallow the
   exception, but a detector that throws on malformed fields is effectively
   disabled from then on. Validate defensively.
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

Each has a matching test in `tests/detectors/` showing the synthetic-sequence
pattern (injected failure fires; healthy sequence stays silent). Copy that
test shape for your own detector.
