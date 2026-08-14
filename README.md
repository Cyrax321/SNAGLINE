# SNAGLINE

**Lightweight, dependency-free companion library that watches any agent's
execution stream and flags loops and error cascades in real time** — cheap
enough to run on every step of a week-long unattended run, with a hard
fail-open guarantee so it can never crash or stall the agent it monitors.

Current build phase: **v0.1** (loop + error-cascade detection, console sink,
`raw` adapter, `replay` CLI). The latency/CUSUM detector, overhead benchmark,
and framework adapters are scheduled for later build steps.

## What it detects

- **Loops** — the same logical action repeating `N` times within a sliding
  window (catches retry storms and stuck agents).
- **Error cascades** — `N` consecutive errors, or `N` errors within a recent
  window (catches both fast and slow-burn failures).

Detection is deterministic and `O(1)` amortized per step. It runs with no
network calls and no LLM calls.

## What it does NOT do (explicit non-goals)

- **No goal-drift detection** — that needs embeddings and is a real
  dependency; planned for v2 at earliest.
- **No automatic repair** — detection and escalation only. Repair is a
  distinct, harder problem.
- **No claim to beat the source paper** (arXiv:2608.02464). v1 uses
  deterministic detectors; the paper's statistical ensemble is an optional
  later extra. No detection-accuracy number is claimed here until it has been
  measured honestly.
- **Not a replacement for human review** on high-stakes actions — it is a cheap
  first-pass signal that routes to existing escalation paths.

## Relationship to CONTINUUM

SNAGLINE is a **separate project**, not a module inside CONTINUUM. It is
framework-agnostic: it ingests a canonical `StepEvent` stream from any agent
runtime (a raw Python loop, LangChain, LangGraph, AutoGen, CrewAI, or
CONTINUUM's ledger). Keeping it separate preserves CONTINUUM's zero-dependency
guarantee and makes the tool adoptable by anyone running agents, not only
CONTINUUM users.

## Privacy

Detectors reason only about **hashes, timings, counts, and booleans** — never
raw prompt or response content. `action_signature` is a one-way SHA-256 digest,
and the `FailureRisk` payload a sink receives deliberately carries no
`metadata` field, so an alerting channel (webhook, Slack) cannot become an
accidental data-exfiltration path.

## Install

Core has **zero third-party dependencies** and works on Python 3.10+:

```
pip install .
```

(Extension extras like `[langchain]`, `[ml]`, `[slack]` are optional and add
dependencies only when you opt in.)

## Quick start (raw loop)

```python
from snagline import Monitor
from snagline.adapters.raw import watch

monitor = Monitor.default()  # loop + error-cascade detectors, console sink
with watch(monitor, "ep-1") as step:
    step("tool_call", tool_name="search", args="query", latency_ms=120, error=False)
    # ... your agent loop body ...
```

Any detected failure is printed as a JSON line to stderr, e.g.:

```json
{"episode_id": "ep-1", "step_id": "3", "score": 0.5, "trigger": "loop", "detail": "action repeated 3x in last 4 steps", "timestamp": 1718300000.0}
```

See `examples/raw_loop_example.py` for a runnable end-to-end example.

## Replay (offline analysis)

The same detectors run over an exported trajectory file:

```
snagline replay trajectory.jsonl --summary
```

Each line must be a JSON object with the `StepEvent` fields (see
`tests/fixtures/trajectories/` for worked examples).

## Fail-open guarantee

If a detector or sink raises, the exception is caught and logged, never
propagated into the host agent. Set `fail_open=False` only when you want strict
behavior (e.g. tests). This property is covered by `tests/test_monitor_fail_open.py`.

## License

MIT.
