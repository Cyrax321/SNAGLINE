# SNAGLINE

**Lightweight, dependency-free companion library that watches any agent's
execution stream and flags loops and error cascades in real time** — cheap
enough to run on every step of a week-long unattended run, with a hard
fail-open guarantee so it can never crash or stall the agent it monitors.

Current build phase: **v0.1** (loop + error-cascade + latency/CUSUM detection,
console sink, `raw` adapter, `replay` + `bench` CLIs). Framework adapters
(LangChain, LangGraph, CONTINUUM, etc.) and the optional ML/goal-drift extras
are scheduled for later, explicitly-ordered build steps.

## What it detects

- **Loops** — the same logical action repeating `N` times within a sliding
  window (catches retry storms and stuck agents).
- **Error cascades** — `N` consecutive errors, or `N` errors within a recent
  window (catches both fast and slow-burn failures).
- **Latency anomalies** — a sustained deviation of a tool's `latency_ms` from
  its own running baseline, via a Welford/CUSUM statistic (stdlib only, no
  numpy). A short warm-up learns the baseline before any alarm can fire, so
  normal run-to-run jitter does not produce false positives.

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

## Runnable examples

All examples live under `examples/` and run against the source tree with
`PYTHONPATH=src`:

```
# Plain agent loop (the `raw` adapter); shows loop + error-cascade + latency risks
PYTHONPATH=src python3 examples/raw_loop_example.py
PYTHONPATH=src python3 examples/raw_loop_example.py --healthy   # clean run, no risks

# Offline trajectory replay (loop + error-cascade + latency)
PYTHONPATH=src python3 examples/replay_offline_trajectory.py

# Real LangChain run via the callback handler (needs the langchain extra)
pip install snagline-agent[langchain]
PYTHONPATH=src python3 examples/langchain_example.py            # repeated prompt -> loop

# Real-framework chaos harness: prove each detector fires on a genuine
# LangChain agent loop (no API key required; uses a real langchain_core Tool
# + callback path, with a scripted fake model). One mode per detector:
PYTHONPATH=src python3 examples/real_agent_demo.py --mode healthy   # expect SILENCE
PYTHONPATH=src python3 examples/real_agent_demo.py --mode loop      # loop
PYTHONPATH=src python3 examples/real_agent_demo.py --mode error     # error_cascade
PYTHONPATH=src python3 examples/real_agent_demo.py --mode latency   # latency_anomaly
# With OPENAI_API_KEY / ANTHROPIC_API_KEY + the matching package installed, the
# healthy run uses a REAL chat model as a false-positive check.
```

The same chaos scenarios, but driven through a **real LangChain 1.x
`create_agent` agent** (a LangGraph `CompiledStateGraph`) instead of a hand-rolled
loop, so the adapter is proven against the actual framework:

```
pip install langchain snagline-agent[langchain]
PYTHONPATH=src python3 examples/real_agent_executor_demo.py --mode loop
PYTHONPATH=src python3 examples/real_agent_executor_demo.py --mode error
PYTHONPATH=src python3 examples/real_agent_executor_demo.py --mode latency
PYTHONPATH=src python3 examples/real_agent_executor_demo.py --mode healthy
```

## Replay (offline analysis)

The same detectors run over an exported trajectory file:

```
snagline replay trajectory.jsonl --summary
```

Each line must be a JSON object with the `StepEvent` fields (see
`tests/fixtures/trajectories/` for worked examples, including an injected loop,
an injected error cascade, and an injected latency spike).

## Overhead

`ingest()` is cheap enough to run on every step of a long unattended run. The
claim is measured, not asserted -- `benchmarks/overhead_benchmark.py` (run via
`snagline bench`) times `Monitor.ingest()` over 200,000 synthetic steps:

```
median 1.4 us/step, p99 ~5-25 us/step   (measured on a 2026 Apple Silicon dev machine; run `snagline bench` to reproduce on yours)
```

This is comfortably under the sub-100-microsecond per-step target.

## Fail-open guarantee

If a detector or sink raises, the exception is caught and logged, never
propagated into the host agent. Set `fail_open=False` only when you want strict
behavior (e.g. tests). This property is covered by `tests/test_monitor_fail_open.py`.

## License

MIT.
