# SNAGLINE — Real-Time Failure Detection for AI Agents

**One-liner:** a lightweight, dependency-free companion library that watches
any agent's execution stream — LangChain, LangGraph, AutoGen, CrewAI, a raw
API loop, or CONTINUUM — and flags loops, error cascades, and behavioral
anomalies in real time, cheaply enough to run on every step of a week-long
unattended run.

*(Working name — rename freely. "SnagLine" was chosen because the core
detectors are deliberately dumb, cheap, and immediate, like an actual
snagline, not a smart sensor. Check PyPI name availability before committing.)*

---

## 0. What this is, what it is not, and how it relates to CONTINUUM

**What it is:** the productionized version of the "cheap, real-time,
framework-agnostic failure detection" idea, built as its own repo so it can
be adopted by anyone running agents — not only people who also run CONTINUUM.

**What it is not:** not a fork or extension of CONTINUUM, not a replacement
for CONTINUUM's crash-recovery guarantees, and not a claim to beat the
detection numbers in the source paper (arXiv:2608.02464, *Real-Time
Detection and Repair of LLM Agent Failures*) — v1 doesn't even attempt their
ML approach; it starts with deterministic detectors and treats the paper's
statistical ensemble as a later, optional extra.

**Relationship to CONTINUUM, and why this is a separate repo:** the
reasoning your other agent gave you is correct and this spec is built on
top of it, not around it:
- CONTINUUM's SDK is stdlib-only. This tool's eventual statistical detector
  (echo-state network, CUSUM) wants `numpy`/`scikit-learn`. Keeping it
  separate means CONTINUUM's zero-dependency guarantee never breaks.
- The "free telemetry" advantage survives the split: the CONTINUUM adapter
  here reads CONTINUUM's `Storage` by sequence — CONTINUUM's own public,
  already-stable API — so it's still zero new instrumentation on the
  CONTINUUM side.
- A standalone tool that ingests any structured step stream is a stronger,
  more general story than a third bolt-on extension, and it's the one most
  likely to attract outside users/stars independent of CONTINUUM's adoption.
- It mirrors the scope discipline already established: recovery (CONTINUUM
  core) and observability (this tool) are different concerns with different
  dependency profiles and different audiences. Keep the boundary clean.

**Non-goals for v1, explicitly:**
- Not attempting goal-drift detection (needs embeddings, a real dependency,
  and is genuinely harder — v2 at earliest).
- Not attempting to *repair* failures automatically — detection and
  escalation only. Repair is a distinct, harder problem (see DARC,
  arXiv:2608.11772, for where that line of work is headed) and folding it
  in now would blow the scope of a companion tool.
- Not claiming to replace human review for high-stakes actions — it's a
  cheap first-pass signal that routes to existing escalation paths
  (CONTINUUM's `REQUIRES_REVIEW`, a webhook, a Slack alert), not a
  final arbiter.

---

## 1. Design principles (non-negotiable — a coding agent should treat these as constraints, not suggestions)

1. **Zero mandatory dependencies in core.** `import snagline` must work with
   nothing but the Python standard library. Every framework integration,
   every ML detector, every notification sink beyond console/webhook is an
   optional extra (`pip install snagline-agent[langchain]`, etc.).
2. **Fail-open, always.** If a detector or sink raises an exception, it is
   caught, logged, and ignored — it must never propagate into the host
   agent and never block or slow the agent's actual work. A monitoring
   library that can crash or stall the thing it's monitoring is a
   non-starter for adoption. This is the single most important property
   in the whole project and must have a dedicated test.
3. **Framework-agnostic core, framework-coupled adapters.** All detector and
   sink logic operates only on the canonical `StepEvent` schema. All
   framework-specific code (LangChain callbacks, LangGraph node hooks, etc.)
   lives in isolated `adapters/` modules and nowhere else.
4. **No content retention by default.** Detectors reason about structure —
   hashes, timing, counts, boolean flags — not prompt or response content.
   This is both a privacy property and an adoption requirement: teams
   should be able to drop this into a production agent without a
   data-handling review.
5. **Cheap enough to run on every step.** Tier-1 detectors (loop,
   error-cascade, latency-anomaly) must be O(1) amortized per step, no
   network calls, no LLM calls, targeting sub-100-microsecond overhead per
   `ingest()` call. This must be benchmarked and the number published in
   the README, not just asserted.
6. **Streaming-first, batch-capable.** Primary use is live monitoring of a
   running agent, but the same event schema and detectors must also work
   over an exported trajectory file for offline analysis/testing
   (`snagline replay trajectory.jsonl`).
7. **Small enough to extend in an afternoon.** Writing a new adapter or a
   new detector should be achievable in under ~50 lines against a documented
   protocol — this is what makes "any agent in the world" a realistic claim
   rather than a slogan.

---

## 2. Architecture overview

```mermaid
flowchart LR
    subgraph Agents["Any Agent Runtime"]
        A1[Raw API loop]
        A2[LangChain]
        A3[LangGraph]
        A4[AutoGen / CrewAI]
        A5[CONTINUUM ledger]
        A6[Claude Code hooks]
    end

    subgraph Adapters["Adapters — the only framework-coupled code"]
        AD1[raw.py]
        AD2[langchain.py]
        AD3[langgraph.py]
        AD4[autogen.py / crewai.py]
        AD5[continuum.py]
        AD6[claude_code.py]
    end

    subgraph Core["Core — zero dependencies"]
        SE[StepEvent schema]
        MON[Monitor orchestrator<br/>fail-open guarantee]
        D1[Loop detector]
        D2[Error-cascade detector]
        D3[Latency/CUSUM detector]
    end

    subgraph Optional["Optional extras"]
        ML[ML ensemble<br/>snagline[ml]]
        DR[Goal-drift detector<br/>snagline[drift]]
    end

    subgraph Sinks["Sinks — pluggable escalation"]
        S1[Console — default]
        S2[Webhook]
        S3[CONTINUUM REQUIRES_REVIEW]
        S4[Slack / PagerDuty]
    end

    A1 --> AD1 --> SE
    A2 --> AD2 --> SE
    A3 --> AD3 --> SE
    A4 --> AD4 --> SE
    A5 --> AD5 --> SE
    A6 --> AD6 --> SE
    SE --> MON
    MON --> D1 & D2 & D3
    MON -.optional.-> ML
    MON -.optional.-> DR
    D1 & D2 & D3 & ML & DR --> MON
    MON --> S1 & S2
    MON -.optional.-> S3
    MON -.optional.-> S4
```

**Data flow, in words:** an adapter observes something happening in the
host agent (a tool call, a node execution, a ledger append) and normalizes
it into a `StepEvent`. It calls `monitor.ingest(event)`. The monitor runs
every registered detector against that event and the detector's own
internal per-episode state (a sliding window, a running mean). If a
detector's `observe()` returns a `FailureRisk`, the monitor dispatches it to
every registered sink. Nothing in this path calls an LLM, makes a network
request (except an explicitly-configured webhook/Slack sink, which is
itself fire-and-forget and never blocks `ingest()`), or reads message
content.

---

## 3. Repository structure

```
snagline/
├── README.md
├── ARCHITECTURE.md                 # this document, trimmed for repo consumption
├── LICENSE                         # MIT
├── pyproject.toml
├── src/
│   └── snagline/
│       ├── __init__.py             # exports Monitor, StepEvent, watch()
│       ├── events.py                # StepEvent, EpisodeMeta, make_signature()
│       ├── monitor.py               # Monitor orchestrator, fail-open wrapper
│       ├── risk.py                  # FailureRisk dataclass
│       ├── config.py                # Config dataclass, all tunable thresholds
 │       ├── cli.py                   # `snagline watch|replay|baseline|bench`
 │       ├── baseline.py              # fit/save/load healthy BaselineProfile (stdlib)
 │       ├── detectors/
 │       │   ├── __init__.py
 │       │   ├── base.py              # Detector protocol
 │       │   ├── loop.py
 │       │   ├── error_cascade.py
 │       │   ├── latency_anomaly.py   # Welford + CUSUM, stdlib `statistics` only
 │       │   ├── goal_drift.py        # OPT-IN: live vs healthy BaselineProfile (built)
 │       │   └── ml_ensemble.py       # OPT-IN: MLOrchestrator noisy-OR (built; model= hook)
 │       ├── ml/                      # optional extra: snagline-agent[ml] (research path, NOT YET BUILT)
 │       │   ├── __init__.py
 │       │   └── esn_ensemble.py      # echo-state-network detector
 │       ├── drift/                   # optional extra: snagline-agent[drift] (research path, NOT YET BUILT)
 │       │   ├── __init__.py
 │       │   └── goal_drift.py        # semantic-drift (sentence-transformers) detector
│       ├── sinks/
│       │   ├── __init__.py
│       │   ├── base.py              # AlertSink protocol
│       │   ├── console.py           # default, zero dep
│       │   ├── webhook.py           # stdlib urllib, zero dep
│       │   ├── continuum_sink.py    # optional extra: snagline-agent[continuum]
│       │   └── slack.py             # optional extra: snagline-agent[slack]
│       ├── adapters/
│       │   ├── __init__.py
│       │   ├── raw.py               # context manager + decorator, zero dep
│       │   ├── langchain_adapter.py  # optional extra: snagline-agent[langchain]
│       │   ├── langgraph_adapter.py  # optional extra: snagline-agent[langgraph]
 │       │   ├── autogen_adapter.py    # built as adapters/autogen.py (duck-typed)
 │       │   ├── crewai_adapter.py     # built as adapters/crewai.py (duck-typed)
│       │   ├── openai_adapter.py     # optional extra: snagline-agent[openai]
│       │   ├── anthropic_adapter.py  # optional extra: snagline-agent[anthropic]
│       │   ├── claude_code_adapter.py # optional; verify current hooks API first — see §5.7
│       │   └── continuum_adapter.py  # reads CONTINUUM Storage by sequence
│       └── server/                   # optional sidecar mode for non-Python agents
│           ├── __init__.py
│           └── http_server.py        # stdlib http.server, POST /events
├── tests/
│   ├── test_events.py
│   ├── test_monitor_fail_open.py     # the single most important test file
│   ├── detectors/
│   │   ├── test_loop.py
│   │   ├── test_error_cascade.py
│   │   └── test_latency_anomaly.py
│   ├── adapters/
│   │   ├── test_raw_adapter.py
│   │   ├── test_langchain_adapter.py
│   │   └── test_continuum_adapter.py
│   ├── test_replay_cli.py
│   └── fixtures/
│       └── trajectories/
│           ├── healthy_run.jsonl
│           ├── injected_loop.jsonl
│           └── injected_error_cascade.jsonl
├── examples/
│   ├── raw_loop_example.py
│   ├── langchain_example.py
│   ├── langgraph_example.py
│   ├── continuum_example.py
│   └── replay_offline_trajectory.py
├── benchmarks/
│   ├── overhead_benchmark.py         # proves the sub-100µs/step claim
│   └── detection_accuracy.py         # replay against fixtures, report recall/false-positive rate
└── docs/
    ├── ADAPTER_GUIDE.md              # how to write a new adapter in <50 lines
    ├── DETECTOR_GUIDE.md             # how to write a new detector
    └── INTEGRATION_CONTINUUM.md
```

---

## 4. Core schemas and protocols

### 4.1 `events.py` — the canonical wire format (stdlib only)

```python
from dataclasses import dataclass, field
import hashlib

@dataclass(frozen=True, slots=True)
class StepEvent:
    step_id: str
    episode_id: str
    timestamp: float                  # unix epoch seconds, float for sub-second precision
    action_type: str                  # "tool_call" | "message" | "plan_step" | "observation" | adapter-defined
    action_signature: str              # normalized hash — see §4.2 for construction rules
    tool_name: str | None = None
    latency_ms: float | None = None
    error: bool = False
    error_type: str | None = None
    tokens_in: int | None = None
    tokens_out: int | None = None
    metadata: dict = field(default_factory=dict)  # adapter-specific extras; detectors never read this

@dataclass(frozen=True, slots=True)
class EpisodeMeta:
    episode_id: str
    agent_name: str | None = None
    started_at: float | None = None
    tags: dict = field(default_factory=dict)
```

**Only five fields are load-bearing for tier-1 detection:** `step_id`,
`episode_id`, `timestamp`, `action_signature`, `error`. Everything else is
optional and improves detector quality but nothing in core requires it.
This is what keeps the adapter-writing bar low.

### 4.2 `make_signature()` — normalization rules (write this down explicitly; it's the part people get wrong)

```python
def make_signature(action_type: str, tool_name: str | None, *stable_parts: str) -> str:
    """
    Build a loop-detectable signature. Rules for adapter authors:
    - Include the logical action (tool name, target element, endpoint) —
      the things that make two actions "the same attempt."
    - EXCLUDE volatile fields: timestamps, request/session ids, nonces,
      retry counters — including these defeats loop detection by making
      every retry look unique.
    - Hashing already-sensitive values (a full prompt, a password field) is
      fine for privacy (SHA-256 is one-way) but prefer hashing only the
      minimum needed to detect repetition, not the full payload, to keep
      signatures meaningful and short.
    """
    raw = "||".join([action_type, tool_name or "", *stable_parts])
    return hashlib.sha256(raw.encode()).hexdigest()[:16]
```

### 4.3 `risk.py`

```python
from dataclasses import dataclass
from typing import Literal

TriggerType = Literal["loop", "error_cascade", "latency_anomaly", "goal_drift", "ml_ensemble"]

@dataclass(frozen=True, slots=True)
class FailureRisk:
    episode_id: str
    step_id: str
    score: float           # 0.0 - 1.0
    trigger: TriggerType
    detail: str             # short human-readable explanation, no raw content
    timestamp: float
```

### 4.4 `detectors/base.py` — the extension point

```python
from typing import Protocol
from snagline.events import StepEvent
from snagline.risk import FailureRisk

class Detector(Protocol):
    name: str
    def observe(self, event: StepEvent) -> FailureRisk | None: ...
    def reset(self, episode_id: str) -> None: ...
```

### 4.5 `sinks/base.py` — the other extension point

```python
from typing import Protocol
from snagline.risk import FailureRisk

class AlertSink(Protocol):
    def emit(self, risk: FailureRisk) -> None: ...
```

### 4.6 `monitor.py` — the orchestrator, fail-open by construction

```python
import logging
import threading
from snagline.events import StepEvent
from snagline.detectors.base import Detector
from snagline.sinks.base import AlertSink

logger = logging.getLogger("snagline")

class Monitor:
    def __init__(self, detectors: list[Detector], sinks: list[AlertSink], fail_open: bool = True):
        self._detectors = detectors
        self._sinks = sinks
        self._fail_open = fail_open
        self._lock = threading.Lock()   # per-instance; supports concurrent multi-episode use

    def ingest(self, event: StepEvent) -> None:
        with self._lock:
            for detector in self._detectors:
                try:
                    risk = detector.observe(event)
                except Exception:
                    logger.exception("snagline detector %s raised; ignoring (fail-open)", detector.name)
                    if not self._fail_open:
                        raise
                    continue
                if risk is not None:
                    self._dispatch(risk)

    def _dispatch(self, risk) -> None:
        for sink in self._sinks:
            try:
                sink.emit(risk)
            except Exception:
                logger.exception("snagline sink %s raised; ignoring (fail-open)", type(sink).__name__)
                if not self._fail_open:
                    raise

    async def ingest_async(self, event: StepEvent) -> None:
        # thin wrapper for async adapters (LangGraph/AutoGen async mode);
        # runs the same sync path — detectors are cheap enough this is safe
        self.ingest(event)
```

---

## 5. Detectors

### 5.1 Loop detector (`detectors/loop.py`)

Deterministic, O(1) amortized. Per-episode sliding window (`collections.deque`)
of recent `action_signature` values. If the same signature appears
`repeat_threshold` times within `window_size` steps, emit a risk.

```python
class LoopDetector:
    name = "loop"
    def __init__(self, window_size: int = 12, repeat_threshold: int = 3):
        self._windows: dict[str, deque[str]] = {}
        self.window_size = window_size
        self.repeat_threshold = repeat_threshold

    def observe(self, event: StepEvent) -> FailureRisk | None:
        w = self._windows.setdefault(event.episode_id, deque(maxlen=self.window_size))
        w.append(event.action_signature)
        count = w.count(event.action_signature)
        if count >= self.repeat_threshold:
            score = min(1.0, count / self.repeat_threshold * 0.5)
            return FailureRisk(event.episode_id, event.step_id, score, "loop",
                                f"action repeated {count}x in last {len(w)} steps", event.timestamp)
        return None

    def reset(self, episode_id: str) -> None:
        self._windows.pop(episode_id, None)
```

### 5.2 Error-cascade detector (`detectors/error_cascade.py`)

Same sliding-window shape, tracks `error` booleans instead of signatures.
Fires when error rate within the window crosses a threshold (default: 3 of
last 10, or 3 consecutive — implement both, consecutive catches fast
cascades, windowed catches slow-burn ones).

### 5.3 Latency/CUSUM anomaly detector (`detectors/latency_anomaly.py`)

Stdlib-only. Maintains running mean/variance per `tool_name` via Welford's
algorithm (no numpy needed), then a CUSUM statistic:

```python
class LatencyAnomalyDetector:
    name = "latency_anomaly"
    def __init__(self, k: float = 0.5, h: float = 5.0):
        self._stats: dict[tuple[str, str], _WelfordCUSUM] = {}
        self.k = k   # slack parameter
        self.h = h   # alarm threshold

    def observe(self, event: StepEvent) -> FailureRisk | None:
        if event.latency_ms is None:
            return None
        key = (event.episode_id, event.tool_name or "default")
        state = self._stats.setdefault(key, _WelfordCUSUM(self.k, self.h))
        alarmed = state.update(event.latency_ms)
        if alarmed:
            return FailureRisk(event.episode_id, event.step_id, 0.6, "latency_anomaly",
                                f"latency deviates from baseline for {event.tool_name}", event.timestamp)
        return None
```

`_WelfordCUSUM` implementation: standard Welford running mean/std update,
then `cusum = max(0, cusum + (x - mean)/std - k)`, alarm when
`cusum > h`. This is the CUSUM-with-alarms approach the source paper uses,
minus their echo-state-network layer — that's the optional `ml` extra, not
tier-1.

### 5.4 Config (`config.py`)

```python
@dataclass
class Config:
    loop_window_size: int = 12
    loop_repeat_threshold: int = 3
    cascade_window_size: int = 10
    cascade_error_threshold: int = 3
    cascade_consecutive_threshold: int = 3
    cusum_k: float = 0.5
    cusum_h: float = 5.0
    fail_open: bool = True
```

All thresholds tunable at `Monitor` construction; ship sensible defaults so
`Monitor.default()` works with zero configuration.

---

## 6. Adapters

Each adapter's only job: turn a framework-specific event into a
`StepEvent` and call `monitor.ingest()`. None of them contain detection
logic.

### 6.1 `raw.py` — for anyone with a plain loop (no framework)

```python
from contextlib import contextmanager

@contextmanager
def watch(monitor, episode_id: str):
    step_counter = itertools.count()
    def step(action_type: str, tool_name: str | None = None, **kwargs):
        sig = make_signature(action_type, tool_name, str(kwargs.get("args", "")))
        event = StepEvent(step_id=str(next(step_counter)), episode_id=episode_id,
                           timestamp=time.time(), action_type=action_type,
                           action_signature=sig, tool_name=tool_name, **kwargs)
        monitor.ingest(event)
    yield step
```
Usage: `with snagline.watch(monitor, episode_id) as step: ... step("tool_call", tool_name="search", latency_ms=120, error=False)`.
This is likely the single most-used adapter — most real agent code today is
a custom loop, not a framework.

### 6.2 `langchain_adapter.py`

Implements `BaseCallbackHandler`; maps `on_tool_start`/`on_tool_end`/
`on_tool_error`/`on_agent_action`/`on_agent_finish` to `StepEvent`s. Verify
current callback method signatures against the installed LangChain version
at build time — this API has changed across versions before.

### 6.3 `langgraph_adapter.py`

LangGraph exposes a stream of state updates per node/superstep via
`graph.stream(...)`. Wrap each node execution (start/end timestamps →
latency, node name → tool_name, node output containing an error key →
error flag). Confirm current streaming API shape against the installed
LangGraph version before implementing — this is a fast-moving library.

### 6.4 `autogen.py`, `crewai.py` (built)

Shipped as `adapters/autogen.py` and `adapters/crewai.py` (the spec's
`autogen_adapter.py` / `crewai_adapter.py` naming was shortened). Both are
duck-typed: they read framework event objects via `model_dump()` with attribute/
dict fallbacks, so they import without the framework installed and never
hard-couple to a release.

- Autogen: `SnaglineAutogenHandler` (observer fed each event) plus
  `run_and_monitor()` wrapping `agent.run_stream`.
- CrewAI: `snagline_step_callback()` returns the `Agent(step_callback=...)`
  hook; `observe_crewai_step()` is the manual equivalent.

Install with `pip install snagline-agent[autogen]` / `[crewai]`.

### 6.5 `openai_adapter.py`, `anthropic_adapter.py`

Explicit wrapper functions around client `.create()`/`.messages.create()`
calls (not monkeypatching) for people using a raw SDK without any
orchestration framework at all — this is the "raw loop" case's most common
concrete form.

### 6.6 `continuum_adapter.py`

Reads CONTINUUM's `Storage` by sequence (its existing public API) and
translates ledger entries — including the `PERCEPTION_OBSERVED`,
`BRANCH_RESOLVED`, and action-ledger claim/perform/complete events from the
security-extension work already done — into `StepEvent`s. This is the
"free telemetry" case: zero new instrumentation on the CONTINUUM side,
just a consumer of the existing hash-chained ledger. **Verify the exact
`Storage` read-by-sequence method signature against the current CONTINUUM
source before implementing** — don't assume the API shape described here
is exact; confirm against the real repo.

### 6.7 `claude_code_adapter.py`

Claude Code supports lifecycle hooks (commonly including events around
tool use and session lifecycle) that can run a shell command. This adapter
would translate hook invocations into `StepEvent`s by having the hook
script POST to the sidecar server (§7) or write JSONL that `snagline
replay` can consume live. **Treat this adapter as the lowest-confidence
item in this spec** — the current hook names, payload shape, and
configuration mechanism should be verified against Claude Code's live
documentation before implementation, since this is exactly the kind of
product surface detail that changes over time. Build this one last.

---

## 7. Sidecar server mode (`server/http_server.py`)

For non-Python agents (a TypeScript LangGraph.js app, a Node-based
orchestrator, anything that isn't Python) — a minimal stdlib
`http.server`-based endpoint:

```
POST /events        body: StepEvent as JSON      → monitor.ingest()
GET  /health         → 200 OK
```

No framework (no Flask/FastAPI) — stdlib `http.server` is enough for a
low-throughput internal sidecar and keeps the zero-dependency principle
intact even for the server mode. Document that high-throughput production
use should front this with a real ASGI server if needed — that's the
user's infra choice, not this library's concern.

---

## 8. Sinks

- **`console.py`** (default): writes `FailureRisk` as a JSON line to
  stderr. Zero dependency.
- **`webhook.py`**: POSTs `FailureRisk` JSON via stdlib `urllib.request`,
  fire-and-forget with a short timeout, never blocks `ingest()`. Zero
  dependency.
- **`continuum_sink.py`** (optional extra): converts a `FailureRisk` into a
  `REQUIRES_REVIEW` event and appends it to CONTINUUM's ledger via its
  existing `request_human` mechanism — this is the closing-the-loop piece,
  giving all three CONTINUUM security extensions (branch-steering gate,
  periodic revalidation, and this) one shared human-escalation path.
- **`slack.py`** (optional extra): posts to a configured webhook URL with
  a formatted message. Only ever transmits `FailureRisk` fields (score,
  trigger, ids, timestamps) — never raw `StepEvent.metadata` — by default,
  so an alerting channel can't become an accidental data-exfiltration path.

---

## 9. CLI (`cli.py`)

```
snagline watch --adapter raw --sink console          # live mode, dev/debug
snagline replay trajectory.jsonl                      # offline batch analysis
snagline baseline <trajectory>.jsonl [--output baseline.json]   # fit a healthy profile
snagline bench                                        # runs overhead_benchmark.py, prints µs/step
```

---

## 10. Testing and benchmark strategy

- **`test_monitor_fail_open.py` first, before anything else.** A detector
  and a sink that deliberately raise on every call must not propagate the
  exception out of `Monitor.ingest()`, with `fail_open=True` (default) and
  must propagate with `fail_open=False`. This is the property the whole
  adoption pitch rests on.
- Per-detector unit tests with synthetic `StepEvent` sequences constructed
  to contain a known loop / known cascade / known latency spike — assert
  detection and, equally important, assert **no false positive** on a
  healthy synthetic sequence.
- `fixtures/trajectories/`: hand-built JSONL files with injected failures,
  used by both tests and `benchmarks/detection_accuracy.py` — this doubles
  as your demo artifact.
- `benchmarks/overhead_benchmark.py`: runs `Monitor.ingest()` in a tight
  loop over N synthetic events, reports median/p99 microseconds per call.
  Publish the number in the README; this is the credibility claim for
  "cheap enough to run on every step," so it needs to be a real, reproduced
  measurement, not an assumption carried over from the source paper.
- Adapter tests use mocked framework callback invocations (mock LangChain
  `BaseCallbackHandler` calls, etc.) — don't require a live LangChain agent
  to run in CI.

---

## 11. Privacy and security design notes

- Detectors operate on hashes, timings, and boolean flags — never on
  prompt/response content. State this explicitly in the README; it's an
  actual adoption blocker if left ambiguous.
- `action_signature` is a one-way hash (SHA-256), so even if an adapter
  author includes sensitive values in the hash input, the signature itself
  isn't reversible — but document the normalization guidance in §4.2 so
  authors don't put volatile fields in and accidentally defeat detection.
- The `metadata` dict on `StepEvent` is the one place raw content could
  leak if an adapter author puts it there. Document clearly: detectors
  never read `metadata`, and sinks should not forward it by default —
  `FailureRisk` deliberately does not carry a `metadata` field, so there's
  no path for it to reach a webhook or Slack channel unless a custom sink
  is written to do so explicitly.

---

## 12. Packaging

```toml
[project]
name = "snagline-agent"
requires-python = ">=3.10"
dependencies = []   # core: zero required deps, non-negotiable

[project.optional-dependencies]
langchain  = ["langchain-core>=0.2"]
langgraph  = ["langgraph>=0.2"]
autogen    = ["pyautogen>=0.2"]
crewai     = ["crewai>=0.30"]
openai     = ["openai>=1.0"]
anthropic  = ["anthropic>=0.30"]
continuum  = []   # depends on CONTINUUM's actual package name/version — confirm before release
ml         = ["numpy>=1.24", "scikit-learn>=1.3"]
drift      = ["sentence-transformers>=2.2"]
slack      = ["httpx>=0.27"]
all        = ["snagline-agent[langchain,langgraph,autogen,crewai,openai,anthropic,continuum,ml,drift,slack]"]
```

MIT license (matches the open-source, low-friction-adoption goal — avoid
anything copyleft here, since the whole point is "embed this into any
project easily").

---

## 13. Build sequencing — do not reorder

1. `events.py` + `risk.py` + `Detector`/`AlertSink` protocols +
   `Monitor` with the fail-open guarantee. Write `test_monitor_fail_open.py`
   before writing a single detector.
2. Loop detector + error-cascade detector + console sink + `raw.py`
   adapter + `snagline replay` CLI. **This is v0.1 — ship it.** Zero
   dependencies, installable, immediately useful to a stranger with no
   training data and no setup beyond `pip install`.
3. Latency/CUSUM detector (still zero deps).
4. `benchmarks/overhead_benchmark.py` — publish the microseconds/step
   number. This is your credibility artifact; don't skip it or leave it
   for later.
5. `langchain_adapter.py` — highest-leverage framework for adoption.
6. `continuum_adapter.py` + `continuum_sink.py` — closes the loop back to
   the main project; verify the real `Storage` API first.
7. `langgraph_adapter.py`.
8. `docs/ADAPTER_GUIDE.md` + `docs/DETECTOR_GUIDE.md` — written once
   you've built enough of each to document the pattern honestly.
9. `ml` extra (echo-state-network ensemble) — only after the deterministic
   core has real usage or real benchmark numbers to compare against. This
   is the research-grade piece; don't let it block the useful, simple
   v0.1.
 10. `autogen_adapter.py`, `crewai_adapter.py`, `claude_code_adapter.py`,
     sidecar server mode, `drift` extra — build in response to actual
     demand, not preemptively.

### 13.1 Status of the sequencing (as of 2026-08-19)

- Steps 1-8 (v0.1 core: events/risk/Monitor, loop + error-cascade +
  latency detectors, console sink, `raw` adapter, `replay`, overhead
  benchmark, LangChain, LangGraph, the two detector/adapter guides) are
  **done**.
- Step 9 (`ml` extra, echo-state-network ensemble) and the `drift` extra
  (semantic goal-drift) are **not yet built** — the deterministic
  `goal_drift` and `ml_ensemble` detectors in `detectors/` were shipped
  first as the simpler, dependency-free path (see §15).
- Step 10: Autogen and CrewAI adapters are **done** (duck-typed, see §6.4).
  `claude_code_adapter.py`, `continuum_adapter.py`/`continuum_sink.py`,
  `openai_adapter.py`, `anthropic_adapter.py`, and the sidecar server mode
  are **still pending**.

---

## 14. Definition of done for v0.1 (the ship-it milestone)

Status: **satisfied as of 2026-08-19** (see §15). Checkboxes reflect the
actual state.

- [x] Core installs with zero third-party dependencies
- [x] `test_monitor_fail_open.py` passes: detector/sink exceptions never
      propagate when `fail_open=True`
- [x] Loop, error-cascade, and latency-anomaly detectors pass unit tests
      against both injected-failure and healthy synthetic trajectories
      (no false positives on the healthy case)
- [x] `snagline replay` works against the fixture trajectories
- [x] `benchmarks/overhead_benchmark.py` runs and reports a real,
      reproducible microseconds/step number in the README
- [x] `raw.py` adapter documented with a working example in
      `examples/raw_loop_example.py`
- [x] README states plainly: what this detects, what it doesn't (goal
      drift, automatic repair), and that it's a companion to, not a part
      of, CONTINUUM

No claim anywhere in the docs that this matches or beats the source
paper's detection numbers until `benchmarks/detection_accuracy.py` has
actually been run against a comparable dataset and the result is honestly
reported either way.

---

## 15. Status (as of 2026-08-19)

Implementation status against the spec above. All merged to `master` with
green CI (ruff + mypy + pytest, py3.10-3.13; 129 tests passing, 1 skipped).

### Shipped

- **Core v0.1** (steps 1-2, 4): `events`, `risk`, `Monitor` (fail-open),
  loop, error-cascade, latency/CUSUM detectors, console + webhook sinks,
  `raw` adapter, `snagline replay`, overhead benchmark. Zero required deps.
- **`snagline baseline` command** (§9): `src/snagline/baseline.py` fits a
  `BaselineProfile` (per-tool latency mean/std/min/max + error rate) from a
  JSONL trajectory; `save_baseline`/`load_baseline` round-trip it. Exposed at
  top level.
- **`GoalDriftDetector`** (opt-in, `detectors/goal_drift.py`): compares a live
  run to a persisted `BaselineProfile`; flags rising error rate, latency
  blowing past the healthy mean by `goal_drift_latency_k` sigmas, or unseen
  tools. Zero-variance baselines use a floored spread so tiny deviations are
  not treated as infinite z.
- **`MLOrchestrator`** (opt-in, `detectors/ml_ensemble.py`): wraps the base
  detectors and combines their scores with a transparent noisy-OR; a real
  model can be injected via `model=` (the `ml` extra provides scikit-learn).
- **Autogen adapter** (`adapters/autogen.py`): `SnaglineAutogenHandler` +
  `run_and_monitor` wrapping `agent.run_stream`. Duck-typed. Hardened to raise
  clear errors and guarantee per-episode teardown.
- **CrewAI adapter** (`adapters/crewai.py`): `snagline_step_callback` for
  `Agent(step_callback=...)` plus `observe_crewai_step`. Duck-typed. Hardened
  with unified latency extraction and a close hook.
- **Docs**: README detector/integration tables + "Baseline and advanced
  detection" section; `docs/DETECTOR_GUIDE.md` and `docs/ADAPTER_GUIDE.md`
  updated; `examples/baseline_to_monitor.py` runnable end-to-end walkthrough.

### Still pending (lower priority, per §13 step 9-10)

- `ml` extra echo-state-network ensemble (`ml/esn_ensemble.py`); the
  `ml_ensemble` shipped detector is the deterministic stand-in.
- `drift` extra semantic goal-drift (`drift/goal_drift.py`,
  sentence-transformers).
- `claude_code_adapter.py`, `continuum_adapter.py`/`continuum_sink.py`,
  `openai_adapter.py`, `anthropic_adapter.py`, and the sidecar server mode
  (`server/http_server.py`).
- `benchmarks/detection_accuracy.py` (paper-number honesty gate in §14).