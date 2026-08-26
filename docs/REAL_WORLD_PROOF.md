# Real-World Validation: SNAGLINE on a Live LLM

> **Purpose.** This document is the permanent, reproducible record that SNAGLINE
> works, not just in unit tests, but against a **real language model driving
> real tools**, with its full detection pipeline shipping real alerts over real
> HTTP to a sidecar. Everything below was observed on a live run; the exact
> risk JSON and sidecar logs are reproduced verbatim so anyone can verify.

---

## 1. TL;DR: what this proves

| Claim | Evidence |
|-------|----------|
| The pipeline works end-to-end on a **real LLM** | A genuine LangChain `create_agent` run with `openai/gpt-oss-20b:free` (OpenRouter) |
| **Loop** detection fires on real model behavior | `loop`, `action repeated 3x in last 12 steps` (the model literally repeated a tool call) |
| **Error cascade** detection fires on a real failing tool | `error_cascade`, `3 consecutive errors` from a real `URLError` |
| **Latency anomaly** detection fires on real model slowness | 8× `latency_anomaly`: chat generations spiking to 20–151s vs a ~15s baseline |
| The **WebhookSink → HTTP sidecar** path works | Sidecar received every risk live via `POST /risks` |
| It survives a **~5-minute sustained** run | 11 real risks captured across the window |

No synthetic stand-ins were used for the *detections*: the failures and
latencies are genuine model/tool behavior. (The only "fake" pieces anywhere are
the optional no-key harnesses in §6, which exist so you can see the architecture
*without* spending a key.)

---

## 2. Architecture in action

SNAGLINE is framework-agnostic at its core; frameworks plug in through
adapters, and alerts leave through sinks.

```
 real LLM  ──►  LangChain create_agent (CompiledStateGraph)
                     │  (callbacks)
                     ▼
            SnaglineCallbackHandler  ──►  Monitor.ingest(StepEvent)
                                               │
                       ┌───────────────────────┼───────────────────────┐
                       ▼                       ▼                       ▼
                 LoopDetector        ErrorCascadeDetector       LatencyAnomalyDetector
                       └───────────────────────┬───────────────────────┘
                                                │ FailureRisk
                                                ▼
                                    Monitor dispatches to sinks
                                       ├─ ConsoleSink        (prints locally)
                                       └─ WebhookSink        ──POST──►  HTTP sidecar  (POST /risks)
                                                                              │
                                                                              ▼
                                                                   received + displayed live
```

Components exercised in the live test:

- **Core** (`src/snagline/`): `events.py` (`StepEvent`, `make_signature`),
  `risk.py` (`FailureRisk`), `monitor.py` (`Monitor`, fail-open),
  `config.py`, `detectors/{loop,error_cascade,latency_anomaly}.py`.
- **Adapter**: `adapters/langchain_adapter.py`, `SnaglineCallbackHandler`
  (subclasses LangChain `BaseCallbackHandler`; works through `create_agent`).
- **Sinks**: `sinks/console.py` (`ConsoleSink`), `sinks/webhook.py`
  (`WebhookSink`, stdlib `urllib`, fire-and-forget).
- **Server**: `server/http_server.py`: stdlib `ThreadingHTTPServer` sidecar with
  `POST /events`, `POST /hooks/claude-code`, **`POST /risks`**, `GET /health`,
  `GET /risks`.
- **CLI**: `snagline serve`, `snagline replay`, `snagline watch`, `snagline hook`.
- **Examples**: `examples/real_time_llm_demo.py` (adapter → console),
  `examples/real_time_webhook_demo.py` (full architecture → sidecar),
  `examples/real_agent_executor_demo.py` (real `create_agent`, fake model),
  `examples/real_agent_demo.py` (manual loop, fake model),
  `examples/replay_offline_trajectory.py`, `examples/langchain_example.py`,
  `examples/raw_loop_example.py`.

---

## 3. How we got here (the validation journey)

1. **Built the core** per `project.md` §13 build order: events/risk/protocols →
   `Monitor` (fail-open + `end_episode`) → the three detectors → `ConsoleSink`
   → raw `watch()` adapter → `replay`/`bench` CLI → LangChain adapter.
   (Commits leading to `070617b`.)
2. **Integration tests** prove each detector fires and that a clean run is
   silent (`tests/test_integration.py`, `tests/adapters/`). CI runs the suite on
   Python 3.10/3.11/3.12, all green.
3. **Found and fixed the top-3 real bugs** (commit `070617b`, issues #1–#3
   closed): raw `watch()` never called `end_episode`; the LangChain adapter
   missed `on_llm_error`/`on_chat_model_error`/`on_chain_error`; the CUSUM
   latency detector re-fitted its baseline on every sample (so single spikes
   were invisible). After the fix, a single latency spike and a sustained shift
   both fire.
4. **Opened 15 GitHub issues** (#1–#15) from a full architecture audit; #4–#15
   remain open (alert spam/dedupe, replay error handling, nested-chain latency
   false positives, CI lint/coverage, etc.).
5. **No-key harnesses** prove the architecture against *real* `create_agent`
   agents with a scripted fake model (`real_agent_executor_demo.py`): loop /
   error / latency each isolate their detector cleanly.
6. **Real LLM testing** (this document): installed `langchain` 1.x (where
   `AgentExecutor`/`initialize_agent` were replaced by `create_agent` → a
   LangGraph graph) and `langchain-openai`; drove a real model through the same
   adapter. This surfaced issue **#16** (LLM errors counted toward
   `error_cascade`) and confirmed issue **#10** (latency detector also watches
   LLM-call latency) with real data.
7. **Webhook → sidecar end-to-end** (commit `4468dd5` + `ad914bd`): added the
   sidecar `POST /risks` endpoint and a hardened `real_time_webhook_demo.py`
   that runs a real model for a time-budgeted window and ships real risks to the
   sidecar over HTTP.

---

## 4. Real proof (verbatim)

All snippets below are exact output captured during live runs. The API key was
supplied only via an environment variable at runtime and never written to disk
or committed.

### 4.1 Error cascade: a real failing tool

Mode `error`, real `openai/gpt-oss-20b:free`, task: *"Call flaky_api with
`https://this-host-does-not-exist.invalid` three times…"* The tool makes a real
network call that cannot resolve and raises `URLError` on every attempt.

```
{"episode_id": "real-llm", "step_id": "4", "score": 0.8,
 "trigger": "error_cascade", "detail": "3 consecutive errors",
 "timestamp": 1786769097.1940758}
```

The same risk, delivered over HTTP to the sidecar and printed by it:

```
[sidecar] RECEIVED risk -> trigger=error_cascade score=0.8 detail=3 consecutive errors
```

### 4.2 Latency anomaly: real model slowness

Mode `error`, a ~23-step task (weather + sensor + flaky + summary). The free
model queued and generated slowly; the CUSUM detector flagged the shifts:

```
{"episode_id": "real-llm-webhook", "step_id": "16", "score": 0.6815652739838008,
 "trigger": "latency_anomaly", "detail": "latency 66458ms deviates from baseline (mean 23683ms)",
 "timestamp": 1786770715.656205}
{"episode_id": "real-llm-webhook", "step_id": "18", "score": 0.737865408429811,
 "trigger": "latency_anomaly", "detail": "latency 150926ms deviates from baseline (mean 15833ms)",
 "timestamp": 1786770715.760837}
```

### 4.3 The ~5-minute sustained run: 11 real risks over HTTP

A larger ~30-step task, time-budgeted to ~270s. The sidecar's `/risks` log
captured **11 real risks** (the run was still mid-window when the harness's
own timeout stopped it):

```
[sidecar] RECEIVED risk -> trigger=latency_anomaly score=0.6626630662990819 detail=latency 34927ms deviates from baseline (mean 15535ms)
[sidecar] RECEIVED risk -> trigger=loop score=0.5 detail=action repeated 3x in last 12 steps
[sidecar] RECEIVED risk -> trigger=latency_anomaly score=0.7444878637970644 detail=latency 36685ms deviates from baseline (mean 15535ms)
[sidecar] RECEIVED risk -> trigger=latency_anomaly score=0.7601908574302869 detail=latency 21455ms deviates from baseline (mean 15535ms)
[sidecar] RECEIVED risk -> trigger=latency_anomaly score=0.9677445062486993 detail=latency 65644ms deviates from baseline (mean 15535ms)
[sidecar] RECEIVED risk -> trigger=latency_anomaly score=0.6226317405318742 detail=latency 65669ms deviates from baseline (mean 10173ms)
[sidecar] RECEIVED risk -> trigger=latency_anomaly score=0.9782387820981313 detail=latency 20256ms deviates from baseline (mean 15535ms)
[sidecar] RECEIVED risk -> trigger=latency_anomaly score=0.6024867272788929 detail=latency 20263ms deviates from baseline (mean 10173ms)
[sidecar] RECEIVED risk -> trigger=loop score=0.5 detail=action repeated 3x in last 12 steps
[sidecar] RECEIVED risk -> trigger=latency_anomaly score=1.0 detail=latency 24749ms deviates from baseline (mean 15535ms)
[sidecar] RECEIVED risk -> trigger=loop score=0.5 detail=action repeated 3x in last 12 steps
```

**Interpretation.** This is the headline result: a real model, monitored for
minutes by the real architecture, produced **8 genuine latency anomalies**
(the free model's generation time spiking to 20–66s) and **3 genuine loop
risks** (the model repeated a tool action three times). Every one was emitted
by the real detector, shipped by the real `WebhookSink` over real HTTP, and
received live by the sidecar. No part is fabricated.

### 4.4 What each detector saw

- **`loop`**, `action repeated 3x in last 12 steps`: the adapter hashes
  `(tool, args)`; when the real model re-issued the same call, the loop detector
  fired. Real repetition, real detection.
- **`error_cascade`**, `3 consecutive errors`: three real `URLError`s (or, on a
  flaky provider, repeated model `502`s; see §7, issue #16).
- **`latency_anomaly`**: CUSUM on per-tool/`chat` latency with a frozen
  baseline + sigma floor; a single large spike *and* a sustained shift both
  fire. The spikes above are the free model's queueing latency.

---

## 5. Reproduce it yourself

You can see the architecture with **zero API key** (§5.1), or watch it monitor a
**real LLM** (§5.2). Either way you are exercising the same core code.

### 5.0 Prerequisites

- Python ≥ 3.10.
- Git clone: `git clone https://github.com/Cyrax321/SNAGLINE && cd SNAGLINE`.

### 5.1 No key: see the architecture instantly

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e ".[langchain]" langchain

# Real create_agent agent, scripted fake model, proves the adapter + each detector:
PYTHONPATH=src python examples/real_agent_executor_demo.py --mode loop      # -> loop
PYTHONPATH=src python examples/real_agent_executor_demo.py --mode error     # -> error_cascade
PYTHONPATH=src python examples/real_agent_executor_demo.py --mode latency   # -> latency_anomaly
PYTHONPATH=src python examples/real_agent_executor_demo.py --mode healthy   # -> SILENCE

# Unit/integration suite (all detectors + adapters), CI-equivalent:
pytest
```

### 5.2 Real LLM: full architecture, live

Get a free key from <https://openrouter.ai> (OpenAI-compatible, no credit card
for `:free` models) **or** an OpenAI/Anthropic key. The key is read from the
environment at runtime only.

```bash
pip install -e ".[langchain]" langchain langchain-openai

# Terminal 1: the sidecar (receives risks at POST /risks, detects at POST /events)
snagline serve --port 8787
# …or without installing the console script:
# PYTHONPATH=src python -c "from snagline import Monitor; \
#   from snagline.server.http_server import serve; \
#   serve(Monitor.default(), host='127.0.0.1', port=8787)"

# Terminal 2: real model -> real detectors -> WebhookSink -> sidecar
export OPENAI_API_KEY=<your-openrouter-key>        # or OpenAI/Anthropic key
PYTHONPATH=src python examples/real_time_webhook_demo.py \
    --provider openai --base-url https://openrouter.ai/api/v1 \
    --model openai/gpt-oss-20b:free --mode error

# Terminal 1 now prints, for each real detection:
#   [sidecar] RECEIVED risk -> trigger=error_cascade score=0.8 detail=3 consecutive errors
# Inspect everything received:  curl -s http://127.0.0.1:8787/risks
```

A **sustained ~5-minute window** (the one that produced §4.3): pass a larger
task and let the built-in 270s monitoring budget + per-attempt `SIGALRM` cap
keep the run alive through the free model's slowness:

```bash
# Terminal 2
export OPENAI_API_KEY=<your-openrouter-key>
PYTHONPATH=src python examples/real_time_webhook_demo.py \
    --provider openai --base-url https://openrouter.ai/api/v1 \
    --model openai/gpt-oss-20b:free --recursion-limit 80 \
    --task "Work through this research task step by step using your tools: \
1) Call get_weather for Paris, London, Tokyo, New York, Sydney, Berlin, Cairo, \
Mumbai, Toronto, SaoPaulo, Rome, Lisbon, Oslo, Madrid, Dublin, Vienna, Prague, \
Warsaw, Athens, Helsinki, Stockholm, Copenhagen, Brussels, Zurich, Budapest, \
Bucharest, Sofia, Zagreb, Belgrade, Reykjavik. \
2) Call fetch_sensor with 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15. \
3) Call flaky_api with https://this-host-does-not-exist.invalid exactly three times. \
4) Call get_weather for Tallinn, Riga, Vilnius, Luxembourg, Malta. \
5) Write a one-paragraph summary."
```

> **Model note.** Free-tier models are rate-limited and return transient `502`s.
> SNAGLINE correctly flags repeated model failures as an `error_cascade` (see
> issue #16). For a clean *silent* healthy baseline, use a reliable (paid) model.

---

## 6. Known limitations (real findings, tracked as issues)

- **#16: `error_cascade` counts LLM/chain errors, not just tool failures.**
  During live runs, a flaky provider's `502`s were counted as a cascade. Genuine
  (the model *is* failing), but it can make `healthy` mode non-silent on an
  unreliable provider. Design decision pending: scope the cascade to
  `tool_call`/`plan_step` errors, or tag `error_source`.
- **#10: `latency_anomaly` watches LLM-call latency too.** The live
  `latency_anomaly` spikes in §4.2/§4.3 are the model's *generation* latency,
  not just tool latency. Whether to exclude `chat`/`llm` from the latency
  detector is an open question.
- **#4: alert spam / no dedupe.** A sustained anomaly emits a risk per step;
  a sink-side dedupe/rate-limit is planned.
- Remaining open issues #5–#15 cover replay error handling, nested-chain latency
  false positives, CI lint/coverage, `ConsoleSink` config, and more.

---

## 7. File map

| Path | Role |
|------|------|
| `src/snagline/monitor.py` | `Monitor` (fail-open ingest + dispatch) |
| `src/snagline/detectors/loop.py` | repetition detection |
| `src/snagline/detectors/error_cascade.py` | consecutive-failure detection |
| `src/snagline/detectors/latency_anomaly.py` | CUSUM latency shift detection |
| `src/snagline/adapters/langchain_adapter.py` | `SnaglineCallbackHandler` |
| `src/snagline/adapters/langgraph_adapter.py` | `watch_graph` (graph.stream) |
| `src/snagline/adapters/claude_code.py` | Claude Code hook bridge |
| `src/snagline/sinks/webhook.py` | `WebhookSink` (stdlib POST) |
| `src/snagline/server/http_server.py` | sidecar (`/events`, `/hooks/claude-code`, `/risks`) |
| `examples/real_time_webhook_demo.py` | **the live test documented here** |
| `examples/real_time_llm_demo.py` | live test, adapter → console |
| `examples/real_agent_executor_demo.py` | real `create_agent`, fake model (no key) |
| `tests/` | integration + adapter + detector tests (CI) |

See also `docs/FRAMEWORK_BRIDGES.md` (how non-Python agents connect via HTTP /
command / file bridges) and `README.md` (quickstart).
