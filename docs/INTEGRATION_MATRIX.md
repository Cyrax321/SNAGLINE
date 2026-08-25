# SNAGLINE integration matrix

How to attach SNAGLINE to any agent system. Everything below is
dependency-free unless noted; the core is always zero-required-dependency.

## Languages and runtimes

| Path | Mechanism | Languages | Code change |
|------|-----------|-----------|-------------|
| HTTP sidecar | `snagline serve` + `POST /events` (or batched array) | any | none in host; send JSON |
| Claude Code hooks | `snagline hook` as a `PostToolUse`/`PreToolUse` hook | any CLI agent | one hook line |
| File / command bridge | `snagline watch --file` or `hook --out` | any | forward one JSON line |
| Python auto-instrumentation | `snagline.auto` (OpenAI/Anthropic/LangChain) | Python | one import |
| Framework adapters | `adapters.raw`, `langchain`, `langgraph`, `autogen`, `crewai`, `claude_code` | Python | wrap your loop |

## Transport security

| Path | Plaintext segment | TLS story |
|------|-------------------|-----------|
| In-process adapters / auto-instrumentation | none (no network) | not applicable |
| HTTP sidecar, direct | client to loopback only | keep the default `127.0.0.1` bind; never expose 8787 off-host |
| HTTP sidecar behind nginx or Caddy | proxy to sidecar over loopback | public TLS terminates at the proxy; copy-paste configs in [ATTACH_ANY_SYSTEM.md](ATTACH_ANY_SYSTEM.md#sidecar-tls-reverse-proxy-termination-issue-103) |
| Webhook / Slack / PagerDuty sinks | none outbound | stdlib `urllib` speaks HTTPS to remote endpoints |

In-process TLS for the sidecar itself is the documented future option
(issue #103); it would remove even the loopback plaintext hop and enable
mutual TLS, with zero new dependencies either way.

## Python auto-instrumentation (`snagline.auto`)

Import-safe: a wrapper is a clean no-op when the SDK is absent, and handles
sync + async clients.

```python
from snagline.auto import instrument_openai, instrument_anthropic, instrument_langchain
from snagline.monitor import Monitor

monitor = Monitor.default()          # or Monitor.default(config=Config.from_env())
instrument_openai(monitor)           # patches openai.OpenAI / AsyncOpenAI globally
instrument_anthropic(monitor)        # patches anthropic.Anthropic / AsyncAnthropic
instrument_langchain(monitor)        # patches langchain BaseLLM / Chain entrypoints
```

## Framework adapters

```python
from snagline.adapters.raw import watch          # generic StepEvent stream
from snagline.adapters.langchain import LLMChainObserver
from snagline.adapters.langgraph import GraphObserver
from snagline.adapters.autogen import SnaglineAutogenHandler
from snagline.adapters.crewai import snagline_step_callback
```

Each adapter maps the framework's native callback/hook into a `StepEvent` and
hands it to a `Monitor`. They are duck-typed so they work without the
framework installed at import time.

## Sinks (escalation)

| Sink | Purpose | Deps |
|------|---------|------|
| `ConsoleSink` | local stderr | stdlib |
| `WebhookSink` | POST `FailureRisk` JSON | stdlib |
| `SlackSink` | Slack incoming webhook | stdlib |
| `PagerDutySink` | PagerDuty Events API v2 | stdlib |
| `DedupSink` | cooldown/dedup wrapper (issue #4) | stdlib |
| `BatchingSink` | async, rate-limited dispatch | stdlib |

CLI: `snagline watch --sink {console,webhook,slack,pagerduty}` (plus
`--cooldown-seconds`, `--min-severity`). Wrap any sink in `DedupSink` and/or
`BatchingSink` for storm protection and non-blocking delivery.

## Configuration (12-factor)

`Config.from_env()` / `Config.load_file()` / `Config.resolve()` layer
built-in defaults -> config file -> `SNAGLINE_*` env vars. The CLI reads
`--config <json|toml>` and `SNAGLINE_*` automatically.

## State

`Monitor` locks per `episode_id` via a `StateBackend`. `MemoryStateBackend`
is the default; `RedisStateBackend` (optional, behind the `redis` extra)
coordinates across workers.

## Baselines

`snagline baseline <traj>` fits a healthy-run profile. With `--store-dir`
the profile is stored versioned and per-tenant via `BaselineStore`, and
`BaselineCollector` captures one live during a healthy run. The goal-drift
detector compares live traffic against it.

## Write your own adapter in ~10 lines

If your agent is not covered above, map each step to a `StepEvent` and feed
it to a `Monitor`. Minimal example:

```python
from snagline.events import StepEvent, make_signature
from snagline.monitor import Monitor

monitor = Monitor.default()  # fail-open by construction

def observe(step_id, tool, latency_ms, error=False, episode_id="run"):
    monitor.ingest(StepEvent(
        step_id=step_id,
        episode_id=episode_id,
        timestamp=__import__("time").time(),
        action_type="tool_call",
        action_signature=make_signature("tool_call", tool, step_id),
        tool_name=tool,
        latency_ms=latency_ms,
        error=error,
    ))
```

Call `observe(...)` once per tool/LLM call in your loop (or wrap the call).
Detection and escalation happen automatically; a fault in detection never
breaks your agent.
