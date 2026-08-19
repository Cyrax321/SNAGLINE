# Attaching SNAGLINE to any system: limitations and the path to readiness

This document records where the project stands against the bar of "attach to
any agent system, in any language, at production scale, with trustworthy
detection." Findings are grounded in the current code (verified, not
speculative). The prioritized plan at the bottom is the route from "a library
you wire in" to "something you attach."

## What "attach to any system" demands

Zero-code or near-zero-code integration into any agent runtime (Python or
not), safe at production scale, with trustworthy detection and
enterprise-grade alerting. Concretely:

- A host can be observed without editing its internals.
- Any language/runtime can send telemetry through a common bridge.
- Detector state and baselines survive restarts and span workers.
- Alerting is deduplicated, routed, and authenticated.
- Configuration and secrets follow 12-factor practices.
- Detection quality is measurable and honest.

## Current limitations (verified in code)

### Integration surface

- No auto-instrumentation. Every host must be edited to call `ingest` or
  register an adapter. There is no monkey-patch or middleware path, so
  "attach without touching the host" does not exist yet.
- Adapters present: `raw`, `langchain`, `langgraph`, `autogen`, `crewai`,
  `claude_code`. Missing: `openai`, `anthropic`, `continuum` (all still
  pending per spec).
- Non-Python systems depend on the HTTP sidecar (`server/http_server.py`
  exists) or file/command bridges, but the sidecar lacks auth, documented
  schema validation, and is flagged inconsistently between README
  ("Complete") and `project.md` ("pending").
- No schema auto-discovery: the host must map its telemetry to `StepEvent`
  fields (`tool_name`, `latency_ms`, `error`, `signature`). Systems that do
  not expose latency or tool names degrade silently.

### State, scale, and distribution

- `Monitor` keeps all detector state in memory, single process. No
  persistence across restarts, no shared state across workers, so horizontal
  scaling is impossible without an external store.
- `monitor.py:47` takes one global `threading.Lock()` per instance, so
  `ingest` serializes. Fine for a single agent; a bottleneck if one Monitor
  serves a high-throughput service.
- `goal_drift` needs a manually captured baseline file. No versioning, no
  auto-collection cadence, no per-tenant or per-deployment baselines, no
  retraining.

### Detection quality

- The research differentiators are unbuilt: `ml/esn_ensemble` (echo-state
  network) and `drift/goal_drift` (semantic embeddings). Shipped
  `ml_ensemble` is a deterministic noisy-OR; `goal_drift` is structural
  (error rate and latency vs baseline). Solid, but not the paper's accuracy.
- No automatic threshold tuning from a baseline; defaults may false-positive
  or miss on unfamiliar systems.
- No evaluation harness (`benchmarks/detection_accuracy.py`), so the §14
  honesty gate is unmet.

### Production hardening

- Sinks: only `console` and `webhook`. Slack and PagerDuty are still
  "planned". No dedup or cooldown (issue #4), so alert storms are possible.
- `webhook.py` sets only `Content-Type`; no auth token, no TLS verification,
  no signing. Not enterprise-ready.
- Config is a Python dataclass; no env/yml/toml loader, no secrets management
  (not 12-factor).
- Not on PyPI (`version = 0.1.0`, README says "Not on PyPI"). `pip install
  snagline-agent` does not work yet.
- No self-observability: no metrics endpoint, no health check, no structured
  logs for the monitor itself.

## What is needed, prioritized

### P0 (blockers for universal attach)

1. **Mature, document, and authenticate the HTTP sidecar.** This is the
   lingua franca for any language or runtime. Validate it round-trips
   `StepEvent`, add auth (HMAC or bearer) and TLS verification, and document
   the schema.
   - **Done:** bearer/token auth (#31), body-size cap + batched `POST /events`
     (#32), 12-factor config + CLI wiring (#36). TLS termination is delegated
     to a reverse proxy (not in-process) - documented as such.
2. **Add Python auto-instrumentation.** Wrap OpenAI/Anthropic clients and
   LangChain automatically so a user adds one line (`import snagline.auto`)
   instead of editing every call. This is the real "attach to any system"
   lever.
   - **Done:** `snagline.auto.openai` (#33), `snagline.auto.anthropic` (#34),
     `snagline.auto.langchain` (#35). All import-safe (no-op when SDK absent)
     and handle sync + async.
3. **Publish to PyPI** with versioning and per-framework extras; load config
   from env/yml with secret handling (12-factor).
   - **Done (code side):** PyPI-ready metadata + verified `python -m build`
     sdist/wheel (#37); `Config.from_env` / `load_file` / `resolve` 12-factor
     loader (#30) wired into the CLI (#36). Actual `pip install snagline-agent`
     upload still needs a PyPI token (not performed automatically).

### P1 (production readiness)

4. **Pluggable state backend** (in-memory default, Redis/DB optional) so
   detector state and baselines survive restarts and span workers; shard the
   lock by `episode_id` to remove the global ingest bottleneck.
5. **Alerting maturity:** dedup/cooldown (issue #4), severity, Slack and
   PagerDuty sinks, batched and rate-limited async dispatch.
6. **Baseline lifecycle:** auto-capture cadence, versioned store, per-tenant
   baselines, scheduled retrain.

### P2 (detection quality, the differentiator)

7. Build the `ml` and `drift` extras (ESN ensemble, semantic goal-drift)
   with training pipelines (`train.py`) so accuracy approaches the paper.
8. Auto-threshold calibration from the baseline; per-system tuning.
9. Eval harness plus a labeled dataset to honestly report detection numbers.

### P3 (trust and observability)

10. Monitor self-metrics (Prometheus), health endpoint, structured logs.
11. Integration matrix document plus a prominent "10-line custom adapter"
    path.

## Suggested starting point

P0 items 1 to 3 are what turn this from "a library you wire in" into
"something you attach." They are now implemented (see Progress below). The
next highest-leverage move is P1 item 4 (pluggable state backend) and P1
item 5 (alerting dedup/cooldown + Slack/PagerDuty sinks).

## Progress log

- **sidecar auth + hardening + 12-factor config + auto-instrumentation + PyPI
  metadata** (P0). Granular PRs, all merged with green CI:
  - #30 `feat/config-env-loader` - `Config.from_env` / `load_file`.
  - #31 `feat/sidecar-auth` - bearer/token auth on the sidecar.
  - #32 `feat/sidecar-hardening` - body-size cap + batched events.
  - #33 `feat/auto-openai` - OpenAI auto-instrumentation.
  - #34 `feat/auto-anthropic` - Anthropic auto-instrumentation.
  - #35 `feat/auto-langchain` - LangChain auto-instrumentation.
  - #36 `feat/config-cli-wiring` - `Config.resolve` wired into the CLI/Monitor.
  - #37 `feat/pypi-metadata` - PyPI-ready metadata; `python -m build` verified.
- **Remaining for true "attach anywhere":** actual PyPI upload (needs token),
  in-process TLS (or documented reverse-proxy), then P1 items 4-6.
