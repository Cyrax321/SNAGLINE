# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-08-27

This is the first tagged release. It comprises 87 merge commits on `origin/master`
from the first commit through `d80a686` (`feat --semantic baseline flag`, PR #189).
Every user-visible change below was verified against `git show <sha> --stat` for
its merge commit, not against PR titles. CI at this commit is green with
reproduced counts: **651 passed, 3 skipped** (`pytest`, py3.10 through 3.13),
**88.04% line coverage**, `ruff check src tests` and `ruff format --check src tests`
clean, `mypy src` clean. Overhead measured on this checkout:
`benchmarks/overhead_benchmark.py` reports **median 5.31 us/step, p99 42.22 us/step**
over 200,000 synthetic steps. Detection accuracy measured on this checkout:
`benchmarks/detection_accuracy.py` reports **macro-F1 1.000** over 76 episodes
(40 labeled, 36 healthy controls), zero healthy-control false positives.

### Added

#### Core and configuration
- Canonical schemas `StepEvent` / `EpisodeMeta` / `FailureRisk` and `make_signature`
  with SHA-256 normalization (PR #24 through #30, #15, #28).
- `Monitor` orchestrator with fail-open guarantee, per-episode lock sharding via
  `StateBackend` / `MemoryStateBackend`, and `Metrics` self-observability counters
  (PR #44, #47).
- `Config` dataclass with 12-factor env/file layering (`SNAGLINE_*` env vars,
  `Config.resolve`, optional JSON/TOML file), tunable thresholds for every
  detector, and CLI wiring (`snagline --config`) (PR #30, #36, #37).
- `BaselineProfile` fitting and persistence (`fit_baseline_from_jsonl`,
  `save_baseline` / `load_baseline`, `BaselineStore` versioned store per
  tenant/deployment) (PR #24, #45, #46).
- `Monitor.snapshot` / `restore` with versioned JSON, atomic tmp+replace,
  strict composition check, and per-detector `dump_state` / `load_state`
  (PR #168).

#### Detectors
- **Loop detector** with sliding window and repeat threshold, plus opt-in
  hardening modes: near-duplicate (volatile ID collapsing), cycle (periodic
  scan), stall (consecutive identical signatures) (PR #114, ce5e449).
- **Error-cascade detector** with windowed and consecutive modes (core), with
  config `cascade_count_non_tool_errors` (PR #16-era, #27 baseline).
- **Latency-anomaly detector** (Welford + CUSUM, stdlib only) with per-tool
  baselines, sigma floors, warm-up, and optional baseline re-fit
  (`cusum_refit_every`) (PR #25, #138, #167).
- **Goal-drift detector** (opt-in, compares live run to persisted
  `BaselineProfile` on error rate and latency z-score) (PR #25).
- **ML ensemble** `MLOrchestrator` (opt-in, noisy-OR over tier-1 detectors,
  `model=` hook) (PR #26).
- **Token-runaway detector** (opt-in, CUSUM over token volume plus
  `episode_token_budget` envelope, triggers `token_runaway` / `budget_breach`)
  (PR #107).
- **Meltdown detector** (opt-in, sliding-window Shannon entropy, low/high
  thresholds) (PR #107, #135).
- **Silent-abort detector** (opt-in, `EpisodeFinalizer` evaluated at
  `end_episode`, trigger `silent_abort`) (PR #107).
- **Stagnation detector** (opt-in, novelty-rate collapse) (PR #127).
- **Side-effect guard** (opt-in, duplicate non-idempotent action detection,
  `side_effect` field on `StepEvent`) (PR #140).
- **Compaction tripwire** (opt-in, governance-decay across context
  compactions, `compaction` / `constraint_present` contract) (PR #147).
- **Horizon-scale time axis** (opt-in, PR #167): `max_episode_wall_seconds`
  with wall-clock budget (`wall_clock_budget`), `idle_warn_seconds`
  (`idle_gap`), window auto-scaling (`window_scale_steps`, `max_window`),
  `HeartbeatSink` and `snagline watch --follow --heartbeat` liveness file.
- **ML extra: ESN ensemble** (`ml/esn_ensemble.py`, `snagline-agent[ml]`,
  one-class ESN + CUSUM + Mahalanobis baseline) (PR #135).
- **Drift extra: semantic goal-drift** (`drift/goal_drift.py`,
  `snagline-agent[drift]`, sentence-transformers, PR #81 / 09ae888).
- **Auto-calibration** from `BaselineProfile` (`calibration="auto"`,
  `CalibrationPlan`, `resolve_baseline_profile`) (PR #138).
- **Scheduled baseline retrain** `snagline baseline retrain` contract with
  `--windows-dir` / `--jsonl` / `--store-dir` / `--max-age` staleness guard
  and `docs/RETRAIN_CADENCE.md` (PR #126, #102).
- **Enforcement policy** `Monitor(policy="observe"|"callback"|"halt_webhook")`
  with `on_risk` callback (fail-open), `halt_url`/`halt_timeout_s`
  (default 250 ms) / `min_severity_for_halt` (default 0.8),
  `HaltDirective` / `last_directive` thread-safe, and CLI
  `snagline serve --halt-forward` (PR #166, #93).
- Follow-up **directive endpoint** `GET /directive` and `policy_errors`
  Prometheus family `snagline_monitor_policy_errors_total` (PR #178, #169).

#### Sinks
- Console (default, JSON line to stderr), webhook (stdlib `urllib`, PR #32),
  Slack (PR #41), PagerDuty (PR #42), dedup / cooldown (`DedupSink`, PR #39,
  #40), batching (`BatchingSink`, PR #48), logging sink
  (`LoggingSink`, PR #111, #99), continuum sink (`REQUIRES_REVIEW`, PR #164),
  heartbeat liveness sink (PR #167), public `Monitor.add_sink` / `remove_sink`
  (PR #124).

#### Adapters
- Raw loop `watch` context manager (PR #24).
- LangChain callback handler and LangGraph node wrapper (PR #27, #77).
- AutoGen and CrewAI adapters (duck-typed, PR #27).
- OpenAI and Anthropic adapters: explicit wrappers plus auto-instrumentation
  `snagline/auto/*` with streaming telemetry deferred until exhaustion
  (PR #33, #34, #35, #83, 74e6049).
- Claude Code hook adapter via `HookTracker` / `payload_to_event`
  (`adapters/claude_code.py`) with latency derivation (PR #71, #64).
- **CONTINUUM bridge** adapter + sink (`adapters/continuum_adapter.py`,
  `sinks/continuum_sink.py`, extra `snagline-agent[continuum]`, duck-typed
  against verified `read_events` / `last_sequence` API) (PR #164, #79).

#### Server (sidecar, stdlib `http.server`)
- `POST /events` (single and batched), `GET /health`, `POST /risks` /
  `GET /risks`, `POST /hooks/claude-code`, `GET /metrics` with Prometheus
  text exposition 0.0.4 and legacy JSON (`?format=`) (PR #31, #32, #98 / #115).
- `GET /episodes` active-episode listing with TTL expiry (PR #160, #123).
- Auth via `Authorization: Bearer` / `X-Snagline-Token` and
  `--auth-token` / `SNAGLINE_SERVE_AUTH_TOKEN`, with `GET /health` open
  (PR #31, #75).
- Hardening: body size cap `max_body_bytes` (default 1 MB), over-cap drain
  so 413 is delivered without RST (PR #125, #121), malformed
  `Content-Length` now returns 400 instead of crashing handler thread
  (PR #185, #129), per-connection read timeout `read_timeout_s` so stalled
  senders cannot pin threads (PR #186, #130), episodes active expiry
  (PR #160).
- TLS: documented reverse-proxy configs and in-process stdlib `ssl`
  termination via `--certfile` / `--keyfile` (PR #110 / #103, #146 / #120).

#### CLI
- `snagline replay` (offline trajectory replay, now with `end_episode`
  teardown, PR #5-era, #18 fix).
- `snagline watch` (stdin or `--file` with `--follow`, `--episode-id`,
  `--sink` / `--cooldown-seconds`, heartbeat file) (PR #40, #43, #167).
- `snagline serve` (sidecar, `--host`/`--port`/`--auth-token`,
  `--max-body-bytes`/`--max-risks`, TLS flags, halt-forward flags)
  (PR #31, #75, #146, #166).
- `snagline hook` universal bridge (Claude Code payload detection, `--url`
  / `--out` / `--timeout`, fail-open) (PR #64-era).
- `snagline baseline` and `snagline baseline retrain` (PR #24, #45, #46,
  #126).
- `snagline bench` (overhead benchmark, PR #24, #6 fix).
- Global 12-factor config: `--config` plus env overrides, sink selection
  (`console`/`webhook`/`slack`/`pagerduty`/`continuum`), `min_severity` and
  cooldown (PR #36, #43, #119).

#### Benchmarks and harness
- `benchmarks/overhead_benchmark.py` and `benchmarks/enforcement_benchmark.py`
  with published median/p99 numbers (PR #24, #166).
- `benchmarks/detection_accuracy.py` honesty gate over 76 episodes (40 labeled
  with 4 episodes per trigger, 36 healthy controls), `harness_config` per
  detector, `benchmarks/fixtures/generate_fixtures.py` corpus generator,
  accuracy gate in CI (`benchmark-accuracy` job, PR #112 / #82, #141 / #118,
  #148, #117 table in README).
- Corpus fixtures `injected_*` and healthy controls committed as JSONL
  (PR #107, #141).

#### Packaging and docs
- `pyproject.toml` zero-dep core with optional extras `langchain`,
  `langgraph`, `autogen`, `crewai`, `openai`, `anthropic`, `continuum`,
  `ml`, `drift`, `all` (PR #37, #164, #81).
- Guides: `docs/DETECTOR_GUIDE.md`, `docs/ADAPTER_GUIDE.md`,
  `docs/FRAMEWORK_BRIDGES.md`, `docs/ATTACH_ANY_SYSTEM.md`,
  `docs/INTEGRATION_MATRIX.md`, `docs/RETRAIN_CADENCE.md`,
  `docs/REAL_WORLD_PROOF.md`, `docs/BENCHMARK_CALIBRATION.md` (PR #28,
  #49, #126, #110).
- `py.typed` marker  and `Development Status :: 4 - Beta` classifier
  (PR #37).

### Fixed

- Alert spam: loop and error-cascade dedupe now re-arms per window and
  `DedupSink` retains only live keys with severity-aware keys and
  tick-based retention (PR #94, #39).
- `risk.severity` sentinel: frozen dataclass now derives `critical` /
  `warning` / `info` from score without tripping the default check
  (PR #95, #39).
- `BatchingSink` `max_batch` enforcement and `close()` flush semantics
  (PR #72).
- `Config.resolve` env precedence: env vars now win over config file even
  when equal to defaults (PR #73, #66).
- `MemoryStateBackend` leak: `release(episode_id)` now frees the per-episode
  `RLock` at `end_episode` (PR #74, #67).
- Sidecar `GET /metrics` and `GET /risks` auth bypass, and `snagline serve`
  `--auth-token` wiring (PR #75, #68).
- `snagline.__version__` missing (PR #76, #70).
- OpenAI / Anthropic adapters: explicit wrappers and deferred streaming
  telemetry (PR #83 / #78, 74e6049).
- CrewAI signature now built from `tool_input` not output (PR #88a0886, #61).
- Claude Code hook latency derivation restored (PR #71, #64).
- `make_signature` now uses full 64-char SHA-256 hex with JSON-stable
  separators instead of truncated 16-char join (PR #15 / ce5e449).
- Error-cascade now counts tool failures by default, not LLM/chain errors
  (PR #16-era, Detour via docs fix).
- Replay now calls `end_episode` per episode to avoid state leakage
  (PR #18).
- Console sink now swallows broken-stream errors (PR #19).
- Fail-open log spam: `logger.exception` on every fault replaced by
  `log_fault_once` (PR #14).
- `snagline bench` fragile import when installed as wheel (PR #6).
- Nested chain `plan_step` latency no longer bleeds into CUSUM (PR #10).
- Latency detector warm-up lowered to 5 samples with sigma floors so
  low-volume tools are monitored (PR #9).
- `LatencyAnomalyDetector` single-spike and sustained-shift handling
  (PR #3).
- Monitor ingest lock no longer held while dispatching to sinks
  (PR #8).
- Raw `watch` now calls `end_episode` and no longer builds dead
  `EpisodeMeta` (PR #1, #7).
- `raw.watch` / `Monitor.default` docstring drift fixed (PR #20, #21).
- `FRAMEWORK_BRIDGES` SHA-256 claim clarified for external processes
  (PR #22).
- `BatchingSink` `close` / `max_batch` (PR #72) and `StateBackend` sharding
  (PR #44) verified.
- Dedup double-wrap in `watch` vs `serve` (PR #159, #152).
- Dedup reboot over-suppression: restored monotonic timestamps no longer sit
  in the clock's future (PR #163, #136).
- Default clock quantized on Windows: every adapter now defaults to
  `time.perf_counter` (PR #161, #155).
- Over-cap POST handling: 413 now drains `max_body_bytes + 64 KiB` then
  replies, avoiding peer reset (PR #125, #121).
- Malformed `Content-Length` no longer crashes handler thread (PR #185,
  #129).
- Stalled sender no longer pins handler thread: `read_timeout_s` default
  5 s with 408 path (PR #186, #130).
- Log format: `SNAGLINE_LOG_FORMAT` / `Config.log_format` now operational in
  `Monitor.default` and CLI sink selection, with validation
  (`validate_log_format`, PR #144, #119).
- Stagnation validation edge cases: `min_novelty=0` now warns / range
  violations raise clearly, env-range crashes fixed (PR #187, #132).
- Baseline `fitted_at` recorded so `--max-age` works with custom version ids
  (PR #188? actually 49dd463, #128).

### Changed

- Dependabot automation bumps for GitHub Actions (`labeler`, `stale`,
  `first-interaction`, `checkout`, `github-script`, `setup-python`) (PR #51
  through #56).
- CI: `langchain-core` installed so integration tests run instead of
  silently skipping (PR #109, #100), Windows matrix added (`windows-latest`
  py3.12/3.13, PR #139, #116), `zizmor` permissions block (`contents: read`)
  on every workflow (PR #175, #154).
- Docs truth sweeps: test counts, badges, closed-issue limitation text,
  `project.md` horizon + snapshots parity (PR #108 / #96, #97, ce5e449,
  #165 wave-4: #133, #153, #151, #158, #137).
- Benchmark corpus: extended to goal-drift and ml-ensemble, healthy controls,
  `goal_drift_baseline.json` (PR #141, #118).
- Detection-accuracy table published in README with harness config table
  (PR #3656235, #117).
- Overhead numbers updated as detectors were added (README provenance lines
  retained; latest reproduced median 5.31 us/step on this checkout).
- StateBackend release semantics documented, `snapshot`/`restore` added to
  detector guide (PR #165, #97).

[0.1.0]: https://github.com/Cyrax321/SNAGLINE/releases/tag/v0.1.0
