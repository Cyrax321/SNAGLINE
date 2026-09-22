# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Nothing yet.
- `snagline hook --out` / `--url` now serializes all thirteen `StepEvent`
  fields. `_event_to_json` wrote eleven, so a host marking a step
  non-idempotent (#88) or emitting compaction pins (#90) through the
  command-hook bridge fed detectors that could never fire — the
  `side_effect` and `metadata` it keys off were dropped in transit.
- Interactive landing-page animation for the project site (comets,
  horizontal-swipe pipeline, tube-light logo effect) (#291).

### Fixed
- `snagline baseline --list-versions` is now honored on both the fit and
  `retrain` paths and is read-only everywhere: it lists and exits 0 without
  fitting, writing `baseline.json`, or storing a new version. Without
  `--store-dir` it now fails closed with `--list-versions requires --store-dir`
  (exit 2) instead of silently writing a file or bumping the store (#293).
- `episode_token_budget` and `token_budget_warn_fraction` are now range-checked
  at construction and after env/file layering, like the horizon and stagnation
  knobs. A zero or negative budget used to fire a score-1.0 `budget_breach` on
  the first token-bearing step, and a `token_budget_warn_fraction` of `0.0` a
  score-0.8 warning; values above `1.0` made the pre-breach warning
  unreachable. An out-of-range value is now a configuration error naming the
  knob (#317). 57305ac (fix(cli): make --list-versions read-only on fit and retrain paths)
- `Monitor.snapshot` now sorts detector keys by slot index, and runs the
  `strict_names` composition check *before* applying any state. `sorted()`
  on the raw `"<index>:<name>"` keys is lexicographic, so with 11+ detectors
  slot `10` sorted between `1` and `2`; two differently-ordered name lists
  were compared, rejecting a matching composition and accepting a
  mismatched one. A rejected restore now also applies no detector state at
  all (#217).
- `MemoryStateBackend.release(episode_id)` no longer orphans parked
  waiters. `episode_lock` fetched the lock under `_meta` then released
  `_meta` before parking, so `release()` could pop the entry while a waiter
  was already parked on the old `RLock`; when the holder released, the
  waiter entered the critical section on the orphaned lock while a later
  fetcher allocated a fresh one — two threads inside one episode's state.
- `MLOrchestrator` now isolates base-detector faults. Enabling
  `ml_ensemble` collapses the base detectors into one Monitor slot, so the
  Monitor's per-slot fail-open guard had nothing to isolate; the first
  detector to raise aborted the delegation loop and every remaining base
  detector was skipped for the rest of the run (#227).
- The snapshot and teardown walks of the cross-episode dicts
  (`Monitor._clocks`, `LatencyAnomalyDetector._states`) are now locked.
  Both are keyed across episodes rather than partitioned by one, so the
  per-episode lock never covered a concurrent insert for a *different*
  episode (#228, #229).
- `dump_state` now copies per-episode state before serializing it. The
  values are partitioned by `episode_id` and safe under the per-episode
  lock, but the key set is shared, and that lock cannot serialize a walk
  against an insert for a different episode (#231).
- `FailureRisk` now keeps an explicitly requested severity of `warning`.
  `__post_init__` treated *any* `severity == "warning"` as "unset" and
  overwrote it with the score-derived value, so a `warning` requested on a
  0.95-score risk silently became `critical` and still paged on-call.
- Boolean environment values are now validated. Anything not in
  `1/true/yes/on/t` or `0/false/no/off/f` is a configuration error naming
  the value, instead of silently meaning `False` — a typo in
  `SNAGLINE_DETECTORS_ENABLED=ture` turned every detector off with no
  signal.
#### Detectors
- `StagnationDetector.load_state` now restores the *effective* window for
  the restored scaler position, and the fill gate is now separate from the
  novelty gate under window scaling. The base-sized slice dropped exactly
  the flags the conditional `maxlen` one line below exists to keep (#218),
  and a window that had not yet filled its *scaled* target reported a
  novelty rate over the target size, firing a spurious stagnation risk
  (#272).
- `MeltdownDetector`, `ErrorCascadeDetector`, and `LoopDetector`
  `load_state` now rebuild windows at the effective size for the restored
  scaler position rather than the base `window_size`, which silently
  truncated the oldest history mid-episode when auto-scaling had grown the
  window live (#268).
- `GoalDriftDetector` now re-arms when the drift score recovers, instead of
  latching the first crossing until `end_episode`. Per #184, hosts that
  never call `end_episode` are a supported deployment, so a long-lived
  episode that recovered from one drift stayed flagged forever.
- A `LatencyAnomalyDetector` alarm coinciding with a periodic baseline
  adoption is now scored from the alarm-time snapshot. The alarm was
  computed before the re-fit advanced but scored after it, so adoption
  on the same step had already reset the accumulator and the risk shipped
  with a flat 0.6 score and a detail string describing the post-reset
  state.
- The ESN live scoring path now advances the reservoir after the readout,
  matching the pairing `fit()` and the warm-up learner solved for. The
  live path advanced first and then asked the readout to predict the
  current step from the post-advance state — the step being judged leaked
  into its own input.
- Claude Code `PreToolUse` / `PostToolUse` events now get distinct
  signatures. Both fire for one logical call with the same `tool_name` and
  `tool_input`, differing only in the volatile `tool_use_id` the signature
  deliberately excludes, so one logical attempt hashed twice and the loop
  detector counted it as two.
#### Monitor and time axis
- `Monitor.restore` now enforces the per-episode retention cap before
  loading detector state, not after. A snapshot holding more episodes than
  `max_live_episodes` used to be fully restored and only then trimmed,
  transiently exceeding the cap (#273).
- The restored time axis is now anchored instead of trusting a dead epoch.
  `_EpisodeClock.last_ts` is a raw `StepEvent.timestamp` and the
  auto-instrumentation stamps events with `perf_counter`, which has no
  meaningful epoch across processes — a snapshot/restore exists for
  restarts, so the restored `last_ts` was meaningless in the new process
  and the first post-restart event measured its delta from it. That failed
  both ways: a young process read a huge bogus elapsed span, and an old one
  could see the breach as already passed (#314, #315).
- The episode clock's `last_ts` now only moves forward. `_advance_clock`
  excluded a negative delta from elapsed but still rewound `last_ts` to
  the event timestamp, so the next event measured its delta from the
  rewound reference and accumulated a span already counted once. Sources
  with out-of-order or backwards-skewed timestamps (merged adapter
  streams, clock skew between hook processes) double-counted wall-clock
  time into elapsed, and `wall_clock_budget` — a hard score-1.0 trigger
  with halt-policy routing — could breach early on an episode that had
  consumed less than it reported (#249).
- The wall-clock pre-breach warning is now suppressed once the budget has
  actually been breached. A single delta jumping straight past the budget
  fired the score-1.0 breach but left `warned` false, so the next step
  fell through to the `elif` and emitted the stale 0.7 warning — severity
  running backwards, with self-contradicting text for an episode already
  reported as exceeded (#224).
#### Adapters and auto-instrumentation
- Global-mode `instrument_openai` / `instrument_anthropic` now patch the
  SDK resource classes directly. Since OpenAI SDK 1.0 and current Anthropic
  SDKs, `OpenAI.chat` / `Anthropic.messages` are
  `functools.cached_property` descriptors: the attribute walk resolved a
  descriptor, wrapped nothing, and still returned `True`, so the
  documented one-liner produced a completely unmonitored process that
  reported instrumentation success (#270).
- The `snagline.auto` wrappers now measure call latency on
  `perf_counter`, not the wall clock, matching the PR #161 audit of
  `snagline.adapters`. On Windows the wall clock advances in ~15.6 ms
  ticks on py3.10–3.12, so a call shorter than one tick recorded
  `latency_ms == 0.0` and starved the CUSUM detector of usable samples,
  and a non-monotonic wall clock can fabricate negative or huge latencies
  (#280).
- Streamed `create()` calls in the `openai` / `anthropic` wrappers now
  emit one event at exhaustion, close, or iteration failure. With
  `stream=True` the call returns immediately, so emitting at return time
  recorded a ~0 ms success before the first chunk and mid-iteration
  failures went unobserved (#242).
- `HookTracker` state shared across the sidecar's handler threads is now
  serialized. `note()` mutated `_starts` unlocked, so eviction raised
  `RuntimeError` mid-walk (swallowed downstream, dropping latency
  samples) and rebound the dict, discarding concurrent inserts (#245).
#### Server, CLI, and sinks
- Batched `POST /events` is now fully validated and constructed before
  anything is ingested. Validation and ingest ran in the same loop, so a
  bad item at position `k` answered 400 with items `0..k-1` already inside
  the Monitor; a retrying client re-fed the accepted prefix on every
  attempt, duplicating steps into the loop/cascade/CUSUM detectors and
  double-counting metrics. Also fixed path-component routing and
  `serve --config` layering for the read-timeout and episode-TTL knobs
  (#239, #240, #241).
- `snagline watch` now finalizes the episode ids actually present in the
  ingested events, not the id derived from the filename. Finalize-based
  detectors were dead in `watch`: `SilentAbortDetector` was asked about an
  episode that never existed while the real one was never finalized.
- `--min-severity` is now honored for `--sink webhook` and validated, and
  the value is range-checked. The webhook branch built a bare
  `WebhookSink(url)`, so `--sink webhook --min-severity critical` still
  POSTed info-level risks (#248).
- `BatchingSink` now preserves its rate limit across flushes. The gap was
  enforced per batch with no memory of the last delivery time, so
  consecutive flushes could fire back-to-back and defeat the cooldown.
- `SNAGLINE_STATE_BACKEND=redis` with no `SNAGLINE_STATE_REDIS_URL` now
  warns before falling back to in-memory state. The redis backend exists
  to coordinate episodes across processes; in-memory state is
  per-process, so a misconfigured deployment silently got N workers each
  holding their own lock — no coordination and nothing in the logs
  (#308).
- `benchmarks/detection_accuracy.py` table no longer misaligns when a
  trigger name exceeds the hardcoded 16-character column width
  (`side_effect_duplicate` is 21).

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
- **ML extra: ESN ensemble** (`ml/esn_ensemble.py`, `snagline[ml]`,
  one-class ESN + CUSUM + Mahalanobis baseline) (PR #135).
- **Drift extra: semantic goal-drift** (`drift/goal_drift.py`,
  `snagline[drift]`, sentence-transformers, PR #81 / 09ae888).
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
  `sinks/continuum_sink.py`, extra `snagline[continuum]`, duck-typed
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
