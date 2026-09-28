# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- `python -m snagline` now works: a `snagline/__main__.py` package entry point
  dispatches into `cli.main`, matching the `snagline` console script. Previously
  only the bare `snagline` command and `python -m snagline.cli` ran; `python -m
  snagline` failed with "'snagline' is a package and cannot be directly executed",
  which is the usual way to invoke a tool when its Scripts directory is not on
  PATH (#487).

### Security
- The sidecar's auth token is now compared with `hmac.compare_digest` instead
  of `==`. `str.__eq__` returns at the first differing character, so its
  runtime grew with the shared prefix and the token was recoverable by anyone
  able to average response times. The gate protects `/events` injection,
  `/metrics` and `/risks` reads, and `--halt-forward`'s directive path. Both
  header values are compared as UTF-8 bytes: the naive string form of
  `compare_digest` raises `TypeError` on a non-ASCII value, and
  `Authorization` is attacker-controlled, so that would have turned an
  unauthenticated probe into a handler crash (#375).

### Changed
- CI hygiene: the accuracy-gate comment in `ci.yml` now says "109-episode
  corpus" to match the committed fixtures (68 labeled + 41 healthy controls,
  the README already reflected 109) (#470); `stale.yml` no longer exempts a
  `security` label that the repo does not define — the token was inert, so the
  exempt lists now name only labels that exist (`help wanted`, `good first
  issue`) (#471).
- `pyproject.toml` now advertises the shipped inline type information with the
  `Typing :: Typed` trove classifier (the `py.typed` marker was already
  shipped, but PyPI never surfaced the package as typed), plus
  `Environment :: Console` given the `snagline` console script. A packaging
  test now fails if the marker ships without the classifier (#456).
- `[project.urls]` now includes a `Changelog` link pointing at `CHANGELOG.md`,
  so PyPI renders a changelog entry in the project sidebar (#467).

### Fixed
- The `error_cascade` score is graded again. The alarm fires *at* the
  threshold, so `n / threshold` was always `>= 1` and the `min` clamp was dead
  code: every alert, from a marginal "3 errors in 10 steps" to a total tool
  outage, was emitted at `1.0` / `critical`, and the pre-breach warning band
  was unreachable. The score now grades in bands on how far the cascade has run
  past its threshold (`0.5` at the crossing, `0.8` at 2x, `1.0` at 3x), and
  the dedupe marker tracks the band instead of a bare boolean so a cascade that
  keeps deepening escalates rather than being silenced for the rest of the
  episode. A first crossing is now a `warning`; a genuine outage still reaches
  `min_severity_for_halt` (`0.8`), which the flat-`1.0` score and a
  first-crossing-only `0.5` both failed to distinguish (#538).
- `TokenRunawayDetector.load_state` now publishes its restored state only once
  the whole snapshot has parsed, so a malformed entry leaves the detector on
  its live state like every other detector (#417 hardened them; this one was
  missed). Each parsed episode was written straight into `_states` and the
  envelope counters were assigned only after the loop, so an entry that parsed
  early was applied before a later one raised — the monitor's "detector keeps
  its live state" warning was then false, and on a fresh restore an episode
  ended up with a frozen CUSUM baseline but no budget total, so an
  already-breached episode could re-warn and re-breach on its next step (#552).
- The explicit `wrap_openai_client` / `wrap_anthropic_client` stream wrappers
  now support the documented `with stream as s:` and `async with stream as s:`
  forms. Implicit special-method lookup goes through the type slots and
  bypasses `__getattr__`, so the wrappers' delegation never ran and a host
  using the context-manager form against a wrapped client got a `TypeError`
  out of the instrumentation layer -- breaking the host's streaming call, not
  just the telemetry (#426).
- `wrap_openai_client` / `wrap_anthropic_client` are now idempotent: wrapping
  an already-wrapped client is a no-op instead of installing a wrapper around
  the wrapper, which double-counted every host call into the monitor. A host
  that wired both global and per-client instrumentation silently inflated
  every per-step metric and tripped detectors on phantom repeats (#427).
- `snagline replay`, `snagline watch --file` and `snagline baseline <file>` now
  report a missing input file as one line on stderr and exit 2 instead of
  raising an uncaught `FileNotFoundError` traceback. The sibling `baseline
  retrain` path already caught `OSError`; these three were the outliers, and a
  mistyped path read as a crash of the tool itself (#428).
- `snagline baseline retrain --semantic` is now rejected with exit 2 instead of
  being accepted and silently discarded. The retrain contract fits a structural
  profile only, so an operator who asked for a semantic baseline discovered
  only when goal-drift stayed inert that they had stored a structural one
  (#429).
- `snagline serve` now range-checks `--port` (0-65535) and verifies that
  `--certfile`, `--keyfile` and `--client-ca` are readable *before* printing
  the listening banner. A bad port or a missing cert file used to be announced
  as a live listener and then die in a traceback inside the bind or the SSL
  handshake, so a supervisor reading the banner believed the sidecar was up
  while it had already died (#430).
- `LoggingSink` no longer drops every alert whose detail cannot be encoded on
  the operator's log stream. The JSON line is rendered with `ensure_ascii=False`
  so the common UTF-8 case stays readable, but a handler whose stream targets a
  narrower codepage used to fail inside `StreamHandler.emit` -- and `logging`'s
  own `handleError` absorbed that failure, dropping the record and printing a
  per-alert traceback to stderr while the sink's fail-open guard never saw an
  exception. When any attached stream handler (including an ancestor logger's,
  since `logging` propagates) declares an encoding that cannot hold the line,
  the sink now emits the ASCII-escaped variant instead; `json.loads` yields the
  identical string, so no pipeline loses data (#431).
- `--cooldown-seconds` is now rejected when it is not finite. `inf` made the
  suppression test always true, so the first alert per key silenced every
  repeat forever -- the indefinite silence the dedup wrapper exists to prevent
  -- and the sweep guard could never fire, so the cooldown table grew without
  bound; `nan` silently disabled the cooldown an operator asked for. Both now
  exit 2 with a message. A non-positive value remains the documented way to
  disable the wrapper (#432).
- The sidecar now answers `HEAD` on `/health` and `/metrics` with the same
  status and headers as `GET` and an empty body. `_Handler` defined only
  `do_GET` and `do_POST`, so every other method fell through to
  `BaseHTTPRequestHandler`'s built-in 501 -- and the module docstring markets
  `/health` as the liveness endpoint for "k8s, ELB, docker healthcheck", so
  ELB/HAProxy-style probes that use HEAD could never go green. A method that a
  route does not accept is now answered 405 with an `Allow: GET, HEAD, POST`
  header instead of 501, and the request body is drained before either
  responds: an unread body made the kernel RST the connection before the
  client read the status, exactly the hazard the 401/413 paths already drain
  for (#433).
- Auto-calibration now warns on an unknown `calibration` value instead of
  falling back to the hand-tuned thresholds silently. Folding an unrecognized
  mode to manual is deliberate (fail-open: never worse than today), but a
  typo'd `"auto"` (`"automatic"`, `"atuo"`) left an operator who opted into
  auto-calibration running the generic defaults with no signal —
  `_auto_calibration_plan`'s docstring listed the unknown-value path as one
  that logs, yet it returned silently. It now emits one WARNING naming the
  value while still behaving as manual; `"manual"` and the default stay silent
  (#448).
- Near-duplicate loop mode (`loop_near_duplicate_enabled`) now re-arms a
  decayed normalized key from the sliding window on every step, matching the
  plain loop path. Previously it only re-armed the key observed on the current
  step, so a key that fired and then aged below `repeat_threshold` while other
  actions were observed stayed latched forever and its next genuine loop was
  suppressed — near-duplicate mode silently missed every loop after the first.
  Because the raw signatures in this mode are distinct by construction, it is
  often the only detector that can see such loops, so the miss was total (#450).
- `MeltdownDetector` no longer goes blind when window auto-scaling is enabled
  with `0 < window_scale_steps < meltdown_window_size`. Its readiness gate
  compared the fill count against the *scaled* target (`base*ceil(n/steps)`),
  which grows faster than the fill count in that regime, so the window never
  reached the target: the first entropy check slid from step `meltdown_window_size`
  to step `max_window` and was suppressed entirely for shorter episodes. The gate
  now opens once the base window has filled, while the window still grows toward
  the scaled target to retain more history (#477).
- `GoalDriftDetector` no longer fires at score 1.00 on an ordinary episode that
  reasons across several `message` steps and then makes one tool call that
  happens to error. Its min-samples gate counted every event (`total_steps`) but
  the drift score is built only from per-tool `tool_call` stats, so message
  padding could satisfy the gate and the score then measured a tool's error rate
  on a single sample. The gate now counts tool-call observations — the samples
  the score actually consumes — so drift is only scored once there is enough tool
  history for a rate to mean anything. (The semantic detector in
  `drift/goal_drift.py` is unaffected; its centroid is computed over all events
  by design.) (#478)
- `LatencyAnomalyDetector` periodic re-fit (`cusum_refit_every > 0`) no longer
  silently learns away a sustained regression. `adopt_candidate()` always moves
  `mu0` and zeroes the CUSUM, but the "baseline shifted" report was gated on the
  single-step `h*sigma0` bar — an order of magnitude stricter than the CUSUM's
  actual sustained-shift sensitivity `k*sigma0`. A sustained regression whose
  per-step move fell in the band `(k*sigma0, h*sigma0]` was adopted (mu0 slid up
  to the regressed latency, CUSUM zeroed) yet emitted no risk of any kind, so
  the detector went quiet on a tool pinned several sigma above its healthy
  baseline. The report bar is now `k*sigma0`, matching the sustained-shift
  sensitivity the CUSUM actually alarms on, so "baseline drifted" is exactly as
  hard to claim as a sustained deviation is. The shift is measured *signed*, not
  by magnitude: a downward move (latency improved) is never something this
  one-sided (upper) CUSUM alarms on, so an improvement now adopts the faster
  baseline silently instead of surfacing a spurious "baseline shifted" risk
  (#482).
- `run_and_monitor` in the Autogen adapter now drives a real Autogen agent. The
  real `autogen-agentchat` `run_stream` is an async *generator* function with a
  keyword-only `task`, so `await agent.run_stream(task)` raised
  `TypeError: object async_generator can't be used in 'await' expression`, and
  passing `task` positionally raised `takes 1 positional argument but 2 were
  given` on both `run_stream` and the `run` fallback. The adapter now iterates
  the returned async iterator directly (awaiting only when the result is
  actually awaitable, for duck-typed agents) and passes `task=` by keyword.
  Verified against autogen-agentchat 0.7.5 (#512).
- The LangGraph adapter (`watch_graph`) now surfaces a node that *raises*.
  A real LangGraph node error is not delivered as an update item — LangGraph
  propagates the exception out of `graph.stream()` — so the previous
  pass-through loop re-raised before emitting anything and the node crash, the
  very signal the error detectors exist to catch, never reached the Monitor.
  `watch_graph` now emits one error `StepEvent` for the failure and then
  re-raises, leaving the caller's own exception handling unchanged. The
  adapter's docstring claim that LangGraph "signals node errors" as yielded
  Exception updates was also corrected.
- The CrewAI adapter no longer emits a phantom `agent_step` for every tool
  call. CrewAI's default sync executor invokes `step_callback` twice per tool
  step (verified against `crewai==1.15.22`): first with a bare `ToolResult`
  (`result`/`result_as_answer` only), then with the `AgentAction`. The adapter
  mapped the `ToolResult` to a content-less `agent_step` (`tool_name=None`,
  `error=False`), doubling the step count and feeding the count/rate detectors
  (meltdown, stagnation, loop) noise. `snagline_step_callback` now drops the
  `ToolResult` invocation -- the paired `AgentAction` already produces the
  `tool_call` and a `result_as_answer=True` final answer arrives as an
  `AgentFinish`, so no event captured elsewhere is lost, and no payload content
  is inspected (#522).
- `snagline.auto` now extracts token counts on the **non-streaming** path, so a
  host instrumented via `instrument_openai` / `instrument_anthropic` /
  `wrap_client` gets the same token-runaway and budget coverage as the explicit
  adapters. The non-streaming success path emitted `tokens_in=tokens_out=None`
  even though `result.usage` was present, and `TokenRunawayDetector.observe`
  early-returns when both are None — so the zero-config entrypoint the README
  pushes had no token-burn coverage at all, silently. Both the sync and async
  OpenAI/Anthropic paths now pass the extracted counts through, mirroring the
  stream wrappers (#529).
- `side_effect_allowed_repeats` is now range-checked and required to be a whole
  number at construction and after env/file layering, like the other detector
  knobs (#322, #317). The guard fires on
  `count == side_effect_allowed_repeats + 1` and the occurrence count is always
  an integer, so a fractional value such as `1.5` could never be equal and the
  duplicate non-idempotent action detector was silently blind for the entire
  run -- repeated sends/payments/deploys never alerted. (`2.0` was never
  affected: an int-equal float compares equal.) The file layer handed the
  field straight through as a float, so `Config(...)` and a JSON config file
  both accepted `1.5` while the env coercioner dropped the token; an
  out-of-range or fractional value is now a configuration error naming the
  knob (#439).
- `MeltdownDetector` no longer truncates its fallback action identity to 16
  characters. When a `tool_call` event carries no `tool_name`, the detector
  keyed its per-window tool-choice distribution on `action_signature[:16]` —
  reintroducing the collision that issue #15 removed from `make_signature`.
  Two distinct endpoint-style signatures sharing a 16-char prefix (e.g.
  `search:database:alpha` / `search:database:omega`, or `/api/v1/users/1` /
  `/api/v1/users/2`) collapsed to one identity, so healthy alternation read as
  zero-entropy repetition and fired a false score-0.7 `meltdown` collapse. The
  identity is only ever compared for equality and counted, never displayed, so
  the full signature is now used with no downside (#493).
- `Monitor.ingest` now rejects a non-string `episode_id` up front instead of
  risking a crash on an unhashable key (a `list`/`dict` decoded from an
  untrusted body would raise `TypeError` before the fail-open guard, taking
  down the sidecar handler thread): under `fail_open=True` the event is dropped
  and logged once (bumping a new `events_dropped` counter); under
  `fail_open=False` it raises a clear `TypeError`. `POST /events` also
  400-rejects a batch whose `episode_id` or `step_id` is non-string before
  ingest. The new `events_dropped` counter is exposed on both the classic JSON
  and the default Prometheus `/metrics` surface
  (`snagline_monitor_events_dropped_total`), so a dropped event stays
  observable rather than silent (#479).
- `snagline watch --file --follow` no longer drops an event whose producer
  flushes it in two writes. `_iter_lines` now holds a partial line (one with no
  trailing newline) in a buffer and yields it only once the newline arrives, so
  a record split across two flushes reaches `monitor.ingest` as one line instead
  of two unparseable `skipping malformed line` fragments. A final unterminated
  line at EOF in non-follow mode is still yielded, and idle follow-polls still
  fire the heartbeat `on_wait` (#508).
- `cusum_k` and `token_cusum_k` are now range-checked at construction and after
  env/file layering. The CUSUM slack is subtracted from the accumulator every
  scored step, so a negative value *adds* `|k|` unconditionally and the
  accumulator climbs by `|k|` per step no matter what the data does -- the
  alarm becomes a function of step count alone and storms false positives on
  perfectly healthy traffic (measured: 20 risks in 30 steps on a constant 100
  tokens/step, score climbing to 1.0). `0` stays legitimate: it means no
  slack, maximally sensitive but not inverted (#421).
- The seven `_WelfordCUSUM` counters restored by `TokenRunawayDetector.load_state`
  and `LatencyAnomalyDetector._state_from_dict` are now coerced to their real
  types. `dump_state` copies them off a live object and a snapshot is only JSON,
  so a hand edit, a torn write, or a schema change between two versions can hand
  back an entry whose keys are all present but whose values are not numbers.
  Such an entry is structurally complete -- it clears every guard and is
  published -- and the failure then moved out of restore and into the next
  event: `int("x") + 1` raised inside `learn_only`, `Monitor.ingest` swallowed
  it fail-open, and the corrupted state stayed put so every later event from
  that episode scored nothing for the rest of its life. A non-numeric field is
  now rejected at restore, where the restore containment already handles it.
  `mu0` is still accepted as `None`, which is legitimate for a state snapshotted
  mid warm-up (#424).
- `BaselineStore` version retention is now chronological and version ids are
  path-validated. Retention sorted `versions/*.json` lexicographically, so a
  variable-width custom id pruned the wrong version — and at `max_versions=1`,
  writing `"b"` then `"a"` deleted `"a"` (the version just written and pointed
  to by `latest.json`), losing the newest baseline for rollback. Pruning and
  `list_versions` now order by write time and the just-written version is never
  pruned. Separately, a `version` id was used unsanitized as a path component:
  a `/` crashed `save()` mid-write *after* it returned the id, and `..` wrote
  outside the scope directory. Ids are now validated up front and a bad one
  raises `ValueError` before anything is written. The default path (synthesized
  fixed-width timestamp id) is unaffected (#451).
- The `adapters/openai.py` and `adapters/anthropic.py` sync stream wrappers no
  longer fabricate a telemetry event from `__del__` for an abandoned stream. A
  stream created but never iterated, closed, or failed reached no completion
  boundary, yet `__del__` emitted it as an `error=False` success (understating
  the error rate) with a `latency_ms` measured as time-until-garbage-collection
  (a bogus sample that poisoned the `LatencyAnomalyDetector` Welford/CUSUM
  baseline for that tool). The `__del__` is removed, so an abandoned stream now
  emits nothing — matching the async wrappers and the `snagline.auto` stream
  wrappers, which never emitted on abandonment (#531).
- `fit_baseline_from_jsonl` no longer aborts the whole fit on a wrong-typed
  field. A line such as `{"latency_ms": "120ms", ...}` is valid JSON and a
  valid `StepEvent` (the dataclass has no `__post_init__` validation), so it
  slipped past the fail-soft loop and only blew up in `ToolBaseline.add`'s
  arithmetic, killing the fit over one bad line in a long trajectory.
  `profile.add_event` is now inside the `try` and coerces the numeric fields
  *before* any accumulator is touched, so a bad line is skipped whole (not
  half-counted) -- matching the documented contract and `replay()` (#540).
- `snagline baseline --list-versions` is now honored on both the fit and
  `retrain` paths and is read-only everywhere: it lists and exits 0 without
  fitting, writing `baseline.json`, or storing a new version. Without
  `--store-dir` it now fails closed with `--list-versions requires --store-dir`
  (exit 2) instead of silently writing a file or bumping the store (#293).
- `TokenRunawayDetector` and `StagnationDetector` now build their restored
  state into locals and publish it in one assignment. A malformed snapshot
  entry raised halfway through `load_state` after live state had already been
  cleared or partially overwritten, so a rejected snapshot left the detector
  half-restored -- some episodes rebuilt and the rest gone, `_totals` /
  `_breached` still describing the live episodes that no longer existed, or
  `_counts` from the snapshot paired with windows emptied of every live
  episode, so the scaler believed episodes had history their windows no longer
  carried. `restore_dict` reports such a detector as keeping its live state,
  which was not true for these two. A rejected snapshot now leaves them
  exactly as they were (#417; `MeltdownDetector` was already fixed in #406).
- the Welford/CUSUM counters restored by `TokenRunawayDetector` and
  `LatencyAnomalyDetector` are now coerced to their real types. A snapshot
  whose entries are all present but whose values are not numbers -- a hand
  edit, a torn write, or a version skew -- used to be accepted silently and
  then poison the detector: the next event made `learn_only`'s `self.n += 1`
  a `TypeError`, which `ingest` swallows fail-open, so the episode scored
  nothing for the rest of its life and the fault was logged only once. Such an
  entry is now rejected at restore, where the #417 containment already handles
  it, and a value written as a numeric *string* is coerced to its real type
  rather than stored raw: `LatencyAnomalyDetector.load_state` re-assigned the
  seven core counters straight from the snapshot right after `from_snapshot`
  had coerced them, silently undoing the coercion for exactly the
  int/float-parseable-but-mistyped values it was meant to fix; the redundant
  re-assignment is gone so the coercion holds (#424).
- The `ml` extra's ensemble no longer pages on a healthy run when the
  configured baseline was fitted from a stream that carried no `latency_ms`.
  `EsnCusumDetector`'s Mahalanobis term gated on `ToolBaseline.count` while
  `mean_latency` / `std_latency` are computed from `latency_count` only, so a
  profile with `count=100, latency_count=0` legitimately read mean 0.0 ms /
  std 0.0. Any real latency was then scored against the 1 ms sigma floor as a
  ~100-sigma deviation, saturating that term to 1.0; it feeds
  `max(anomaly, mahalanobis)`, so it armed the CUSUM on every step and the
  detector fired a false `ml_ensemble` risk every few steps of an otherwise
  healthy episode. The latency term now gates on `latency_count >= 2`, the
  same fix the zero-dependency latency detector already had (#348). The
  error term is unaffected -- measuring error rates on untimed streams is the
  reason the two counts are kept separate (#101).
- `snagline baseline` now reports a tool whose steps carried no timing as
  `n=10 timed=0 mean=n/a std=n/a` instead of `n=10 mean=0.0ms std=0.0ms`.
  The old line claimed a hundred samples averaged zero milliseconds, which
  is what hid a baseline the latency detectors cannot use. Tools whose every
  step was timed print exactly as before (#348).
- `episode_token_budget` and `token_budget_warn_fraction` are now range-checked
  at construction and after env/file layering, like the horizon and stagnation
  knobs. A zero or negative budget used to fire a score-1.0 `budget_breach` on
  the first token-bearing step, and a `token_budget_warn_fraction` of `0.0` a
  score-0.8 warning; values above `1.0` made the pre-breach warning
  unreachable. An out-of-range value is now a configuration error naming the
  knob (#317). 57305ac (fix(cli): make --list-versions read-only on fit and retrain paths)
- `snagline.auto` stream wrappers now support the context-manager form. Both
  SDKs document streaming as `with client....create(stream=True) as stream:`,
  but the wrappers only proxied the raw stream's attributes through
  `__getattr__`, and implicit special-method lookup for `with` / `async with`
  resolves on the *type*, never through `__getattr__` -- so the form raised
  `TypeError` and emitted zero events, a monitored call that was not monitored
  at all. `_SyncStreamWrapper` / `_AsyncStreamWrapper` in both `auto/openai.py`
  and `auto/anthropic.py` now define `__enter__`/`__exit__` and
  `__aenter__`/`__aexit__`, returning themselves so iteration still flows
  through the wrapper and the deferred event still fires at exhaustion or
  close (#335).
- `wrap_client` (the per-client `snagline.auto` path) is now idempotent like
  global mode. It had no `__snagline_wrapped__` guard, so composing global and
  per-client instrumentation -- or calling `wrap_client` twice -- stacked a
  second wrapper layer and emitted one event per layer per call, silently
  biasing every counting detector (`error_cascade` tripping on ~2 real
  failures, doubled loop/stagnation repetition, every latency in the CUSUM
  window twice). `_wrap_one` now marks its own output and returns an already
  wrapped callable verbatim across `auto/openai.py`, `auto/anthropic.py` and
  `auto/langchain.py` (whose `_wrap_one` never set the sentinel at all), and a
  second `wrap_client` that finds everything already wrapped stays quiet
  instead of warning that nothing was patchable (#336).
- The `raw`, `langgraph` and `claude_code` adapters now stamp `StepEvent.timestamp`
  from the same monotonic clock as the other live-instrumentation adapters. They
  were the three holdouts reading `time.time()` (unix epoch, ~1.79e9) while the
  rest read `time.perf_counter` (process-local, ~1e3-1e5), so an episode that
  mixed them -- a raw-adapter host loop plus a claude_code bridge, for instance --
  produced a ~1.79-billion-second delta on a single step. That fired both
  time-axis risks as critical on one event and, because `elapsed` can never be
  un-spent, latched `breached` forever: every later genuine step was scored
  against an already exhausted `wall_clock_budget` and the episode could not
  recover (#532).
- The time axis now re-anchors instead of scoring a step interval no single step
  can span (a day). A delta past that bound means the two events came from
  different clock domains -- a mixed-adapter episode, a cross-process replay, or
  an adapter reading times out of an external record -- and scoring it invented
  an idle gap and a permanent breach out of one healthy event. The span is
  dropped, budget already spent is kept, and the next event measures from a sane
  reference. The bound is five orders of magnitude below the jump it guards
  against, so the slowest real tool call still fires genuinely (#532).
- The CONTINUUM adapter intentionally still stamps unix epoch: it pairs a
  claim's observed time with its terminal record to report a real
  claim-to-terminal duration, which is only meaningful when both sides come
  from the same ledger clock. Combined with the re-anchoring bound above, a
  CONTINUUM episode now degrades to a dropped interval rather than a fabricated
  breach, so mixing it with a monotonic adapter is safe (#532).

### Security
- The sidecar's mutating `POST` endpoints now check where a request came from,
  not just that it carries the token (#388). `snagline serve` defaults to no
  token ("unset means all endpoints are open"), and authenticating the token
  never authenticated the *sender*: any page the operator was visiting could
  issue a cross-site POST to the loopback sidecar as a CORS "simple request"
  (no preflight), so the write landed. A forged `POST /episodes/end` silently
  discarded an in-flight episode's detection state, and forged `POST /events`
  telemetry drove the halt policy and the `/metrics` gauges. `POST` is now
  gated on two things a forged cross-site request cannot supply together: a
  JSON content type (the CORS-safelisted types are the only ones a cross-site
  `fetch` can send without a preflight; a bad one gets 415), and a same-site
  origin when the client declares one (`Sec-Fetch-Site` / `Origin` are absent
  from non-browser clients, so a missing header is not an error; a request
  declaring itself cross-site gets 403). Both refusals log the origin, so an
  attempt is distinguishable from a mistyped token instead of vanishing into
  the general 401 noise. Every shipped client already sends JSON.

### Changed
- CI: removed the leftover `disallowScopes: [name:none]` placeholder from the
  semantic-PR-title workflow. `amannn/action-semantic-pull-request` treats each
  `disallowScopes` line as a regex auto-wrapped in `^…$`, so `[name:none]`
  compiled to a character class that would reject any legitimate single-letter
  scope (`fix(a): …`) while blocking nothing intended (#472).
- `Monitor(detectors, sinks, config=cfg)` now applies the enforcement knobs the
  `Config` carries -- `policy`, `halt_url`, `halt_timeout_s`,
  `min_severity_for_halt`, and `fail_open` -- instead of only its own
  arguments. The direct constructor and `Monitor.default()` disagreed about
  the same configuration, so an operator wiring `SNAGLINE_POLICY=halt_webhook`
  into the library API silently got observation mode with no warning, while
  the same config through `snagline serve` armed the webhook. A `None`
  argument means "not given" and defers to the config; an explicitly passed
  value still wins, and `Config`'s defaults are the same literals the
  arguments used to carry, so every pre-existing call constructs identically
  (#352).
- `BatchingSink` now rejects a non-positive `flush_interval` at construction
  with `ValueError`. `_wake.wait()` returns immediately for one, so the flusher
  thread spun through an empty queue roughly 780,000 times per second, pinning a
  full core for the life of the process while alerts still delivered and
  nothing else looked wrong; `close()` also used the interval as its join
  timeout, so a negative one starved the shutdown drain. `max_batch` is still
  clamped, since any size still paces -- a non-positive interval has no
  meaningful reading (#358).
- `BaselineCollector.snapshot()` now returns a `copy.deepcopy` of the live
  profile under a lock shared with `observe()`/`commit()`, instead of the
  accumulator itself. The name promises a point-in-time view, but a caller that
  inspected and mutated the result (`p.tools.clear()` while deciding whether to
  `commit()`) corrupted the profile a later `commit()` persisted, and a reader
  iterating it raced `observe()` with no lock on either side. `commit()` copies
  before the fsyncing save so a concurrent ingest cannot reshape the profile
  mid-serialization either (#357).
- `snagline watch --episode-id X` now attributes ingested events to X. The
  flag's help text promised attribution, but the parsed event's own
  `episode_id` was ingested and the flag only named the zero-events fallback
  at teardown, so an operator scoping a multi-tenant stdin stream to one
  episode silently got per-event attribution with no warning. The override
  is applied at parse time (via `dataclasses.replace`, since `StepEvent` is
  frozen), so ingest and the teardown set see one id (#354).
- `SilentAbortDetector.load_state` no longer overwrites its own
  `output_action_types` from the restored snapshot. That field is operator
  configuration, not per-episode state, so a snapshot written on a
  differently-configured host silently changed which final steps counted as
  "output" on this one -- a real silent abort could be missed, or a clean
  ending flagged. `dump_state` still records the field for diagnostics, as
  `MeltdownDetector` does for `window_size`; only the restore path now
  ignores it (#347).
- A `compaction` event carrying no usable pins of its own (missing or empty
  `pinned`, or a malformed non-collection value) no longer discards an
  in-flight grace window from an earlier pin-bearing compaction. Such an event
  describes a truncation that tracked no constraints and says nothing about
  the previous window's pins, but the pending set was unconditionally
  overwritten with `None`, so a `governance_decay` risk that had not yet
  reached its deadline vanished silently. The previous window now stands and
  can still fire or be confirmed; only a compaction that pins constraints of
  its own replaces it (#356).
- `min_severity_for_halt` and `halt_timeout_s` are now range-checked in `Config`
  at construction and after env/file layering, like every other ranged knob.
  They were previously validated only inside `Monitor._configure_policy`, which
  `snagline serve` reaches *after* printing its "listening on" banner and inside
  `with suppress(KeyboardInterrupt)` -- so an out-of-range
  `--min-severity-for-halt` (or the same via `SNAGLINE_*` / config file) printed
  a banner announcing a live server and then died with a traceback and exit code
  1 instead of the clean exit-2 usage error every other malformed flag produces.
  The same bad value also silently took down `snagline watch` / `replay`. All
  three commands now exit 2 with a message naming the knob, before anything is
  printed. `halt_timeout_s` is also checked unconditionally now, not only when
  `policy` is already `halt_webhook`: a non-positive timeout has no valid
  reading under any policy, and gating it let a negative timeout set while
  observing surface only when the policy was later armed (#353).
- `Monitor.restore` now contains a malformed detector entry instead of
  aborting the restore around it. Each detector's `load_state` hard-subscripts
  fields an older release did not write, so a snapshot that crossed a version
  boundary raised `KeyError`/`AttributeError` mid-loop: detectors already
  loaded held the snapshot's state, the one that raised kept its live state,
  and the sink and time-axis restoration that follow the loop never ran --
  leaving a monitor whose components disagreed about which episodes existed,
  with no indication of it. The entry is now logged and skipped, leaving that
  detector on its live state while the rest of the restore completes. A
  rejected slot is marked consumed, so it is not also reported as an unknown-
  slot orphan (#384).
- Detector `load_state` is now transactional. It builds the windows, scaler
  positions, streaks and fired flags into locals and publishes them only once
  every field has parsed; a snapshot rejected partway used to leave the
  detector half on the snapshot and half on its live state -- the windows
  replaced while the counts and fired flags kept their live values, so the
  restored window contradicted the scaler position `observe` then used.
- `SNAGLINE_STATE_REDIS_URL` pointing at an unusable URL now warns and falls
  back to in-memory state instead of crashing startup. redis-py's URL parser
  rejects any scheme other than redis/rediss/unix at construction, before a
  socket is ever opened, so `SNAGLINE_STATE_REDIS_URL=postgres://...`, a bare
  `host:6379`, or an unsubstituted secrets-manager placeholder raised
  `ValueError` out of `default_state_backend()` and through
  `Monitor.default()` -- the one arm of an otherwise optional knob that
  escaped. The caught set now matches what the constructor can raise, and the
  warning names the offending target. The URL is the credential -- redis-py
  reads the password out of it -- so it appears in that line redacted to
  scheme and host, and once it carries a secret the exception's own text is
  withheld too, since redis-py's parse failures can quote the value back
  (#392).
- `ConsoleSink(stream=...)` now rejects a binary or already-closed stream at
  construction with a `TypeError` naming the stream and advising
  `open(path, 'w')` or the logging module. `open(p, "wb")` and
  `sys.stdout.buffer` are the natural ways to route alerts to a file, and a
  `str` write to either raises `TypeError` -- not an `OSError` subclass, so
  the fire-and-forget guard in `emit()` never caught it and every alert was
  silently discarded for the whole run behind the fail-open contract (#391).
- Halt-webhook enforcement now falls the directive back to `continue` when a
  consultation fails, instead of leaving a previously latched `pause` in force.
  The error path (timeout, dead endpoint, malformed body, unknown action) reset
  `policy_errors` but never touched `last_directive`, so once a severe risk
  latched `pause` an unreachable halt service held the host paused indefinitely
  on a stale decision it could no longer confirm -- the fail-CLOSED outcome the
  `last_directive` docstring, the method docstring, and the module header all
  promise against. It now resets to `continue` under `fail_open=True` (a genuine
  pause is re-issued on the next successful consult); `fail_open=False` still
  propagates (#523).
- `TokenRunawayDetector`'s pre-breach warning is now scored `0.7` (was `0.8`),
  so it derives `warning` severity instead of `critical`. At `0.8` merely
  reaching `token_budget_warn_fraction` of the budget paged critical -- the same
  band as the `1.0` breach it precedes -- and, under `policy="halt_webhook"`,
  performed a halt consult (default `min_severity_for_halt=0.8`) that can return
  an ABORT reserved for the actual breach. The score now matches the
  `wall_clock_budget` twin envelope, which already grades its pre-breach signal
  at `0.7` (#537).
- `EsnCusumDetector` (the `ml` extra) now range-checks its constructor knobs:
  `reservoir_size >= 1`, `cusum_k >= 0.0`, `cusum_h > 0.0`. `cusum_h = 0.0`
  divided by zero in the score formula on every step, so the detector was
  permanently dark while logging a traceback per step; a negative `cusum_h`
  could never be crossed and silently disabled detection; a negative
  `cusum_k` inflated the accumulator and fired false alarms on healthy traffic.
  The knobs have no `Config` field, so the constructor is the only gate. A
  persistent internal fault is also now logged once instead of per step:
  `observe()` swallows the exception, so `MLOrchestrator._log_fault_once`
  could never dedupe it (#386).
- `Config` now rejects non-finite float knobs (`inf`, `-inf`, `nan`, and
  overflow literals like `1e400`) from both `SNAGLINE_*` environment variables
  and JSON config files. A non-finite CUSUM knob never crashed: `cusum > inf`
  is never true, and `max(0.0, nan)` is `0.0` in CPython, which pinned the
  accumulator at zero -- so the detector went inert for the entire run while
  the process started cleanly and printed nothing to say the safety net was
  off. The env path logs and drops the value (the existing un-coercible-value
  contract, falling back to the built-in default) and a JSON file now raises
  with the offending key named, since `json.loads` accepts the bare
  `Infinity` / `NaN` tokens by default (#383).
- `semantic_drift_cusum_h` and `semantic_drift_cusum_k` are now range-checked
  at construction and after env/file layering, alongside the deterministic
  CUSUM bars. `cusum_h` is the denominator of the semantic goal-drift alarm
  score, so `0` raised `ZeroDivisionError` on every scored step; the
  detector's fail-open wrapper swallowed it and re-logged a traceback once per
  step while no `goal_drift` risk ever fired. A negative `cusum_h` or
  `cusum_k` inverted the CUSUM and stormed a false positive on nearly every
  step. Both are now startup configuration errors naming the knob (#370).
- The latency detector's calibrated start now gates on the profile's
  `latency_count` rather than its total `count`. The seeded statistics
  (`mean_latency` / `std_latency`) are computed from the latency-bearing subset
  only, so a profile fitted from a stream whose adapter reported no timings --
  which auto-calibration explicitly supports -- had `count=100,
  latency_count=0, mean_latency=0.0`. Seeding from it froze onto a zero
  baseline: the reference spread stays finite (sigma floors), but a real call
  then measures as a large multiple of that floor and crosses the CUSUM
  threshold on the first step, paging critical on step 0 of every episode.
  Such tools now keep the learn-then-freeze warm-up (#348).
- A `SNAGLINE_*` key naming an object-typed field now logs a warning when
  ignored, instead of vanishing silently. `Config.from_env_overrides` warned
  only for values that failed coercion; a key naming `goal_drift_baseline` or
  `calibration_baseline` (both `BaselineProfile | None`, deliberately out of
  reach of string coercion) was skipped with no log line, so the module's own
  promise that ignored keys are "ignored (logged at warning)" did not hold.
  An operator setting either env form got no feedback that it could never
  apply, and found out only by noticing the detector staying inert. The
  warning now also points at the right alternative: `SNAGLINE_CALIBRATION_
  BASELINE_PATH` for the one field with a path form, or passing the object in
  code for `goal_drift_baseline`, which has none (#355).
- `goal_drift_score_threshold`, `ml_ensemble_score_threshold`,
  `goal_drift_error_tolerance` and `goal_drift_latency_k` are now range-checked
  at construction and after env/file layering, alongside the divisor and count
  knobs. Both detectors gate their own output on a score threshold, and scores
  are clamped to [0, 1], so a threshold above `1.0` made the gate always true
  and the detector could never fire on any input -- reporting
  `detector_errors == 0` while monitoring nothing -- while `0.0` or less fired
  on a score of exactly `0.0`, paging healthy traffic. A negative tolerance or
  latency k added to the drift instead of absorbing it, since both are slack
  subtracted from a non-negative measure. None of them divide or count, which
  is why the earlier validators missed them (#385).
- `WebhookSink` and `SlackSink` no longer log the destination URL when a POST
  fails. The URL is the credential in both cases: a Slack incoming webhook
  embeds its secret as the final path segment
  (`https://hooks.slack.com/services/T.../B.../<secret>`), and an arbitrary
  webhook URL routinely carries basic auth (`https://user:pass@host/`). A sink
  that cannot reach its destination is exactly the moment an operator reads the
  log, so the raw URL was being handed to whoever was already looking at a
  failed integration. Both sinks now log only scheme + host + port via a shared
  `redacted_destination` helper in `sinks/base.py`, matching `PagerDutySink`,
  which never logged its routing key. Their `__repr__` is redacted too, so the
  URL cannot reach a diagnostic capture or an unhandled-exception report
  (#390).
- The same two sinks now name a POST failure by its class and status code
  (`HTTP 503`, `URLError`) rather than logging the exception itself. A
  `URLError` embeds the destination URL in its reason for some failures (`no
  host given: <url>`), and the traceback a `logger.exception` call attaches
  carries it out to the log verbatim -- a second route for the same leak, from
  inside the fail-open `except` block where a further failure would be the
  last thing the operator is told (#390).
- `redacted_destination` no longer raises on a destination with an unbalanced
  bracket in the authority (`https://[::1`). `urlsplit` itself raises
  `ValueError` on that input, so the validity guard beneath it never ran and
  the exception escaped through the two callers the helper exists to keep safe.
  Such a destination is now reported as `<invalid destination url>`, without
  echoing the malformed input back (#390).
- The network sinks (`WebhookSink`, `SlackSink`, `PagerDutySink`) now bound the
  whole POST by a wall-clock deadline instead of passing `timeout=` through to
  `urlopen`. That argument is applied per socket operation, and only after name
  resolution has already completed, so a stuck resolver held the call for as
  long as it liked and a server trickling its body one byte at a time just
  under the interval never tripped a read timeout at all. Issue #395 measured a
  configured 2.0 s budget taking 36.2 s on exactly that trickle, on a request
  the sink reported as successful. All three sinks now go through a shared
  `bounded_post` in `sinks/base.py`, which raises `TimeoutError` when the
  deadline passes and abandons the in-flight request on a daemon thread; the
  sinks log it fail-open as before (#395).
- Anthropic streaming (`stream=True`) auto-instrumentation now records token
  usage. The SDK splits usage across raw events -- `message_start` carries
  `input_tokens` on `event.message.usage` and `message_delta` carries the
  running `output_tokens` on `event.usage`, while the terminal `message_stop`
  event and the `Stream` object carry none -- so probing only the last chunk
  always yielded `tokens_in=tokens_out=None`, silently disabling token-runaway
  and budget coverage for every Anthropic streaming call. Usage is now
  accumulated across the stream, with the pre-existing last-chunk/stream
  extraction kept as a fallback for shapes that stamp usage on the terminal
  chunk (#530).
- `snagline.auto.langchain` global mode now patches the classes the declared
  dependency actually ships. It imported `langchain.chains.base.Chain` and
  `langchain_core.language_models.BaseLanguageModel` in one `try`, but
  `langchain.chains` was removed in the langchain 1.0 restructure and lived in
  the top-level `langchain` package, which `snagline[langchain]` never declared
  (it depends on `langchain-core` only). On every modern install the import
  failed, the bare `except Exception` reported it as "LangChain not installed",
  and no LangChain call was ever observed -- silently, with no error. Global
  mode now probes the package for presence first, patches `BaseChatModel` and
  `BaseLanguageModel` from `langchain_core` (with `langchain.chains.base.Chain`
  as a langchain < 1.0 fallback), keeps the batch `generate` entrypoints out of
  global mode since `BaseChatModel.invoke` delegates to `self.generate`, and
  reports a present-but-reshaped SDK as a monitoring failure rather than an
  absent dependency (#339).
  which never logged its routing key. Their `__repr__` is redacted too, since
  the default attribute dump would leak the same URL into any diagnostic
  capture (#390).
- `snagline serve --max-body-bytes 0` (or any negative) is now rejected at
  startup with exit 2 instead of being accepted. `do_POST` compares the
  declared length against the cap with a strict `>`, so a cap of 0 rejects
  *every* POST with 413: the sidecar came up cleanly, reported `GET /health`
  green, and silently dropped 100% of inbound telemetry. The neighbouring
  `max_risks` knob was already clamped with `max(1, ...)` while this one was
  not, and the flag has no `None`/unlimited value -- worse, `--episode-ttl-seconds`
  on the next line documents `0 disables`, so 0 was the value an operator
  reaching for "no body limit" was most likely to type. `make_handler` /
  `make_server` now raise on a non-positive cap too, so library callers cannot
  build a green-but-deaf server either (#394).
- `BatchingSink.close()` no longer hangs forever when the wrapped sink is
  stuck. It joined the flusher for `flush_interval + 1.0` s and then called
  `_flush()` unconditionally, which takes `_delivery_lock` with no bound -- so
  when the background thread was parked inside `_deliver` on a hung endpoint,
  shutdown parked on the very lock that thread held. The flusher clears the
  queue before taking the lock, so the deadlock needed an alert enqueued
  *while* it was parked, which is what a live alert stream does; a hung
  shutdown produced no diagnostic at all. `close()` now waits once more with
  the same budget and, if delivery still does not return, logs how many alerts
  went undelivered instead of blocking indefinitely (#393). The acquired lock
  is also now released *before* the fallback `_flush()`: `_flush` -> `_deliver`
  re-acquires that non-reentrant lock, so holding it across the call
  self-deadlocked shutdown whenever the probe succeeded and the queue was
  non-empty -- reachable whenever a delivery finished just as the deadline
  expired and an alert arrived mid-shutdown.
- The network sinks no longer follow a redirect. `urllib` honours a
  `301`/`302`/`303` by re-issuing the request to the `Location` URL as a GET
  with no body, and hands the final `2xx` back to the caller, so a sink that hit
  a redirect reported a successful delivery while its payload travelled with the
  POST the server rejected and went to a destination the server chose. That is
  reachable without a misconfigured destination -- a trailing-slash hop, an
  HTTP->HTTPS or proxy canonicalisation, or, worst, an auth redirect from an
  expired credential that would otherwise have shown as a `401`. All three sinks
  now POST through an opener whose `_NoRedirect` handler makes urllib raise
  `HTTPError` for the `3xx`, which is logged fail-open as any other delivery
  failure is; a `307`/`308` already raised, so the failure mode is now consistent
  across redirect codes instead of silent for exactly the common ones (#389).
- `bounded_post` now caps how many sink POSTs may be in flight across the
  process at once. The `#395` deadline abandons a stalled POST but cannot
  cancel it: the worker thread, and the socket it holds, lives on until the
  exchange resolves on its own. Against a merely slow endpoint that thread
  drains and exits shortly after, but against a *dead* one -- a black hole
  that neither answers nor resets -- every alert spawned a worker that never
  returned, so a monitor under sustained load accumulated one parked thread and
  file descriptor per emit without bound (#423). A process-global non-blocking
  semaphore (`_MAX_INFLIGHT_POSTS`, default 64) now bounds the parked workers:
  a worker holds a slot for its whole life and frees it in a `finally`, so the
  ceiling counts live threads rather than calls. When every slot is taken the
  next POST is dropped with a distinct `SinkBusyError` -- fast, and logged
  fail-open at the sink like any other delivery failure -- rather than adding
  another parked thread. The cap is deliberately generous, so only a genuinely
  dead endpoint under load ever trips it (#423).
- The halt webhook and `snagline hook --url` no longer hold the caller past
  their configured budget. Both used bare `urllib.request.urlopen`, whose
  `timeout=` is applied per socket operation and only after name resolution,
  so a stalled resolver or a body trickled one byte per interval just under
  the timeout parked the exchange far past the configured `halt_timeout_s`
  with no `policy_error` ever counted, because nothing ever raised. For the
  halt webhook this is the enforcement policy, called synchronously from
  `ingest()`, so the stall was the host agent's own step while the operator's
  budget read 250 ms. Both call sites now go through the shared `bounded_post`
  in `sinks/base.py`, which bounds the whole exchange by a wall-clock deadline
  and raises `TimeoutError` on overrun -- counted as a `policy_error` and left
  fail-open, as any other halt failure already was (#415).
- The halt webhook no longer follows a redirect. `urllib` rewrites a
  `301`/`302`/`303` POST into a bodyless GET to the `Location` URL and hands
  the final `2xx` back, and the halt path parses that reply into a
  `HaltDirective` -- so a redirect let the enforcement decision arrive from a
  server the operator never configured, on a round trip that looked
  successful. An auth redirect from an expired credential, which would
  otherwise have shown as a `401`, became a silent misrouting instead. The
  `snagline hook --url` forward had the same shape, leaking the step event to
  the redirect target. Both now POST through the opener carrying the
  `_NoRedirect` handler added for the sinks (#389), which raises `HTTPError`
  for the `3xx` and logs fail-open (#416).
- The halt webhook and `snagline hook --url` no longer log the destination URL
  when the POST fails. The halt URL can carry basic auth (`user:pass@host`)
  and the failure line is what an operator reads when enforcement stops
  working. Both now log only scheme + host + port via `redacted_destination`,
  matching the network sinks. The `snagline serve` startup banner is redacted
  too, since it prints the same URL to stderr, which a process supervisor
  captures and keeps after the process is gone (#390).
  `ErrorCascadeDetector` published its windows and counts before parsing the
  `consecutive` streaks and `fired` flags, so a malformed streak raised with
  the snapshot's windows/counts already installed on top of the live streaks;
  it now parses every field into locals before publishing, like its siblings.
- Scaled-window restore now seeds the auto-scaler position it inferred, not
  just the window width. `load_state` sized a restored window from the
  position it read off the payload but rebuilt the scaler's `counts` dict
  from the payload alone, so an episode present in `windows` but missing from
  `counts` (a pre-#92 snapshot, or a partial one) restored at the correct
  width and then lost it: `MeltdownDetector`'s window was left stuck wide for
  the rest of the episode because `push` popped at most one item per call,
  while `ErrorCascadeDetector` and `LoopDetector` refit their deques down to
  the base on the very next observe, discarding the history the restore
  applied. All three detectors now seed the position they inferred, and
  `MeltdownDetector.push` pops while over instead of once (#403).
- The sidecar now decodes a `Transfer-Encoding: chunked` POST body instead of
  reading it as empty and answering 400 "invalid StepEvent JSON": the JSON was
  fine, and the real body was left unread in the socket, resetting the
  connection on close. The framed body is capped at `max_body_bytes` exactly
  like a Content-Length request, malformed framing is a 400, and the
  401/404/413 paths still drain a chunked body before replying so the response
  is not lost to RST (#434).
- `SemanticGoalDriftDetector.load_state` now rejects a restored running sum
  whose embedding dimension does not match the live baseline centroid: the
  live path's dimension check compares the embedder against the centroid and
  passes when the operator refit both together, so a snapshot fitted under an
  older embedder was indexed out of range on every step -- IndexError,
  swallowed by the fail-open wrapper, zero risks and a traceback per step for
  the rest of the run. Only the affected episode is dropped and it
  re-accumulates from scratch (#435).
- `loop_repeat_threshold`, `cascade_error_threshold` and the cycle-window
  pair are now range-checked against the window that holds the count. A
  threshold wider than its sliding window is unreachable on every input, so
  the detector never fired and reported healthy traffic forever, with no error
  or warning; with window scaling on the reachable bound is the scaled cap.
  A bad combination is now a configuration error at construction and after
  env/file layering (#436).
- Values from a JSON/TOML config file are now type-checked. Env values were
  coerced from text but file values were used verbatim, so `{"fail_open":
  "false"}` stayed a truthy string and kept fail-open on, `{"fail_open": 0}`
  silently turned it off, and `{"max_live_episodes": "5000"}` blew up with an
  off-point `TypeError` inside a range validator. A quoted scalar is now
  coerced to the declared type and anything unparseable or of the wrong shape
  is rejected at load with the field name and both types (#437).
- A non-finite or negative `latency_ms` is no longer absorbed into a
  `ToolBaseline`'s running moments: a single NaN sample made `mean_latency`
  NaN, collapsed `std_latency` to 0.0 (`max(0.0, NaN)` is 0.0 in CPython), and
  the corrupted profile was then *persisted*, poisoning every monitor and
  detector that loads it. The call and its error are still counted; only the
  timing sample is dropped, which is the fail-soft handling
  `fit_baseline_from_jsonl`'s docstring already promised (#438).

### Changed
- Docs accuracy sweep. `docs/ATTACH_ANY_SYSTEM.md`'s "Current limitations
  (verified in code)" block described auto-instrumentation, the
  `openai`/`anthropic`/`continuum` adapters, sidecar auth/TLS, the versioned
  `BaselineStore` with retrain and auto-calibration, the Slack/PagerDuty/dedup
  sinks, and `/metrics` + `/health` as unbuilt -- all of which ship and are
  itemised as merged in the same file's Progress log, so the "(verified in
  code)" header was actively misleading. The block now lists only genuinely
  open limitations (no schema auto-discovery, opt-in cross-worker state,
  untuned ml/drift extras, no PyPI upload yet) and points at the Progress log
  (#453). The `docs/FRAMEWORK_BRIDGES.md` OpenClaw `curl` recipe emitted
  invalid JSON -- a bare `"timestamp",` key with no value made the sidecar
  reject the pasted snippet with 400; it now sends `"timestamp":'$(date +%s)'`
  (#454).
- The calibrated CUSUM seed now gates on `latency_count` instead of `count`.
  A profile fitted from a trajectory that called a tool often enough but never
  with a `latency_ms` (the shape auto-calibration produces on streams whose
  steps report no timing) used to seed `mu0 = 0` with the bare sigma floor, so
  the first ordinary call came back score 1.0 / `critical` with a
  self-contradictory "baseline (mean 0ms)". Such tools now keep the
  learn-then-freeze warm-up. The ML ensemble's Mahalanobis latency term is
  likewise gated on two timed samples, so it scores the error term alone
  instead of dividing by the 1ms floor (#539).

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
