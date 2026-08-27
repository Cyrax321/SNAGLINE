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
  snagline` does not work yet.
- No self-observability: no metrics endpoint, no health check, no structured
  logs for the monitor itself.

## What is needed, prioritized

### P0 (blockers for universal attach)

1. **Mature, document, and authenticate the HTTP sidecar.** This is the
   lingua franca for any language or runtime. Validate it round-trips
   `StepEvent`, add auth (HMAC or bearer) and TLS verification, and document
   the schema.
   - **Done:** bearer/token auth (#31), body-size cap + batched `POST /events`
     (#32), 12-factor config + CLI wiring (#36). TLS terminates at a
     reverse proxy (configs below) or in-process via `--certfile/--keyfile`
     (issue #120).
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
     loader (#30) wired into the CLI (#36). Actual `pip install snagline`
     upload still needs a PyPI token (not performed automatically).

### P1 (production readiness)

4. **Pluggable state backend** (in-memory default, Redis/DB optional) so
   detector state and baselines survive restarts and span workers; shard the
   lock by `episode_id` to remove the global ingest bottleneck. **Done:**
   `StateBackend` (memory + optional Redis) + per-episode lock sharding (#44).
5. **Alerting maturity:** dedup/cooldown (issue #4), severity, Slack and
   PagerDuty sinks, batched and rate-limited async dispatch. **Done:**
   `DedupSink` + severity (#39, #40), `SlackSink`/`PagerDutySink` + CLI
   exposure (#41-#43), `BatchingSink` (#48).
6. **Baseline lifecycle:** auto-capture cadence, versioned store, per-tenant
   baselines, scheduled retrain. **Done (storage + collector + CLI):** versioned
   per-tenant `BaselineStore`, `BaselineCollector`, and `snagline baseline
   --store-dir` (#45, #46). **Done (retrain contract):** `snagline baseline
   retrain` refits from the newest JSONL window with an atomic store bump,
   plus a documented cron/systemd cadence and `--max-age` staleness warning;
   see docs/RETRAIN_CADENCE.md (#102). The schedule itself stays host-owned.

### P2 (detection quality, the differentiator)

7. Build the `ml` and `drift` extras (ESN ensemble, semantic goal-drift)
   with training pipelines (`train.py`) so accuracy approaches the paper.
8. Auto-threshold calibration from the baseline; per-system tuning.
9. Eval harness plus a labeled dataset to honestly report detection numbers.

### P3 (trust and observability)

10. Monitor self-metrics (Prometheus), health endpoint, structured logs.
    **Done (mostly):** `Monitor.metrics()` counters (#47) and the `GET
    /metrics` sidecar endpoint now speak Prometheus text exposition 0.0.4 by
    default (#98): `snagline_events_total`,
    `snagline_risks_total{trigger,severity}`, `snagline_episodes_active`, and
    the `snagline_ingest_seconds` count/sum pair. The legacy JSON body stays
    available via `?format=classic`, an `Accept: application/json` header,
    or `SNAGLINE_METRICS_FORMAT=classic`. Scrape config: point Prometheus at
    the sidecar with `metrics_path: /metrics` (same port and auth token as
    the event endpoints). Structured logging remains.

    **End-of-episode signal (#123):** `snagline_episodes_active` counts
    distinct episode ids seen through the sidecar since process start
    (capped at 10,000, least-recently-seen eviction). A host that knows an
    episode is finished should tell the sidecar so the gauge drops the id
    immediately instead of waiting for cap eviction:

    ```bash
    curl -X POST http://127.0.0.1:8787/episodes/end \
      -H "Authorization: Bearer $SNAGLINE_SERVE_AUTH_TOKEN" \
      -H "Content-Type: application/json" \
      -d '{"episode_id": "run-42"}'
    # -> {"status": "ended", "episode_id": "run-42"}
    ```

    The endpoint runs behind the same auth token as every other POST,
    forwards to `Monitor.end_episode()` (running detector finalizers and
    releasing per-episode detector state), and answers `400` on malformed
    bodies without crashing or retaining anything beyond the id itself.
    Ending an unknown id is an idempotent no-op.

    **Episode TTL expiry (#173):** When hosts cannot or forget to send the
    end signal, the gauge would otherwise rely solely on cap eviction.
    Configure an idle-based TTL so ids not seen for N seconds expire
    automatically: `snagline serve --episode-ttl-seconds 3600` or
    `SNAGLINE_EPISODE_TTL_SECONDS=3600` / `Config(episode_ttl_seconds=3600)`.
    `0` or `None` disables (default, byte-identical to #123). Expiry uses
    monotonic time, so NTP jumps cannot mass-expire or freeze entries; the
    table stays ids-plus-a-float and the same `_MAX_TRACKED_EPISODES` cap
    still bounds memory. Reseeing an id after expiry just re-registers it.
    Trade-off: explicit `POST /episodes/end` is precise and immediate, while
    TTL is self-healing for uncooperative or legacy senders but cannot
    distinguish a quiet-but-still-running episode from a finished one; an
    idle gap longer than the TTL looks like completion. Replay note:
    replaying an old trajectory through a live sidecar will resurrect stale
    ids regardless of TTL, so point replay at a metrics-disabled sidecar or
    a throwaway port.
11. Integration matrix document plus a prominent "10-line custom adapter"
    path. **Done:** `docs/INTEGRATION_MATRIX.md` (#49).

## Suggested starting point

P0 items 1 to 3 are what turn this from "a library you wire in" into
"something you attach." They are now implemented (see Progress below). The
next highest-leverage move is P1 item 4 (pluggable state backend) and P1
item 5 (alerting dedup/cooldown + Slack/PagerDuty sinks).

## Sidecar TLS: reverse-proxy termination (issue #103)

The sidecar (`snagline serve`) speaks plain HTTP and listens on
`127.0.0.1:8787` by default. Loopback-only binding is safe on a shared host,
but the moment telemetry crosses a network segment you do not fully trust
(pod to pod, host to host, office to datacenter), the connection needs TLS.
The supported production pattern is TLS termination at a reverse proxy:
public traffic arrives at the proxy over HTTPS, the proxy strips TLS and
forwards plaintext to the sidecar over loopback. Both configs below are
copy-paste ready and introduce zero new Python dependencies. If you cannot
run a proxy on the same host, the sidecar can also terminate TLS itself;
see the threat model below for the trade-off and the invocation.

Assumptions used throughout: the sidecar runs on the same host as the proxy,
on port 8787, with a bearer token set via `$SNAGLINE_SERVE_AUTH_TOKEN` (or
`--auth-token`). Adjust the hostname and cert paths to yours.

### nginx (stable, copy-paste)

```nginx
# /etc/nginx/sites-available/snagline

# Standard WebSocket upgrade mapping (from the nginx proxy module docs).
map $http_upgrade $connection_upgrade {
    default upgrade;
    ''      close;
}

server {
    listen 443 ssl;
    http2 on;                          # nginx >= 1.25.1; older: "listen 443 ssl http2;"
    server_name snagline.example.com;

    ssl_certificate     /etc/letsencrypt/live/snagline.example.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/snagline.example.com/privkey.pem;
    ssl_protocols       TLSv1.2 TLSv1.3;

    # Mirror the sidecar's --max-body-bytes (default 1000000). Unitless means
    # bytes. Change both together: the edge should reject what the sidecar
    # would reject so oversized batches die with 413 before reaching Python.
    client_max_body_size 1000000;

    location / {
        proxy_pass http://127.0.0.1:8787;

        proxy_set_header Host              $host;
        proxy_set_header X-Real-IP         $remote_addr;
        proxy_set_header X-Forwarded-For   $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # Preserve the shared secret. nginx forwards Authorization unchanged
        # by default; pinning it here makes that guarantee explicit and
        # survives later refactors that add other proxy_set_header lines.
        # The alternate X-Snagline-Token header also passes through untouched.
        proxy_set_header Authorization $http_authorization;

        # The sidecar exposes no WebSocket endpoints today; these two lines
        # plus the map above keep the block correct if streaming endpoints
        # are added later. Harmless for plain request/response POSTs.
        proxy_http_version 1.1;
        proxy_set_header Upgrade    $http_upgrade;
        proxy_set_header Connection $connection_upgrade;

        # Let a large batched POST drain slowly instead of being cut off at
        # the 60s default.
        proxy_read_timeout 300s;
    }
}
```

Enable it, then verify from outside the host:

```bash
sudo ln -s /etc/nginx/sites-available/snagline /etc/nginx/sites-enabled/
sudo nginx -t && sudo systemctl reload nginx

curl -fsS https://snagline.example.com/health
curl -fsS -X POST https://snagline.example.com/events \
  -H "Authorization: Bearer $SNAGLINE_SERVE_AUTH_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"step_id":"1","episode_id":"run","timestamp":1735689600.0,"action_type":"tool_call","action_signature":"deadbeefdeadbeef"}'
```

The health probe returns `{"status": "ok"}` and the POST returns 202 with an
`ingested` status. A missing or wrong token gets 401 exactly as it would
against the sidecar directly; termination does not weaken auth because the
proxy never inspects or strips the credential.

### Caddy v2 (copy-paste)

Caddy obtains and renews certificates automatically (ACME against Let's
Encrypt or ZeroSSL), so TLS needs no configuration at all; the whole site
block below is six lines:

```caddy
# /etc/caddy/Caddyfile
snagline.example.com {
	request_body {
		max_size 1000000
	}
	reverse_proxy 127.0.0.1:8787
}
```

- `reverse_proxy` forwards all request headers, `Authorization` included,
  untouched, and proxies WebSockets and streaming responses (SSE) with no
  extra configuration.
- `request_body`'s `max_size` takes a byte count (unit suffixes like `MB`
  also work); it mirrors the sidecar's `--max-body-bytes` default.
- For a host that is not reachable from the public internet, swap automatic
  ACME certificates for Caddy's own internal CA by adding `tls internal`
  inside the site block.

Reload with `caddy reload --config /etc/caddy/Caddyfile`, then run the same
two `curl` checks as under nginx.

### Threat model

TLS termination solves wire confidentiality and integrity on every segment
between the sending agent and the proxy host: passive observers cannot read
event payloads or capture the bearer token, and the certificate proves the
caller reached the intended endpoint. What it does not solve is the final
hop: proxy to sidecar remains plaintext. On one trusted host across
loopback, that residual exposure is usually acceptable. It is not acceptable
when the proxy runs on a different machine than the sidecar, or when the
shared host runs processes you do not control; in those cases terminate TLS
in the sidecar process itself. That is built in as of issue #120: pass
`--certfile`/`--keyfile` and `serve()` wraps its listening socket with a
stdlib `ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)`, closing the last plaintext
hop:

```bash
# Any PEM pair works; to generate a throwaway self-signed one:
#   openssl req -x509 -newkey rsa:2048 -nodes -days 365 \
#     -subj "/CN=snagline.internal" -keyout key.pem -out cert.pem
SNAGLINE_SERVE_AUTH_TOKEN=... snagline serve \
  --host 127.0.0.1 --port 8787 \
  --certfile /etc/snagline/cert.pem \
  --keyfile /etc/snagline/key.pem

# Acceptance check against a self-signed pair (-k). With a real certificate,
# verify the chain instead of skipping it (--cacert) and map the certificate's
# hostname to loopback for local checks (--resolve); a plain
# https://127.0.0.1:8787 would fail hostname validation against any real cert:
curl -k https://127.0.0.1:8787/health
curl --cacert /etc/snagline/ca.pem \
  --resolve snagline.internal:8787:127.0.0.1 \
  https://snagline.internal:8787/health
```

Auth semantics are unchanged over TLS: `Authorization: Bearer` and
`X-Snagline-Token` stay required on everything except `GET /health`, exactly
as on plain HTTP. For mutual TLS pass `--client-ca /path/to/ca.pem` alongside
`--certfile/--keyfile`: the sidecar loads the CA bundle via
`load_verify_locations()` and sets `verify_mode = CERT_REQUIRED`, so every
TLS handshake must present a client certificate chaining to that CA. Handshakes
without one or with an untrusted one fail at the TLS layer before any HTTP is
spoken. Both directions cost zero new dependencies: `ssl` is standard library,
and a reverse proxy adds nothing to the Python environment, so the choice
between them is purely operational.

### Hardening checklist

- **Rotate the bearer token on a schedule.** The token is read once at
  startup, so rotation means a sidecar restart; plan for it. Pass the token
  via `$SNAGLINE_SERVE_AUTH_TOKEN` rather than `--auth-token` so the secret
  stays out of process listings and shell history.
- **Keep the loopback bind.** The default `host="127.0.0.1"` exists so only
  the proxy can reach the sidecar. Do not expose port 8787 off-host; if the
  proxy must run on another machine, either tunnel that segment
  (VPN/WireGuard) or terminate TLS in the sidecar itself with
  `--certfile/--keyfile` (see the threat model above).
- **Keep the body-size caps in sync.** The sidecar rejects requests above
  `--max-body-bytes` (default 1000000) with 413; mirror that number in
  `client_max_body_size` (nginx) or `request_body max_size` (Caddy) so
  oversized payloads are dropped at the edge instead of consuming sidecar
  memory and CPU. The cap exists to bound per-request memory while still
  admitting batched `StepEvent` arrays.
- **Remember nothing sensitive is retained.** Events carry hashes, timings,
  counts, and booleans, never prompt or response content. `GET /risks`
  retains the most recent 1000 `FailureRisk` records (ids, scores, trigger
  names); proxies' access logs record paths and status codes, not bodies.
  Do not add `Authorization` (or any header logging) to your proxy log
  format, and the deployment leaks nothing the library does not already
  refuse to store.
- **`GET /health` is intentionally unauthenticated** so liveness probes work
  without secrets; it reveals reachability only.

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
- **P1 production readiness - alerting maturity + state backend.** Merged:
  - #39 `feat/alerting-dedup-severity` - `FailureRisk.severity` + `DedupSink`
    cooldown (issue #4).
  - #40 `feat/cli-cooldown` - `--cooldown-seconds` on `watch`/`serve`.
  - #41 `feat/slack-sink` - `SlackSink` (stdlib, min_severity filter).
  - #42 `feat/pagerduty-sink` - `PagerDutySink` (Events API v2).
  - #43 `feat/cli-sinks` - `--sink` exposes slack/pagerduty on watch/serve.
  - #44 `feat/state-backend-lock-sharding` - `StateBackend` (memory + optional
    Redis) and per-episode ingest lock sharding (removes the global lock).
  - #45 `feat/baseline-store` - versioned, per-tenant `BaselineStore`.
  - #46 `feat/baseline-cli-store` - `BaselineCollector` + store-backed
    `snagline baseline` CLI.
  - #47 `feat/monitor-metrics` - `Monitor.metrics()` + `GET /metrics`.
  - #48 `feat/batching-sink` - `BatchingSink` async/rate-limited dispatch.
  - #49 `docs/integration-matrix` - `docs/INTEGRATION_MATRIX.md` (P3 item 11).
- **Status: P0, P1, and most of P3 are complete.** Remaining:
  - P3 item 10 tail: Prometheus export + structured logging for the monitor.
  - Actual PyPI upload (needs a token). In-process TLS shipped via
    #120 alongside the reverse-proxy pattern from #110; mutual TLS shipped
    in #145 via `--client-ca`.
  - Then P2 (ml/drift extras, calibration, eval harness) - the research
    differentiators that approach the paper's accuracy.
