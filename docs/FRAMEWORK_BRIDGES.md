# Connecting SNAGLINE to external agent frameworks

Claude Code, OpenClaw, and Hermes are not Python libraries you can import:
they are separate processes. So the bridge is process-level, not an in-process
adapter. Every agent framework can do at least one of three things: run a
shell command, make an HTTP call, or write to a file. SNAGLINE supports all
three, so any framework (including ones that do not exist yet) can plug in
with no glue code beyond configuration.

| Bridge | Mechanism | Best for |
|---|---|---|
| HTTP | `POST /events` (canonical StepEvent) or `POST /hooks/claude-code` (native Claude Code payload) to the sidecar | frameworks with HTTP hooks (Claude Code `http` hooks, OpenClaw automations) |
| Command | `snagline hook` reads one hook payload from stdin, maps it, forwards it | frameworks with command hooks (Claude Code `command` hooks, OpenClaw hooks, anything shell-capable) |
| File | framework appends StepEvent JSON lines; `snagline watch --file --follow` tails it | frameworks that can only write logs |

Start the receiver once (it is stdlib-only, no dependencies):

```
snagline serve --host 127.0.0.1 --port 8787
```

## Claude Code (fully native, zero glue)

Claude Code hooks fire on `PreToolUse`, `PostToolUse`, `PostToolUseFailure`,
and many lifecycle events, delivering a JSON payload on stdin. SNAGLINE maps
that payload itself, so you only edit settings.

**Option A: native `http` hooks (no scripts at all).** Add to
`.claude/settings.json` (or `~/.claude/settings.json`):

```json
{
  "hooks": {
    "PreToolUse": [
      {
        "hooks": [
          { "type": "http", "url": "http://127.0.0.1:8787/hooks/claude-code" }
        ]
      }
    ],
    "PostToolUse": [
      {
        "hooks": [
          { "type": "http", "url": "http://127.0.0.1:8787/hooks/claude-code" }
        ]
      }
    ],
    "PostToolUseFailure": [
      {
        "hooks": [
          { "type": "http", "url": "http://127.0.0.1:8787/hooks/claude-code" }
        ]
      }
    ]
  }
}
```

The sidecar pairs Pre/Post by `tool_use_id` to compute `latency_ms` and maps
failures to `error=True`. Unmapped lifecycle events are acknowledged and
ignored.

**Option B: `command` hooks (works even without HTTP reachability).**
`snagline hook` reads the same stdin payload and forwards it as a canonical
StepEvent:

```json
{
  "hooks": {
    "PostToolUse": [
      {
        "hooks": [
          {
            "type": "command",
            "command": "snagline hook --url http://127.0.0.1:8787/events"
          }
        ]
      }
    ]
  }
}
```

`snagline hook` ALWAYS exits 0 and never blocks on failure, so it can never
break or stall Claude Code. To batch instead of streaming, append to a file
and analyze later: `snagline hook --out /tmp/snagline-events.jsonl`, then
`snagline replay /tmp/snagline-events.jsonl --summary`.

**Verification, end to end:**

```
# terminal 1: receiver
snagline serve --port 8787
# terminal 2: simulate exactly what Claude Code sends
curl -s -X POST http://127.0.0.1:8787/hooks/claude-code \
  -d '{"session_id":"s1","hook_event_name":"PreToolUse","tool_name":"Bash","tool_input":{"command":"npm test"},"tool_use_id":"t1"}'
# repeat the same curl 3 more times (same tool_input) -> a loop risk
# prints in terminal 1
```

## OpenClaw

OpenClaw hooks are shell scripts discovered from hook directories (managed
with `openclaw hooks`) and fire on session lifecycle and message-flow events;
in-process plugin hooks can additionally intercept every tool call (see the
OpenClaw plugin SDK). The command bridge works for any hook that runs a
script with the event available as JSON:

```bash
#!/usr/bin/env bash
# .openclaw/hooks/on-agent-event (example name; see `openclaw hooks` docs
# for the real directory and invocation contract)
cat | snagline hook --url http://127.0.0.1:8787/events
```

If your OpenClaw build passes the event JSON as `$1` instead of stdin:

```bash
#!/usr/bin/env bash
printf '%s' "$1" | snagline hook --url http://127.0.0.1:8787/events
```

Note: `snagline hook` expects either a Claude Code hook payload (detected via
`hook_event_name`) or an already-canonical StepEvent object. For OpenClaw's
native event schema, wrap it in a one-line `jq` mapping into StepEvent fields
(documented recipe below), or write a small OpenClaw plugin hook that POSTs a
canonical StepEvent directly:

```bash
curl -s -X POST http://127.0.0.1:8787/events -H 'Content-Type: application/json' \
  -d '{"step_id":"'$RANDOM'","episode_id":"openclaw-session","timestamp",'\
'"action_type":"tool_call","action_signature":"<hash-of-tool+args>",'\
'"tool_name":"<tool>","latency_ms":<ms>,"error":false}'
```

Only five fields are required (`step_id`, `episode_id`, `timestamp`,
`action_type`, `action_signature`); everything else is optional. Compute the
signature with any stable hash over (tool name + arguments), excluding
timestamps and ids.

## Hermes (or any framework with shell access)

Hermes Agent exposes plugin hooks around its execution loop. The same two
recipes apply: either have the hook POST a canonical StepEvent to
`/events` (the curl line above), or pipe any JSON through the bridge. The
generic recipe, using `jq` to normalize any event into a StepEvent:

```bash
some-hook-output.json | jq -c '
  {
    step_id: (.id // input_filename),
    episode_id: (.session // "hermes"),
    timestamp: (now),
    action_type: "tool_call",
    action_signature: (.tool + (.args // "") | @base64),
    tool_name: .tool,
    error: (.error != null)
  }' | snagline hook --url http://127.0.0.1:8787/events
```

`action_signature` just needs to be a stable string: identical logical
attempts must produce identical signatures, or loop detection cannot see
repetition. Any deterministic function of (tool, arguments) works.

Important: SHA-256 hashing of the action signature is a property of the
**Python adapters** (`raw.watch`, `SnaglineCallbackHandler`,
`watch_graph`, the Claude Code hook bridge). When an external process sends
events directly over the bridge (via `POST /events`, `snagline hook`, or
`snagline watch`), the `action_signature` is whatever that process puts in the
JSON. SNAGLINE accepts it as-is and does no hashing on the HTTP/CLI side. So
if you bridge a non-Python agent, hash the signature yourself (for example
with a one-way digest of the tool plus arguments) before sending, or loops
will not be detected correctly.

## File bridge (frameworks that can only write logs)

If the framework can append JSON lines but cannot run commands or POST:

```
tail -f /var/log/agent/events.jsonl | snagline watch
# or natively:
snagline watch --file /var/log/agent/events.jsonl --follow
```

Each line must be a canonical StepEvent JSON object (see
`docs/ADAPTER_GUIDE.md` for the five required fields).

## What detection sees through the bridge

- Loops: same tool + same arguments repeated (a Claude Code retry storm
  shows up as identical `tool_input` hashes).
- Error cascades: `PostToolUseFailure` events map to `error=True`.
- Latency anomalies: only when Pre/Post pairs are both forwarded (the
  sidecar pairs them by `tool_use_id`); the file/command bridge without
  pairing still detects loops and cascades.
