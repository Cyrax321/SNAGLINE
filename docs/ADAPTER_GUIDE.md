# Writing a SNAGLINE adapter

An adapter's only job: translate your framework's events into `StepEvent`s
and call `monitor.ingest(event)`. No detection logic, ever - that all lives in
core behind the canonical schema.

## The 10-line pattern

```python
import itertools, time
from snagline.events import StepEvent, make_signature

class MyAdapter:
    def __init__(self, monitor, episode_id):
        self._monitor, self._episode_id = monitor, episode_id
        self._counter = itertools.count()

    def on_something(self, tool_name, args, latency_ms=None, error=False):
        event = StepEvent(
            step_id=str(next(self._counter)),
            episode_id=self._episode_id,
            timestamp=time.time(),
            action_type="tool_call",
            action_signature=make_signature("tool_call", tool_name, str(args)),
            tool_name=tool_name,
            latency_ms=latency_ms,
            error=error,
        )
        self._monitor.ingest(event)
```

When the run finishes, call `monitor.end_episode(episode_id)` so per-episode
detector state (loop windows, CUSUM baselines) doesn't leak into the next run.
If your framework is a context manager or generator, put that teardown in the
`finally` - see `adapters/raw.py` and `adapters/langchain_adapter.py:close`.

## Signature rules (the part people get wrong)

`make_signature(action_type, tool_name, *stable_parts)` is what loop detection
sees. Two steps with the same signature count as "the same attempt."

- **Include** the logical action: tool name, target element, endpoint, the
  shape of the arguments.
- **Exclude** anything volatile: timestamps, request/session ids, nonces,
  retry counters. Including them makes every retry look unique and silently
  defeats loop detection. (Excluding them entirely is equally wrong in the
  other direction - a constant `args=""` makes *every* LLM call look like a
  loop; hash the real prompt text like `langchain_adapter.on_llm_start` does.)

## Privacy rules

Never put raw prompt/response content into `metadata` unless you have a reason
and a reviewed sink; detectors never read it, but a custom sink could forward
it. Everything detection needs fits in the hash + timings + booleans.
(The one documented exception is the compaction tripwire below, which reads
exactly two metadata keys.)

## Side-effect marking for callback adapters (issue #150)

The `SideEffectGuardDetector` only fires when `StepEvent.side_effect=True`,
and only host-declared knowledge may set it (#88 rule: never invent or
guess, never read `metadata["side_effect"]`). Callback adapters now
expose a constructor allowlist so hosts can declare which tools are
non-idempotent without touching payloads:

```python
from snagline.adapters.langchain_adapter import SnaglineCallbackHandler
from snagline.adapters.autogen import SnaglineAutogenHandler
from snagline.adapters.crewai import snagline_step_callback

handler = SnaglineCallbackHandler(monitor, "ep", side_effect_tools={"charge_card", "send_email"})
autogen_handler = SnaglineAutogenHandler(monitor, "ep", side_effect_tools={"charge_card"})
crewai_cb = snagline_step_callback(monitor, "ep", side_effect_tools={"charge_card"})
# Autogen also supports the same via run_and_monitor(..., side_effect_tools={...})
```

The adapter matches emitted `tool_call` steps by their mapped `tool_name`;
allowlisted names become `side_effect=True`, everything else stays `False`
with the default `_emit(side_effect=False)`. Nothing is inferred from
payloads, args, or content.

Pure payload adapters (`watch_graph` in `langgraph_adapter.py` and
`payload_to_event` / `ingest_payload` in `claude_code.py`) derive everything
from framework payloads and have no flag source in those payloads, so they
keep the schema default `False`. If you need side-effect guarding for those
runtimes, route the marked action through the raw adapter or an
`observe_*` helper that takes an explicit `side_effect=True`. This
limitation is intentional and documented.

## Optional: compaction tripwire events (issue #90)

If your harness exposes compaction hooks (LangGraph pre-compaction callbacks,
Claude Code auto-compact hooks, your own summarization step), you can opt into
`CompactionTripwireDetector` by emitting two extra action types:

```python
import hashlib

# when the harness compacts context: pin the constraints that MUST survive
step("compaction", tool_name=None, metadata={"pinned": [
    hashlib.sha256(c.encode()).hexdigest() for c in constraints
]})

# whenever introspection shows a constraint still present in live context
step("constraint_present", tool_name=None, metadata={"pin":
    hashlib.sha256(constraint.encode()).hexdigest()
})
```

Rules:

- **Hash the constraint text yourself** (SHA-256). Constraint text never
  reaches snagline; send only hex digests. Emitted risks carry 16-hex
  prefixes only.
- **Metadata exception:** `pinned` and `pin` are the one documented pair of
  metadata keys any detector reads. Nothing else on the event is touched.
- **Re-confirm within the grace window:** every pinned hash needs a
  `constraint_present` event within `compaction_tripwire_grace_steps`
  (default 3) events of the compaction, or exactly one `score=0.9`
  `governance_decay` risk fires.

Be honest about coverage: if your host offers no compaction hook, or gives no
way to observe whether a constraint survived, do not emit these events. The
detector then stays permanently silent (inert by design); it never pretends
to protect constraints it cannot see. Enable it with
`Config(compaction_tripwire_enabled=True)`; default off everywhere.

## Reference adapters

| Adapter | Host | Pattern |
|---|---|---|
| `adapters/raw.py` | plain Python loop | context manager yielding a `step()` callable |
| `adapters/langchain_adapter.py` | LangChain / LangGraph `create_agent` | `BaseCallbackHandler` subclass |
| `adapters/langgraph_adapter.py` | LangGraph `graph.stream()` | pass-through iterator wrapper |
| `adapters/autogen.py` | Autogen `agent.run_stream(task)` | `SnaglineAutogenHandler` observer + `run_and_monitor` wrapper |
| `adapters/crewai.py` | CrewAI `Agent(step_callback=...)` | `snagline_step_callback` returning the hook callable |
| `adapters/continuum_adapter.py` | CONTINUUM `Storage` event log | `ContinuumAdapter` poll/push translating ledger entries |
| `server/http_server.py` | any non-Python runtime | stdlib HTTP sidecar, `POST /events` |

Both the Autogen and CrewAI adapters are duck-typed: they read event objects via
`model_dump()` with attribute/dict fallbacks, so they import without the
framework installed and never hard-couple to a specific release. The CONTINUUM
adapter (`pip install snagline-agent[continuum]`, issue #79) is duck-typed the
same way: it calls only the verified public read pair
(`read_events(run_id, *, after_sequence=0, upto=None)` + `last_sequence(run_id)`)
and never imports `continuum`. It maps `PERCEPTION_OBSERVED` to an observation,
`BRANCH_RESOLVED` to a plan step, and the action-ledger lifecycle
(`ACTION_RECORDED` started/completed/failed, plus reconcile/compensate) to tool
calls with paired latency. `raw_claim` text is dropped at translation time;
only hashes, ids, statuses and counts cross over.

## CONTINUUM sink (closing the loop)

`sinks/continuum_sink.py` escalates a detected risk back into CONTINUUM via its
existing request-human path: `ActionLedger.flag_for_review(key, reason)`, which
sets the action's status to `REQUIRES_REVIEW` and appends an auditable ledger
event (CONTINUUM's recovery planner then routes it to a person).

```python
from snagline.sinks.continuum_sink import ContinuumSink

sink = ContinuumSink(
    storage,                      # a CONTINUUM Storage handle
    run_id,
    key_from_risk=lambda risk: my_action_keys.get(risk.step_id),
)
monitor.add_sink(sink)
```

Honest limits, verified against current CONTINUUM source:

- Run-level `request_human` is a *derived* recovery mode there; the only
  writable escalation API is the action-level `flag_for_review`. A risk with no
  resolvable action key is therefore logged and dropped, not fabricated onto
  the ledger.
- `key_from_risk` must return the resolved idempotency key or `action_id`; a
  bare action name is refused by the real ledger with `LedgerError`.
- Exercised against real CONTINUUM code only in the skip-guarded test
  `tests/sinks/test_continuum_sink.py::test_real_actionledger_flags_review`
  (claim -> flag_for_review -> REQUIRES_REVIEW); no live CONTINUUM instance was
  used for end-to-end polling.

## Testing your adapter

Don't require a live framework in CI. Either call your hooks directly with
mocked framework objects (see `tests/adapters/test_langchain_adapter.py`) or,
for stream-shaped APIs, feed hand-built items with the same shape the real
framework yields (see `tests/adapters/test_langgraph_adapter.py`). Assert
two things: events reach the Monitor with the right fields, and a known-bad
sequence (repeated identical action, consecutive errors) actually trips a
detector through your adapter.
