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

## Reference adapters

| Adapter | Host | Pattern |
|---|---|---|
| `adapters/raw.py` | plain Python loop | context manager yielding a `step()` callable |
| `adapters/langchain_adapter.py` | LangChain / LangGraph `create_agent` | `BaseCallbackHandler` subclass |
| `adapters/langgraph_adapter.py` | LangGraph `graph.stream()` | pass-through iterator wrapper |
| `server/http_server.py` | any non-Python runtime | stdlib HTTP sidecar, `POST /events` |

Adapters for AutoGen, CrewAI, raw OpenAI/Anthropic SDKs, and CONTINUUM are
planned (project.md §13 step 10 / §6) - the raw adapter is usually a fine
stand-in for any of them, since they all bottom out in a tool-calling loop.

## Testing your adapter

Don't require a live framework in CI. Either call your hooks directly with
mocked framework objects (see `tests/adapters/test_langchain_adapter.py`) or,
for stream-shaped APIs, feed hand-built items with the same shape the real
framework yields (see `tests/adapters/test_langgraph_adapter.py`). Assert
two things: events reach the Monitor with the right fields, and a known-bad
sequence (repeated identical action, consecutive errors) actually trips a
detector through your adapter.
