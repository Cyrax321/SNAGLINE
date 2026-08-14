"""Real-framework SNAGLINE test harness using a genuine LangChain 1.x agent.

LangChain 1.x replaced ``AgentExecutor``/``initialize_agent`` with
``langchain.agents.create_agent``, which builds a LangGraph agent (a
``CompiledStateGraph``). This example runs a REAL agent created by
``create_agent`` with the ``SnaglineCallbackHandler`` attached, so the adapter
sees genuine LangChain/LangGraph callbacks (on_chat_model_start / on_llm_end /
on_tool_start / on_tool_end / on_tool_error) propagated through the actual
framework, not a hand-rolled loop.

Because LangChain 1.x agents call ``model.bind_tools``, this example uses a tiny
``FakeMessagesListChatModel`` subclass that implements ``bind_tools`` (the
scripted responses already carry ``tool_calls``). No API key is required.

Modes (which detector each one is meant to trip):

    healthy   -> expect SILENCE (false-positive check)
    loop      -> loop detector (agent repeats the same tool call)
    error     -> error_cascade (tool always raises)
    latency   -> latency_anomaly (a tool's latency shifts from fast to slow)

Requires: pip install langchain  (and snagline-agent[langchain])

Run:
    PYTHONPATH=src python3 examples/real_agent_executor_demo.py --mode loop
    PYTHONPATH=src python3 examples/real_agent_executor_demo.py --mode error
    PYTHONPATH=src python3 examples/real_agent_executor_demo.py --mode latency
    PYTHONPATH=src python3 examples/real_agent_executor_demo.py --mode healthy
"""

from __future__ import annotations

import argparse
import sys
import time

from snagline import Monitor
from snagline.adapters.langchain_adapter import SnaglineCallbackHandler
from snagline.config import Config
from snagline.detectors.error_cascade import ErrorCascadeDetector
from snagline.detectors.latency_anomaly import LatencyAnomalyDetector
from snagline.detectors.loop import LoopDetector
from snagline.sinks.console import ConsoleSink


# --------------------------------------------------------------------------
# Optional LangChain dependency
# --------------------------------------------------------------------------
try:
    from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
    from langchain_core.messages import AIMessage
    from langchain_core.tools import Tool
    from langchain.agents import create_agent

    LANGCHAIN_AGENT_AVAILABLE = True
except Exception:  # pragma: no cover - exercised only without langchain
    LANGCHAIN_AGENT_AVAILABLE = False


class _FakeToolCallingModel(FakeMessagesListChatModel):
    """FakeMessagesListChatModel lacks bind_tools, which create_agent calls."""

    def bind_tools(self, tools, **kwargs):
        return self


# --------------------------------------------------------------------------
# Tools (real langchain_core Tool objects, arg named to match tool_calls)
# --------------------------------------------------------------------------
def _search(arg: str) -> str:
    return f"search results for: {arg}"


def _boom(arg: str) -> str:
    raise RuntimeError("tool failed on purpose (chaos)")


def _slow(arg: str) -> str:
    _slow.state["n"] += 1
    if _slow.state["n"] <= 5:
        time.sleep(0.01)  # fast baseline
    else:
        time.sleep(0.5)  # sustained latency shift
    return "ok"


_slow.state = {"n": 0}


TOOLS = [
    Tool(name="search", func=_search, description="Search the web for a query."),
    Tool(name="boom", func=_boom, description="A tool that always fails."),
    Tool(name="slow", func=_slow, description="A tool whose latency shifts under load."),
]


def _tool_call(name: str, arg: str) -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[{"name": name, "args": {"arg": arg}, "id": f"call-{name}-{arg}"}],
    )


def _build_fake_model(mode: str) -> "_FakeToolCallingModel":
    if mode == "loop":
        responses = [_tool_call("search", "repeat") for _ in range(4)]
    elif mode == "error":
        responses = [_tool_call("boom", str(i)) for i in range(4)]
    elif mode == "latency":
        _slow.state["n"] = 0
        responses = [_tool_call("slow", str(i)) for i in range(12)]
    else:  # healthy: one benign tool call, then a final answer
        responses = [_tool_call("search", "weather"), AIMessage(content="The weather is sunny.")]
    responses.append(AIMessage(content="[agent finished]"))
    return _FakeToolCallingModel(responses=responses)


def _expected(mode: str) -> str:
    return {
        "healthy": "SILENCE (no risks expected)",
        "loop": "loop detector",
        "error": "error_cascade detector",
        "latency": "latency_anomaly detector",
    }[mode]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=["healthy", "loop", "error", "latency"], default="loop")
    args = parser.parse_args()

    if not LANGCHAIN_AGENT_AVAILABLE:
        print(
            "This example needs langchain (>=1.x). Install it with:\n"
            "    pip install langchain snagline-agent[langchain]",
            file=sys.stderr,
        )
        return

    # Small warm-up so the latency demo can shift after only a few calls.
    cfg = Config(cusum_min_samples=5)

    # Each mode monitors only its target detector so the output is unambiguous
    # (the adapter also emits latency for LLM "chat" calls, which can be noisy
    # on a fake model; see issue #10). The healthy run monitors everything and
    # must stay silent, which doubles as the false-positive check.
    detectors = {
        "loop": [LoopDetector(config=cfg)],
        "error": [ErrorCascadeDetector(config=cfg)],
        "latency": [LatencyAnomalyDetector(config=cfg)],
        "healthy": [
            LoopDetector(config=cfg),
            ErrorCascadeDetector(config=cfg),
            LatencyAnomalyDetector(config=cfg),
        ],
    }
    monitor = Monitor(detectors[args.mode], [ConsoleSink()])
    handler = SnaglineCallbackHandler(monitor, f"exec-{args.mode}")

    model = _build_fake_model(args.mode)
    agent = create_agent(model=model, tools=TOOLS)

    print(f"[demo] mode={args.mode!r} -> expected: {_expected(args.mode)}")
    print("[demo] running a REAL create_agent (LangGraph) agent with the handler attached...")
    print("[demo] risks print as JSON to stderr below...\n")

    agent.invoke(
        {"messages": [("user", "Run the assigned task.")]},
        config={"callbacks": [handler], "recursion_limit": 40},
    )
    handler.close()

    print("\n[demo] done. If a risk line above matches the expected detector, it works.")


if __name__ == "__main__":
    main()
