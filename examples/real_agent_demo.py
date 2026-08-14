"""Real-framework SNAGLINE test harness for LangChain (chaos injection).

This drives a genuine LangChain agent loop (real ``langchain_core`` ``Tool``
objects and a real chat model) with the ``SnaglineCallbackHandler`` attached, so
the adapter sees REAL LangChain callbacks (on_chat_model_start / on_llm_end /
on_tool_start / on_tool_end / on_tool_error). It is the practical way to prove
each detector actually fires on a real framework path.

Modes (which detector each one is meant to trip):

    healthy   -> expect SILENCE (false-positive check)
    loop      -> loop detector (agent repeats the same tool call)
    error     -> error_cascade (tool always raises)
    latency   -> latency_anomaly (a tool's latency shifts from fast to slow)

Model selection:
    * If OPENAI_API_KEY / ANTHROPIC_API_KEY is set AND the matching package is
      installed, a real model is used for the `healthy` run (real false-positive
      check). Installing those packages is optional.
    * Otherwise a FakeMessagesListChatModel is scripted to produce the exact
      tool-call sequence for the chosen chaos mode, so the run is fully
      deterministic and needs no API key.

Requires: pip install snagline-agent[langchain]

Run:
    PYTHONPATH=src python3 examples/real_agent_demo.py --mode healthy
    PYTHONPATH=src python3 examples/real_agent_demo.py --mode loop
    PYTHONPATH=src python3 examples/real_agent_demo.py --mode error
    PYTHONPATH=src python3 examples/real_agent_demo.py --mode latency
"""

from __future__ import annotations

import argparse
import sys
import time

from snagline import Monitor
from snagline.adapters.langchain_adapter import SnaglineCallbackHandler
from snagline.config import Config


# --------------------------------------------------------------------------
# Optional LangChain dependency
# --------------------------------------------------------------------------
try:
    from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
    from langchain_core.messages import AIMessage, HumanMessage
    from langchain_core.tools import Tool

    LANGCHAIN_AVAILABLE = True
except Exception:  # pragma: no cover - exercised only without LangChain
    LANGCHAIN_AVAILABLE = False


# --------------------------------------------------------------------------
# Tools (real langchain_core Tool objects)
# --------------------------------------------------------------------------
def _search(q: str) -> str:
    return f"search results for: {q}"


def _boom(_: str) -> str:
    raise RuntimeError("tool failed on purpose (chaos)")


def _slow(_: str) -> str:
    # Latency flips from fast baseline to slow after the first few calls so the
    # CUSUM baseline is learned, then a sustained shift occurs.
    _slow.state["n"] += 1
    if _slow.state["n"] <= 5:
        time.sleep(0.01)
    else:
        time.sleep(0.5)
    return "ok"


_slow.state = {"n": 0}


TOOLS = [
    Tool(name="search", func=_search, description="Search the web for a query."),
    Tool(name="boom", func=_boom, description="A tool that always fails."),
    Tool(name="slow", func=_slow, description="A tool whose latency shifts under load."),
]
TOOLS_BY_NAME = {t.name: t for t in TOOLS}


def _tool_call(name: str, arg: str = "x") -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[{"name": name, "args": {"input": arg}, "id": f"call-{name}"}],
    )


def _build_fake_model(mode: str) -> "FakeMessagesListChatModel":
    if mode == "loop":
        # Constant args on purpose -> identical signatures -> loop detector.
        responses = [_tool_call("search", "repeat") for _ in range(6)]
    elif mode == "error":
        # Vary args so only error_cascade fires (no spurious loop).
        responses = [_tool_call("boom", str(i)) for i in range(4)]
    elif mode == "latency":
        # Vary args so only latency_anomaly fires (no spurious loop).
        _slow.state["n"] = 0
        responses = [_tool_call("slow", str(i)) for i in range(12)]
    else:  # healthy: one benign tool call, then a final answer
        responses = [_tool_call("search", "weather"), AIMessage(content="The weather is sunny.")]
    responses.append(AIMessage(content="[agent finished]"))
    return FakeMessagesListChatModel(responses=responses)


def _build_real_model() -> object | None:
    if "OPENAI_API_KEY" in __import__("os").environ:
        try:
            from langchain_openai import ChatOpenAI

            return ChatOpenAI(model="gpt-4o-mini", temperature=0)
        except Exception:
            pass
    if "ANTHROPIC_API_KEY" in __import__("os").environ:
        try:
            from langchain_anthropic import ChatAnthropic

            return ChatAnthropic(model="claude-3-5-haiku-latest", temperature=0)
        except Exception:
            pass
    return None


# --------------------------------------------------------------------------
# Minimal agent loop (real LangChain callbacks fire as the model / tools run)
# --------------------------------------------------------------------------
def run_agent(model, handler, max_steps: int = 25) -> None:
    messages = [HumanMessage(content="Run the assigned task.")]
    for _ in range(max_steps):
        ai = model.invoke(messages, config={"callbacks": [handler]})
        if not getattr(ai, "tool_calls", None):
            print(f"[agent] final answer: {ai.content!r}")
            break
        for tc in ai.tool_calls:
            tool = TOOLS_BY_NAME[tc["name"]]
            try:
                result = tool.invoke(tc["args"]["input"], config={"callbacks": [handler]})
                messages.append(HumanMessage(content=str(result)))
            except Exception as exc:  # on_tool_error already fired via callback
                messages.append(HumanMessage(content=f"tool error: {exc}"))
    else:
        print("[agent] reached max steps without finishing")


def _expected(mode: str) -> str:
    return {
        "healthy": "SILENCE (no risks expected)",
        "loop": "loop detector",
        "error": "error_cascade detector",
        "latency": "latency_anomaly detector",
    }[mode]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--mode",
        choices=["healthy", "loop", "error", "latency"],
        default="healthy",
        help="Which scenario to run.",
    )
    args = parser.parse_args()

    if not LANGCHAIN_AVAILABLE:
        print(
            "This example needs langchain-core. Install it with:\n"
            "    pip install snagline-agent[langchain]",
            file=sys.stderr,
        )
        return

    # Use a small warm-up so the latency demo can shift after only a few calls.
    cfg = Config(cusum_min_samples=5)
    monitor = Monitor.default(config=cfg)
    handler = SnaglineCallbackHandler(monitor, f"demo-{args.mode}")

    if args.mode == "healthy" and (real := _build_real_model()) is not None:
        print("[demo] using REAL chat model for the healthy false-positive check")
        model = real
    else:
        if args.mode == "healthy":
            print("[demo] no API key / model package; using a fake model for the healthy run")
        model = _build_fake_model(args.mode)

    print(f"[demo] mode={args.mode!r} -> expected: {_expected(args.mode)}")
    print("[demo] watching agent run (risks print as JSON to stderr below)...\n")

    run_agent(model, handler)
    handler.close()

    print("\n[demo] done. If a risk line above matches the expected detector, it works.")


if __name__ == "__main__":
    main()
