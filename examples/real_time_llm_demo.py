"""Real-time SNAGLINE test with a REAL LLM.

Runs a genuine LangChain 1.x ``create_agent`` agent backed by a real chat model
(OpenAI / Anthropic / OpenRouter-compatible) with REAL tools, and attaches the
SnaglineCallbackHandler so detections print to stderr as the agent runs. This is
the live, uncontrolled test: you watch a real model drive real tools and see
SNAGLINE flag failures in real time.

Modes:
  healthy  -> benign tool calls only (false-positive check: should be SILENT)
  error    -> a tool that raises on every call (triggers error_cascade)
  latency  -> a single tool with variable latency (triggers latency_anomaly)

Requirements:
  * pip install langchain langchain-openai snagline-agent[langchain]
    (or langchain-anthropic)
  * For OpenAI/Anthropic: export OPENAI_API_KEY / ANTHROPIC_API_KEY
  * For OpenRouter (OpenAI-compatible, free models): 
        export OPENAI_API_KEY=sk-or-...
        run with --base-url https://openrouter.ai/api/v1 --model <free-slug>

Run:
    export OPENAI_API_KEY=sk-or-...
    PYTHONPATH=src /tmp/lc-venv/bin/python examples/real_time_llm_demo.py \\
        --provider openai --base-url https://openrouter.ai/api/v1 \\
        --model openai/gpt-oss-20b:free --mode latency
"""

from __future__ import annotations

import argparse
import os
import time
import urllib.request

from snagline import Monitor
from snagline.adapters.langchain_adapter import SnaglineCallbackHandler
from snagline.config import Config
from snagline.detectors.error_cascade import ErrorCascadeDetector
from snagline.detectors.latency_anomaly import LatencyAnomalyDetector
from snagline.detectors.loop import LoopDetector
from snagline.sinks.console import ConsoleSink


# --------------------------------------------------------------------------
# Real tools
# --------------------------------------------------------------------------
def _get_weather(city: str) -> str:
    return f"Weather in {city}: 22C, sunny."


def _flaky_api(url: str) -> str:
    # A real network call to a non-resolving host -> raises on every attempt.
    with urllib.request.urlopen(url, timeout=5) as resp:
        return resp.read()[:200].decode()


def _fetch_sensor(n: int) -> str:
    # Variable latency on a SINGLE tool: lets CUSUM detect a latency shift.
    time.sleep((n % 10) * 0.3)
    return f"sensor({n}) reading = {n * 7}"


TOOLS = [
    {"name": "get_weather", "func": _get_weather, "desc": "Get the weather for a city."},
    {"name": "flaky_api", "func": _flaky_api, "desc": "Fetch a URL and return the first 200 bytes."},
    {"name": "fetch_sensor", "func": _fetch_sensor, "desc": "Read a sensor by integer id (1-100); returns a reading."},
]


def _build_tools():
    from langchain_core.tools import Tool

    return [Tool(name=t["name"], func=t["func"], description=t["desc"]) for t in TOOLS]


def _build_model(provider: str, model: str, base_url: str | None):
    if provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model=model, temperature=0, base_url=base_url)
    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(model=model, temperature=0)
    raise SystemExit(f"unknown provider: {provider}")


def _pick_provider(cli: str | None) -> str:
    if cli:
        return cli
    if os.environ.get("OPENAI_API_KEY"):
        return "openai"
    if os.environ.get("ANTHROPIC_API_KEY"):
        return "anthropic"
    raise SystemExit("No provider. Set OPENAI_API_KEY or ANTHROPIC_API_KEY, or pass --provider.")


MODE_TASKS = {
    "healthy": "Call get_weather for 'Paris', then 'Tokyo', then 'New York', then summarize the results.",
    "error": "Call flaky_api with 'https://this-host-does-not-exist.invalid' three times and report each result.",
    "latency": "Call fetch_sensor with 1, then 4, then 2, then 9, then 7, then 3, then summarize the readings.",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--provider", choices=["openai", "anthropic"], default=None)
    parser.add_argument("--model", default=None, help="Model name (e.g. openai/gpt-oss-20b:free for OpenRouter).")
    parser.add_argument("--base-url", default=None, help="API base URL (e.g. https://openrouter.ai/api/v1).")
    parser.add_argument("--mode", choices=list(MODE_TASKS), default="healthy")
    args = parser.parse_args()

    provider = _pick_provider(args.provider)
    model = args.model or ("gpt-4o-mini" if provider == "openai" else "claude-3-5-haiku-latest")
    task = MODE_TASKS[args.mode]
    print(f"[demo] provider={provider} model={model} mode={args.mode}; attaching SNAGLINE...\n")

    cfg = Config(cusum_min_samples=3)
    detectors = [LoopDetector(config=cfg), ErrorCascadeDetector(config=cfg), LatencyAnomalyDetector(config=cfg)]
    monitor = Monitor(detectors, [ConsoleSink()])
    handler = SnaglineCallbackHandler(monitor, "real-llm")

    model_obj = _build_model(provider, model, args.base_url)
    agent = __import__("langchain.agents", fromlist=["create_agent"]).create_agent(model=model_obj, tools=_build_tools())

    print(f"[demo] task: {task}\n[demo] risks stream to stderr as they occur.\n")
    for chunk in agent.stream(
        {"messages": [("user", task)]},
        config={"callbacks": [handler], "recursion_limit": 40},
    ):
        if "messages" in chunk and chunk["messages"]:
            msg = chunk["messages"][-1]
            content = getattr(msg, "content", None)
            if content and getattr(msg, "type", "") in ("ai", "human", "tool"):
                print(f"[{msg.type}] {content if isinstance(content, str) else content}")

    handler.close()
    print("\n[demo] run complete. Risk lines (if any) above show what SNAGLINE caught live.")


if __name__ == "__main__":
    main()
