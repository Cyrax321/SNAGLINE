"""Real-time SNAGLINE: real LLM -> real detectors -> WebhookSink -> HTTP sidecar.

End-to-end proof of the NEW webhook sink + HTTP sidecar with REAL detection
(nothing faked):

  1. A genuine LangChain 1.x ``create_agent`` agent runs with a REAL chat model
     and REAL tools.
  2. The SnaglineCallbackHandler feeds real StepEvents into a local Monitor with
     the real detectors.
  3. When the real agent drives the ``flaky_api`` tool (which raises on every
     call), the real ErrorCascadeDetector fires a REAL FailureRisk.
  4. A WebhookSink POSTs that real FailureRisk over HTTP to the local sidecar's
     new ``POST /risks`` endpoint.
  5. The sidecar receives and displays the risk in real time.

So you watch: the agent fails for real -> SNAGLINE detects it for real ->
the alert is delivered over HTTP to the sidecar for real.

Setup (two processes, real HTTP between them):
    # terminal 1: the sidecar (receives risks at POST /risks, detects at POST /events)
    snagline serve --port 8787
    # (or without installing the console script:
    #  PYTHONPATH=src python -c "from snagline import Monitor; \
    #    from snagline.server.http_server import serve; \
    #    serve(Monitor.default(), host='127.0.0.1', port=8787)")
    # terminal 2: this demo
    export OPENAI_API_KEY=sk-or-...
    PYTHONPATH=src python examples/real_time_webhook_demo.py \\
        --provider openai --base-url https://openrouter.ai/api/v1 \\
        --model openai/gpt-oss-20b:free --mode error
"""

from __future__ import annotations

import argparse
import os
import signal
import time
import urllib.request


class _AttemptTimeout(Exception):
    """Raised by SIGALRM when a single agent.stream call runs too long."""


def _on_alarm(signum, frame):  # noqa: ANN001, ARG001 - signal handler signature
    raise _AttemptTimeout()

from snagline import Monitor
from snagline.adapters.langchain_adapter import SnaglineCallbackHandler
from snagline.config import Config
from snagline.detectors.error_cascade import ErrorCascadeDetector
from snagline.detectors.latency_anomaly import LatencyAnomalyDetector
from snagline.detectors.loop import LoopDetector
from snagline.sinks.console import ConsoleSink
from snagline.sinks.webhook import WebhookSink


def _get_weather(city: str) -> str:
    return f"Weather in {city}: 22C, sunny."


def _flaky_api(url: str) -> str:
    # A real network call to a non-resolving host -> raises on every attempt.
    with urllib.request.urlopen(url, timeout=5) as resp:
        return resp.read()[:200].decode()


TOOLS = [
    {"name": "get_weather", "func": _get_weather, "desc": "Get the weather for a city."},
    {"name": "flaky_api", "func": _flaky_api, "desc": "Fetch a URL and return the first 200 bytes."},
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
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--provider", choices=["openai", "anthropic"], default=None)
    parser.add_argument("--model", default=None, help="Model name (e.g. openai/gpt-oss-20b:free for OpenRouter).")
    parser.add_argument("--base-url", default=None, help="API base URL (e.g. https://openrouter.ai/api/v1).")
    parser.add_argument("--mode", choices=list(MODE_TASKS), default="error")
    parser.add_argument("--task", default=None, help="Override the mode's task with a custom real prompt.")
    parser.add_argument("--recursion-limit", type=int, default=40, help="LangGraph recursion limit (max agent steps).")
    parser.add_argument("--webhook-url", default="http://127.0.0.1:8787/risks", help="Where the WebhookSink POSTs risks.")
    args = parser.parse_args()

    provider = _pick_provider(args.provider)
    model = args.model or ("gpt-4o-mini" if provider == "openai" else "claude-3-5-haiku-latest")
    task = args.task or MODE_TASKS[args.mode]
    print(f"[demo] provider={provider} model={model} mode={args.mode}")
    print(f"[demo] WebhookSink -> {args.webhook_url} (the local sidecar)")
    print(f"[demo] recursion_limit={args.recursion_limit}\n")

    cfg = Config(cusum_min_samples=3)
    detectors = [LoopDetector(config=cfg), ErrorCascadeDetector(config=cfg), LatencyAnomalyDetector(config=cfg)]
    # Real detection happens here; ConsoleSink shows it locally AND WebhookSink
    # ships the real FailureRisk over HTTP to the sidecar.
    monitor = Monitor(detectors, [ConsoleSink(), WebhookSink(args.webhook_url)])
    handler = SnaglineCallbackHandler(monitor, "real-llm-webhook")

    model_obj = _build_model(provider, model, args.base_url)
    agent = __import__("langchain.agents", fromlist=["create_agent"]).create_agent(model=model_obj, tools=_build_tools())

    print(f"[demo] task: {task}\n[demo] local detection + HTTP delivery to sidecar...\n")
    signal.signal(signal.SIGALRM, _on_alarm)
    # Time-budgeted monitoring window: keep the real agent running against the
    # real model for up to BUDGET seconds. Each single stream call is capped by
    # SIGALRM (ATTEMPT_MAX) so a slow/queued free-tier model can't hang the run
    # past the window. Transient 502s are retried so the window actually fills.
    BUDGET = 270.0
    ATTEMPT_MAX = 55.0
    deadline = time.time() + BUDGET
    attempt = 0
    while time.time() < deadline:
        attempt += 1
        signal.alarm(int(ATTEMPT_MAX))
        try:
            for chunk in agent.stream(
                {"messages": [("user", task)]},
                config={"callbacks": [handler], "recursion_limit": args.recursion_limit},
            ):
                if "messages" in chunk and chunk["messages"]:
                    msg = chunk["messages"][-1]
                    content = getattr(msg, "content", None)
                    if content and getattr(msg, "type", "") in ("ai", "human", "tool"):
                        print(f"[{msg.type}] {content if isinstance(content, str) else content}")
            signal.alarm(0)
            print(f"\n[demo] agent finished the task on attempt {attempt}.")
            break
        except _AttemptTimeout:
            signal.alarm(0)
            print(f"\n[demo] attempt {attempt} exceeded {ATTEMPT_MAX:.0f}s (slow model); retrying...")
            time.sleep(2)
        except Exception as exc:  # fail-open: survive a transient model/API error
            signal.alarm(0)
            print(f"\n[demo] attempt {attempt} interrupted ({type(exc).__name__}); retrying in 5s...")
            time.sleep(5)
    else:
        print(f"\n[demo] hit the {int(BUDGET)}s monitoring budget without finishing; stopping.")

    handler.close()
    print("\n[demo] done. The sidecar process should have printed '[sidecar] RECEIVED risk' lines.")


if __name__ == "__main__":
    main()
