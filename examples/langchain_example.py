"""Runnable demo: SNAGLINE watching a REAL LangChain run via the callback handler.

Requires langchain-core (optional extra):
    pip install snagline-agent[langchain]
    PYTHONPATH=src python3 examples/langchain_example.py

Four identical prompts are sent to a (fake) chat model; the repeated message
signature trips the loop detector, printed as a JSON line to stderr.
"""

from __future__ import annotations

import sys

from snagline import Monitor
from snagline.adapters.langchain_adapter import SnaglineCallbackHandler


def main() -> None:
    try:
        from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
        from langchain_core.messages import AIMessage, HumanMessage
    except ImportError:
        print(
            "This example needs langchain-core. Install it with:\n"
            "    pip install snagline-agent[langchain]",
            file=sys.stderr,
        )
        return

    monitor = Monitor.default()
    handler = SnaglineCallbackHandler(monitor, "langchain-demo")

    # 6 responses so up to 6 repeats are safe; identical prompt each time.
    model = FakeMessagesListChatModel(
        responses=[AIMessage(content="ok") for _ in range(6)]
    )

    print("[demo] invoking the model 4x with an identical prompt...", file=sys.stderr)
    for _ in range(4):
        model.invoke([HumanMessage(content="repeat this exactly")], config={"callbacks": [handler]})

    handler.close()
    print("[demo] finished; loop detector should have fired above.", file=sys.stderr)


if __name__ == "__main__":
    main()
