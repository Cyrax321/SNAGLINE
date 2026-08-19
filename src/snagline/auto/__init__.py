"""Auto-instrumentation package (ATTACH_ANY_SYSTEM P0)."""

from snagline.auto.anthropic import instrument_anthropic
from snagline.auto.anthropic import wrap_client as wrap_anthropic
from snagline.auto.langchain import (
    instrument_langchain,
)
from snagline.auto.langchain import (
    wrap_client as wrap_langchain,
)
from snagline.auto.openai import instrument_openai, wrap_client

__all__ = [
    "instrument_openai",
    "wrap_client",
    "instrument_anthropic",
    "wrap_anthropic",
    "instrument_langchain",
    "wrap_langchain",
]
