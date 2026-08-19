"""Auto-instrumentation package (ATTACH_ANY_SYSTEM P0)."""

from snagline.auto.openai import instrument_openai, wrap_client

__all__ = ["instrument_openai", "wrap_client"]
