"""Sink extension point. See ``base.py`` for the ``AlertSink`` protocol."""

from snagline.sinks.logging_sink import JsonRiskFormatter, LoggingSink

__all__ = ["JsonRiskFormatter", "LoggingSink"]
