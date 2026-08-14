"""Adapter package -- the only framework-coupled code (project.md §1.3).

Each adapter turns a framework-specific event into a ``StepEvent`` and calls
``monitor.ingest``. The ``raw`` adapter is stdlib-only and always available;
framework adapters are optional extras.
"""

from snagline.adapters.raw import watch

__all__ = ["watch"]
