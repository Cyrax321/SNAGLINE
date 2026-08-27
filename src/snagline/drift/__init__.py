"""Optional ``drift`` extra: semantic goal-drift detection (issue #81).

Provides ``pip install snagline[drift]`` (sentence-transformers). This
subpackage is never imported by core code paths unless
``Config.semantic_drift_enabled`` is set; ``import snagline`` works with
nothing but the standard library (project.md section 1.1). Even this module
itself imports cleanly without the extra installed: the heavy dependency is
touched only inside the lazy model loader, and any failure there leaves the
detector inert (caught, logged, ignored) instead of raising into the host.
"""
