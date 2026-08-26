"""Window auto-scaling shared by window-based detectors (issue #92).

Fixed windows tuned for interactive runs (``loop_window_size=12``) mean
nothing across a 500k-step week-long episode: they either fire constantly or
are gone in the first minute of noise. When ``Config.window_scale_steps`` is
set (> 0), each detector's effective window grows as

    base * ceil(episode_len / window_scale_steps)

capped at ``Config.max_window``, and is refitted lazily: growing a deque keeps
its most recent items, so no per-step reprocessing ever happens.

Defaults preserve current behavior exactly: with ``window_scale_steps == 0``
the effective size is always ``base`` and every resize call below is a no-op.
"""

from __future__ import annotations

from collections import deque


def effective_window_size(
    base: int, episode_len: int, scale_steps: int, max_window: int
) -> int:
    """Scaled window length for an episode of ``episode_len`` observed events.

    Monotonically non-decreasing in ``episode_len``, always within
    ``[base, max_window]``. With ``scale_steps <= 0`` (scaling disabled) this
    is exactly ``base`` for every input, which is what keeps the default
    configuration byte-identical to pre-#92 behavior.
    """
    if scale_steps <= 0 or episode_len <= 1:
        return base
    # Integer ceiling division: ceil(episode_len / scale_steps).
    factor = -(-episode_len // scale_steps)
    # The cap never pulls the window below its base: a detector configured
    # with base > max_window keeps its base rather than shrinking.
    return min(max(base * factor, base), max(max_window, base))


def next_window(
    windows: dict[str, deque],
    counts: dict[str, int],
    episode_id: str,
    base: int,
    scale_steps: int,
    max_window: int,
) -> deque:
    """Return the episode's window, resized lazily when scaling demands it.

    Encapsulates the ``dict[str, deque]`` bookkeeping several detectors share:
    counts the episode's observed events, computes the effective size, and
    swaps in a larger deque (keeping the most recent items) only when the
    target actually changed. Replacement (rather than in-place append) is safe
    because callers fetch the window fresh from the dict on every observe().
    """
    n = counts.get(episode_id, 0) + 1
    counts[episode_id] = n
    w = windows.get(episode_id)
    if w is None:
        w = deque(maxlen=base)
        windows[episode_id] = w
        return w
    target = effective_window_size(base, n, scale_steps, max_window)
    if w.maxlen != target:
        w = deque(w, maxlen=target)
        windows[episode_id] = w
    return w
