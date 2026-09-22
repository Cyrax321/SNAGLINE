"""Regression tests for non-finite adapter input reaching the detectors (#349, #350).

#349  TokenRunawayDetector converted token counts with a bare ``int(...)``:
      ``int(NaN)`` raises ValueError and ``int(Inf)`` overflows, and both
      escaped ``observe`` into ``Monitor.ingest``'s fail-open handler -- the
      detector then stayed installed and reported nothing for the rest of the
      run. ``tokens_in=-50`` was silently accepted and *reduced* the budget
      total.
#350  One NaN ``latency_ms`` poisoned ``mu0`` irreversibly: CPython's
      ``max(0.0, NaN)`` returns ``0.0``, so ``cusum > h`` was False forever --
      the detector went dark with no crash and no log line for fail-open to
      catch. ``inf`` alarmed on every step instead.

Adapters derive these values from provider ``usage`` objects and duration
arithmetic, so a malformed blob or a zero-elapsed division is a realistic
runtime failure, not a synthetic one. The contract these tests pin: one bad
sample costs at most that step, never the episode.
"""

from __future__ import annotations

from typing import Any

import pytest

from snagline.config import Config
from snagline.detectors.latency_anomaly import LatencyAnomalyDetector
from snagline.detectors.token_runaway import TokenRunawayDetector
from snagline.events import StepEvent, make_signature


def _ev(
    i: int,
    latency_ms: float | None = None,
    # Deliberately malformed values: NaN/Inf are floats, not ints, and the
    # point of these tests is exactly that a detector must survive values the
    # declared type disallows. Any keeps mypy from rejecting the repro itself.
    tokens_in: Any = None,
    tokens_out: Any = None,
) -> StepEvent:
    return StepEvent(
        step_id=str(i),
        episode_id="ep",
        timestamp=1000.0 + i,
        action_type="tool_call",
        action_signature=make_signature("tool_call", "t", "a"),
        tool_name="t",
        latency_ms=latency_ms,
        tokens_in=tokens_in,
        tokens_out=tokens_out,
    )


# --- issue #349: the token detector ----------------------------------------


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_token_counts_do_not_crash_the_detector(bad):
    """The reported failure: int(NaN) raised ValueError, int(Inf) raised
    OverflowError, and fail-open left the detector dead for the run."""
    det = TokenRunawayDetector(config=Config(token_runaway_enabled=True))
    # Must not raise -- the whole point is that one bad sample is survivable.
    assert det.observe(_ev(0, tokens_in=bad)) is None
    assert det.observe(_ev(1, tokens_out=bad)) is None
    assert det.observe(_ev(2, tokens_in=bad, tokens_out=bad)) is None


def test_negative_token_count_does_not_shrink_the_budget():
    """tokens_in=-50 used to reduce the running total, making a later breach
    look smaller than it was."""
    det = TokenRunawayDetector(
        config=Config(token_runaway_enabled=True, episode_token_budget=100)
    )
    for i in range(5):
        assert det.observe(_ev(i, tokens_in=-50)) is None
    # None of the negative counts counted toward the 100-token budget.
    assert det.observe(_ev(6, tokens_in=150)) is not None


def test_good_token_steps_still_fire_after_bad_ones():
    """The detector must stay live: the envelope still fires once usable
    samples arrive, and the breach latches on the first crossing."""
    det = TokenRunawayDetector(
        config=Config(token_runaway_enabled=True, episode_token_budget=100)
    )
    det.observe(_ev(0, tokens_in=float("nan")))
    assert det.observe(_ev(1, tokens_in=100)) is not None
    # The breach latches: later steps stay quiet, but the detector is healthy.
    assert det.observe(_ev(2, tokens_in=100)) is None


def test_partial_token_signal_is_kept():
    """One usable field still contributes its half: a NaN tokens_in must not
    discard a real tokens_out."""
    det = TokenRunawayDetector(
        config=Config(token_runaway_enabled=True, episode_token_budget=100)
    )
    assert det.observe(_ev(0, tokens_in=float("nan"), tokens_out=100)) is not None


# --- issue #350: the latency detector --------------------------------------


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_latency_does_not_blind_the_detector(bad):
    """The reported failure was silent rather than loud: a NaN propagated into
    mu0 and ``max(0.0, NaN)`` returned 0.0, so cusum never rose again."""
    det = LatencyAnomalyDetector(config=Config(cusum_min_samples=3))
    for i in range(3):
        assert det.observe(_ev(i, latency_ms=100.0)) is None  # healthy warm-up
    # A bad sample is dropped, not absorbed into the baseline.
    assert det.observe(_ev(3, latency_ms=bad)) is None
    fired = [det.observe(_ev(i, latency_ms=999.0)) for i in range(4, 11)]
    assert any(r is not None for r in fired), "a clear anomaly must still page"


def test_negative_latency_is_ignored():
    det = LatencyAnomalyDetector(config=Config(cusum_min_samples=3))
    for i in range(3):
        det.observe(_ev(i, latency_ms=100.0))
    assert det.observe(_ev(3, latency_ms=-500.0)) is None


def test_non_finite_warm_up_does_not_satisfy_the_sample_gate():
    """The warm-up gate must not be satisfied by garbage (issue #350): with
    every warm-up sample unusable the baseline is never frozen, so nothing is
    scored against a baseline built from nothing. Real samples must still
    complete a clean warm-up afterwards."""
    det = LatencyAnomalyDetector(config=Config(cusum_min_samples=3))
    for i in range(5):
        assert det.observe(_ev(i, latency_ms=float("nan"))) is None
    # Warm-up on healthy samples only, so the frozen baseline is clean.
    for i in range(5, 8):
        assert det.observe(_ev(i, latency_ms=100.0)) is None
    assert det.observe(_ev(8, latency_ms=999.0)) is not None


def test_nan_via_monitor_fail_open_keeps_detector_alive():
    """End-to-end shape from the issue: through Monitor.default a bad event
    used to leave detector_errors == 1 and every later event unmonitored."""
    from snagline.monitor import Monitor

    mon = Monitor.default(
        config=Config(token_runaway_enabled=True, token_min_samples=3)
    )
    for i in range(4):
        mon.ingest(_ev(i, tokens_in=float("nan")))
    for i in range(10):
        mon.ingest(_ev(i, tokens_in=1000))
    m = mon.metrics()
    assert m["detector_errors"] == 0, "one bad sample must not deaden the run"
    assert m["risks_emitted"] > 0, "the burn must still be reported"
