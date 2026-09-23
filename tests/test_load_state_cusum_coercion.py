"""A structurally-complete but non-numeric restored CUSUM state must be rejected
at restore, not accepted and detonated on the next observe (issue #424)."""

from __future__ import annotations

import pytest

from snagline.config import Config
from snagline.detectors.latency_anomaly import LatencyAnomalyDetector
from snagline.detectors.token_runaway import TokenRunawayDetector
from snagline.events import StepEvent

_GOOD = {
    "n": 3,
    "mean": 5.0,
    "m2": 4.0,
    "cusum": 0.0,
    "mu0": 5.0,
    "sigma0": 2.0,
    "frozen": True,
}


def _event(
    step: int = 1, episode: str = "ep", tokens_out: int = 10, latency_ms: float = 12.0
) -> StepEvent:
    return StepEvent(
        step_id=str(step),
        episode_id=episode,
        timestamp=float(step),
        action_type="tool_call",
        action_signature="s1",
        tokens_out=tokens_out,
        latency_ms=latency_ms,
    )


def _token_state(**over) -> dict:
    return {
        "states": {"ep": {**_GOOD, **over}},
        "totals": {},
        "warned": {},
        "breached": {},
    }


def _latency_state(**over) -> dict:
    return {"states": [[["ep", "s1"], {**_GOOD, **over}]]}


@pytest.mark.parametrize(
    "bad",
    [
        {"n": "not-a-number"},
        {"mean": "x"},
        {"m2": None},
        {"cusum": object()},
        {"sigma0": "wide"},
    ],
)
def test_token_runaway_rejects_a_non_numeric_cusum_field_at_restore(bad):
    """Pre-fix the entry was accepted and the episode wedged on the next step."""
    d = TokenRunawayDetector(min_samples=3)
    with pytest.raises((TypeError, ValueError)):
        d.load_state(_token_state(**bad))


def test_token_runaway_recovers_after_a_rejected_entry():
    """The whole point of rejecting at restore: the detector stays usable."""
    d = TokenRunawayDetector(min_samples=3)
    d.load_state(_token_state())
    # No raise, and the detector still scores.
    d.observe(_event(tokens_out=1000))
    assert len(d.dump_state()["states"]) == 1


def test_token_runaway_mid_warmup_mu0_none_is_legitimate():
    """mu0 is None for a state snapshotted before freeze(); it must survive."""
    d = TokenRunawayDetector(min_samples=3)
    d.load_state(_token_state(mu0=None, frozen=False))
    s = d.dump_state()["states"]["ep"]
    assert s["mu0"] is None
    assert s["frozen"] is False
    # And the detector can still warm up from that state.
    d.observe(_event(tokens_out=10))
    assert d.dump_state()["states"]["ep"]["n"] == 4


@pytest.mark.parametrize(
    "bad", [{"n": "not-a-number"}, {"mean": "x"}, {"sigma0": None}]
)
def test_latency_anomaly_rejects_a_non_numeric_cusum_field_at_restore(bad):
    d = LatencyAnomalyDetector(config=Config(), min_samples=3)
    with pytest.raises((TypeError, ValueError)):
        d.load_state(_latency_state(**bad))


def test_latency_anomaly_recovers_after_a_rejected_entry():
    d = LatencyAnomalyDetector(config=Config(), min_samples=3)
    d.load_state(_latency_state())
    d.observe(_event())
    # No raise, and the restored baseline is still intact and live.
    states = {tuple(k): v for k, v in d.dump_state()["states"]}
    assert states[("ep", "s1")]["n"] == 3
    assert states[("ep", "s1")]["frozen"] is True


def test_latency_anomaly_mid_warmup_mu0_none_is_legitimate():
    d = LatencyAnomalyDetector(config=Config(), min_samples=3)
    d.load_state(_latency_state(mu0=None, frozen=False))
    # Accepted as-is: mu0 is legitimately absent before freeze().
    s = d.dump_state()["states"][0][1]
    assert s["mu0"] is None
    assert s["frozen"] is False


def test_round_trip_of_a_live_state_is_unchanged():
    """Coercion must not alter what a real dump/load cycle produces."""
    d = LatencyAnomalyDetector(config=Config(), min_samples=2)
    for i in range(5):
        d.observe(_event(step=i, latency_ms=10.0 + i))
    dumped = d.dump_state()
    d2 = LatencyAnomalyDetector(config=Config(), min_samples=2)
    d2.load_state(dumped)
    assert d2.dump_state() == dumped
