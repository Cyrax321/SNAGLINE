"""Regression tests for the unvalidated CUSUM slack (issue #421).

``cusum_k`` (latency anomaly) and ``token_cusum_k`` (token runaway) are the
slack subtracted from the CUSUM accumulator on every scored step. A negative
value *adds* ``|k|`` unconditionally, so the accumulator climbs by ``|k|`` per
step no matter what the data does and the alarm becomes a function of step
count alone -- a false-positive storm on perfectly healthy traffic. Measured:
20 risks in 30 steps on an exactly constant 100 tokens/step, score climbing to
1.0; the latency detector does the same on constant latency.

``cusum_h`` / ``token_cusum_h`` (the alarm bars) are #331; this covers the
slack, which #412's non-finite check does not reach -- a finite
``SNAGLINE_CUSUM_K=-1.0`` sailed through every layer. ``k = 0`` is legitimate
("no slack": maximally sensitive but not inverted), so the bound is open at
zero, matching #371's treatment of ``semantic_drift_cusum_k``.

Each bad value must be rejected at every config layer (constructor, env,
``resolve``) and at detector construction, matching the _validated_stagnation
precedent (issue #132): broken monitoring config aborts startup loudly instead
of running silently mis-configured.
"""

from __future__ import annotations

import pytest

from snagline.config import Config
from snagline.events import StepEvent


def _env(name: str, value) -> dict[str, str]:
    return {f"SNAGLINE_{name.upper()}": str(value)}


def _lat_ev(latency_ms: float, tool_name: str = "search") -> StepEvent:
    # action_type must be tool_call: the detector skips aggregate steps
    # (issue #10) and would otherwise observe nothing at all.
    return StepEvent(
        step_id="s",
        episode_id="ep",
        timestamp=1.0,
        action_type="tool_call",
        action_signature="SIG",
        tool_name=tool_name,
        latency_ms=latency_ms,
    )


def _tok_ev(tokens: int) -> StepEvent:
    """A token-bearing event with no latency -- the token detector ignores
    events carrying neither token field, so it needs its own builder."""
    return StepEvent(
        step_id="s",
        episode_id="ep",
        timestamp=1.0,
        action_type="tool_call",
        action_signature="SIG",
        tokens_in=tokens,
    )


# --- the config layers ------------------------------------------------------


@pytest.mark.parametrize("name", ["cusum_k", "token_cusum_k"])
@pytest.mark.parametrize("bad", [-0.001, -1.0, -100.0])
def test_slack_rejects_negative(name, bad):
    with pytest.raises(ValueError, match=name):
        Config(**{name: bad})
    with pytest.raises(ValueError, match=name):
        Config.from_env(environ=_env(name, bad))
    with pytest.raises(ValueError, match=name):
        Config.resolve(environ=_env(name, bad))


@pytest.mark.parametrize("name", ["cusum_k", "token_cusum_k"])
@pytest.mark.parametrize("good", [0.0, 0.001, 0.5, 2.0, 100.0])
def test_slack_accepts_zero_and_positive(name, good):
    """The bound is open at zero: no slack is maximally sensitive but not
    inverted, so it stays a legal configuration."""
    assert getattr(Config(**{name: good}), name) == good
    assert getattr(Config.from_env(environ=_env(name, good)), name) == good


# --- the detectors, reached directly (kwargs skip the Config check) ---------


@pytest.mark.parametrize("bad", [-1.0, -0.001])
def test_latency_detector_rejects_a_negative_slack(bad):
    from snagline.detectors.latency_anomaly import LatencyAnomalyDetector

    with pytest.raises(ValueError, match="k must be >= 0"):
        LatencyAnomalyDetector(k=bad)


@pytest.mark.parametrize("bad", [-1.0, -0.001])
def test_token_detector_rejects_a_negative_slack(bad):
    from snagline.detectors.token_runaway import TokenRunawayDetector

    with pytest.raises(ValueError, match="k must be >= 0"):
        TokenRunawayDetector(k=bad)


# --- the positive control: a zero slack must not storm ---------------------
# This is the assertion that pins the bound choice. It passes both before and
# after the fix, which is what makes it a control rather than a regression
# test: it proves the rejection is narrow and does not forbid a working value.


def test_zero_slack_does_not_storm_healthy_traffic():
    from snagline.detectors.latency_anomaly import LatencyAnomalyDetector
    from snagline.detectors.token_runaway import TokenRunawayDetector

    lat = LatencyAnomalyDetector(k=0.0, min_samples=5)
    lat_risks = [lat.observe(_lat_ev(100.0)) for _ in range(30)]
    assert [r for r in lat_risks if r is not None] == [], (
        "a constant latency with no slack must not page"
    )

    tok = TokenRunawayDetector(k=0.0, min_samples=5)
    tok_risks = [tok.observe(_tok_ev(100)) for _ in range(30)]
    assert [r for r in tok_risks if r is not None] == [], (
        "a constant token volume with no slack must not page"
    )


# --- the end-to-end layer an env var actually hits -------------------------


def test_negative_slack_aborts_startup_before_any_risk_is_dispatched():
    """The reported symptom is an alert storm. Post-fix the config is refused
    at ``Monitor.default`` construction, so no risk can be dispatched from a
    rejected configuration."""
    from snagline.monitor import Monitor
    from snagline.sinks.base import AlertSink

    class Collect(AlertSink):
        def __init__(self) -> None:
            self.risks: list = []

        def emit(self, risk) -> None:
            self.risks.append(risk)

    sink = Collect()
    with pytest.raises(ValueError, match="token_cusum_k"):
        Monitor.default(
            config=Config(token_runaway_enabled=True, token_cusum_k=-1.0),
            sinks=[sink],
        )
    assert sink.risks == [], "no risk may be dispatched from a rejected config"
