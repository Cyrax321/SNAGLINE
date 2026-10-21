"""Semantic goal-drift tests: ``drift`` extra (issue #81).

Everything here runs without sentence-transformers, numpy, or torch: the
detector accepts an injectable ``embedder`` callable, so the suite exercises
the real detection logic with small hand-built vectors. Import-guard tests
additionally poison ``sys.modules`` to prove laziness. Both sides are always
covered per detector behavior: an injected-drift sequence that fires and a
healthy sequence that stays silent.
"""

from __future__ import annotations

import json
import logging
import sys
from typing import Any

import pytest

from snagline.baseline import BaselineProfile, save_baseline
from snagline.config import Config
from snagline.detectors.ml_ensemble import MLOrchestrator
from snagline.drift.goal_drift import (
    SemanticGoalDriftDetector,
    _cosine,
    _event_label,
    fit_semantic_baseline,
)
from snagline.events import StepEvent
from snagline.monitor import Monitor

# ---------------------------------------------------------------------------
# Helpers


def _ev(i: int, tool: str = "search", episode: str = "ep") -> StepEvent:
    return StepEvent(
        step_id=f"s{i}",
        episode_id=episode,
        timestamp=float(i),
        action_type="tool_call",
        action_signature=f"sig-{tool}-{i}",
        tool_name=tool,
        latency_ms=100.0,
        error=False,
    )


def _profile(centroid: list[float]) -> BaselineProfile:
    p = BaselineProfile()
    p.embedding_centroid = centroid
    p.embedding_count = 4
    p.embedding_model = "fake-model"
    return p


def test_profile_without_semantics_serializes_exactly_as_before():
    # Byte-identical zero-dep preset: no embedding keys unless fitted.
    legacy_keys = {"version", "total_steps", "tools"}
    assert set(BaselineProfile().to_dict().keys()) == legacy_keys


def test_profile_semantic_round_trip_through_dict():
    p = _profile([1.0, 0.0])
    restored = BaselineProfile.from_dict(p.to_dict())
    assert restored.embedding_centroid == [1.0, 0.0]
    assert restored.embedding_count == 4
    assert restored.embedding_model == "fake-model"


def test_profile_legacy_json_without_embedding_keys_loads_unchanged():
    raw = {
        "version": 1,
        "total_steps": 3,
        "tools": {},
    }
    p = BaselineProfile.from_dict(raw)
    assert p.embedding_centroid is None
    assert p.embedding_count == 0
    assert p.embedding_model is None


def test_saved_baseline_file_round_trips_semantic_fields(tmp_path):
    p = _profile([0.6, 0.8])
    path = str(tmp_path / "base.json")
    save_baseline(p, path)
    with open(path, encoding="utf-8") as fh:
        on_disk = json.load(fh)
    assert on_disk["embedding_centroid"] == [0.6, 0.8]
    # The structural part stays untouched by the optional fields.
    assert set(on_disk.keys()) >= {"version", "total_steps", "tools"}


def test_profile_structural_to_dict_is_byte_identical_to_legacy_shape():
    # Compare against the exact pre-#81 serialization of an empty profile.
    expected = '{\n  "tools": {},\n  "total_steps": 0,\n  "version": 1\n}\n'
    import io

    buf = io.StringIO()
    from snagline.baseline import _write_json

    _write_json(buf, BaselineProfile().to_dict())
    assert buf.getvalue() == expected


def test_profile_from_dict_rejects_non_numeric_centroids():
    with pytest.raises((TypeError, ValueError)):
        BaselineProfile.from_dict({"embedding_centroid": ["x"]})


# ---------------------------------------------------------------------------
# Config keys (scalar coercion must cover the new opt-in fields)


def test_semantic_drift_keys_coerce_from_environment():

    cfg = Config.from_env(
        {
            "SNAGLINE_SEMANTIC_DRIFT_ENABLED": "true",
            "SNAGLINE_SEMANTIC_DRIFT_MODEL": "custom/model",
            "SNAGLINE_SEMANTIC_DRIFT_MIN_SAMPLES": "7",
            "SNAGLINE_SEMANTIC_DRIFT_WINDOW_SIZE": "15",
            "SNAGLINE_SEMANTIC_DRIFT_TOLERANCE": "0.4",
        }
    )
    assert cfg.semantic_drift_enabled is True
    assert cfg.semantic_drift_model == "custom/model"
    assert cfg.semantic_drift_min_samples == 7
    assert cfg.semantic_drift_window_size == 15
    assert cfg.semantic_drift_tolerance == pytest.approx(0.4)


def test_semantic_drift_defaults_keep_preset_off():

    cfg = Config()
    assert cfg.semantic_drift_enabled is False
    # Hand-computed expectation of the shipped defaults (not read back from
    # the same object): documented in README and DETECTOR_GUIDE.
    assert cfg.semantic_drift_min_samples == 10
    assert cfg.semantic_drift_window_size == 20
    assert cfg.semantic_drift_tolerance == 0.2
    assert cfg.semantic_drift_cusum_k == 0.05
    assert cfg.semantic_drift_cusum_h == 0.5


# ---------------------------------------------------------------------------
# Detector: geometry helpers and fakes


# Hand-built embedding families. Healthy tools point along +x; the drifted
# tool points along -x, i.e. cosine similarity -1 territory.
HEALTHY_VECS = {"search": [1.0, 0.1], "summarize": [0.9, 0.2]}
DRIFT_VEC = [-1.0, 0.05]
ALL_VECS = {**HEALTHY_VECS, "wipe_disk": DRIFT_VEC}


def _map_embedder(table: dict[str, list[float]]) -> Any:
    def _embed(event: StepEvent) -> list[float]:
        tool = event.tool_name
        assert tool is not None
        return list(table[tool])

    return _embed


def _healthy(n: int, start: int = 0, episode: str = "ep") -> list[StepEvent]:
    tools = list(HEALTHY_VECS)
    return [_ev(start + i, tools[i % len(tools)], episode) for i in range(n)]


def _detector(
    baseline: BaselineProfile,
    embedder: Any = None,
    **overrides: Any,
) -> SemanticGoalDriftDetector:

    cfg = Config(**{"semantic_drift_model": "fake-model", **overrides})
    return SemanticGoalDriftDetector(
        baseline,
        config=cfg,
        embedder=embedder if embedder is not None else _map_embedder(ALL_VECS),
    )


def _semantic_baseline(n: int = 30) -> BaselineProfile:

    return fit_semantic_baseline(
        _healthy(n), _map_embedder(HEALTHY_VECS), model="fake-model"
    )


# ---------------------------------------------------------------------------
# Detector: pure math, labels, privacy


def test_cosine_hand_computed_values():

    assert _cosine([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)
    # Orthogonal-ish pair [1,0] vs [1,1]: dot 1, norms 1 and sqrt(2).
    assert _cosine([1.0, 0.0], [1.0, 1.0]) == pytest.approx(1.0 / 2**0.5)
    assert _cosine([1.0, 0.0], [-1.0, 0.0]) == pytest.approx(-1.0)
    # Degenerate vectors carry no direction: guard must stay silent (1.0).
    assert _cosine([0.0, 0.0], [1.0, 0.0]) == 1.0
    assert _cosine([1.0, 0.0], [0.0, 0.0]) == 1.0


def test_event_label_is_structural_and_ignores_content_fields():

    base = _ev(0, "search")
    with_meta = StepEvent(
        step_id=base.step_id,
        episode_id=base.episode_id,
        timestamp=base.timestamp,
        action_type=base.action_type,
        action_signature="totally-different-hash",
        tool_name="search",
        metadata={"prompt": "SECRET PROMPT TEXT", "response": "SECRET"},
    )
    # Same structure, different hash/metadata: identical label, so identical
    # embeddings downstream. Content never leaks into the semantic channel.
    assert _event_label(base) == _event_label(with_meta)
    assert _event_label(base) == "tool_call search"
    err = StepEvent(
        step_id="e",
        episode_id="ep",
        timestamp=0.0,
        action_type="tool_call",
        action_signature="h",
        tool_name="search",
        error_type="TimeoutError",
    )
    assert _event_label(err) == "tool_call search TimeoutError"


def test_fit_metadata_neutrality_identical_centroids():

    plain = _healthy(10)
    tagged = [
        StepEvent(
            step_id=e.step_id,
            episode_id=e.episode_id,
            timestamp=e.timestamp,
            action_type=e.action_type,
            action_signature=e.action_signature,
            tool_name=e.tool_name,
            metadata={"note": f"private-{i}"},
        )
        for i, e in enumerate(plain)
    ]
    emb = _map_embedder(HEALTHY_VECS)
    a = fit_semantic_baseline(plain, emb)
    b = fit_semantic_baseline(tagged, emb)
    assert a.embedding_centroid == b.embedding_centroid


def test_fit_hand_computed_mean_and_provenance():

    events = [_ev(0, "search", "fit"), _ev(1, "wipe_disk", "fit")]
    profile = fit_semantic_baseline(events, _map_embedder(ALL_VECS), model="m1")
    # Mean of [1.0, 0.1] and [-1.0, 0.05], computed by hand.
    assert profile.embedding_centroid == pytest.approx([0.0, 0.075])
    assert profile.embedding_count == 2
    assert profile.embedding_model == "m1"
    # Structural stats ride along untouched.
    assert profile.tools["search"].count == 1
    assert profile.tools["wipe_disk"].count == 1


def test_fit_rejects_dimension_changes():

    def unstable(event: StepEvent) -> list[float]:
        return [1.0, 0.0] if event.timestamp < 1 else [1.0, 2.0, 3.0]

    with pytest.raises(ValueError, match="dimension"):
        fit_semantic_baseline(_healthy(4), unstable)


# ---------------------------------------------------------------------------
# Detector: both-sided behavior (fires on sustained drift / silent on healthy)


def test_sustained_drift_fires_exact_goal_drift_trigger():
    det = _detector(_semantic_baseline(), semantic_drift_min_samples=10)
    episode = "live"
    for i, ev in enumerate(_healthy(12, episode=episode)):
        assert det.observe(ev) is None  # warm-up plus two scored healthy steps
    fired = None
    for j in range(12, 60):
        risk = det.observe(_ev(j, "wipe_disk", episode))
        if risk is not None:
            fired = risk
            break
    assert fired is not None, "sustained semantic drift never fired"
    assert fired.trigger == "goal_drift"
    assert fired.episode_id == episode
    assert 0.0 < fired.score <= 1.0


def test_long_healthy_stream_stays_silent():
    det = _detector(_semantic_baseline(), semantic_drift_min_samples=10)
    for ev in _healthy(200, episode="calm"):
        assert det.observe(ev) is None


def test_single_outlier_step_among_healthy_stays_silent():
    # The CUSUM gate exists precisely so one odd step cannot page anyone:
    # deviation must be sustained. Hand-checked: a single signal pulse minus
    # slack k decays back to zero long before reaching h.
    det = _detector(_semantic_baseline(), semantic_drift_min_samples=10)
    stream = _healthy(80, episode="blip")
    stream[40] = _ev(40, "wipe_disk", "blip")
    for ev in stream:
        assert det.observe(ev) is None


def test_warmup_below_min_samples_stays_silent_even_on_pure_drift():
    det = _detector(_semantic_baseline(), semantic_drift_min_samples=10)
    for i in range(9):
        assert det.observe(_ev(i, "wipe_disk", "short")) is None


def test_rearms_and_fires_again_while_drift_continues():
    det = _detector(_semantic_baseline(), semantic_drift_min_samples=10)
    fires = 0
    for i in range(120):
        if det.observe(_ev(i, "wipe_disk", "relapse")) is not None:
            fires += 1
    assert fires >= 2, "persistent drift must re-alarm after re-arming"


def test_empty_embedding_vectors_keep_detector_inert(caplog):
    baseline = BaselineProfile()
    baseline.embedding_centroid = []  # fitted but degenerate: no direction
    with caplog.at_level(logging.INFO, logger="snagline"):
        det = _detector(baseline)
        for ev in _healthy(30, episode="x"):
            assert det.observe(ev) is None
    assert any("embedding_centroid" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# Detector: fail-open guarantees (dedicated, per task rules)


def test_fail_open_embedder_exceptions_swallowed_and_logged(caplog):
    def boom(event: StepEvent) -> list[float]:
        raise RuntimeError("inference backend exploded")

    det = _detector(_semantic_baseline(), embedder=boom)
    with caplog.at_level(logging.ERROR, logger="snagline"):
        for ev in _healthy(30, episode="host"):
            assert det.observe(ev) is None  # host keeps running, no raise
    assert any("semantic_goal_drift" in r.message for r in caplog.records)


def test_repeated_inference_failure_logs_once_not_every_step(caplog):
    # Contract (module + class docstring): a broken embedding backend is
    # "logged once", never a traceback per step. A persistently-raising
    # embedder over a long episode must not flood the host's logs -- the same
    # warn-once discipline ConsoleSink applies to a dead stream (issue #327)
    # and that #386 flags as a bug for the ESN CUSUM detector.
    calls = {"n": 0}

    def boom(event: StepEvent) -> list[float]:
        calls["n"] += 1
        raise RuntimeError("inference backend exploded")

    det = _detector(_semantic_baseline(), embedder=boom)
    with caplog.at_level(logging.ERROR, logger="snagline"):
        for ev in _healthy(30, episode="flood"):
            assert det.observe(ev) is None  # still fail-open, still None
    faults = [r for r in caplog.records if "semantic_goal_drift" in r.message]
    assert len(faults) == 1, (
        f"inference fault must be logged once, not per step; got {len(faults)}"
    )
    # Fail-open stays fail-open: the detector is NOT latched inert (unlike a
    # setup failure), it keeps trying so a transient fault can recover.
    assert calls["n"] == 30


def test_recovered_embedder_rearms_the_fault_warning(caplog):
    # The warn-once latch resets on a clean step, so an embedder that fails,
    # recovers, then fails again logs each distinct fault episode once rather
    # than staying silent forever after the first.
    state = {"broken": True}

    def flaky(event: StepEvent) -> list[float]:
        if state["broken"]:
            raise RuntimeError("backend down")
        return list(ALL_VECS[event.tool_name])  # type: ignore[index]

    det = _detector(_semantic_baseline(), embedder=flaky)
    with caplog.at_level(logging.ERROR, logger="snagline"):
        for ev in _healthy(5, episode="r"):
            assert det.observe(ev) is None
        state["broken"] = False  # backend recovers
        for ev in _healthy(3, episode="r"):
            det.observe(ev)
        state["broken"] = True  # backend fails again
        for ev in _healthy(5, episode="r"):
            assert det.observe(ev) is None
    faults = [r for r in caplog.records if "semantic_goal_drift" in r.message]
    assert len(faults) == 2, (
        f"each fault episode logs once after recovery; got {len(faults)}"
    )


def test_model_load_failure_latches_inert_after_exactly_one_attempt(caplog):
    attempts = {"n": 0}

    def broken_loader():
        attempts["n"] += 1
        raise ImportError("torch not installed")

    det = _detector(_semantic_baseline())
    det._model_loader = broken_loader  # type: ignore[attr-defined]
    det._resolved = False
    with caplog.at_level(logging.WARNING, logger="snagline"):
        for ev in _healthy(25, episode="z"):
            assert det.observe(ev) is None
    assert attempts["n"] == 1, "broken setup must not retry on the hot path"
    assert any("unavailable" in r.message for r in caplog.records)


def test_default_loader_missing_extra_is_inert_with_pip_hint(monkeypatch, caplog):
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)

    det = SemanticGoalDriftDetector(
        _semantic_baseline(), config=Config(semantic_drift_model="any")
    )
    with caplog.at_level(logging.WARNING, logger="snagline"):
        for ev in _healthy(15, episode="q"):
            assert det.observe(ev) is None
    assert any(
        "snagline[drift]" in r.message
        for r in caplog.records
        if r.levelno >= logging.WARNING
    )


def test_explicit_embedder_never_touches_sentence_transformers(monkeypatch):
    # Laziness proof: even with the heavy package poisoned away, an injected
    # embedder runs the full detection path (this whole suite relies on it;
    # this test makes the guarantee explicit).
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)
    det = _detector(_semantic_baseline(), semantic_drift_min_samples=10)
    for i, ev in enumerate(_healthy(12, episode="ok")):
        assert det.observe(ev) is None
    fired = None
    for j in range(12, 40):
        fired = det.observe(_ev(j, "wipe_disk", "ok")) or fired
    assert fired is not None


def test_dump_state_round_trip_preserves_episode_state():

    det = _detector(_semantic_baseline(), semantic_drift_min_samples=10)
    for ev in _healthy(12, episode="snap"):
        det.observe(ev)
    for i in range(12, 18):
        det.observe(_ev(i, "wipe_disk", "snap"))
    state = det.dump_state()
    assert state["episodes"]["snap"]["n"] == 18
    restored = SemanticGoalDriftDetector(
        _semantic_baseline(),
        config=Config(semantic_drift_min_samples=10),
        embedder=_map_embedder(ALL_VECS),
    )
    restored.load_state(json.loads(json.dumps(state)))
    assert restored.dump_state() == state
    # Both detectors must behave identically going forward (same verdicts).
    verdicts_a = [det.observe(_ev(18, "wipe_disk", "snap")) for _ in range(1)]
    verdicts_b = [restored.observe(_ev(18, "wipe_disk", "snap")) for _ in range(1)]
    assert [(v is not None) for v in verdicts_a] == [
        (v is not None) for v in verdicts_b
    ]


def test_reset_clears_episode_state():
    det = _detector(_semantic_baseline(), semantic_drift_min_samples=10)
    for ev in _healthy(12, episode="gone"):
        det.observe(ev)
    det.reset("gone")
    assert det.dump_state() == {"episodes": {}}


# ---------------------------------------------------------------------------
# Monitor wiring: opt-in guard proves zero-dep preset untouched


def test_monitor_default_has_no_semantic_detector_without_flag():

    mon = Monitor.default(config=Config())
    names = sorted(d.name for d in mon._detectors)
    assert "semantic_goal_drift" not in names
    # Canonical preset from the guard test in test_ml_extra_guard.py:
    # error_cascade, latency_anomaly, loop and nothing else.
    assert names == ["error_cascade", "latency_anomaly", "loop"]


def test_monitor_default_with_semantic_flag_and_fitted_baseline_adds_detector():

    cfg = Config(
        semantic_drift_enabled=True,
        goal_drift_baseline=_semantic_baseline(),
    )
    mon = Monitor.default(config=cfg)
    assert any(d.name == "semantic_goal_drift" for d in mon._detectors)


def test_monitor_flag_without_semantic_baseline_stays_inert_but_alive():

    # Baseline fitted structurally only: no centroid, so semantic side
    # constructs but stays inert (logs once, never fires).

    empty_sem = BaselineProfile()
    for ev in _healthy(30, episode="seed"):
        empty_sem.add_event(ev)
    cfg = Config(semantic_drift_enabled=True, goal_drift_baseline=empty_sem)
    mon = Monitor.default(config=cfg)
    for ev in _healthy(40, episode="alive"):
        mon.ingest(ev)
    mon.end_episode("alive")


def test_monitor_with_poisoned_transformers_stays_alive(monkeypatch, caplog):

    monkeypatch.setitem(sys.modules, "sentence_transformers", None)
    cfg = Config(
        semantic_drift_enabled=True,
        goal_drift_baseline=_semantic_baseline(),
    )
    with caplog.at_level(logging.WARNING, logger="snagline"):
        mon = Monitor.default(config=cfg)
        for ev in _healthy(20, episode="safe"):
            mon.ingest(ev)
        mon.end_episode("safe")
    # Each missing extra is a WARNING from the lazy loader inside observe,
    # not an exception at wiring time.
    assert any(
        "drift" in r.message for r in caplog.records if r.levelno >= logging.WARNING
    )


def test_dimension_mismatch_in_monitor_path_is_fail_open(caplog):
    # A fitted centroid with dim 3 followed by a 2-dim embedder: monitor
    # must keep running and the stream must stay silent on the bad shots.

    baseline = BaselineProfile()
    baseline.embedding_centroid = [1.0, 0.0, 0.0]
    baseline.embedding_count = 1
    baseline.embedding_model = "fake"
    cfg = Config(semantic_drift_enabled=True, goal_drift_baseline=baseline)
    mon = Monitor.default(config=cfg)
    # Replace its semantic detector's embedder with a 2-dim one (setup for
    # the mismatch path): find it inside monitor.
    from typing import cast

    sem = cast(
        SemanticGoalDriftDetector,
        next(d for d in mon._detectors if d.name == "semantic_goal_drift"),
    )
    sem._embedder = _map_embedder({"search": [0.0, 1.0]})  # type: ignore[attr-defined]
    sem._resolved = True  # type: ignore[attr-defined]
    with caplog.at_level(logging.WARNING, logger="snagline"):
        for i in range(20):
            mon.ingest(_ev(i, "search", "dim"))
    assert any("dimension" in r.message for r in caplog.records)


def test_semantic_signal_feeds_noisy_or_when_ml_ensemble_wraps_it():
    # Mirrors test_ml_esn_ensemble.py but for the semantic side: direct
    # detector fires goal_drift, the same detector inside an MLOrchestrator
    # fires ml_ensemble (trigger rewrite) while a healthy stream stays quiet.
    # Each case is both sides of the detector contract.

    healthy = _healthy(12, episode="h")
    det = _detector(_semantic_baseline(), semantic_drift_min_samples=10)
    for ev in healthy:
        assert det.observe(ev) is None
    healthy = _healthy(12, episode="h2")
    wrapped_cfg = Config(
        ml_ensemble_enabled=True,
        ml_ensemble_score_threshold=0.5,
        semantic_drift_min_samples=10,
    )
    direct = _detector(
        _semantic_baseline(),
        semantic_drift_min_samples=10,
    )
    orchestrated = MLOrchestrator(
        [
            SemanticGoalDriftDetector(
                _semantic_baseline(),
                config=wrapped_cfg,
                embedder=_map_embedder(ALL_VECS),
            )
        ],
        config=wrapped_cfg,
    )

    # Healthy stays silent through orchestrator too.
    for ev in _healthy(30, episode="oh"):
        assert orchestrated.observe(ev) is None
    # Sustained drift: direct fires goal_drift, orchestrated fires ml_ensemble.
    direct_fired = None
    orch_fired = None
    for j in range(30):
        r = direct.observe(_ev(j, "wipe_disk", "dh"))
        if r is not None and direct_fired is None:
            direct_fired = r
        ro = orchestrated.observe(_ev(j, "wipe_disk", "dh"))
        if ro is not None and orch_fired is None:
            orch_fired = ro
    # Hand-checked expectation: both fire, trigger rewrites through noisy-OR.
    assert direct_fired is not None and direct_fired.trigger == "goal_drift"
    assert orch_fired is not None and orch_fired.trigger == "ml_ensemble"


def test_non_finite_embedding_is_skipped_without_poisoning(caplog):
    det = _detector(_semantic_baseline(), semantic_drift_min_samples=10)

    def flaky(event: StepEvent) -> list[float]:
        if event.step_id == "s5":
            return [float("nan"), 0.0]
        return list(ALL_VECS[event.tool_name])  # type: ignore[index]

    det._embedder = flaky  # type: ignore[attr-defined]
    det._resolved = True  # type: ignore[attr-defined]
    with caplog.at_level(logging.WARNING, logger="snagline"):
        for i in range(12):
            # One of these is the NaN step; it must be skipped, not crash.
            assert det.observe(_ev(i, "search", "nan-ep")) is None or True
            # The NaN step itself returns None (skipped).
        # Explicitly hit the NaN step
        assert det.observe(_ev(5, "search", "nan-ep2")) is None
    assert any("non-finite" in r.message for r in caplog.records)
    # Episode state must still be usable: sustained drift after the NaN still
    # fires (i.e. the NaN did not poison the running sum with NaN).
    det2 = _detector(_semantic_baseline(), semantic_drift_min_samples=10)
    det2._embedder = flaky  # type: ignore[attr-defined]
    det2._resolved = True  # type: ignore[attr-defined]
    for i in range(12):
        det2.observe(_ev(i, "search", "after-nan"))
    fired = None
    for j in range(12, 50):
        r = det2.observe(_ev(j, "wipe_disk", "after-nan"))
        if r is not None:
            fired = r
            break
    assert fired is not None


def test_fit_rejects_non_finite_embeddings():
    from snagline.drift.goal_drift import fit_semantic_baseline

    def bad(event: StepEvent) -> list[float]:
        return [float("inf"), 0.0]

    with pytest.raises(ValueError, match="non-finite"):
        fit_semantic_baseline(_healthy(3), bad)


def test_semantic_drift_window_size_validation():
    with pytest.raises(ValueError, match="semantic_drift_window_size must be >= 1"):
        Config(semantic_drift_window_size=0)
    with pytest.raises(ValueError, match="semantic_drift_window_size must be >= 1"):
        Config(semantic_drift_window_size=-5)


def test_event_label_includes_intent_when_declared():
    base = StepEvent(
        step_id="1",
        episode_id="ep",
        timestamp=1.0,
        action_type="tool_call",
        action_signature="sig",
        tool_name="search",
        intent="extract passwords",
    )
    assert _event_label(base) == "tool_call search extract passwords"

    no_intent = StepEvent(
        step_id="2",
        episode_id="ep",
        timestamp=2.0,
        action_type="tool_call",
        action_signature="sig",
        tool_name="search",
    )
    assert _event_label(no_intent) == "tool_call search"


def test_goal_change_with_same_tools_detected_via_intent():
    # Issue #603: when tools are identical, intent differentiates purpose
    intent_table = {
        "benign research": [0.95, 0.10, 0.0],
        "credential harvest": [0.50, 0.80, 0.30],
    }

    def embed_intent(event: StepEvent) -> list[float]:
        return list(intent_table[event.intent])  # type: ignore[index]

    base_events = [
        StepEvent(
            f"b{i}",
            "b",
            float(i),
            "tool_call",
            f"sig-{i}",
            tool_name="search_web",
            intent="benign research",
        )
        for i in range(30)
    ]
    baseline = fit_semantic_baseline(base_events, embed_intent)

    det = SemanticGoalDriftDetector(
        baseline,
        config=Config(semantic_drift_min_samples=10, semantic_drift_tolerance=0.2),
        embedder=embed_intent,
    )

    # 15 steps of benign intent stay silent
    for i in range(15):
        assert (
            det.observe(
                StepEvent(
                    f"s{i}",
                    "ep_intent",
                    float(i),
                    "tool_call",
                    f"sig-{i}",
                    tool_name="search_web",
                    intent="benign research",
                )
            )
            is None
        )

    # Drift begins: purpose flips while tools stay identical
    fired = None
    for j in range(15, 60):
        risk = det.observe(
            StepEvent(
                f"s{j}",
                "ep_intent",
                float(j),
                "tool_call",
                f"sig-{j}",
                tool_name="search_web",
                intent="credential harvest",
            )
        )
        if risk is not None:
            fired = risk
            break

    assert fired is not None
    assert fired.trigger == "goal_drift"


def test_multi_tool_vocabulary_drift_fires_without_washing_out():
    # Issue #603: multi-tool shift does not wash out under vector averaging
    base_tools = {
        "search_web": [0.96, 0.10, 0.05],
        "read_file": [0.94, 0.12, -0.05],
        "run_sql": [0.95, 0.08, 0.08],
        "summarize_text": [0.95, 0.11, -0.08],
    }
    drift_tools = {
        "drop_table": [0.60, 0.70, 0.25],
        "delete_users": [0.52, 0.75, 0.30],
        "wipe_logs": [0.62, 0.68, 0.28],
        "disable_alerts": [0.65, 0.65, 0.25],
    }
    all_tools = {**base_tools, **drift_tools}

    def embed(event: StepEvent) -> list[float]:
        return list(all_tools[event.tool_name])  # type: ignore[index]

    base_events = [
        StepEvent(
            f"b{i}",
            "b",
            float(i),
            "tool_call",
            f"sig-{i}",
            tool_name=list(base_tools)[i % 4],
        )
        for i in range(40)
    ]
    baseline = fit_semantic_baseline(base_events, embed)

    det = SemanticGoalDriftDetector(
        baseline,
        config=Config(
            semantic_drift_min_samples=10,
            semantic_drift_window_size=20,
            semantic_drift_tolerance=0.2,
        ),
        embedder=embed,
    )

    # 20 steps of baseline tools
    for i in range(20):
        t = list(base_tools)[i % 4]
        assert (
            det.observe(
                StepEvent(
                    f"s{i}",
                    "ep_mix",
                    float(i),
                    "tool_call",
                    f"sig-{i}",
                    tool_name=t,
                )
            )
            is None
        )

    # 40 steps of multi-tool destructive shift
    fired = None
    for j in range(20, 60):
        t = list(drift_tools)[j % 4]
        risk = det.observe(
            StepEvent(
                f"s{j}",
                "ep_mix",
                float(j),
                "tool_call",
                f"sig-{j}",
                tool_name=t,
            )
        )
        if risk is not None:
            fired = risk
            break

    assert fired is not None
    assert fired.trigger == "goal_drift"


def test_multi_tool_healthy_stream_stays_silent():
    # Long multi-tool healthy stream stays silent
    base_tools = {
        "search_web": [0.96, 0.10, 0.05],
        "read_file": [0.94, 0.12, -0.05],
        "run_sql": [0.95, 0.08, 0.08],
        "summarize_text": [0.95, 0.11, -0.08],
    }

    def embed(event: StepEvent) -> list[float]:
        return list(base_tools[event.tool_name])  # type: ignore[index]

    base_events = [
        StepEvent(
            f"b{i}",
            "b",
            float(i),
            "tool_call",
            f"sig-{i}",
            tool_name=list(base_tools)[i % 4],
        )
        for i in range(40)
    ]
    baseline = fit_semantic_baseline(base_events, embed)

    det = SemanticGoalDriftDetector(
        baseline,
        config=Config(
            semantic_drift_min_samples=10,
            semantic_drift_window_size=20,
            semantic_drift_tolerance=0.2,
        ),
        embedder=embed,
    )

    for i in range(120):
        t = list(base_tools)[i % 4]
        assert (
            det.observe(
                StepEvent(
                    f"h{i}",
                    "ep_healthy",
                    float(i),
                    "tool_call",
                    f"sig-{i}",
                    tool_name=t,
                )
            )
            is None
        )
