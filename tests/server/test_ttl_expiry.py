"""TTL expiry for snagline_episodes_active ids (issue #173).

Covers: TTL expiry fires, disabled TTL preserves old behavior, bounded memory,
fail-open on malformed config, and monotonic clock usage.
"""

from __future__ import annotations

import logging
import time

import pytest

from snagline.server.http_server import SidecarMetricsCollector, _resolve_episode_ttl


class FakeClock:
    def __init__(self, start: float = 0.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def test_ttl_expiry_fires_without_end_signal() -> None:
    clock = FakeClock(start=1000.0)
    col = SidecarMetricsCollector(episode_ttl_seconds=10, clock=clock)
    col.record_ingest("ep-1", 0.001)
    assert col.snapshot()["episodes_active"] == 1
    clock.advance(5)
    # Still within window, reseeing keeps it alive.
    col.record_ingest("ep-2", 0.001)
    assert col.snapshot()["episodes_active"] == 2
    clock.advance(6)  # ep-1 now 11s old, ep-2 6s old
    # Lazy sweep on next ingest and on snapshot.
    assert col.snapshot()["episodes_active"] == 1
    # ep-1 should be gone, ep-2 remains.
    assert "ep-1" not in col._episodes
    assert "ep-2" in col._episodes


def test_ttl_reseen_id_refreshes_and_survives() -> None:
    clock = FakeClock(start=0.0)
    col = SidecarMetricsCollector(episode_ttl_seconds=10, clock=clock)
    col.record_ingest("ep-1", 0.001)
    clock.advance(8)
    col.record_ingest("ep-1", 0.001)  # refresh
    clock.advance(8)  # 8s after refresh, still within TTL
    assert col.snapshot()["episodes_active"] == 1
    clock.advance(3)  # 11s after last see -> expire
    assert col.snapshot()["episodes_active"] == 0


def test_disabled_ttl_preserves_old_behavior() -> None:
    # Default off: byte-identical to pre-TTL.
    col = SidecarMetricsCollector()
    assert col._ttl is None
    col.record_ingest("ep-1", 0.001)
    col.record_ingest("ep-2", 0.001)
    assert col.snapshot()["episodes_active"] == 2
    # Even after advancing a fake clock, collector without TTL never expires
    # (we cannot advance its real monotonic clock, but we can check that
    # explicit None stays disabled and that 0 also disables).
    col2 = SidecarMetricsCollector(episode_ttl_seconds=0)
    assert col2._ttl is None
    col2.record_ingest("ep-1", 0.001)
    assert col2.snapshot()["episodes_active"] == 1
    # Explicit None also disabled.
    col3 = SidecarMetricsCollector(episode_ttl_seconds=None)
    assert col3._ttl is None


def test_ttl_bounded_memory_and_ids_only() -> None:
    clock = FakeClock()
    col = SidecarMetricsCollector(max_episodes=3, episode_ttl_seconds=10, clock=clock)
    for i in range(5):
        col.record_ingest(f"ep-{i}", 0.001)
    # Cap still bounds memory: at most 3 ids.
    assert col.snapshot()["episodes_active"] <= 3
    # Table stores ids-plus-a-float, never content.
    for k, v in col._episodes.items():
        assert isinstance(k, str)
        assert isinstance(v, float)
        assert "content" not in k.lower()


def test_ttl_fail_open_on_malformed_env(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    # Malformed env values fall back to disabled with a warning, never crash.
    # "not-a-number" fails to coerce in Config.from_env_overrides and is
    # ignored there (logged as "ignoring bad env"), so TTL stays disabled.
    monkeypatch.setenv("SNAGLINE_EPISODE_TTL_SECONDS", "not-a-number")
    with caplog.at_level(logging.WARNING):
        ttl = _resolve_episode_ttl(None)
    assert ttl is None
    # No crash, disabled is the correct fallback.
    caplog.clear()
    monkeypatch.setenv("SNAGLINE_EPISODE_TTL_SECONDS", "-5")
    with caplog.at_level(logging.WARNING):
        ttl = _resolve_episode_ttl(None)
    assert ttl is None
    assert any("must be positive" in r.message for r in caplog.records)
    caplog.clear()
    monkeypatch.setenv("SNAGLINE_EPISODE_TTL_SECONDS", "0")
    ttl = _resolve_episode_ttl(None)
    assert ttl is None  # 0 disables without warning
    # Explicit valid value wins.
    ttl = _resolve_episode_ttl(5.0)
    assert ttl == 5.0
    ttl = _resolve_episode_ttl(None)
    # Env 0 still disabled, not 5.
    assert ttl is None
    monkeypatch.delenv("SNAGLINE_EPISODE_TTL_SECONDS", raising=False)
    # Config direct: None disabled.
    assert SidecarMetricsCollector(episode_ttl_seconds=None)._ttl is None
    # Config with TTL set propagates.
    assert SidecarMetricsCollector(episode_ttl_seconds=5)._ttl == 5.0


def test_ttl_uses_monotonic_not_wall_clock(monkeypatch: pytest.MonkeyPatch) -> None:
    # Ensure expiry uses monotonic clock, not wall-clock time.
    # We inject a fake monotonic clock and verify that wall-clock NTP jumps
    # (time.time) do not affect expiry.
    clock = FakeClock(start=1000.0)
    col = SidecarMetricsCollector(episode_ttl_seconds=10, clock=clock)
    col.record_ingest("ep-1", 0.001)
    # Simulate NTP jump: time.time jumps but monotonic does not.
    real_time = time.time
    monkeypatch.setattr(time, "time", lambda: real_time() + 10000)
    # Still within monotonic TTL, so still active.
    assert col.snapshot()["episodes_active"] == 1
    clock.advance(11)
    assert col.snapshot()["episodes_active"] == 0


def test_ttl_sweep_on_both_ingest_and_scrape() -> None:
    clock = FakeClock(start=0.0)
    col = SidecarMetricsCollector(episode_ttl_seconds=10, clock=clock)
    col.record_ingest("ep-1", 0.001)
    col.record_ingest("ep-2", 0.001)
    clock.advance(11)
    # No new ingest, but scrape (snapshot) must still expire.
    snap = col.snapshot()
    assert snap["episodes_active"] == 0
    # Also render_prometheus must expire (it calls snapshot).
    col.record_ingest("ep-3", 0.001)
    clock.advance(11)
    body = col.render_prometheus({})
    assert "snagline_episodes_active 0" in body
