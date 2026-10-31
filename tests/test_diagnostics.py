import pytest
from snagline.diagnostics import DiagnosticMetrics

def test_diagnostic_metrics_counters():
    metrics = DiagnosticMetrics()
    metrics.record_counter("step_count")
    metrics.record_counter("step_count", 2)
    assert metrics.counters["step_count"] == 3

def test_diagnostic_metrics_timings():
    metrics = DiagnosticMetrics()
    metrics.record_timing("step_latency_ms", 12.5)
    metrics.record_timing("step_latency_ms", 18.0)
    assert len(metrics.timings["step_latency_ms"]) == 2

def test_diagnostic_metrics_ring_buffer_cap():
    metrics = DiagnosticMetrics(max_samples=3)
    for i in range(5):
        metrics.record_timing("latency", float(i))
    assert metrics.timings["latency"] == [2.0, 3.0, 4.0]

def test_diagnostic_metrics_percentile():
    metrics = DiagnosticMetrics()
    for val in [10.0, 20.0, 30.0, 40.0, 50.0]:
        metrics.record_timing("lat", val)
    assert metrics.percentile("lat", 50.0) == 30.0
    assert metrics.percentile("lat", 0.0) == 10.0
    assert metrics.percentile("lat", 100.0) == 50.0
    assert metrics.percentile("missing", 50.0) is None

def test_diagnostic_metrics_invalid_max_samples():
    for invalid_val in [0, -1, -100, float("nan"), float("inf"), float("-inf"), 3.14, True, False, "1000", None]:
        with pytest.raises(ValueError, match="max_samples must be a positive integer"):
            DiagnosticMetrics(max_samples=invalid_val)  # type: ignore

def test_diagnostic_metrics_valid_max_samples_boundary():
    metrics = DiagnosticMetrics(max_samples=1)
    metrics.record_timing("lat", 10.0)
    metrics.record_timing("lat", 20.0)
    assert metrics.timings["lat"] == [20.0]

