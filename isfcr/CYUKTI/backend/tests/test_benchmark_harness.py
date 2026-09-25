"""
Tests for benchmarks/run_benchmarks.py's own measurement logic
(percentile computation, skip-vs-measure detection) -- not for any
specific subsystem's real-world latency, which is environment-
dependent and reported separately in benchmarks/results/*.json.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "benchmarks"))

from run_benchmarks import _percentile, _time_it  # noqa: E402


def test_percentile_of_empty_list_is_zero_not_a_crash():
    assert _percentile([], 50) == 0.0


def test_percentile_p50_of_sorted_values():
    values = [1.0, 2.0, 3.0, 4.0, 5.0]
    assert _percentile(values, 50) == 3.0


def test_percentile_p100_is_the_maximum():
    values = [1.0, 2.0, 3.0]
    assert _percentile(values, 100) == 3.0


def test_percentile_p0_is_the_minimum():
    values = [1.0, 2.0, 3.0]
    assert _percentile(values, 0) == 1.0


def test_time_it_reports_measured_for_a_function_that_succeeds():
    result = _time_it(lambda: None, iterations=5)
    assert result["status"] == "measured"
    assert result["iterations"] == 5
    assert result["p50_ms"] >= 0
    assert result["throughput_per_sec"] is not None


def test_time_it_reports_skipped_not_a_crash_when_the_function_fails_on_warmup():
    def _always_fails():
        raise RuntimeError("simulated unavailable subsystem")

    result = _time_it(_always_fails, iterations=5)
    assert result["status"] == "skipped"
    assert "simulated unavailable subsystem" in result["reason"]


def test_time_it_never_fabricates_a_latency_for_a_skipped_benchmark():
    def _always_fails():
        raise RuntimeError("down")

    result = _time_it(_always_fails, iterations=5)
    assert "p50_ms" not in result
    assert "p95_ms" not in result
