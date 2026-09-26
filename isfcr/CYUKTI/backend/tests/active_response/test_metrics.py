from active_response.metrics import latency_stats, rate_stats, compute_lifecycle_latencies, ResponseLifecycleTimestamps


def test_latency_stats_with_no_observations_is_not_measured():
    result = latency_stats([])
    assert result.status == "NOT_MEASURED"
    assert result.p50_seconds is None


def test_latency_stats_with_real_observations():
    pairs = [
        ("2026-01-01T00:00:00+00:00", "2026-01-01T00:00:01+00:00"),
        ("2026-01-01T00:00:00+00:00", "2026-01-01T00:00:03+00:00"),
        ("2026-01-01T00:00:00+00:00", "2026-01-01T00:00:05+00:00"),
    ]
    result = latency_stats(pairs)
    assert result.status == "MEASURED"
    assert result.n == 3
    assert result.p50_seconds == 3.0
    assert result.mean_seconds == (1 + 3 + 5) / 3


def test_rate_stats_with_no_observations_is_not_measured():
    result = rate_stats([])
    assert result.status == "NOT_MEASURED"
    assert result.rate is None


def test_rate_stats_computes_real_wilson_interval():
    result = rate_stats([True, True, True, False])
    assert result.status == "MEASURED"
    assert result.successes == 3
    assert result.rate == 0.75
    assert result.confidence_interval_95["low"] < 0.75 < result.confidence_interval_95["high"]


def test_lifecycle_latencies_report_not_measured_per_stage_independently():
    records = [ResponseLifecycleTimestamps(correlation_id="c1", detected_at="2026-01-01T00:00:00+00:00",
                                            decided_at="2026-01-01T00:00:02+00:00")]
    result = compute_lifecycle_latencies(records)
    assert result["detection_to_decision"]["status"] == "MEASURED"
    assert result["containment_execution"]["status"] == "NOT_MEASURED"
