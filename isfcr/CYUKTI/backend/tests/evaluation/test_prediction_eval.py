import pytest

from evaluators import prediction_eval
from evaluators.base import MetricStatus
from tests.evaluation.fake_neo4j import FakeDriver


def test_small_live_graph_reuses_historical_phase18_result_honestly(monkeypatch):
    def router(query, params):
        if "count(r)" in query:
            return [{"n": 3}]
        return [{"src": "T1110.001", "dst": "T1110", "count": 6, "confidence": 1.0}]

    monkeypatch.setattr("neo4j_client.driver", FakeDriver(router))
    result = prediction_eval.evaluate()
    assert result.status == MetricStatus.UNMEASURABLE
    assert result.metrics["historical_phase18_result"]["correct"] == 4
    assert result.metrics["historical_phase18_result"]["evaluable_predictions"] == 12
    assert "INSUFFICIENT_FOR_SUPERVISED_ML" in result.reason


def test_grown_graph_reports_not_measured_not_a_fabricated_score(monkeypatch):
    def router(query, params):
        if "count(r)" in query:
            return [{"n": 50}]
        return []

    monkeypatch.setattr("neo4j_client.driver", FakeDriver(router))
    result = prediction_eval.evaluate()
    assert result.status == MetricStatus.NOT_MEASURED
    assert result.metrics["live_next_technique_edge_count"] == 50
