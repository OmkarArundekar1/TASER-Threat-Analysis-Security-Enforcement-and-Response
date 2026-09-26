import json

import pytest

from evaluators import investigation_eval
from evaluators.base import MetricStatus


@pytest.fixture(autouse=True)
def _isolate_verdicts_path(tmp_path, monkeypatch):
    path = tmp_path / "investigation_verdicts.json"
    monkeypatch.setattr(investigation_eval, "_VERDICTS_PATH", str(path))
    yield path


def test_no_verdicts_reports_not_measured():
    result = investigation_eval.evaluate()
    assert result.status == MetricStatus.NOT_MEASURED


def test_unreviewed_verdict_requires_review(_isolate_verdicts_path):
    _isolate_verdicts_path.write_text(json.dumps([
        {"investigation_id": "I1", "verdict": "CORRECT", "reviewer": "unreviewed", "investigation_confidence": 0.8},
    ]))
    result = investigation_eval.evaluate()
    assert result.status == MetricStatus.GROUND_TRUTH_REVIEW_REQUIRED


def test_reviewed_verdicts_compute_real_metrics(_isolate_verdicts_path):
    _isolate_verdicts_path.write_text(json.dumps([
        {"investigation_id": "I1", "verdict": "CORRECT", "reviewer": "jane", "investigation_confidence": 0.9},
        {"investigation_id": "I2", "verdict": "INCORRECT", "reviewer": "jane", "investigation_confidence": 0.2},
        {"investigation_id": "I3", "verdict": "PARTIALLY_CORRECT", "reviewer": "jane", "investigation_confidence": 0.5},
    ]))
    result = investigation_eval.evaluate()
    assert result.status == MetricStatus.MEASURED
    assert result.metrics["correct"] == 1
    assert result.metrics["incorrect"] == 1
    assert result.metrics["partially_correct"] == 1
    assert result.metrics["accuracy_on_binary_subset"] == 0.5
    assert "brier_score" in result.metrics
