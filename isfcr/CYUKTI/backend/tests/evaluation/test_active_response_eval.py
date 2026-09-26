from evaluators.active_response_eval import evaluate, ActiveResponseGroundTruthSchema
from evaluators.base import MetricStatus


def test_active_response_eval_reports_blocked_by_environment_not_a_fabricated_number():
    result = evaluate()
    assert result.status == MetricStatus.BLOCKED_BY_ENVIRONMENT
    assert result.n == 0
    assert "metrics" not in result.to_dict() or not result.to_dict().get("metrics")


def test_ground_truth_schema_has_no_default_labels_prefilled():
    schema = ActiveResponseGroundTruthSchema(sample_id="S1", correlation_id="corr-1", scenario_id="SCN-1")
    assert schema.detection_success is None
    assert schema.reviewer == "unreviewed"
