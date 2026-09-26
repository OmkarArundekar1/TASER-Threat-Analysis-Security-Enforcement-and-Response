import pytest

import ground_truth.store as store
from evaluators import campaign_correlation_eval
from evaluators.base import MetricStatus
from ground_truth.schema import GroundTruthRecord, ReviewStatus


@pytest.fixture(autouse=True)
def _isolate_store(tmp_path, monkeypatch):
    fake_dirs = {
        ReviewStatus.AUTO_PROPOSED: str(tmp_path / "provisional"),
        ReviewStatus.HUMAN_REVIEWED: str(tmp_path / "reviewed"),
        ReviewStatus.LOCKED: str(tmp_path / "locked"),
    }
    monkeypatch.setattr(store, "_DIRS", fake_dirs)
    monkeypatch.setattr(campaign_correlation_eval.store, "_DIRS", fake_dirs)
    yield


def _boundary_record(sample_id, session_key, campaign_ids):
    return GroundTruthRecord(
        sample_id=sample_id, source="scenario_session_boundary", timestamp="",
        scenario_id="SCN-TEST", raw_event_id="|".join(campaign_ids),
        expected_campaign_id=session_key,
    )


def test_no_dataset_reports_not_measured():
    result = campaign_correlation_eval.evaluate(dataset_name="does_not_exist")
    assert result.status == MetricStatus.NOT_MEASURED


def test_single_real_campaign_across_all_sessions_is_unmeasurable():
    store.save_records("cc_tiny", [_boundary_record("S1", "SESSION::A", ["CAMP_1"])])
    result = campaign_correlation_eval.evaluate(dataset_name="cc_tiny", allow_preliminary=True)
    assert result.status == MetricStatus.UNMEASURABLE


def test_perfect_merge_scores_one_on_all_metrics():
    """Ground truth says CAMP_1 and CAMP_2 are the same real session, but
    CYUKTI put them in different Campaign nodes -- this dataset shape
    can only ever show fragmentation, never a 'perfect merge' in
    CYUKTI's own output (each row IS a distinct real Campaign node by
    construction) -- so this test instead verifies the true-fragmentation
    case is scored honestly, not silently smoothed over."""
    store.save_records("cc_frag", [_boundary_record("S1", "SESSION::A", ["CAMP_1", "CAMP_2"])])
    result = campaign_correlation_eval.evaluate(dataset_name="cc_frag", allow_preliminary=True)
    assert result.status == MetricStatus.MEASURED_PRELIMINARY
    assert result.metrics["n_real_campaigns"] == 2
    assert result.metrics["fragmentation_per_true_session"]["SESSION::A"] == 2
    # never silently claims a merge that didn't happen:
    assert result.metrics["pairwise_recall"] == 0.0


def test_two_distinct_sessions_each_correctly_kept_separate():
    store.save_records("cc_separate", [
        _boundary_record("S1", "SESSION::A", ["CAMP_1"]),
        _boundary_record("S2", "SESSION::B", ["CAMP_2"]),
    ])
    result = campaign_correlation_eval.evaluate(dataset_name="cc_separate", allow_preliminary=True)
    assert result.metrics["cluster_purity"] == 1.0


def test_strict_mode_requires_review():
    store.save_records("cc_strict", [_boundary_record("S1", "SESSION::A", ["CAMP_1", "CAMP_2"])])
    result = campaign_correlation_eval.evaluate(dataset_name="cc_strict", allow_preliminary=False)
    assert result.status == MetricStatus.GROUND_TRUTH_REVIEW_REQUIRED
