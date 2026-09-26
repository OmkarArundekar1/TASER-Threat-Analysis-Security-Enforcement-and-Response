import pytest

import ground_truth.store as store
from evaluators import threat_qualification_eval
from evaluators.base import MetricStatus
from ground_truth.schema import GroundTruthRecord, ReviewStatus
from tests.evaluation.fake_neo4j import FakeDriver


@pytest.fixture(autouse=True)
def _isolate_store(tmp_path, monkeypatch):
    fake_dirs = {
        ReviewStatus.AUTO_PROPOSED: str(tmp_path / "provisional"),
        ReviewStatus.HUMAN_REVIEWED: str(tmp_path / "reviewed"),
        ReviewStatus.LOCKED: str(tmp_path / "locked"),
    }
    monkeypatch.setattr(store, "_DIRS", fake_dirs)
    monkeypatch.setattr(threat_qualification_eval.store, "_DIRS", fake_dirs)
    yield


def _record(sample_id, campaign_id, expected):
    return GroundTruthRecord(
        sample_id=sample_id, source="neo4j_campaign", timestamp="",
        scenario_id="SCN-NMAP-001", raw_event_id=campaign_id,
        expected_threat_status=expected,
    )


def test_no_dataset_reports_not_measured():
    result = threat_qualification_eval.evaluate(dataset_name="does_not_exist")
    assert result.status == MetricStatus.NOT_MEASURED


def test_perfect_agreement_with_real_thresholds(monkeypatch):
    store.save_records("tq_agree", [_record("S1", "CAMP_A", "QUALIFIED_THREAT")])
    monkeypatch.setattr("neo4j_client.driver", FakeDriver(lambda q, p: [{"cti_score": 90.0}]))
    result = threat_qualification_eval.evaluate(dataset_name="tq_agree", allow_preliminary=True)
    assert result.status == MetricStatus.MEASURED_PRELIMINARY
    assert result.metrics["accuracy"] == 1.0


def test_disagreement_is_reported_honestly_not_hidden(monkeypatch):
    store.save_records("tq_disagree", [_record("S1", "CAMP_A", "SUSPICIOUS")])
    monkeypatch.setattr("neo4j_client.driver", FakeDriver(lambda q, p: [{"cti_score": 90.0}]))
    result = threat_qualification_eval.evaluate(dataset_name="tq_disagree", allow_preliminary=True)
    assert result.metrics["accuracy"] == 0.0


def test_campaign_with_no_live_score_is_skipped_and_reported(monkeypatch):
    store.save_records("tq_noscore", [_record("S1", "CAMP_A", "SUSPICIOUS")])
    monkeypatch.setattr("neo4j_client.driver", FakeDriver(lambda q, p: [{"cti_score": None}]))
    result = threat_qualification_eval.evaluate(dataset_name="tq_noscore", allow_preliminary=True)
    assert result.status == MetricStatus.UNMEASURABLE
    assert "1 ground-truth record" in result.reason


def test_strict_mode_requires_review():
    store.save_records("tq_strict", [_record("S1", "CAMP_A", "SUSPICIOUS")])
    result = threat_qualification_eval.evaluate(dataset_name="tq_strict", allow_preliminary=False)
    assert result.status == MetricStatus.GROUND_TRUTH_REVIEW_REQUIRED
