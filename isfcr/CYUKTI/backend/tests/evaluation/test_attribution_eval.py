import pytest

import ground_truth.store as store
from evaluators import attribution_eval
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
    monkeypatch.setattr(attribution_eval.store, "_DIRS", fake_dirs)
    yield


def _record(sample_id, campaign_id, expected_ip):
    return GroundTruthRecord(
        sample_id=sample_id, source="neo4j_campaign", timestamp="",
        scenario_id="SCN-TEST", raw_event_id=campaign_id,
        expected_attribution=expected_ip,
    )


def test_no_dataset_reports_not_measured():
    result = attribution_eval.evaluate(dataset_name="does_not_exist")
    assert result.status == MetricStatus.NOT_MEASURED


def test_campaign_with_no_techniques_is_skipped(monkeypatch):
    store.save_records("attr_empty", [_record("S1", "CAMP_A", "1.2.3.4")])

    def router(query, params):
        if "attacker_ip" in query and "campaign_id:$id" in query:
            return [{"attacker_ip": "1.2.3.4", "victim_ip": "v", "last_technique": None}]
        if "MATCHES" in query:
            return []  # no techniques observed -- attribute() has nothing to score
        return []

    monkeypatch.setattr("neo4j_client.driver", FakeDriver(router))
    result = attribution_eval.evaluate(dataset_name="attr_empty", allow_preliminary=True)
    assert result.status == MetricStatus.UNMEASURABLE


def test_strict_mode_requires_review():
    store.save_records("attr_strict", [_record("S1", "CAMP_A", "1.2.3.4")])
    result = attribution_eval.evaluate(dataset_name="attr_strict", allow_preliminary=False)
    assert result.status == MetricStatus.GROUND_TRUTH_REVIEW_REQUIRED
