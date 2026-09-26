import os

import pytest

import ground_truth.store as store
from ground_truth.schema import GroundTruthRecord, ReviewStatus


@pytest.fixture(autouse=True)
def _isolate_store_dirs(tmp_path, monkeypatch):
    """Redirect the store's directories into a tmp_path so tests never
    touch the real evaluation/ground_truth/{provisional,reviewed,locked}
    directories that hold real project data."""
    fake_dirs = {
        ReviewStatus.AUTO_PROPOSED: str(tmp_path / "provisional"),
        ReviewStatus.HUMAN_REVIEWED: str(tmp_path / "reviewed"),
        ReviewStatus.LOCKED: str(tmp_path / "locked"),
    }
    monkeypatch.setattr(store, "_DIRS", fake_dirs)
    yield


def _record(**overrides) -> GroundTruthRecord:
    defaults = dict(
        sample_id="S1", source="wazuh_alert", timestamp="t",
        scenario_id="SCN-NMAP-001", raw_event_id="rule=100500",
        expected_mitre_techniques=["T1595"],
    )
    defaults.update(overrides)
    return GroundTruthRecord(**defaults)


def test_save_and_load_round_trip():
    records = [_record(sample_id="S1"), _record(sample_id="S2")]
    store.save_records("mitre_v0", records)
    loaded = store.load_records("mitre_v0", ReviewStatus.AUTO_PROPOSED)
    assert {r.sample_id for r in loaded} == {"S1", "S2"}


def test_load_missing_dataset_returns_empty_list_not_crash():
    assert store.load_records("does_not_exist", ReviewStatus.AUTO_PROPOSED) == []


def test_mixed_review_statuses_in_one_save_call_is_rejected():
    r1 = _record(sample_id="S1", review_status=ReviewStatus.AUTO_PROPOSED)
    r2 = _record(sample_id="S2", review_status=ReviewStatus.HUMAN_REVIEWED, reviewer="jane")
    with pytest.raises(ValueError, match="mixed review statuses"):
        store.save_records("bad", [r1, r2])


def test_load_best_available_prefers_locked_over_reviewed_over_auto_proposed():
    store.save_records("layered", [_record(sample_id="A", review_status=ReviewStatus.AUTO_PROPOSED)])
    records, status = store.load_best_available("layered")
    assert status == ReviewStatus.AUTO_PROPOSED

    store.save_records("layered", [_record(sample_id="B", review_status=ReviewStatus.HUMAN_REVIEWED, reviewer="jane")])
    records, status = store.load_best_available("layered")
    assert status == ReviewStatus.HUMAN_REVIEWED
    assert records[0].sample_id == "B"


def test_lock_dataset_refuses_when_nothing_is_human_reviewed():
    store.save_records("unreviewed_only", [_record(sample_id="S1")])  # AUTO_PROPOSED
    with pytest.raises(ValueError, match="nothing to lock"):
        store.lock_dataset("unreviewed_only", reviewer="jane", source_commit="abc123")


def test_lock_dataset_succeeds_on_human_reviewed_data_and_writes_manifest():
    store.save_records("ready", [_record(sample_id="S1", review_status=ReviewStatus.HUMAN_REVIEWED, reviewer="jane")])
    manifest = store.lock_dataset("ready", reviewer="jane", source_commit="abc123")
    assert manifest.record_count == 1
    assert manifest.reviewer == "jane"
    locked = store.load_records("ready", ReviewStatus.LOCKED)
    assert len(locked) == 1
    assert locked[0].review_status == ReviewStatus.LOCKED
