import csv
import json

import pytest

import ground_truth.store as store
from ground_truth.schema import GroundTruthRecord, ReviewStatus
from review import import_reviewed


@pytest.fixture(autouse=True)
def _isolate(tmp_path, monkeypatch):
    fake_dirs = {
        ReviewStatus.AUTO_PROPOSED: str(tmp_path / "provisional"),
        ReviewStatus.HUMAN_REVIEWED: str(tmp_path / "reviewed"),
        ReviewStatus.LOCKED: str(tmp_path / "locked"),
    }
    monkeypatch.setattr(store, "_DIRS", fake_dirs)
    monkeypatch.setattr(import_reviewed, "_DIR", str(tmp_path))
    monkeypatch.setattr(import_reviewed, "_LABELS_DIR", str(tmp_path / "labels"))
    yield tmp_path


def _mitre_record(sample_id):
    return GroundTruthRecord(
        sample_id=sample_id, source="wazuh_alert", timestamp="",
        scenario_id="SCN-NMAP-001", raw_event_id=f"rule=100500|{sample_id}",
        expected_mitre_techniques=["T1595"],
    )


def _write_mitre_csv(tmp_path, rows):
    path = tmp_path / "mitre_review_queue_mitre_mapping_v0.csv"
    fields = ["sample_id", "decision (fill in: ACCEPT|MODIFY|REJECT|UNCERTAIN)",
              "corrected_techniques (fill in only if MODIFY, comma-separated ATT&CK IDs)", "reviewer"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return path


def test_mitre_import_promotes_only_rows_with_reviewer_and_accept_or_modify(tmp_path):
    store.save_records("mitre_mapping_v0", [_mitre_record("S1"), _mitre_record("S2"), _mitre_record("S3")])
    _write_mitre_csv(tmp_path, [
        {"sample_id": "S1", "decision (fill in: ACCEPT|MODIFY|REJECT|UNCERTAIN)": "ACCEPT",
         "corrected_techniques (fill in only if MODIFY, comma-separated ATT&CK IDs)": "", "reviewer": "jane"},
        {"sample_id": "S2", "decision (fill in: ACCEPT|MODIFY|REJECT|UNCERTAIN)": "UNCERTAIN",
         "corrected_techniques (fill in only if MODIFY, comma-separated ATT&CK IDs)": "", "reviewer": "jane"},
        {"sample_id": "S3", "decision (fill in: ACCEPT|MODIFY|REJECT|UNCERTAIN)": "",
         "corrected_techniques (fill in only if MODIFY, comma-separated ATT&CK IDs)": "", "reviewer": ""},
    ])

    promoted, skipped = import_reviewed.import_mitre_reviews()
    assert promoted == 1
    assert skipped == 2

    reviewed = store.load_records("mitre_mapping_v0", ReviewStatus.HUMAN_REVIEWED)
    assert len(reviewed) == 1
    assert reviewed[0].sample_id == "S1"
    assert reviewed[0].reviewer == "jane"


def test_mitre_import_modify_overrides_expected_techniques(tmp_path):
    store.save_records("mitre_mapping_v0", [_mitre_record("S1")])
    _write_mitre_csv(tmp_path, [
        {"sample_id": "S1", "decision (fill in: ACCEPT|MODIFY|REJECT|UNCERTAIN)": "MODIFY",
         "corrected_techniques (fill in only if MODIFY, comma-separated ATT&CK IDs)": "T1110, T1110.001",
         "reviewer": "jane"},
    ])
    promoted, _ = import_reviewed.import_mitre_reviews()
    assert promoted == 1
    reviewed = store.load_records("mitre_mapping_v0", ReviewStatus.HUMAN_REVIEWED)
    assert reviewed[0].expected_mitre_techniques == ["T1110", "T1110.001"]


def test_mitre_import_empty_csv_promotes_nothing(tmp_path):
    store.save_records("mitre_mapping_v0", [_mitre_record("S1")])
    _write_mitre_csv(tmp_path, [])
    promoted, skipped = import_reviewed.import_mitre_reviews()
    assert promoted == 0
    assert skipped == 0
    assert store.load_records("mitre_mapping_v0", ReviewStatus.HUMAN_REVIEWED) == []


def test_mitre_import_missing_csv_raises_not_silently_pretends_reviewed():
    with pytest.raises(FileNotFoundError):
        import_reviewed.import_mitre_reviews()


def test_campaign_correlation_pairwise_import_skips_uncertain(tmp_path):
    path = tmp_path / "campaign_correlation_review_queue_campaign_correlation_v0.csv"
    fields = ["campaign_a", "campaign_b", "same_campaign_ground_truth (fill in: SAME|DIFFERENT|UNCERTAIN)", "reviewer"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerow({"campaign_a": "CAMP_1", "campaign_b": "CAMP_2",
                          "same_campaign_ground_truth (fill in: SAME|DIFFERENT|UNCERTAIN)": "SAME", "reviewer": "jane"})
        writer.writerow({"campaign_a": "CAMP_3", "campaign_b": "CAMP_4",
                          "same_campaign_ground_truth (fill in: SAME|DIFFERENT|UNCERTAIN)": "UNCERTAIN", "reviewer": "jane"})

    n = import_reviewed.import_campaign_correlation_reviews()
    assert n == 1
    out_path = tmp_path / "labels" / "campaign_correlation_v0_pairwise_reviewed.json"
    data = json.loads(out_path.read_text())
    assert len(data) == 1
    assert data[0]["same_campaign"] is True
