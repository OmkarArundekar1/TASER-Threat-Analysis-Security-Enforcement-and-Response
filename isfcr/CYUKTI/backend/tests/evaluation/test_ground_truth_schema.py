import pytest

from ground_truth.schema import GroundTruthRecord, ReviewStatus, record_hash


def _make_record(**overrides) -> GroundTruthRecord:
    defaults = dict(
        sample_id="S1",
        source="wazuh_alert",
        timestamp="2026-09-26T00:00:00+00:00",
        scenario_id="SCN-NMAP-001",
        raw_event_id="rule=100500|agent=001",
        attacker_identity="192.168.56.106",
        victim_identity="192.168.56.105",
        expected_attack="Nmap scan",
        expected_mitre_techniques=["T1595"],
        expected_threat_status="SUSPICIOUS",
    )
    defaults.update(overrides)
    return GroundTruthRecord(**defaults)


def test_valid_auto_proposed_record_passes_validation():
    record = _make_record()
    record.validate()  # must not raise


def test_system_output_marker_in_expected_field_is_rejected():
    record = _make_record(expected_attack="__FROM_MITRE_RESOLVER__")
    with pytest.raises(ValueError, match="system-output marker"):
        record.validate()


def test_system_output_marker_in_technique_list_is_rejected():
    record = _make_record(expected_mitre_techniques=["__FROM_MITRE_RESOLVER__"])
    with pytest.raises(ValueError, match="system-output marker"):
        record.validate()


def test_promoting_past_auto_proposed_without_reviewer_is_rejected():
    record = _make_record(review_status=ReviewStatus.HUMAN_REVIEWED, reviewer="unreviewed")
    with pytest.raises(ValueError, match="no reviewer is named"):
        record.validate()


def test_promoting_past_auto_proposed_with_named_reviewer_is_accepted():
    record = _make_record(review_status=ReviewStatus.HUMAN_REVIEWED, reviewer="jane_analyst")
    record.validate()  # must not raise


def test_round_trip_to_dict_and_back():
    record = _make_record()
    restored = GroundTruthRecord.from_dict(record.to_dict())
    assert restored == record


def test_record_hash_is_stable_across_metadata_only_changes():
    r1 = _make_record()
    r2 = _make_record(reviewer="someone_else", created_at="2099-01-01T00:00:00+00:00")
    assert record_hash(r1) == record_hash(r2)


def test_record_hash_changes_when_a_ground_truth_field_changes():
    r1 = _make_record()
    r2 = _make_record(expected_threat_status="QUALIFIED_THREAT")
    assert record_hash(r1) != record_hash(r2)


def test_record_hash_is_order_independent_for_technique_list():
    r1 = _make_record(expected_mitre_techniques=["T1595", "T1110"])
    r2 = _make_record(expected_mitre_techniques=["T1110", "T1595"])
    assert record_hash(r1) == record_hash(r2)
