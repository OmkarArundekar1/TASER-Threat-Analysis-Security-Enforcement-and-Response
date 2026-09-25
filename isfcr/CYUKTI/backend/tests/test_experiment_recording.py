"""
Tests for experiments/record_experiment.py -- the schema is real JSON
Schema (experiments/schema.json), validated with the real `jsonschema`
library where available. These tests exercise the validator's own
correctness (accepts a valid record, rejects specific real defects),
not any particular experiment's content.
"""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "experiments"))

import record_experiment  # noqa: E402


def _minimal_valid_record(**overrides):
    record = {
        "experiment_id": "EXP-TEST-01",
        "timestamp": "2026-01-01T00:00:00+00:00",
        "scenario": "test scenario",
        "input": {"wazuh_alerts": []},
        "expected_behavior": "x",
        "actual_behavior": "y",
        "errors": [],
    }
    record.update(overrides)
    return record


def test_the_real_committed_example_validates_cleanly():
    examples_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "experiments", "examples")
    example_path = os.path.join(examples_dir, "EXP-2026-09-25-nmap-scan-01.json")
    with open(example_path) as f:
        record = json.load(f)
    assert record_experiment.validate_record(record) == []


def test_minimal_valid_record_has_no_validation_problems():
    assert record_experiment.validate_record(_minimal_valid_record()) == []


def test_missing_required_field_is_reported():
    record = _minimal_valid_record()
    del record["scenario"]
    problems = record_experiment.validate_record(record)
    assert any("scenario" in p for p in problems)


def test_wrong_type_for_threat_qualification_classification_is_reported():
    record = _minimal_valid_record(threat_qualification={"classification": "MAYBE"})
    problems = record_experiment.validate_record(record)
    assert len(problems) > 0


def test_null_evidence_and_latency_are_explicitly_allowed_not_flagged():
    record = _minimal_valid_record(evidence=None, latency=None, response_plan=None)
    assert record_experiment.validate_record(record) == []


def test_save_record_writes_a_file_and_rejects_invalid_records(tmp_path, monkeypatch):
    monkeypatch.setattr(record_experiment, "_RECORDS_DIR", str(tmp_path))
    record = _minimal_valid_record(experiment_id="EXP-SAVE-TEST")

    saved_path = record_experiment.save_record(record)
    assert os.path.exists(saved_path)
    with open(saved_path) as f:
        assert json.load(f)["experiment_id"] == "EXP-SAVE-TEST"

    invalid = _minimal_valid_record()
    del invalid["actual_behavior"]
    try:
        record_experiment.save_record(invalid)
        assert False, "save_record should have raised for an invalid record"
    except ValueError:
        pass
