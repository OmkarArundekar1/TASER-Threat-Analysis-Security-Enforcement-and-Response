import pytest

import ground_truth.store as store
import ground_truth.raw_alerts as raw_alerts_module
from evaluators import mitre_eval
from evaluators.base import MetricStatus
from ground_truth.builder import build_from_raw_alert
from scenarios.registry import get_scenario

NMAP_ALERT = {
    "timestamp": "2026-09-25T07:55:00+0000",
    "rule": {"id": "100500", "description": "HIGH SEVERITY: Nmap Reconnaissance Detected",
             "mitre": {"id": ["T1595"]}},
    "agent": {"id": "001"},
    "data": {"src_ip": "192.168.56.106", "dest_ip": "192.168.56.105"},
}

WRONG_TECHNIQUE_ALERT = {
    "timestamp": "2026-09-25T07:56:00+0000",
    "rule": {"id": "100500", "description": "HIGH SEVERITY: Nmap Reconnaissance Detected"},
    # no rule.mitre block at all -- resolver should return UNKNOWN, not T1595
    "agent": {"id": "001"},
    "data": {"src_ip": "192.168.56.106", "dest_ip": "192.168.56.105"},
}


@pytest.fixture(autouse=True)
def _isolate_stores(tmp_path, monkeypatch):
    from ground_truth.schema import ReviewStatus
    fake_dirs = {
        ReviewStatus.AUTO_PROPOSED: str(tmp_path / "provisional"),
        ReviewStatus.HUMAN_REVIEWED: str(tmp_path / "reviewed"),
        ReviewStatus.LOCKED: str(tmp_path / "locked"),
    }
    monkeypatch.setattr(store, "_DIRS", fake_dirs)
    monkeypatch.setattr(raw_alerts_module, "_RAW_DIR", str(tmp_path / "raw"))
    monkeypatch.setattr(mitre_eval.store, "_DIRS", fake_dirs)
    monkeypatch.setattr(mitre_eval.raw_alerts, "_RAW_DIR", str(tmp_path / "raw"))
    yield


def test_no_dataset_reports_not_measured():
    result = mitre_eval.evaluate(dataset_name="does_not_exist")
    assert result.status == MetricStatus.NOT_MEASURED


def test_strict_mode_on_auto_proposed_only_requires_review():
    scenario = get_scenario("SCN-NMAP-001")
    record = build_from_raw_alert(scenario, NMAP_ALERT, sample_id="X1")
    store.save_records("t_strict", [record])
    raw_alerts_module.save_raw_alerts("t_strict", {"X1": NMAP_ALERT})

    result = mitre_eval.evaluate(dataset_name="t_strict", allow_preliminary=False)
    assert result.status == MetricStatus.GROUND_TRUTH_REVIEW_REQUIRED


def test_preliminary_mode_on_matching_alert_reports_perfect_agreement():
    scenario = get_scenario("SCN-NMAP-001")
    record = build_from_raw_alert(scenario, NMAP_ALERT, sample_id="X1")
    store.save_records("t_match", [record])
    raw_alerts_module.save_raw_alerts("t_match", {"X1": NMAP_ALERT})

    result = mitre_eval.evaluate(dataset_name="t_match", allow_preliminary=True)
    assert result.status == MetricStatus.MEASURED_PRELIMINARY
    assert result.metrics["exact_match_ratio"] == 1.0
    assert result.metrics["precision"] == 1.0
    assert result.metrics["recall"] == 1.0


def test_missing_native_mitre_block_surfaces_as_disagreement_not_a_crash():
    scenario = get_scenario("SCN-NMAP-001")
    record = build_from_raw_alert(scenario, WRONG_TECHNIQUE_ALERT, sample_id="X1")
    store.save_records("t_disagree", [record])
    raw_alerts_module.save_raw_alerts("t_disagree", {"X1": WRONG_TECHNIQUE_ALERT})

    result = mitre_eval.evaluate(dataset_name="t_disagree", allow_preliminary=True)
    assert result.metrics["exact_match_ratio"] == 0.0
    assert result.metrics["unknown_rate"] == 1.0
    assert result.metrics["recall"] == 0.0


def test_ground_truth_with_no_matching_raw_alert_is_unmeasurable():
    scenario = get_scenario("SCN-NMAP-001")
    record = build_from_raw_alert(scenario, NMAP_ALERT, sample_id="X1")
    store.save_records("t_orphan", [record])
    # deliberately do not save a matching raw alert

    result = mitre_eval.evaluate(dataset_name="t_orphan", allow_preliminary=True)
    assert result.status == MetricStatus.UNMEASURABLE
