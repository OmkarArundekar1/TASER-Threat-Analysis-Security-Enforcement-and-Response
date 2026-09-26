import inspect

import pytest

import ground_truth.builder as builder_module
from ground_truth.builder import build_from_raw_alert, build_session_boundary
from ground_truth.schema import ReviewStatus
from scenarios.registry import get_scenario


NMAP_ALERT = {
    "timestamp": "2026-09-25T07:55:00+0000",
    "rule": {"id": "100500", "description": "HIGH SEVERITY: Nmap Reconnaissance Detected"},
    "agent": {"id": "001", "name": "pes1ug23cs411-VirtualBox"},
    "data": {"src_ip": "192.168.56.106", "dest_ip": "192.168.56.105"},
}


def test_build_from_raw_alert_produces_auto_proposed_record():
    scenario = get_scenario("SCN-NMAP-001")
    record = build_from_raw_alert(scenario, NMAP_ALERT, sample_id="M1")
    assert record.review_status == ReviewStatus.AUTO_PROPOSED
    assert record.expected_mitre_techniques == ["T1595"]
    assert record.attacker_identity == "192.168.56.106"
    assert "mitre_resolver.py" in record.labeling_method


def test_build_from_raw_alert_never_calls_the_real_resolver():
    """If someone later wires mitre_resolver into the builder by
    mistake, this test should start failing loudly: it asserts
    builder.py's own SOURCE never references the resolver module.
    (A sys.modules check would be unreliable here -- other tests in
    the same process legitimately import mitre_resolver for unrelated
    reasons, so global import-cache state is not a valid signal.)"""
    code_without_docstring = inspect.getsource(builder_module).split('"""', 2)[-1]  # drop the module docstring block
    assert "import mitre_resolver" not in code_without_docstring
    assert "from mitre_resolver" not in code_without_docstring
    assert "resolve_mitre(" not in code_without_docstring


def test_build_session_boundary_uses_independent_campaign_key_not_a_cyukti_id():
    scenario = get_scenario("SCN-SESSION-BOUNDARY-001")
    record = build_session_boundary(scenario, sample_id="C1", involved_raw_event_ids=["e1", "e2"])
    assert record.expected_campaign_id.startswith("SESSION::SCN-SESSION-BOUNDARY-001::")
    assert not record.expected_campaign_id.startswith("CAMP_")
    assert record.expected_mitre_techniques == []


def test_build_session_boundary_never_calls_campaign_manager():
    """Same rationale as the resolver check above: inspect builder.py's
    own source rather than global sys.modules state."""
    code_without_docstring = inspect.getsource(builder_module).split('"""', 2)[-1]
    assert "import campaign_manager" not in code_without_docstring
    assert "from campaign_manager" not in code_without_docstring
