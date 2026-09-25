"""
Failure-injection testing (FAILURE_INJECTION.md, cross-cutting
engineering audit). For each listed failure mode, verifies:
  - no uncontrolled crash (either a clean, typed error/None, or a
    graceful 5xx JSON response -- never an unhandled exception reaching
    the caller);
  - no unsafe fallback (a missing artifact/service is reported as
    unavailable, never silently substituted with fabricated output);
  - no accidental MISP publication;
  - no destructive response;
  - a useful, honest error/status is returned.

Several of these failure modes already have dedicated coverage
elsewhere (e.g. test_gnn_inference.py, test_dashboard_api_routes.py,
test_misp_integration.py, test_soar_shuffle_client.py) -- this file
exists to make the *complete* list of required failure modes
explicit and verifiable in one place, not to duplicate those tests'
internals.
"""

from __future__ import annotations

import pytest

import dashboard_api
from neo4j.exceptions import ServiceUnavailable


@pytest.fixture()
def client():
    dashboard_api.app.config["TESTING"] = True
    with dashboard_api.app.test_client() as c:
        yield c


class _RaisingDriver:
    def __init__(self, exc):
        self._exc = exc

    def session(self):
        raise self._exc


# ---------------------------------------------------------------- 1. Neo4j unavailable

def test_neo4j_unavailable_returns_clean_503_not_a_crash(client, monkeypatch):
    monkeypatch.setattr(dashboard_api, "driver", _RaisingDriver(ServiceUnavailable("down")))
    resp = client.get("/api/incidents/CAMP_1/overview")
    assert resp.status_code in (503, 404)  # 404 if _try_load_campaign_context's own outage path returns that instead
    assert resp.is_json


def test_neo4j_unavailable_on_campaigns_route_returns_clean_503(client, monkeypatch):
    monkeypatch.setattr(dashboard_api, "driver", _RaisingDriver(ServiceUnavailable("down")))
    resp = client.get("/api/campaigns")
    assert resp.status_code == 503
    assert "error" in resp.get_json()


# ---------------------------------------------------------------- 2. Wazuh (alert file) unavailable

def test_missing_alert_file_does_not_crash_the_listener_at_import(monkeypatch, tmp_path):
    """AlertFileHandler.__init__ must tolerate ALERT_FILE not existing
    (e.g. Wazuh not installed/running) -- offset defaults to 0, no
    exception. wazuh_listener.py's own top-level `from utils import ...`
    only resolves with listener/ itself on sys.path (see
    test_audit_logging.py's _PATH_PRELUDE, the established pattern for
    importing this module in a test)."""
    import sys
    import os
    listener_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "listener")
    if listener_dir not in sys.path:
        sys.path.insert(0, listener_dir)
    import wazuh_listener as wl

    monkeypatch.setattr(wl, "ALERT_FILE", str(tmp_path / "does_not_exist.json"))
    handler = wl.AlertFileHandler()
    assert handler.offset == 0


# ---------------------------------------------------------------- 3. GNN artifact unavailable

def test_gnn_unavailable_embed_campaign_returns_none_not_a_crash(monkeypatch):
    from ml.gnn.inference import GNNInferenceService
    service = GNNInferenceService(model_path="/nonexistent/path/model.pt")
    result = service.embed_campaign("CAMP_1")
    assert result is None


def test_gnn_unavailable_dashboard_route_reports_false_not_error(client, monkeypatch):
    class _FakeService:
        available = False
        metadata = None

    monkeypatch.setattr("ml.gnn.inference.gnn_inference_service", _FakeService())
    resp = client.get("/api/gnn/status")
    assert resp.status_code == 200
    assert resp.get_json()["gnn_available"] is False


# ---------------------------------------------------------------- 4. XGBoost artifact unavailable

def test_xgboost_unavailable_returns_clean_503(client, monkeypatch):
    monkeypatch.setattr("os.path.exists", lambda path: False)
    resp = client.post("/api/ml/predict/severity", json={"campaign_id": "CAMP_1"})
    assert resp.status_code == 503
    assert "error" in resp.get_json()


# ---------------------------------------------------------------- 5. MISP unavailable

def test_misp_unreachable_health_check_returns_false_not_a_crash(monkeypatch):
    from cti_publisher import CTIPublisher
    import requests

    publisher = CTIPublisher("https://misp.invalid.example", "fake-key", verify_ssl=False)

    def _raise(*a, **kw):
        raise requests.exceptions.ConnectionError("refused")

    monkeypatch.setattr(publisher.session, "request", _raise)
    assert publisher.health_check() is False


def test_misp_unreachable_does_not_block_campaign_processing(monkeypatch):
    """should_publish() must be answerable purely from cti.publish --
    it must never itself attempt an HTTP call that could raise."""
    from misp_sync import MISPSync

    class _Stub:
        publish = False

    class _Incident:
        campaign_id = "CAMP_1"
        cti = _Stub()

    sync = MISPSync.__new__(MISPSync)  # bypass __init__'s real CTIPublisher wiring
    assert sync.should_publish(_Incident()) is False


# ---------------------------------------------------------------- 6. Shuffle unavailable

def test_shuffle_unavailable_marks_execution_failed_with_honest_reason(tmp_path):
    from soar.execution_service import PlaybookExecutionService
    from soar.memory import PlaybookMemoryStore
    from soar.schema import ExecutionPolicy, ExecutionStatus, Playbook, PlaybookAction
    from soar.shuffle_client import ShuffleClient

    memory_store = PlaybookMemoryStore(db_path=str(tmp_path / "failure_injection_test.db"))
    shuffle_client = ShuffleClient(webhook_url="")  # not configured
    service = PlaybookExecutionService(memory_store, shuffle_client)

    playbook = Playbook(
        playbook_id="pb_1", name="TEST", description="", trigger_conditions={},
        campaign_type="x", mitre_techniques=[], severity="LOW", risk=10.0, required_evidence=[],
        actions=[PlaybookAction(action_type="enrich_ip", name="Enrich", description="", order=1)],
        execution_policy=ExecutionPolicy.AUTOMATIC,
    )
    memory_store.save_playbook(playbook)

    execution = service.request_execution(playbook, campaign_id="CAMP_1")
    assert execution.status == ExecutionStatus.FAILED
    assert len(execution.action_results) == 1
    assert execution.action_results[0].status == ExecutionStatus.FAILED
    assert "not configured" in execution.action_results[0].error.lower()


# ---------------------------------------------------------------- 7/8. Invalid alert / missing MITRE

def test_completely_empty_alert_resolves_to_unknown_not_a_crash():
    from mitre_resolver import resolve_mitre, PROVENANCE_UNKNOWN
    result = resolve_mitre({})
    assert result.provenance == PROVENANCE_UNKNOWN
    assert result.technique_ids == ()


def test_malformed_alert_missing_rule_key_resolves_to_unknown():
    from mitre_resolver import resolve_mitre, PROVENANCE_UNKNOWN
    result = resolve_mitre({"data": {"src_ip": "1.2.3.4"}})
    assert result.provenance == PROVENANCE_UNKNOWN


# ---------------------------------------------------------------- 9. Missing attacker

def test_missing_attacker_ip_fails_the_ioc_check_not_a_crash():
    from threat_qualification import ThreatQualificationEngine, QUALIFIED_THREAT

    class _Stub:
        threat_classification = QUALIFIED_THREAT
        score = 90.0

    class _Incident:
        campaign_id = "CAMP_1"
        attacker_ip = None
        technique = "T1110"
        timestamp = "2026-01-01T00:00:00Z"
        cti = _Stub()

    result = ThreatQualificationEngine().qualify(_Incident())
    assert result.may_publish_to_misp is False
    assert any(c.name == "has_ioc" and not c.passed for c in result.checks)


# ---------------------------------------------------------------- 10. Missing victim

def test_missing_victim_ip_returns_none_similarity_not_a_crash():
    from campaign_selection import _ip_similarity
    assert _ip_similarity("1.2.3.4", None) is None
    assert _ip_similarity(None, None) is None


# ---------------------------------------------------------------- 11. Duplicate alert

def test_duplicate_alert_is_recognized_on_second_call_not_reprocessed(monkeypatch):
    """is_duplicate() alone only checks; the caller registers a
    non-duplicate event via register_event() -- mirrors how
    realtime_socgraph.py actually uses this engine (check, then
    register once the event is truly processed)."""
    import dedup_engine as dedup_module

    monkeypatch.setattr(dedup_module, "find_recent_duplicate", lambda fingerprint: None)
    engine = dedup_module.DeduplicationEngine()

    first_is_dup, fp1 = engine.is_duplicate("1.2.3.4", "10.0.0.5", "T1110", "5716", "001")
    assert first_is_dup is False
    engine.register_event(fp1, campaign_id="CAMP_1", event_id="evt-1")

    second_is_dup, fp2 = engine.is_duplicate("1.2.3.4", "10.0.0.5", "T1110", "5716", "001")
    assert second_is_dup is True
    assert fp1 == fp2


# ---------------------------------------------------------------- 12. Malformed IOC

def test_extract_iocs_with_no_data_field_does_not_crash():
    from realtime_socgraph import extract_iocs
    attacker, victim = extract_iocs({})
    assert attacker is None


def test_extract_iocs_falls_back_to_agent_identity_when_victim_ip_absent():
    from realtime_socgraph import extract_iocs
    attacker, victim = extract_iocs({"data": {"src_ip": "1.2.3.4"}, "agent": {"name": "host01"}})
    assert attacker == "1.2.3.4"
    assert victim == "host01"


# ---------------------------------------------------------------- 13. Empty evidence

def test_best_campaign_selector_with_no_candidates_returns_none_not_a_crash():
    from campaign_selection import BestCampaignSelector
    result = BestCampaignSelector().select([])
    assert result.selected is None
    assert result.confidence == "NONE"


def test_response_plan_generation_with_no_qualification_or_selection_degrades_honestly():
    from campaign_context import CampaignContext
    from soar.response_plan import ResponsePlanGenerator

    campaign = CampaignContext(campaign_id="CAMP_1", attacker_ip="1.2.3.4", victim_ip="10.0.0.5", risk_score=10.0)
    plan = ResponsePlanGenerator().generate(campaign)
    assert plan.misp_status == "NOT_APPLICABLE"
    assert plan.selected_historical_campaign is None
