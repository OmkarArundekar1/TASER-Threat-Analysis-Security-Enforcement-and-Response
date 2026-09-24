"""
Behavioral tests for soar/api.py's Flask blueprint, registered onto the
real dashboard_api.app instance (dashboard_api.py does
`app.register_blueprint(soar_bp)` at import time). Each test swaps in
an isolated, tmp-path-backed PlaybookMemoryStore so no test touches the
real soar/playbook_memory.db, mirroring test_dashboard_api_routes.py's
_QueueDriver pattern for Neo4j -- this module's boundary is
soar.api.memory_store / the singleton service objects, not the Neo4j
driver.
"""

from __future__ import annotations

import pytest

import dashboard_api
import soar.api as soar_api
from campaign_context import CampaignContext
from soar.memory import PlaybookMemoryStore
from soar.schema import ExecutionPolicy, ExecutionStatus, Playbook, PlaybookAction


@pytest.fixture()
def client():
    dashboard_api.app.config["TESTING"] = True
    with dashboard_api.app.test_client() as c:
        yield c


@pytest.fixture()
def isolated_store(tmp_path, monkeypatch):
    test_store = PlaybookMemoryStore(db_path=str(tmp_path / "api_test.db"))
    monkeypatch.setattr(soar_api, "memory_store", test_store)
    monkeypatch.setattr(soar_api._execution_service, "memory_store", test_store)
    monkeypatch.setattr(soar_api._matcher, "memory_store", test_store)
    return test_store


def _fake_campaign_context(**overrides):
    kwargs = dict(campaign_id="CAMP_1", attacker_ip="1.2.3.4", victim_ip="10.0.0.5",
                  risk_score=1300.0, last_technique="T1110", techniques={"T1110"})
    kwargs.update(overrides)
    return CampaignContext(**kwargs)


def _mock_campaign_loading(monkeypatch, context=_fake_campaign_context()):
    monkeypatch.setattr(soar_api, "_load_context_or_error", lambda campaign_id: (context, None))


def test_status_reports_shuffle_not_configured_by_default(client, monkeypatch, isolated_store):
    monkeypatch.setattr("config.SHUFFLE_WEBHOOK", "", raising=False)
    monkeypatch.setattr("config.SHUFFLE_BASE_URL", "", raising=False)
    monkeypatch.setattr("config.SHUFFLE_API_KEY", "", raising=False)

    resp = client.get("/api/soar/status")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["shuffle_webhook_configured"] is False
    assert data["shuffle_api_configured"] is False


def test_generate_playbook_requires_campaign_id(client, isolated_store):
    resp = client.post("/api/soar/playbooks/generate", json={})
    assert resp.status_code == 400


def test_generate_playbook_returns_404_for_unknown_campaign(client, monkeypatch, isolated_store):
    monkeypatch.setattr(soar_api, "_load_context_or_error", lambda campaign_id: (None, None))
    resp = client.post("/api/soar/playbooks/generate", json={"campaign_id": "GHOST"})
    assert resp.status_code == 404


def test_generate_playbook_persists_and_returns_playbook(client, monkeypatch, isolated_store):
    monkeypatch.setattr("soar.generator.get_recommendations", lambda tech: [])
    _mock_campaign_loading(monkeypatch)

    resp = client.post("/api/soar/playbooks/generate", json={"campaign_id": "CAMP_1"})
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["source_campaign_id"] == "CAMP_1"
    assert len(isolated_store.list_playbooks()) == 1


def test_list_and_get_playbook(client, isolated_store):
    pb = Playbook(playbook_id="pb_1", name="TEST", description="", trigger_conditions={}, campaign_type="x",
                  mitre_techniques=[], severity="LOW", risk=10.0, required_evidence=[], actions=[])
    isolated_store.save_playbook(pb)

    list_resp = client.get("/api/soar/playbooks")
    assert len(list_resp.get_json()["playbooks"]) == 1

    get_resp = client.get("/api/soar/playbooks/pb_1")
    assert get_resp.get_json()["playbook_id"] == "pb_1"

    missing_resp = client.get("/api/soar/playbooks/does_not_exist")
    assert missing_resp.status_code == 404


def test_execute_playbook_recommend_only_returns_400(client, isolated_store):
    pb = Playbook(playbook_id="pb_1", name="TEST", description="", trigger_conditions={}, campaign_type="x",
                  mitre_techniques=[], severity="LOW", risk=10.0, required_evidence=[], actions=[],
                  execution_policy=ExecutionPolicy.RECOMMEND_ONLY, source_campaign_id="CAMP_1")
    isolated_store.save_playbook(pb)

    resp = client.post("/api/soar/playbooks/pb_1/execute", json={})
    assert resp.status_code == 400


def test_execute_playbook_analyst_approval_creates_pending_execution(client, isolated_store):
    pb = Playbook(playbook_id="pb_1", name="TEST", description="", trigger_conditions={}, campaign_type="x",
                  mitre_techniques=[], severity="HIGH", risk=900.0, required_evidence=[],
                  actions=[PlaybookAction(action_type="block_ip", name="Block", description="", order=1,
                                           destructive=True, requires_approval=True)],
                  execution_policy=ExecutionPolicy.ANALYST_APPROVAL, source_campaign_id="CAMP_1")
    isolated_store.save_playbook(pb)

    resp = client.post("/api/soar/playbooks/pb_1/execute", json={})
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["status"] == "pending_approval"


def test_approve_and_reject_execution_endpoints(client, isolated_store):
    pb = Playbook(playbook_id="pb_1", name="TEST", description="", trigger_conditions={}, campaign_type="x",
                  mitre_techniques=[], severity="HIGH", risk=900.0, required_evidence=[],
                  actions=[PlaybookAction(action_type="block_ip", name="Block", description="", order=1,
                                           destructive=True, requires_approval=True)],
                  execution_policy=ExecutionPolicy.ANALYST_APPROVAL, source_campaign_id="CAMP_1")
    isolated_store.save_playbook(pb)

    exec_resp = client.post("/api/soar/playbooks/pb_1/execute", json={})
    execution_id = exec_resp.get_json()["execution_id"]

    reject_resp = client.post(f"/api/soar/executions/{execution_id}/reject", json={"reason": "not needed"})
    assert reject_resp.status_code == 200
    assert reject_resp.get_json()["status"] == "rejected"

    # already rejected -- approving now must fail cleanly, not 500
    approve_resp = client.post(f"/api/soar/executions/{execution_id}/approve", json={"approved_by": "a1"})
    assert approve_resp.status_code == 400


def test_get_execution_includes_audit_events(client, isolated_store):
    pb = Playbook(playbook_id="pb_1", name="TEST", description="", trigger_conditions={}, campaign_type="x",
                  mitre_techniques=[], severity="LOW", risk=10.0, required_evidence=[], actions=[],
                  execution_policy=ExecutionPolicy.AUTOMATIC, source_campaign_id="CAMP_1")
    isolated_store.save_playbook(pb)

    exec_resp = client.post("/api/soar/playbooks/pb_1/execute", json={})
    execution_id = exec_resp.get_json()["execution_id"]

    get_resp = client.get(f"/api/soar/executions/{execution_id}")
    assert get_resp.status_code == 200
    assert "audit_events" in get_resp.get_json()


def test_get_execution_404_for_unknown_id(client, isolated_store):
    resp = client.get("/api/soar/executions/does_not_exist")
    assert resp.status_code == 404


def test_effectiveness_endpoint_returns_stats_for_all_playbooks(client, isolated_store):
    pb = Playbook(playbook_id="pb_1", name="TEST", description="", trigger_conditions={}, campaign_type="x",
                  mitre_techniques=[], severity="LOW", risk=10.0, required_evidence=[], actions=[])
    isolated_store.save_playbook(pb)
    resp = client.get("/api/soar/effectiveness")
    assert resp.status_code == 200
    assert len(resp.get_json()["effectiveness"]) == 1


def test_recommendations_endpoint_returns_candidate_and_historical_matches(client, monkeypatch, isolated_store):
    monkeypatch.setattr("soar.generator.get_recommendations", lambda tech: [])
    _mock_campaign_loading(monkeypatch)

    class _FakeSession:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    class _FakeDriver:
        def session(self):
            return _FakeSession()

    monkeypatch.setattr("neo4j_client.driver", _FakeDriver())

    resp = client.get("/api/soar/recommendations/CAMP_1")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["campaign_id"] == "CAMP_1"
    assert "candidate_playbook" in data
    assert data["candidate_playbook"]["source_campaign_id"] == "CAMP_1"


def test_recommendations_endpoint_404_for_unknown_campaign(client, monkeypatch, isolated_store):
    monkeypatch.setattr(soar_api, "_load_context_or_error", lambda campaign_id: (None, None))
    resp = client.get("/api/soar/recommendations/GHOST")
    assert resp.status_code == 404


def test_adapt_playbook_endpoint(client, monkeypatch, isolated_store):
    source_pb = Playbook(playbook_id="pb_old", name="OLD", description="", trigger_conditions={}, campaign_type="x",
                          mitre_techniques=["T1110"], severity="HIGH", risk=900.0, required_evidence=[],
                          actions=[PlaybookAction(action_type="enrich_ip", name="Enrich", description="", order=1,
                                                   inputs={"ip": "9.9.9.9"})],
                          source_campaign_id="CAMP_OLD")
    isolated_store.save_playbook(source_pb)
    _mock_campaign_loading(monkeypatch)

    resp = client.post("/api/soar/playbooks/adapt", json={"source_playbook_id": "pb_old", "campaign_id": "CAMP_1"})
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["adapted_from_playbook_id"] == "pb_old"
    assert data["source_campaign_id"] == "CAMP_1"
    assert len(isolated_store.list_playbooks()) == 2
