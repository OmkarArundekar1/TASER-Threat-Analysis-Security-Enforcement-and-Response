"""
Behavioral tests for the incident-centric aggregate routes:
  - GET /api/incidents/<id>/overview
  - GET /api/incidents/<id>/response-plan
"""

from __future__ import annotations

import pytest

import dashboard_api


@pytest.fixture()
def client():
    dashboard_api.app.config["TESTING"] = True
    with dashboard_api.app.test_client() as c:
        yield c


def _fake_campaign_context(**overrides):
    from datetime import datetime, timezone
    from campaign_context import CampaignContext
    kwargs = dict(campaign_id="CAMP_1", attacker_ip="1.2.3.4", victim_ip="10.0.0.5",
                  risk_score=1300.0, last_technique="T1110", techniques={"T1110"},
                  last_seen=datetime.now(timezone.utc), first_seen=datetime.now(timezone.utc))
    kwargs.update(overrides)
    return CampaignContext(**kwargs)


class _FakeResult:
    def __init__(self, row):
        self._row = row

    def single(self):
        return self._row


class _FakeSession:
    def run(self, query, **kwargs):
        if "c2.campaign_id AS campaign_id" in query:
            return []
        if "Operation" in query:
            return _FakeResult(None)
        return _FakeResult({"score": 90.0, "publish": True})

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class _FakeDriver:
    def session(self):
        return _FakeSession()


# ---------------------------------------------------------------- overview

def test_overview_404_for_unknown_campaign(client, monkeypatch):
    monkeypatch.setattr(dashboard_api, "_try_load_campaign_context", lambda cid: (None, None))
    resp = client.get("/api/incidents/GHOST/overview")
    assert resp.status_code == 404


def test_overview_composes_all_sections_for_a_real_campaign(client, monkeypatch):
    monkeypatch.setattr(dashboard_api, "_try_load_campaign_context", lambda cid: (_fake_campaign_context(), None))
    monkeypatch.setattr(dashboard_api, "driver", _FakeDriver())
    monkeypatch.setattr("mitre_resolver.driver", _FakeDriver())

    resp = client.get("/api/incidents/CAMP_1/overview")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["campaign"]["campaign_id"] == "CAMP_1"
    assert data["threat_qualification"]["classification"] == "QUALIFIED_THREAT"
    assert data["campaign_selection"]["selected"] is None  # no candidates in this fake session
    assert data["operation_id"] is None
    assert "gnn" in data
    assert data["investigation_available"] is True
    assert data["rag_available"] is True
    assert isinstance(data["mitre"], list)


def test_overview_reports_operation_id_when_present(client, monkeypatch):
    monkeypatch.setattr(dashboard_api, "_try_load_campaign_context", lambda cid: (_fake_campaign_context(), None))

    class _OpSession(_FakeSession):
        def run(self, query, **kwargs):
            if "Operation" in query:
                return _FakeResult({"operation_id": "OP_99"})
            return super().run(query, **kwargs)

    class _OpDriver:
        def session(self):
            return _OpSession()

    monkeypatch.setattr(dashboard_api, "driver", _OpDriver())
    monkeypatch.setattr("mitre_resolver.driver", _OpDriver())

    resp = client.get("/api/incidents/CAMP_1/overview")
    assert resp.get_json()["operation_id"] == "OP_99"


def test_overview_500s_cleanly_to_json_on_neo4j_outage(client, monkeypatch):
    from neo4j.exceptions import ServiceUnavailable

    class _RaisingDriver:
        def session(self):
            raise ServiceUnavailable("down")

    monkeypatch.setattr(dashboard_api, "_try_load_campaign_context", lambda cid: (_fake_campaign_context(), None))
    monkeypatch.setattr(dashboard_api, "driver", _RaisingDriver())

    resp = client.get("/api/incidents/CAMP_1/overview")
    assert resp.status_code == 503


# ---------------------------------------------------------------- response-plan

def test_response_plan_404_for_unknown_campaign(client, monkeypatch):
    monkeypatch.setattr(dashboard_api, "_try_load_campaign_context", lambda cid: (None, None))
    resp = client.get("/api/incidents/GHOST/response-plan")
    assert resp.status_code == 404


def test_response_plan_generates_a_fresh_candidate_when_none_stored(client, monkeypatch):
    monkeypatch.setattr(dashboard_api, "_try_load_campaign_context", lambda cid: (_fake_campaign_context(), None))
    monkeypatch.setattr(dashboard_api, "driver", _FakeDriver())
    monkeypatch.setattr("soar.generator.get_recommendations", lambda tech: [])

    resp = client.get("/api/incidents/CAMP_1/response-plan")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["campaign_id"] == "CAMP_1"
    assert data["playbook"] is not None
    assert data["misp_status"] == "READY"  # score=90 -> QUALIFIED_THREAT, all checks pass


def test_response_plan_reuses_an_existing_stored_playbook(client, monkeypatch, tmp_path):
    import soar.memory as soar_memory_module
    from soar.memory import PlaybookMemoryStore
    from soar.schema import Playbook

    isolated_store = PlaybookMemoryStore(db_path=str(tmp_path / "response_plan_test.db"))
    pb = Playbook(playbook_id="pb_existing", name="EXISTING", description="", trigger_conditions={},
                  campaign_type="x", mitre_techniques=["T1110"], severity="HIGH", risk=900.0,
                  required_evidence=[], actions=[], source_campaign_id="CAMP_1")
    isolated_store.save_playbook(pb)
    monkeypatch.setattr(soar_memory_module, "memory_store", isolated_store)

    monkeypatch.setattr(dashboard_api, "_try_load_campaign_context", lambda cid: (_fake_campaign_context(), None))
    monkeypatch.setattr(dashboard_api, "driver", _FakeDriver())

    resp = client.get("/api/incidents/CAMP_1/response-plan")
    data = resp.get_json()
    assert data["playbook"]["playbook_id"] == "pb_existing"
