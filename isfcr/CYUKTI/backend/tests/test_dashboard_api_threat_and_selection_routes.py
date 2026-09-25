"""
Behavioral tests for the two routes added for the MITRE/threat-
qualification/campaign-selection phase:
  - GET /api/threat-qualification/<campaign_id>
  - GET /api/campaign-selection/<campaign_id>
"""

from __future__ import annotations

import pytest

import dashboard_api


@pytest.fixture()
def client():
    dashboard_api.app.config["TESTING"] = True
    with dashboard_api.app.test_client() as c:
        yield c


class _FakeResult:
    def __init__(self, row):
        self._row = row

    def single(self):
        return self._row


class _FakeSession:
    def __init__(self, single_row=None, rows=None):
        self._single_row = single_row
        self._rows = rows or []

    def run(self, query, **kwargs):
        if "c2.campaign_id AS campaign_id" in query:
            return self._rows
        return _FakeResult(self._single_row)

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class _FakeDriver:
    def __init__(self, session):
        self._session = session

    def session(self):
        return self._session


def _fake_campaign_context(**overrides):
    from datetime import datetime, timezone
    from campaign_context import CampaignContext
    kwargs = dict(campaign_id="CAMP_1", attacker_ip="1.2.3.4", victim_ip="10.0.0.5",
                  risk_score=1300.0, last_technique="T1110", techniques={"T1110"},
                  last_seen=datetime.now(timezone.utc))
    kwargs.update(overrides)
    return CampaignContext(**kwargs)


# ---------------------------------------------------------------- /api/threat-qualification

def test_threat_qualification_404_for_unknown_campaign(client, monkeypatch):
    monkeypatch.setattr(dashboard_api, "_try_load_campaign_context", lambda cid: (None, None))
    resp = client.get("/api/threat-qualification/GHOST")
    assert resp.status_code == 404


def test_threat_qualification_reports_not_threat_when_no_cti_score_stored(client, monkeypatch):
    monkeypatch.setattr(dashboard_api, "_try_load_campaign_context", lambda cid: (_fake_campaign_context(), None))
    monkeypatch.setattr(dashboard_api, "driver", _FakeDriver(_FakeSession(single_row=None)))

    resp = client.get("/api/threat-qualification/CAMP_1")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["classification"] == "NOT_THREAT"
    assert data["may_publish_to_misp"] is False


def test_threat_qualification_reports_qualified_threat_from_stored_cti_score(client, monkeypatch):
    monkeypatch.setattr(dashboard_api, "_try_load_campaign_context", lambda cid: (_fake_campaign_context(), None))
    monkeypatch.setattr(dashboard_api, "driver", _FakeDriver(_FakeSession(single_row={"score": 90.0, "publish": True})))

    resp = client.get("/api/threat-qualification/CAMP_1")
    data = resp.get_json()
    assert data["classification"] == "QUALIFIED_THREAT"
    assert data["cti_score"] == 90.0
    assert data["may_publish_to_misp"] is True


def test_threat_qualification_reports_suspicious_from_mid_range_stored_score(client, monkeypatch):
    monkeypatch.setattr(dashboard_api, "_try_load_campaign_context", lambda cid: (_fake_campaign_context(), None))
    monkeypatch.setattr(dashboard_api, "driver", _FakeDriver(_FakeSession(single_row={"score": 30.0, "publish": False})))

    resp = client.get("/api/threat-qualification/CAMP_1")
    data = resp.get_json()
    assert data["classification"] == "SUSPICIOUS"
    assert data["may_publish_to_misp"] is False


# ---------------------------------------------------------------- /api/campaign-selection

def test_campaign_selection_404_for_unknown_campaign(client, monkeypatch):
    monkeypatch.setattr(dashboard_api, "_try_load_campaign_context", lambda cid: (None, None))
    resp = client.get("/api/campaign-selection/GHOST")
    assert resp.status_code == 404


def test_campaign_selection_returns_none_selected_with_no_candidates(client, monkeypatch):
    monkeypatch.setattr(dashboard_api, "_try_load_campaign_context", lambda cid: (_fake_campaign_context(), None))
    monkeypatch.setattr(dashboard_api, "driver", _FakeDriver(_FakeSession(rows=[])))

    resp = client.get("/api/campaign-selection/CAMP_1")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["selected"] is None
    assert data["confidence"] == "NONE"


def test_campaign_selection_ranks_and_selects_a_real_candidate(client, monkeypatch):
    monkeypatch.setattr(dashboard_api, "_try_load_campaign_context", lambda cid: (_fake_campaign_context(), None))
    monkeypatch.setattr(dashboard_api, "driver", _FakeDriver(_FakeSession(rows=[{"campaign_id": "CAMP_OLD"}])))
    monkeypatch.setattr(
        "dashboard_api._load_campaign_context",
        lambda session, cid: _fake_campaign_context(campaign_id="CAMP_OLD") if cid == "CAMP_OLD" else None,
    )
    monkeypatch.setattr("ml.gnn.topology_similarity.gnn_topology_similarity_between_campaigns", lambda a, b: 0.9)

    resp = client.get("/api/campaign-selection/CAMP_1")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["selected"]["campaign_id"] == "CAMP_OLD"
    assert data["selected"]["signals"]["topology_similarity"] == 0.9
    assert "CAMP_OLD" in data["explanation"]
