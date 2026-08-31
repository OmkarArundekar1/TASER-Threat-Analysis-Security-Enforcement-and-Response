"""
Integration tests for the new dashboard_api.py endpoints
(/api/investigate, /api/rag/mitre/search, /api/ml/predict/severity),
using Flask's test client.

/api/rag/mitre/search needs no live infrastructure at all (it's the real
MITRE STIX corpus loaded from disk) and is tested fully end to end.
/api/investigate and /api/ml/predict/severity require a live Neo4j
connection to do real work — here we only verify they fail gracefully
(a clean JSON error, not a raw 500 traceback) when Neo4j is unreachable,
since that's the only thing testable without live infrastructure.
"""

import pytest

import dashboard_api


@pytest.fixture()
def client():
    dashboard_api.app.config["TESTING"] = True
    with dashboard_api.app.test_client() as c:
        yield c


def test_query_route_is_registered_exactly_once():
    rules = [r for r in dashboard_api.app.url_map.iter_rules() if r.rule == "/api/query"]
    assert len(rules) == 1
    assert rules[0].endpoint == "execute_query"


def test_query_console_endpoint_no_longer_exists():
    endpoints = {r.endpoint for r in dashboard_api.app.url_map.iter_rules()}
    assert "query_console" not in endpoints


def test_rag_mitre_search_requires_query(client):
    resp = client.post("/api/rag/mitre/search", json={})
    assert resp.status_code == 400


def test_rag_mitre_search_end_to_end_real_corpus(client):
    resp = client.post("/api/rag/mitre/search", json={
        "query": "attacker repeatedly guessing account passwords", "top_k": 3,
    })
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["query"]
    assert len(data["results"]) > 0
    for r in data["results"]:
        assert r["source"] == "mitre"
        assert 0.0 <= r["relevance"] <= 1.0


def test_investigate_requires_json_body_tolerant(client):
    # no body at all -> should not crash with a raw exception; Neo4j is
    # unreachable here, so this should surface as a clean 5xx JSON error,
    # not a stack trace leaking to the client.
    resp = client.post("/api/investigate/nonexistent-campaign")
    assert resp.status_code in (404, 500, 503)
    assert resp.is_json


def test_ml_predict_severity_without_model_returns_503(client, monkeypatch):
    # Force the "no model" path regardless of whether a real model has
    # been trained elsewhere in this environment (ml/train_xgboost.py now
    # trains on the real 33-campaign dataset, so a real model file may
    # legitimately exist on disk when this test runs).
    real_exists = dashboard_api.os.path.exists
    monkeypatch.setattr(
        dashboard_api.os.path, "exists",
        lambda path: False if str(path).endswith("xgb_severity.json") else real_exists(path),
    )
    resp = client.post("/api/ml/predict/severity", json={"campaign_id": "camp-1"})
    assert resp.status_code == 503
    assert "error" in resp.get_json()


def test_ml_predict_severity_requires_campaign_id(client):
    resp = client.post("/api/ml/predict/severity", json={})
    assert resp.status_code == 400
