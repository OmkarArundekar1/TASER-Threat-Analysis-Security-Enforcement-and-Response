"""
Behavioral tests for the four routes added during the "FINAL FULL-SYSTEM
INTEGRATION" phase (see FULL_SYSTEM_INTEGRATION_AUDIT.md):

- POST /api/rag/search      -- unified Multi-RAG query (MITRE + campaign
  narrative + GNN topology), each source's real results kept separate
  and tagged, never blended into one ranking.
- GET  /api/misp/status     -- surfaces the MISP/CTI integration state
  that already exists in realtime_socgraph.py's live pipeline
  (CTIPublisher/MISPSync) but previously had zero dashboard visibility.
- GET  /api/system/health   -- per-subsystem health, distinct from the
  minimal Neo4j-only /api/health.
- GET  /api/audit/logs      -- tails/parses the real listener log file.

All Neo4j/GNN/MISP/filesystem access is mocked at the boundary each
route actually calls (dashboard_api.driver, ml.gnn.inference's module
functions imported inline, cti_publisher.CTIPublisher, misp_cache.cache,
dashboard_api.LISTENER_LOG_PATH) -- no test touches a real database,
model artifact, network socket, or the real, live listener log.
"""

from __future__ import annotations

import json
import os

import pytest

import dashboard_api


@pytest.fixture()
def client():
    dashboard_api.app.config["TESTING"] = True
    with dashboard_api.app.test_client() as c:
        yield c


# ---------------------------------------------------------------- /api/rag/search

def test_rag_search_requires_query_or_campaign_id(client):
    resp = client.post('/api/rag/search', json={})
    assert resp.status_code == 400


def test_rag_search_queries_text_sources_when_query_given(client, monkeypatch):
    class _FakeEvidence:
        def to_dict(self):
            return {"source": "mitre_knowledge", "source_id": "T1078"}

    monkeypatch.setattr(dashboard_api.mitre_retriever, "query", lambda q, top_k=5: [_FakeEvidence()])

    class _FakeNarrativeRetriever:
        def query(self, q, top_k=5):
            return [_FakeEvidence()]

    monkeypatch.setattr(
        "rag.campaign_retriever.CampaignNarrativeRetriever", lambda: _FakeNarrativeRetriever()
    )

    resp = client.post('/api/rag/search', json={"query": "lateral movement", "top_k": 3})
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["query"] == "lateral movement"
    assert len(data["sources"]["mitre_knowledge"]) == 1
    assert len(data["sources"]["campaign_history"]) == 1
    assert "gnn_topology" not in data["sources"]


def test_rag_search_reports_gnn_disabled_without_erroring(client, monkeypatch):
    class _FakeService:
        available = False

    monkeypatch.setattr("ml.gnn.inference.gnn_inference_service", _FakeService())

    resp = client.post('/api/rag/search', json={"campaign_id": "CAMP_1"})
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["sources"]["gnn_topology"] == {"gnn_available": False, "results": []}


def test_rag_search_includes_gnn_results_when_available(client, monkeypatch):
    class _FakeMeta:
        model_version = "v1"

    class _FakeService:
        available = True
        metadata = _FakeMeta()

    class _FakeEvidence:
        def to_dict(self):
            return {"source": "gnn_topology", "source_id": "CAMP_2"}

    monkeypatch.setattr("ml.gnn.inference.gnn_inference_service", _FakeService())
    monkeypatch.setattr(
        "rag.gnn_topology_retriever.GNNTopologyRetriever",
        lambda: type("R", (), {"query": lambda self, cid, top_k=5: [_FakeEvidence()]})(),
    )

    resp = client.post('/api/rag/search', json={"campaign_id": "CAMP_1"})
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["sources"]["gnn_topology"]["gnn_available"] is True
    assert len(data["sources"]["gnn_topology"]["results"]) == 1


def test_rag_search_source_failure_is_isolated_not_fatal(client, monkeypatch):
    def _raise(*a, **kw):
        raise RuntimeError("corpus unavailable")

    monkeypatch.setattr(dashboard_api.mitre_retriever, "query", _raise)
    monkeypatch.setattr(
        "rag.campaign_retriever.CampaignNarrativeRetriever",
        lambda: type("R", (), {"query": lambda self, q, top_k=5: []})(),
    )

    resp = client.post('/api/rag/search', json={"query": "x"})
    assert resp.status_code == 200
    data = resp.get_json()
    assert "error" in data["sources"]["mitre_knowledge"]
    assert data["sources"]["campaign_history"] == []


# ---------------------------------------------------------------- /api/misp/status

def test_misp_status_reports_missing_credential_honestly(client, monkeypatch):
    class _FakeConfig:
        MISP_API_KEY = ""
        MISP_URL = "https://localhost:8443"

    monkeypatch.setattr("config.MISP_API_KEY", "", raising=False)
    monkeypatch.setattr("config.MISP_URL", "https://localhost:8443", raising=False)

    resp = client.get('/api/misp/status')
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["credential_configured"] is False
    assert data["authenticated"] is False
    assert "MISP_API_KEY" in data["status_message"]
    assert "api_key" not in json.dumps(data).lower().replace("misp_api_key", "")


def test_misp_status_reports_connected_when_healthy(client, monkeypatch):
    monkeypatch.setattr("config.MISP_API_KEY", "secret-key-value", raising=False)
    monkeypatch.setattr("config.MISP_URL", "https://localhost:8443", raising=False)
    monkeypatch.setattr("config.VERIFY_MISP_SSL", False, raising=False)

    class _FakePublisher:
        def __init__(self, *a, **kw):
            pass

        def health_check(self):
            return True

    class _FakeCache:
        def all(self):
            return {"CAMP_1": 42}

    monkeypatch.setattr("cti_publisher.CTIPublisher", _FakePublisher)
    monkeypatch.setattr("misp_cache.cache", _FakeCache())

    resp = client.get('/api/misp/status')
    data = resp.get_json()
    assert data["authenticated"] is True
    assert data["cached_campaigns"] == 1
    assert data["campaign_event_map"] == {"CAMP_1": 42}
    assert "secret-key-value" not in json.dumps(data)


def test_misp_status_never_leaks_key_when_health_check_raises(client, monkeypatch):
    monkeypatch.setattr("config.MISP_API_KEY", "super-secret", raising=False)
    monkeypatch.setattr("config.MISP_URL", "https://localhost:8443", raising=False)
    monkeypatch.setattr("config.VERIFY_MISP_SSL", False, raising=False)

    def _raise(*a, **kw):
        raise ConnectionError("unreachable")

    monkeypatch.setattr("cti_publisher.CTIPublisher", _raise)

    resp = client.get('/api/misp/status')
    assert resp.status_code == 200
    data = resp.get_json()
    assert "super-secret" not in json.dumps(data)
    assert "failed" in data["status_message"].lower()


# ---------------------------------------------------------------- /api/system/health

def test_system_health_reports_each_subsystem_independently(client, monkeypatch, tmp_path):
    monkeypatch.setattr(dashboard_api, "driver", type("D", (), {"verify_connectivity": lambda self: None})())
    monkeypatch.setattr("config.GNN_ENABLED", False, raising=False)
    monkeypatch.setattr("config.MISP_API_KEY", "", raising=False)
    monkeypatch.setattr(dashboard_api, "LISTENER_LOG_PATH", str(tmp_path / "missing.log"))

    resp = client.get('/api/system/health')
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["subsystems"]["neo4j"]["status"] == "connected"
    assert data["subsystems"]["gnn"]["status"] == "disabled"
    assert data["subsystems"]["misp"]["status"] == "credential_missing"
    assert data["subsystems"]["wazuh_listener"]["status"] == "no_log_found"
    assert data["status"] == "healthy"


def test_system_health_degraded_when_neo4j_down(client, monkeypatch):
    class _RaisingDriver:
        def verify_connectivity(self):
            raise ConnectionError("down")

    monkeypatch.setattr(dashboard_api, "driver", _RaisingDriver())
    monkeypatch.setattr("config.GNN_ENABLED", False, raising=False)
    monkeypatch.setattr("config.MISP_API_KEY", "", raising=False)

    resp = client.get('/api/system/health')
    data = resp.get_json()
    assert data["subsystems"]["neo4j"]["status"] == "disconnected"
    assert data["status"] == "degraded"


def test_system_health_recently_active_listener(client, monkeypatch, tmp_path):
    log_file = tmp_path / "prerana_listener.log"
    log_file.write_text("2026-09-24 08:35:26,018 | INFO | Alert ID: abc\n")

    monkeypatch.setattr(dashboard_api, "driver", type("D", (), {"verify_connectivity": lambda self: None})())
    monkeypatch.setattr("config.GNN_ENABLED", False, raising=False)
    monkeypatch.setattr("config.MISP_API_KEY", "", raising=False)
    monkeypatch.setattr(dashboard_api, "LISTENER_LOG_PATH", str(log_file))

    resp = client.get('/api/system/health')
    data = resp.get_json()
    assert data["subsystems"]["wazuh_listener"]["status"] == "recently_active"


# ---------------------------------------------------------------- /api/audit/logs

def test_audit_logs_returns_empty_when_file_missing(client, monkeypatch, tmp_path):
    monkeypatch.setattr(dashboard_api, "LISTENER_LOG_PATH", str(tmp_path / "missing.log"))

    resp = client.get('/api/audit/logs')
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["entries"] == []
    assert data["total_lines"] == 0


def test_audit_logs_parses_real_log_line_format(client, monkeypatch, tmp_path):
    log_file = tmp_path / "prerana_listener.log"
    log_file.write_text(
        "2026-09-24 08:35:18,013 | INFO | Alert ID: 5b8961dae8c8\n"
        "2026-09-24 08:35:26,018 | INFO | Offset updated: 737531 -> 737991\n"
    )
    monkeypatch.setattr(dashboard_api, "LISTENER_LOG_PATH", str(log_file))

    resp = client.get('/api/audit/logs')
    data = resp.get_json()
    assert data["total_lines"] == 2
    assert len(data["entries"]) == 2
    assert data["entries"][0]["level"] == "INFO"
    assert data["entries"][0]["message"] == "Alert ID: 5b8961dae8c8"
    assert data["entries"][0]["timestamp"] == "2026-09-24 08:35:18,013"


def test_audit_logs_respects_limit(client, monkeypatch, tmp_path):
    log_file = tmp_path / "prerana_listener.log"
    log_file.write_text("".join(f"L{i} | INFO | line {i}\n" for i in range(50)))
    monkeypatch.setattr(dashboard_api, "LISTENER_LOG_PATH", str(log_file))

    resp = client.get('/api/audit/logs?limit=10')
    data = resp.get_json()
    assert data["total_lines"] == 50
    assert len(data["entries"]) == 10
    assert data["entries"][-1]["message"] == "line 49"


def test_audit_logs_handles_malformed_line_without_crashing(client, monkeypatch, tmp_path):
    log_file = tmp_path / "prerana_listener.log"
    log_file.write_text("this is not a pipe-delimited line\n")
    monkeypatch.setattr(dashboard_api, "LISTENER_LOG_PATH", str(log_file))

    resp = client.get('/api/audit/logs')
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["entries"][0]["timestamp"] is None
    assert data["entries"][0]["message"] == "this is not a pipe-delimited line"
