import os
import tempfile

import pytest

import dashboard_api
import active_response.audit as audit_module
from soar.memory import PlaybookMemoryStore


@pytest.fixture()
def client():
    dashboard_api.app.config["TESTING"] = True
    with dashboard_api.app.test_client() as c:
        yield c


@pytest.fixture()
def isolated_store(monkeypatch):
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    store = PlaybookMemoryStore(db_path=path)
    monkeypatch.setattr(audit_module, "memory_store", store)
    yield store
    os.remove(path)


def test_response_audit_route_returns_empty_list_for_unknown_correlation_id(client, isolated_store):
    resp = client.get("/api/soar/response-audit/never-logged")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["event_count"] == 0
    assert data["events"] == []


def test_response_audit_route_returns_real_logged_events(client, isolated_store):
    audit_module.log_response_event("CONTAINMENT_REQUESTED", correlation_id="corr-api-1", decision_id="dec-1")
    audit_module.log_response_event("CONTAINMENT_VERIFIED", correlation_id="corr-api-1", decision_id="dec-1")

    resp = client.get("/api/soar/response-audit/corr-api-1")
    data = resp.get_json()
    assert data["event_count"] == 2
    assert all(e["detail"]["correlation_id"] == "corr-api-1" for e in data["events"])
