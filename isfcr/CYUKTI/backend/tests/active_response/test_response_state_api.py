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


def test_no_activity_reports_no_response_activity_not_a_fabricated_state(client, isolated_store):
    resp = client.get("/api/soar/response-state/never-logged")
    data = resp.get_json()
    assert data["current_state"] == "NO_RESPONSE_ACTIVITY"


def test_current_state_reflects_the_most_recent_real_event(client, isolated_store):
    audit_module.log_response_event("CONTAINMENT_REQUESTED", correlation_id="corr-1", decision_id="dec-1")
    audit_module.log_response_event("CONTAINMENT_EXECUTED", correlation_id="corr-1", decision_id="dec-1")
    audit_module.log_response_event("CONTAINMENT_VERIFIED", correlation_id="corr-1", decision_id="dec-1")

    resp = client.get("/api/soar/response-state/corr-1")
    data = resp.get_json()
    assert data["current_state"] == "CONTAINMENT_VERIFIED"
    assert data["event_count"] == 3


def test_response_state_route_never_accepts_post():
    """Confirms this is GET-only -- there is no way to trigger anything
    via this endpoint, only to read what already happened in-process."""
    import dashboard_api
    dashboard_api.app.config["TESTING"] = True
    with dashboard_api.app.test_client() as c:
        resp = c.post("/api/soar/response-state/corr-1")
        assert resp.status_code == 405
