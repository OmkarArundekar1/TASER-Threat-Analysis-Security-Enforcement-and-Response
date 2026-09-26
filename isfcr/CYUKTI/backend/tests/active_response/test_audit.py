import os
import tempfile

import pytest

import soar.memory as memory_module
from soar.memory import PlaybookMemoryStore


@pytest.fixture(autouse=True)
def _isolated_memory_store(monkeypatch):
    """Real SQLite store, but pointed at a throwaway temp file so this
    test never touches the real backend/soar/playbook_memory.db."""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    store = PlaybookMemoryStore(db_path=path)
    monkeypatch.setattr(memory_module, "memory_store", store)
    import active_response.audit as audit_module
    monkeypatch.setattr(audit_module, "memory_store", store)
    yield store
    os.remove(path)


def test_log_response_event_is_retrievable_by_correlation_id():
    from active_response.audit import log_response_event, audit_trail_for_correlation

    log_response_event("CONTAINMENT_REQUESTED", correlation_id="corr-42", decision_id="dec-1",
                        campaign_id="CAMP_X")
    log_response_event("CONTAINMENT_VERIFIED", correlation_id="corr-42", decision_id="dec-1",
                        campaign_id="CAMP_X")
    log_response_event("CONTAINMENT_REQUESTED", correlation_id="corr-other", decision_id="dec-2")

    trail = audit_trail_for_correlation("corr-42")
    assert len(trail) == 2
    assert all(e["detail"]["correlation_id"] == "corr-42" for e in trail)


def test_log_response_event_never_persists_secret_looking_keys():
    from active_response.audit import log_response_event, audit_trail_for_correlation

    log_response_event(
        "CONTAINMENT_REQUESTED", correlation_id="corr-99", decision_id="dec-9",
        detail={"api_key": "should-not-be-stored", "note": "safe field"},
    )
    trail = audit_trail_for_correlation("corr-99")
    assert "api_key" not in trail[0]["detail"]
    assert trail[0]["detail"]["note"] == "safe field"
