"""
Integration/regression tests for realtime_socgraph.process_alert()'s
Phase 20 branching: resolved (NATIVE_WAZUH/REVIEWED/DETERMINISTIC) alerts
must follow the exact pre-Phase-20 pipeline; UNKNOWN/AMBIGUOUS alerts must
follow the new unattributed path and must never touch attack-chain state,
prediction, or MISP.

Every subsystem process_alert() touches is monkeypatched with a minimal
fake so these tests run without a live Neo4j/MISP/Wazuh connection --
consistent with the fake-driver/fake-session pattern used throughout the
existing test suite (see test_campaign_reconstruction.py).
"""

from types import SimpleNamespace

import pytest

import realtime_socgraph as rsg
import mitre_resolver


class _FakeCampaignContext:
    def __init__(self, campaign_id):
        self.campaign_id = campaign_id
        self.techniques = set()
        self.attack_chain = []
        self.predicted_next = None


_NOT_CALLED = object()  # distinct from a real return/argument value of None


@pytest.fixture
def stub_everything(monkeypatch):
    """Neutralizes every downstream subsystem process_alert() calls, and
    records which of the two ingestion paths (resolved vs unattributed)
    actually ran."""
    calls = {
        "create_attack_event": None,
        "create_unattributed_attack_event": None,
        "update_attack_chain": None,
        "predict_next": None,
        "publish_campaign": None,
        "get_or_create_campaign": None,
        "append_technique": None,
        "resolve_campaign_context_calls": [],
        "create_campaign_context": _NOT_CALLED,
    }

    monkeypatch.setattr(rsg.dedup_engine, "is_duplicate", lambda **kw: (False, "fp-123"))
    monkeypatch.setattr(rsg.dedup_engine, "register_event", lambda **kw: None)

    def _fake_create_attack_event(**kwargs):
        calls["create_attack_event"] = kwargs
        return "event-resolved-1"

    def _fake_create_unattributed(**kwargs):
        calls["create_unattributed_attack_event"] = kwargs
        return "event-unknown-1"

    monkeypatch.setattr(rsg, "create_attack_event", _fake_create_attack_event)
    monkeypatch.setattr(rsg, "create_unattributed_attack_event", _fake_create_unattributed)
    monkeypatch.setattr(rsg, "update_duplicate_event", lambda **kw: None)

    def _fake_get_or_create_campaign(attacker_ip, victim_ip, technique):
        calls["get_or_create_campaign"] = technique
        return "CAMP_TEST_RESOLVED"

    monkeypatch.setattr(rsg.campaign_manager, "get_or_create_campaign", _fake_get_or_create_campaign)

    def _fake_append_technique(context, technique):
        calls["append_technique"] = technique

    monkeypatch.setattr(rsg.campaign_manager, "append_technique", _fake_append_technique)

    def _fake_resolve_campaign_context(attacker_ip, victim_ip, include_inactive=False):
        # Phase 20E regression: the UNKNOWN path must never request
        # include_inactive=True (that argument reaches the broken
        # campaign_manager.load_from_database -> get_recent_inactive_campaign_db
        # list/dict mismatch -- see realtime_socgraph.py's comment at the
        # call site). Recording every call lets tests assert this directly
        # instead of just trusting the source doesn't pass it.
        calls["resolve_campaign_context_calls"].append(include_inactive)
        return None

    monkeypatch.setattr(rsg.campaign_manager, "resolve_campaign_context", _fake_resolve_campaign_context)

    def _fake_create_campaign_context(attacker_ip, victim_ip, current_technique):
        calls["create_campaign_context"] = current_technique
        return _FakeCampaignContext("CAMP_TEST_UNKNOWN")

    monkeypatch.setattr(rsg.campaign_manager, "create_campaign_context", _fake_create_campaign_context)

    monkeypatch.setattr(rsg.feature_engine, "extract_features", lambda *a, **k: SimpleNamespace())
    monkeypatch.setattr(rsg.severity_engine, "calculate", lambda features: "Low")
    monkeypatch.setattr(
        rsg.dynamic_risk_engine, "calculate",
        lambda features, severity: SimpleNamespace(risk_score=10.0, risk_level="Low", confidence=50, breakdown={}),
    )
    monkeypatch.setattr(rsg, "store_dynamic_risk", lambda campaign_id, dynamic_risk: None)

    def _fake_update_attack_chain(campaign_id, technique):
        calls["update_attack_chain"] = technique

    monkeypatch.setattr(rsg, "update_attack_chain", _fake_update_attack_chain)
    monkeypatch.setattr(rsg, "update_campaign_similarity", lambda campaign_id: None)
    monkeypatch.setattr(rsg, "update_actor_attribution", lambda campaign_id: None)

    def _fake_predict_next(campaign_id, technique):
        calls["predict_next"] = technique
        return None

    monkeypatch.setattr(rsg, "predict_next", _fake_predict_next)
    monkeypatch.setattr(rsg, "get_recommendations", lambda predicted: [])

    monkeypatch.setattr(
        rsg.operation_manager, "build_campaign_context",
        lambda campaign_id: _FakeCampaignContext(campaign_id),
    )
    monkeypatch.setattr(
        rsg.operation_engine, "correlate",
        lambda context: SimpleNamespace(matched=False, score=0.0, confidence=0.0, candidate_count=0, operation_id=None),
    )
    monkeypatch.setattr(rsg.detection_engine, "calculate", lambda event_id: SimpleNamespace(confidence=0.0))
    monkeypatch.setattr(rsg.threat_engine, "calculate", lambda attacker_ip: SimpleNamespace(confidence=0.0))
    monkeypatch.setattr(
        rsg.cti_engine, "calculate",
        lambda **kw: SimpleNamespace(score=0.0, level="Low", publish=False, breakdown={}),
    )
    monkeypatch.setattr(rsg, "store_cti_confidence", lambda campaign_id, cti: None)
    monkeypatch.setattr(rsg, "IncidentContext", lambda **kw: SimpleNamespace(**kw))
    monkeypatch.setattr(
        rsg.attribution_engine, "attribute",
        lambda context: SimpleNamespace(actors=[]),
    )
    monkeypatch.setattr(rsg, "create_operation_db", lambda context: "OP_TEST")
    monkeypatch.setattr(rsg, "attach_campaign_to_operation", lambda operation_id, context: None)
    monkeypatch.setattr(rsg, "update_operation_activity", lambda operation_id, context: None)

    def _fake_publish_campaign(incident):
        calls["publish_campaign"] = incident
        return {"success": False, "reason": "test stub"}

    monkeypatch.setattr(rsg.sync, "publish_campaign", _fake_publish_campaign)

    return calls


def _native_mapped_alert():
    return {
        "rule": {
            "id": "5503",
            "level": 5,
            "description": "PAM: User login failed.",
            "mitre": {"id": ["T1110.001"], "technique": ["Password Guessing"]},
        },
        "agent": {"id": "001", "name": "pes1ug23cs411-VirtualBox"},
        "data": {"srcip": "192.168.56.106", "dstip": "192.168.56.105"},
        "timestamp": "2026-08-29T11:25:45.436+0000",
    }


def _no_mitre_alert(rule_id="40704"):
    return {
        "rule": {
            "id": rule_id,
            "level": 5,
            "description": "Systemd: Service exited due to a failure.",
        },
        "agent": {"id": "001", "name": "pes1ug23cs411-VirtualBox"},
        "data": {"srcip": "192.168.56.106", "dstip": "192.168.56.105"},
        "timestamp": "2026-08-29T11:09:55.311+0000",
    }


# ---------------------------------------------------------------- 13. no longer hard-rejected

def test_alert_without_rule_mitre_is_no_longer_rejected(stub_everything):
    result = rsg.process_alert(_no_mitre_alert())
    assert result is True  # old behavior returned False and dropped the alert entirely


# ---------------------------------------------------------------- 14/15. UNKNOWN path

def test_unknown_alert_follows_unattributed_path(stub_everything):
    calls = stub_everything
    result = rsg.process_alert(_no_mitre_alert())

    assert result is True
    assert calls["create_unattributed_attack_event"] is not None
    assert calls["create_attack_event"] is None

    kwargs = calls["create_unattributed_attack_event"]
    assert kwargs["mitre_provenance"] == mitre_resolver.PROVENANCE_UNKNOWN
    assert kwargs["mitre_confidence"] == mitre_resolver.CONFIDENCE_NONE
    assert kwargs["mitre_technique_ids"] == []
    assert "attack_id" not in kwargs  # structurally can't be set -- not a parameter at all


# ---------------------------------------------------------------- Phase 20E regression: include_inactive=True

def test_unknown_path_never_requests_include_inactive_campaigns(stub_everything):
    """Regression for the Phase 20D live blocker: campaign_manager.
    load_from_database's include_inactive=True branch calls
    get_recent_inactive_campaign_db(), which returns list[dict] while
    load_from_database indexes it as a single dict -- a pre-existing bug
    with no caller before Phase 20B. The fix removes that argument from
    the UNKNOWN path entirely; this test fails loudly if it's ever
    reintroduced."""
    calls = stub_everything
    rsg.process_alert(_no_mitre_alert())

    assert calls["resolve_campaign_context_calls"] == [False]


def test_unknown_alert_with_existing_active_campaign_succeeds(stub_everything, monkeypatch):
    calls = stub_everything
    monkeypatch.setattr(
        rsg.campaign_manager, "resolve_campaign_context",
        lambda attacker_ip, victim_ip, include_inactive=False: (
            calls["resolve_campaign_context_calls"].append(include_inactive)
            or _FakeCampaignContext("CAMP_EXISTING_ACTIVE")
        ),
    )

    result = rsg.process_alert(_no_mitre_alert())

    assert result is True
    assert calls["resolve_campaign_context_calls"] == [False]
    assert calls["create_campaign_context"] is _NOT_CALLED  # existing campaign reused, no new one created
    assert calls["create_unattributed_attack_event"]["campaign_id"] == "CAMP_EXISTING_ACTIVE"


def test_unknown_alert_with_no_active_campaign_creates_shell_and_succeeds(stub_everything):
    calls = stub_everything
    result = rsg.process_alert(_no_mitre_alert())

    assert result is True
    assert calls["resolve_campaign_context_calls"] == [False]
    assert calls["create_campaign_context"] is None  # called, with current_technique=None (not a real technique)
    assert calls["create_unattributed_attack_event"]["campaign_id"] == "CAMP_TEST_UNKNOWN"


def test_unknown_alert_does_not_touch_chain_prediction_or_misp(stub_everything):
    calls = stub_everything
    rsg.process_alert(_no_mitre_alert())

    assert calls["get_or_create_campaign"] is None  # never routed through the technique-driven campaign path
    assert calls["append_technique"] is None
    assert calls["update_attack_chain"] is None
    assert calls["predict_next"] is None
    assert calls["publish_campaign"] is None


# ---------------------------------------------------------------- 16. resolved native path unchanged

def test_native_mapped_alert_follows_resolved_path_unchanged(stub_everything):
    calls = stub_everything
    result = rsg.process_alert(_native_mapped_alert())

    assert result is True
    assert calls["create_unattributed_attack_event"] is None
    assert calls["create_attack_event"] is not None

    kwargs = calls["create_attack_event"]
    assert kwargs["mitre_id"] == "T1110.001"
    assert kwargs["technique"] == "Password Guessing"
    assert kwargs["mitre_provenance"] == mitre_resolver.PROVENANCE_NATIVE_WAZUH
    assert kwargs["mitre_confidence"] == mitre_resolver.CONFIDENCE_CONFIRMED
    assert kwargs["mitre_technique_ids"] == ["T1110.001"]

    assert calls["get_or_create_campaign"] == "T1110.001"
    assert calls["update_attack_chain"] == "T1110.001"
    assert calls["predict_next"] == "T1110.001"
    assert calls["publish_campaign"] is not None


def test_multi_technique_native_alert_uses_first_as_primary_but_preserves_all(stub_everything):
    calls = stub_everything
    alert = _native_mapped_alert()
    alert["rule"]["mitre"]["id"] = ["T1110.001", "T1078"]
    alert["rule"]["mitre"]["technique"] = ["Password Guessing", "Valid Accounts"]

    rsg.process_alert(alert)

    kwargs = calls["create_attack_event"]
    assert kwargs["mitre_id"] == "T1110.001"  # primary, drives chain/prediction/campaign exactly as before
    assert kwargs["mitre_technique_ids"] == ["T1110.001", "T1078"]  # nothing silently discarded
