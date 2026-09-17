"""
Behavioral tests for CYUKTI's MISP integration
(`cti_publisher.py`, `misp_sync.py`, `misp_event_generator.py`,
`misp_cache.py`) -- previously untested (module_status.md rated this
row **C**: "CTIPublisher initializes successfully... live publish
success not confirmed"). These tests exercise the REAL
CTIPublisher/MISPSync/MISPEventGenerator implementations; only the
HTTP transport boundary (`requests.Session.request`) is mocked, plus
`misp_sync.cache` is pointed at an isolated tmp-backed MISPCache so no
test touches the real `misp_cache.json`.

**No accuracy/quality claim is made anywhere in this file.**

Live-verification note (see MISP_INTEGRATION.md for full detail): this
session confirmed a real, reachable MISP instance at the configured
MISP_URL in this environment, and exercised CTIPublisher.health_check()
against it for real (no API key is configured here, so only the
unauthenticated-request failure path was live-exercised — a full
authenticated create/search/update round trip requires credentials this
session does not have and must not request or expose). That live check
is not repeated in this file (it would make the suite depend on
network/service availability); it is recorded in MISP_INTEGRATION.md.

Two real defects found via behavioral testing and fixed this session
(see the "false-positive success" tests below):

1. `CTIPublisher.create_event()`/`update_event()` reported `success:
   True` for any 2xx HTTP response, even one whose body didn't match
   MISP's actual event-creation/edit schema (no `Event.id` / no
   `Event` object) -- meaning a malformed response could be reported
   as a successful publish with no way for the caller to tell.
2. `MISPSync.publish_campaign()` inferred whether a create or an
   update had just happened from `event_id is None` -- ambiguous,
   because `update_event()` never set that key either, so a CREATE
   whose event_id failed to parse was mislabeled "updated".
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass

import pytest

from attribution_models import HistoricalCampaign
from campaign_context import CampaignContext
from cti_publisher import CTIPublisher
from misp_cache import MISPCache
from misp_event_generator import IncidentContext, MISPEventGenerator
from misp_sync import MISPSync
from threat_attribution_engine import ThreatAttributionEngine


# ---------------------------------------------------------------- shared fixtures / fakes

class _FakeResponse:
    def __init__(self, status_code, json_body=None, text="", raise_json_error=False):
        self.status_code = status_code
        self.ok = 200 <= status_code < 400
        self._json_body = json_body
        self.text = text if not raise_json_error else "<html>not json</html>"
        self._raise_json_error = raise_json_error

    def json(self):
        if self._raise_json_error:
            raise ValueError("not JSON")
        return self._json_body


@dataclass
class _Stub:
    """Minimal stand-in for detection/threat/dynamic_risk/cti objects
    IncidentContext/MISPEventGenerator read specific attributes from."""
    confidence: float = 0.5
    level: str = "MEDIUM"
    breakdown: dict = None
    risk_score: float = 50.0
    risk_level: str = "MEDIUM"
    score: float = 50.0
    publish: bool = True

    def __post_init__(self):
        if self.breakdown is None:
            self.breakdown = {k: 1 for k in ("Wazuh", "Suricata", "Zeek", "Sigma", "YARA")}


def _incident(**overrides):
    defaults = dict(
        campaign_id="CAMP_TEST", operation_id="OP_TEST",
        attacker_ip="203.0.113.7", victim_ip="10.0.0.9", event_id="evt-1",
        technique="T1110", stage="Credential Access", prediction="T1078",
        prediction_confidence=0.6,
        detection=_Stub(), threat=_Stub(), dynamic_risk=_Stub(), cti=_Stub(),
        recommendations=["Rotate credentials"], investigation_payload="{}",
        timestamp="2026-01-15T12:00:00+00:00", attribution=None,
    )
    defaults.update(overrides)
    return IncidentContext(**defaults)


@pytest.fixture()
def publisher():
    return CTIPublisher("https://misp.test.invalid", "test-api-key-should-never-leak", verify_ssl=False)


def _mock_request(monkeypatch, pub, fn):
    monkeypatch.setattr(pub.session, "request", fn)


# ============================================================== A. event generation

def test_basic_event_generation_has_required_structure_and_serializes():
    incident = _incident()
    event = MISPEventGenerator().generate(incident)

    assert "Event" in event
    body = event["Event"]
    for key in ("info", "distribution", "threat_level_id", "analysis", "published", "date", "Attribute", "Tag"):
        assert key in body
    assert body["info"] == "CYUKTI Campaign CAMP_TEST"
    assert isinstance(body["Attribute"], list) and body["Attribute"]
    assert isinstance(body["Tag"], list) and body["Tag"]
    json.dumps(event)  # must serialize with no custom hacks


def test_attribution_enrichment_reaches_generated_event():
    engine = ThreatAttributionEngine()
    historical = [HistoricalCampaign(
        campaign_id="CAMP_OLD", attacker="203.0.113.7", victim="10.0.0.1",
        techniques=["T1110", "T1078"], timestamps=["2026-01-01T00:00:00+00:00"] * 2, status="ARCHIVED",
    )]
    campaign = CampaignContext(
        campaign_id="CAMP_TEST", attacker_ip="203.0.113.7", victim_ip="10.0.0.9",
        techniques={"T1110", "T1078"}, attack_chain=["T1110", "T1078"],
    )
    import attribution_context as attribution_context_module
    original = attribution_context_module.context.load_historical_campaigns
    attribution_context_module.context.load_historical_campaigns = lambda: historical
    try:
        attribution = engine.attribute(campaign)
    finally:
        attribution_context_module.context.load_historical_campaigns = original
    assert attribution.actors

    incident = _incident(attribution=attribution)
    event = MISPEventGenerator().generate(incident)

    attributes = event["Event"]["Attribute"]
    tag_names = [t["name"] for t in event["Event"]["Tag"]]
    best = attribution.actors[0]
    assert any(a["value"] == f"Threat Actor : {best.actor}" for a in attributes)
    assert f"actor:{best.actor}" in tag_names
    assert any(a["value"] == e for a in attributes for e in best.evidence)  # rationale text carried through


def test_no_attribution_omits_attribution_fields_without_error():
    incident = _incident(attribution=None)
    event = MISPEventGenerator().generate(incident)
    tag_names = [t["name"] for t in event["Event"]["Tag"]]
    assert not any(t.startswith("actor:") for t in tag_names)


def test_campaign_and_operation_context_propagate():
    incident = _incident(campaign_id="CAMP_XYZ", operation_id="OP_XYZ")
    event = MISPEventGenerator().generate(incident)
    attributes = event["Event"]["Attribute"]
    tag_names = [t["name"] for t in event["Event"]["Tag"]]
    assert any(a["value"] == "Campaign : CAMP_XYZ" for a in attributes)
    assert any(a["value"] == "Operation : OP_XYZ" for a in attributes)
    assert "campaign:CAMP_XYZ" in tag_names
    assert "operation:OP_XYZ" in tag_names


def test_technique_and_stage_propagate_as_tags():
    incident = _incident(technique="T1110.001", stage="Credential Access")
    event = MISPEventGenerator().generate(incident)
    tag_names = [t["name"] for t in event["Event"]["Tag"]]
    assert "mitre:T1110.001" in tag_names
    assert "stage:Credential Access" in tag_names


def test_attacker_ip_is_a_typed_ip_src_ioc_attribute():
    """The only real IOC CYUKTI currently tracks end to end is the
    attacker IP (extract_iocs() in realtime_socgraph.py) -- verify it
    reaches MISP as a properly-typed `ip-src` attribute, the correct
    MISP attribute type for an indicator meant to be actioned on."""
    incident = _incident(attacker_ip="198.51.100.23")
    event = MISPEventGenerator().generate(incident)
    ip_attrs = [a for a in event["Event"]["Attribute"] if a["type"] == "ip-src"]
    assert len(ip_attrs) == 1
    assert ip_attrs[0]["value"] == "198.51.100.23"
    assert ip_attrs[0]["category"] == "Network activity"


# ============================================================== B. publisher: success / failure paths

def test_create_event_success_extracts_event_id(publisher, monkeypatch):
    _mock_request(monkeypatch, publisher, lambda **kw: _FakeResponse(200, {"Event": {"id": "42"}}))
    result = publisher.create_event({"Event": {"info": "test"}})
    assert result["success"] is True
    assert result["event_id"] == 42


def test_authentication_failure_reports_explicit_failure_not_false_success(publisher, monkeypatch):
    """Matches the real, live-verified MISP response shape for a
    missing/invalid API key (403 with a JSON error body) -- see
    MISP_INTEGRATION.md."""
    _mock_request(monkeypatch, publisher, lambda **kw: _FakeResponse(
        403, {"name": "Authentication failed.", "message": "Authentication failed.", "url": "/events/add"},
    ))
    result = publisher.create_event({"Event": {"info": "test"}})
    assert result["success"] is False
    assert result["status"] == 403
    assert result.get("event_id") is None


def test_health_check_false_on_auth_failure(publisher, monkeypatch):
    _mock_request(monkeypatch, publisher, lambda **kw: _FakeResponse(403, {"name": "Authentication failed."}))
    assert publisher.health_check() is False


def test_http_500_is_a_controlled_failure(publisher, monkeypatch):
    _mock_request(monkeypatch, publisher, lambda **kw: _FakeResponse(500, {"error": "Internal Server Error"}))
    result = publisher.create_event({"Event": {"info": "test"}})
    assert result["success"] is False
    assert result["status"] == 500


def test_http_400_is_a_controlled_failure(publisher, monkeypatch):
    _mock_request(monkeypatch, publisher, lambda **kw: _FakeResponse(400, {"errors": "Invalid payload"}))
    result = publisher.create_event({"Event": {"info": "test"}})
    assert result["success"] is False
    assert result["status"] == 400


def test_timeout_is_a_controlled_failure_not_a_raised_exception(publisher, monkeypatch):
    import requests

    def _raise(**kw):
        raise requests.exceptions.Timeout("Connection timed out")

    _mock_request(monkeypatch, publisher, _raise)
    result = publisher.create_event({"Event": {"info": "test"}})
    assert result["success"] is False
    assert result["status"] is None
    assert "error" in result["response"]


def test_connection_failure_is_a_controlled_failure_not_a_raised_exception(publisher, monkeypatch):
    import requests

    def _raise(**kw):
        raise requests.exceptions.ConnectionError("Failed to establish a new connection")

    _mock_request(monkeypatch, publisher, _raise)
    result = publisher.create_event({"Event": {"info": "test"}})
    assert result["success"] is False


def test_missing_configuration_does_not_crash_at_construction():
    pub = CTIPublisher("", "", verify_ssl=False)
    assert pub.url == ""
    assert pub.api_key == ""


def test_malformed_url_produces_a_controlled_failure_not_a_crash():
    pub = CTIPublisher("not a valid url", "key", verify_ssl=False)
    result = pub.create_event({"Event": {"info": "test"}})
    assert result["success"] is False


# ============================================================== C. false-positive success (fixed defects)

def test_create_event_2xx_with_malformed_body_does_not_report_success(publisher, monkeypatch):
    """Regression test for the fix: HTTP 200 whose body doesn't match
    MISP's event-creation schema (no Event.id) must NOT be reported as
    success -- a genuine created-event response always has one."""
    _mock_request(monkeypatch, publisher, lambda **kw: _FakeResponse(200, {"unexpected": "shape"}))
    result = publisher.create_event({"Event": {"info": "test"}})
    assert result["success"] is False
    assert result["event_id"] is None


def test_create_event_2xx_with_non_json_body_does_not_report_success(publisher, monkeypatch):
    _mock_request(monkeypatch, publisher, lambda **kw: _FakeResponse(200, raise_json_error=True))
    result = publisher.create_event({"Event": {"info": "test"}})
    assert result["success"] is False


def test_update_event_2xx_with_malformed_body_does_not_report_success(publisher, monkeypatch):
    _mock_request(monkeypatch, publisher, lambda **kw: _FakeResponse(200, {"unexpected": "shape"}))
    result = publisher.update_event(42, {"Event": {"info": "test"}})
    assert result["success"] is False


def test_update_event_success_with_well_formed_body(publisher, monkeypatch):
    _mock_request(monkeypatch, publisher, lambda **kw: _FakeResponse(200, {"Event": {"id": "42"}}))
    result = publisher.update_event(42, {"Event": {"info": "test"}})
    assert result["success"] is True


def test_misp_sync_action_label_is_unambiguous_for_a_create_with_parse_failure(monkeypatch):
    """Regression test for the action-mislabeling defect: previously,
    a create whose response body could not be parsed for event_id was
    labeled "updated" by publish_campaign() (both cases left event_id
    absent/None). Now create_event's own false-positive fix makes this
    scenario report success=False in the first place, and _create
    sets its own action explicitly -- verify both together."""
    pub = CTIPublisher("https://misp.test.invalid", "key", verify_ssl=False)
    _mock_request(monkeypatch, pub, lambda **kw: _FakeResponse(200, {"unexpected": "shape"}))
    sync = MISPSync(pub)
    fake_cache = MISPCache(cache_file="__unused__.json")
    monkeypatch.setattr(fake_cache, "_save", lambda: None)  # no disk I/O in this unit test
    monkeypatch.setattr("misp_sync.cache", fake_cache)
    monkeypatch.setattr(pub, "search_campaign", lambda campaign_id: None)

    incident = _incident()
    result = sync.publish_campaign(incident)

    assert result["action"] == "created"
    assert result["success"] is False  # the malformed response, not a mislabeled update


# ============================================================== D. security: no credential leakage

def test_api_key_never_appears_in_log_output(publisher, monkeypatch, caplog):
    secret = publisher.api_key
    assert secret == "test-api-key-should-never-leak"

    import requests

    def _raise(**kw):
        raise requests.exceptions.ConnectionError(f"Failed to connect to {kw.get('url')}")

    _mock_request(monkeypatch, publisher, _raise)

    with caplog.at_level(logging.DEBUG):
        publisher.create_event({"Event": {"info": "test"}})
        publisher.health_check()

    all_log_text = "\n".join(record.getMessage() for record in caplog.records)
    assert secret not in all_log_text


def test_api_key_never_appears_in_success_or_error_response_dicts(publisher, monkeypatch):
    _mock_request(monkeypatch, publisher, lambda **kw: _FakeResponse(403, {"name": "Authentication failed."}))
    result = publisher.create_event({"Event": {"info": "test"}})
    assert publisher.api_key not in json.dumps(result)


def test_generated_misp_event_never_embeds_the_api_key():
    incident = _incident()
    event = MISPEventGenerator().generate(incident)
    assert "test-api-key-should-never-leak" not in json.dumps(event)


# ============================================================== E. idempotency / duplicate publication

@pytest.fixture()
def sync_with_fake_cache(monkeypatch, tmp_path):
    pub = CTIPublisher("https://misp.test.invalid", "key", verify_ssl=False)
    sync = MISPSync(pub)
    fake_cache = MISPCache(cache_file=str(tmp_path / "misp_cache_test.json"))
    monkeypatch.setattr("misp_sync.cache", fake_cache)
    return sync, pub, fake_cache


def test_first_publish_of_a_new_campaign_creates_one_event(sync_with_fake_cache, monkeypatch):
    sync, pub, fake_cache = sync_with_fake_cache
    calls = []

    def fake_request(**kw):
        calls.append(kw["url"])
        if kw["url"].endswith("/events/restSearch"):
            return _FakeResponse(200, [])
        if kw["url"].endswith("/events/add"):
            return _FakeResponse(200, {"Event": {"id": "100"}})
        raise AssertionError(f"unexpected call: {kw['url']}")

    _mock_request(monkeypatch, pub, fake_request)

    result = sync.publish_campaign(_incident())

    assert result["success"] is True
    assert result["action"] == "created"
    assert fake_cache.get_event_id("CAMP_TEST") == 100
    assert sum(1 for u in calls if u.endswith("/events/add")) == 1


def test_republishing_the_same_campaign_updates_instead_of_duplicating(sync_with_fake_cache, monkeypatch):
    sync, pub, fake_cache = sync_with_fake_cache
    fake_cache.set_event_id("CAMP_TEST", 100)
    calls = []

    def fake_request(**kw):
        calls.append(kw["url"])
        if kw["url"].endswith("/events/view/100"):
            return _FakeResponse(200, {"Event": {"id": "100"}})
        if kw["url"].endswith("/events/edit/100"):
            return _FakeResponse(200, {"Event": {"id": "100"}})
        raise AssertionError(f"unexpected call: {kw['url']}")

    _mock_request(monkeypatch, pub, fake_request)

    result = sync.publish_campaign(_incident())

    assert result["success"] is True
    assert result["action"] == "updated"
    assert not any(u.endswith("/events/add") for u in calls)  # never a second create


def test_stale_cache_entry_self_heals_via_search_then_recreates_if_absent(sync_with_fake_cache, monkeypatch):
    """Cached event_id points at an event that no longer exists in
    MISP (e.g. deleted externally) -- the cache entry must be evicted
    and a fresh lookup performed, not silently fail or loop."""
    sync, pub, fake_cache = sync_with_fake_cache
    fake_cache.set_event_id("CAMP_TEST", 100)
    calls = []

    def fake_request(**kw):
        calls.append(kw["url"])
        if kw["url"].endswith("/events/view/100"):
            return _FakeResponse(404, {"errors": "Event not found"})
        if kw["url"].endswith("/events/restSearch"):
            return _FakeResponse(200, [])
        if kw["url"].endswith("/events/add"):
            return _FakeResponse(200, {"Event": {"id": "200"}})
        raise AssertionError(f"unexpected call: {kw['url']}")

    _mock_request(monkeypatch, pub, fake_request)

    result = sync.publish_campaign(_incident())

    assert result["success"] is True
    assert result["action"] == "created"
    assert fake_cache.get_event_id("CAMP_TEST") == 200  # evicted stale 100, learned fresh 200


def test_cache_miss_but_event_already_exists_in_misp_updates_not_duplicates(sync_with_fake_cache, monkeypatch):
    """No local cache entry (e.g. cache cleared/lost), but MISP already
    has an event for this campaign (found via search_campaign's text
    search on "CYUKTI Campaign <id>") -- must update that one, not
    create a duplicate."""
    sync, pub, fake_cache = sync_with_fake_cache
    calls = []

    def fake_request(**kw):
        calls.append(kw["url"])
        if kw["url"].endswith("/events/restSearch"):
            return _FakeResponse(200, [{"Event": {"id": "77"}}])
        if kw["url"].endswith("/events/edit/77"):
            return _FakeResponse(200, {"Event": {"id": "77"}})
        raise AssertionError(f"unexpected call: {kw['url']}")

    _mock_request(monkeypatch, pub, fake_request)

    result = sync.publish_campaign(_incident())

    assert result["success"] is True
    assert result["action"] == "updated"
    assert fake_cache.get_event_id("CAMP_TEST") == 77
    assert not any(u.endswith("/events/add") for u in calls)


def test_should_publish_gates_on_cti_policy_not_just_availability(sync_with_fake_cache, monkeypatch):
    sync, pub, fake_cache = sync_with_fake_cache

    def _fail_if_called(**kw):
        raise AssertionError("no HTTP call should happen when CTI policy says don't publish")

    _mock_request(monkeypatch, pub, _fail_if_called)

    incident = _incident(cti=_Stub(publish=False))
    result = sync.publish_campaign(incident)

    assert result["success"] is False
    assert result["action"] == "skipped"


# ============================================================== F. full pipeline integration

def test_full_pipeline_investigation_evidence_confidence_to_misp_publish(monkeypatch, tmp_path):
    """The largest realistic local integration: a real CampaignContext
    -> real ThreatAttributionEngine -> real MISPEventGenerator -> real
    MISPSync -> real CTIPublisher, all real, with only the Neo4j
    historical-campaign lookup and the MISP HTTP transport mocked.
    Proves the event MISPSync actually sends is the SAME event the
    investigation/attribution pipeline produced, not a hand-built
    stand-in.
    """
    import attribution_context as attribution_context_module

    historical = [HistoricalCampaign(
        campaign_id="CAMP_OLD", attacker="203.0.113.7", victim="10.0.0.1",
        techniques=["T1110", "T1078"], timestamps=["2026-01-01T00:00:00+00:00"] * 2, status="ARCHIVED",
    )]
    monkeypatch.setattr(attribution_context_module.context, "load_historical_campaigns", lambda: historical)

    campaign = CampaignContext(
        campaign_id="CAMP_FULL", attacker_ip="203.0.113.7", victim_ip="10.0.0.55",
        techniques={"T1110", "T1078"}, attack_chain=["T1110", "T1078"],
    )
    attribution = ThreatAttributionEngine().attribute(campaign)
    assert attribution.actors

    incident = _incident(
        campaign_id="CAMP_FULL", attacker_ip="203.0.113.7", victim_ip="10.0.0.55",
        attribution=attribution,
    )

    pub = CTIPublisher("https://misp.test.invalid", "key", verify_ssl=False)
    sync = MISPSync(pub)
    fake_cache = MISPCache(cache_file=str(tmp_path / "cache.json"))
    monkeypatch.setattr("misp_sync.cache", fake_cache)

    sent_payloads = []

    def fake_request(**kw):
        if kw["url"].endswith("/events/restSearch"):
            return _FakeResponse(200, [])
        if kw["url"].endswith("/events/add"):
            sent_payloads.append(kw["json"])
            return _FakeResponse(200, {"Event": {"id": "999"}})
        raise AssertionError(f"unexpected call: {kw['url']}")

    _mock_request(monkeypatch, pub, fake_request)

    result = sync.publish_campaign(incident)

    assert result["success"] is True
    assert result["event_id"] == 999
    assert len(sent_payloads) == 1
    sent_event = sent_payloads[0]["Event"]
    assert sent_event["info"] == "CYUKTI Campaign CAMP_FULL"
    tag_names = [t["name"] for t in sent_event["Tag"]]
    assert "actor:CAMP_OLD" in tag_names  # the REAL attribution candidate, not a fixture
    json.dumps(sent_payloads[0])
