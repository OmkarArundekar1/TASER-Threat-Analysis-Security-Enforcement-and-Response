"""
Comprehensive behavioral tests for dashboard_api.py's 22 routes --
previously covered only for the newest evidence-aware endpoints
(tests/test_dashboard_api.py, test_dashboard_api_campaign_context.py,
test_query_console_execution.py). All 22 routes are consumed by the
real React frontend (frontend/src/services/api.ts is the source of
truth cross-checked here) -- there are no dead/unused routes.

Two tiers, both real dashboard_api.app.test_client() calls (never unit
tests of route internals):

1. MOCKED Neo4j -- deterministic, portable, runs everywhere. Covers
   error handling, input validation, and response-shape correctness
   against controlled fixture data.
2. LIVE verification against this environment's real, reachable Neo4j
   instance -- skipped (not failed) via `_require_live_neo4j`/
   `_require_live_model` if unavailable, matching the project's
   established pattern (test_production_model_artifact.py). Identifiers
   are discovered from the live database itself, never hardcoded.

Real defects found (via live testing, not just mocks) and fixed this
session:

1. Most routes queried `driver.session()` directly with no per-route
   try/except, so a Neo4j outage (or any other uncaught exception -- a
   malformed int parameter, an internal engine error) fell through to
   Flask's default HTML error page instead of this API's
   otherwise-universal JSON contract. Fixed with four app-level
   `@app.errorhandler`s in dashboard_api.py: ServiceUnavailable -> 503
   ("database unreachable"), Neo4jError -> 500 ("database responded
   but rejected the query" -- a real, distinct case this session,
   see #2 below), HTTPException -> JSON with the real status code,
   bare Exception -> 500. One place fixes every route uniformly rather
   than touching each of the 15+ affected routes individually.
2. `/api/attribution/actors/<id>`'s Cypher used `max(a, b)` as a
   two-scalar function -- not valid Cypher (max() is aggregate-only) --
   live-confirmed to raise `Neo.ClientError.Statement.SyntaxError` on
   every real call. Fixed by removing the (mathematically unnecessary,
   given the preceding WHERE clause) guard entirely.
3. Three call sites (`event_detail`, `predictions`, `predict`) called
   `prediction_engine.predict_next(technique)` with one argument, but
   that function requires `(campaign_id, technique)` -- a live
   `TypeError` on every call. Worse than a simple arity bug:
   `predict_next` also has a Neo4j WRITE side effect (overwrites the
   campaign's live prediction state) meant for the investigation loop,
   which a read-only GET must never trigger. Fixed by calling the
   existing, purpose-built `predict_next_readonly(technique)` instead
   at all three sites.

See DASHBOARD_API.md for the full write-up.
"""

from __future__ import annotations

import json

import pytest

import dashboard_api
from neo4j.exceptions import Neo4jError, ServiceUnavailable


# ---------------------------------------------------------------- shared fakes (mocked tier)

class _FakeResult:
    def __init__(self, rows):
        self._rows = list(rows)

    def single(self):
        return self._rows[0] if self._rows else None

    def __iter__(self):
        return iter(self._rows)


class _QueueSession:
    """Each dashboard_api route does `with driver.session() as session:`
    once, then issues one or more session.run(...) calls in a fixed,
    known sequence -- this returns pre-configured results from that
    sequence in order, so each test only has to describe what each call
    in the route's real code returns, not reimplement Cypher semantics."""

    def __init__(self, *results):
        self._queue = list(results)

    def run(self, query, **params):
        if not self._queue:
            return _FakeResult([])
        return self._queue.pop(0)

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class _QueueDriver:
    def __init__(self, *results):
        self._results = results

    def session(self):
        return _QueueSession(*self._results)


class _RaisingDriver:
    def __init__(self, exc):
        self._exc = exc

    def session(self):
        raise self._exc


class _FakeNode(dict):
    """Minimal Neo4j Node stand-in: dict(n) works (dict subclass), plus
    the .element_id/.labels attributes route code reads directly."""

    def __init__(self, element_id, labels, props):
        super().__init__(props)
        self.element_id = element_id
        self.labels = labels


class _FakeRel(dict):
    def __init__(self, start_node, end_node, rel_type, props=None):
        super().__init__(props or {})
        self.start_node = start_node
        self.end_node = end_node
        self.type = rel_type


@pytest.fixture()
def client():
    dashboard_api.app.config["TESTING"] = True
    with dashboard_api.app.test_client() as c:
        yield c


@pytest.fixture()
def mock_driver(monkeypatch):
    def _set(*results):
        fake = _QueueDriver(*results)
        monkeypatch.setattr(dashboard_api, "driver", fake)
        return fake
    return _set


def _neo4j_reachable():
    try:
        dashboard_api.driver.verify_connectivity()
        return True
    except Exception:
        return False


requires_live_neo4j = pytest.mark.skipif(
    not _neo4j_reachable(),
    reason="No live Neo4j instance reachable in this environment.",
)


# ============================================================== global error handling (fix)

def test_unmatched_route_returns_json_404_not_html(client):
    resp = client.get("/api/this-route-does-not-exist")
    assert resp.status_code == 404
    assert resp.is_json
    assert "error" in resp.get_json()


def test_neo4j_outage_on_a_previously_unprotected_route_returns_json_503(client, monkeypatch):
    """Regression test for the fix: before this session, only the newer
    evidence-aware endpoints (_try_load_campaign_context) handled a
    Neo4j outage gracefully. /api/overview had no such protection."""
    monkeypatch.setattr(dashboard_api, "driver", _RaisingDriver(ServiceUnavailable("down")))
    resp = client.get("/api/overview")
    assert resp.status_code == 503
    assert resp.is_json
    assert "Database unavailable" in resp.get_json()["error"]


@pytest.mark.parametrize("route", ["/api/campaigns", "/api/attackers", "/api/predictions", "/api/attack-chain"])
def test_neo4j_outage_returns_json_503_across_multiple_routes(client, monkeypatch, route):
    monkeypatch.setattr(dashboard_api, "driver", _RaisingDriver(ServiceUnavailable("down")))
    resp = client.get(route)
    assert resp.status_code == 503
    assert resp.is_json


def test_neo4j_query_error_is_distinguished_from_unavailability(client, monkeypatch):
    """A Neo4jError that isn't ServiceUnavailable means the database IS
    up and responded -- it rejected the query (bad Cypher, a constraint
    violation). Reported as 500 ("database query error"), not 503
    ("database unavailable") -- conflating the two was live-confirmed
    misleading this session (attribution_actors' now-fixed max(a,b) bug
    reported "unavailable" for what was actually a syntax error)."""
    monkeypatch.setattr(dashboard_api, "driver", _RaisingDriver(
        Neo4jError("Too many parameters for function 'max'")
    ))
    resp = client.get("/api/campaigns")
    assert resp.status_code == 500
    assert "query error" in resp.get_json()["error"].lower()
    assert "unavailable" not in resp.get_json()["error"].lower()


def test_malformed_int_query_param_does_not_leak_a_raw_traceback(client, mock_driver):
    """/api/graph/expand casts `depth` via int() with no validation --
    a non-numeric value raises ValueError uncaught by the route itself;
    must now be caught by the generic error handler, not crash into HTML."""
    mock_driver()
    resp = client.get("/api/graph/expand?node_id=abc&depth=not-a-number")
    assert resp.status_code == 500
    assert resp.is_json
    assert resp.get_json()["error"] == "Internal server error"  # no stack trace/internals leaked


def test_unexpected_engine_exception_does_not_leak_internals(client, monkeypatch):
    """A non-Neo4j exception (a bug in an internal engine, not a DB
    outage) must still reach the client as a clean JSON 500."""
    class _WeirdSession:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def run(self, *a, **k):
            raise ValueError("unexpected internal state")

    class _WeirdDriver:
        def session(self):
            return _WeirdSession()

    monkeypatch.setattr(dashboard_api, "driver", _WeirdDriver())
    resp = client.get("/api/campaigns")
    assert resp.status_code == 500
    assert resp.is_json
    assert "unexpected internal state" not in json.dumps(resp.get_json())


# ============================================================== health

def test_health_degraded_json_on_neo4j_failure(client, monkeypatch):
    monkeypatch.setattr(dashboard_api, "driver", _RaisingDriver(ServiceUnavailable("down")))
    resp = client.get("/api/health")
    assert resp.status_code == 500
    assert resp.get_json()["status"] == "degraded"
    assert resp.get_json()["neo4j"] == "disconnected"


@requires_live_neo4j
def test_health_live(client):
    resp = client.get("/api/health")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["status"] == "healthy"
    assert data["neo4j"] == "connected"


# ============================================================== overview (frontend: OverviewMetrics)

def test_overview_response_matches_frontend_contract(client, mock_driver):
    mock_driver(_FakeResult([{
        "total_events": 10, "active_campaigns": 3, "unique_attackers": 2,
        "techniques_detected": 5, "learned_transitions": 1, "total_hosts": 4,
    }]))
    resp = client.get("/api/overview")
    assert resp.status_code == 200
    data = resp.get_json()
    for field in ("total_events", "active_campaigns", "unique_attackers",
                  "techniques_detected", "learned_transitions", "total_hosts", "timestamp"):
        assert field in data


def test_overview_handles_all_null_counts_without_error(client, mock_driver):
    mock_driver(_FakeResult([{
        "total_events": None, "active_campaigns": None, "unique_attackers": None,
        "techniques_detected": None, "learned_transitions": None, "total_hosts": None,
    }]))
    resp = client.get("/api/overview")
    assert resp.status_code == 200
    assert resp.get_json()["total_events"] == 0


@requires_live_neo4j
def test_overview_live(client):
    resp = client.get("/api/overview")
    assert resp.status_code == 200
    data = resp.get_json()
    assert isinstance(data["active_campaigns"], int) and data["active_campaigns"] >= 0


# ============================================================== campaigns list

def test_campaigns_list_response_matches_frontend_contract(client, mock_driver):
    mock_driver(_FakeResult([{
        "campaign_id": "CAMP_1", "attacker_ip": "10.0.0.1", "victim_ip": "10.0.0.2",
        "first_seen": None, "last_seen": None, "event_count": 3,
        "raw_tps": 50.0, "latest_technique": "T1110", "predicted_technique": "T1078",
        "prediction_confidence": 0.8,
    }]))
    resp = client.get("/api/campaigns")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["count"] == 1
    row = data["campaigns"][0]
    for field in ("campaign_id", "campaign_label", "attacker_ip", "victim_ip",
                  "first_seen", "last_seen", "event_count", "risk_score",
                  "risk_level", "latest_technique", "predicted_technique", "prediction_confidence"):
        assert field in row


def test_campaigns_empty_result_returns_empty_list_not_error(client, mock_driver):
    mock_driver(_FakeResult([]))
    resp = client.get("/api/campaigns")
    assert resp.status_code == 200
    assert resp.get_json() == {"campaigns": [], "count": 0}


def test_campaigns_missing_victim_ip_defaults_to_unknown_not_none(client, mock_driver):
    mock_driver(_FakeResult([{
        "campaign_id": "CAMP_1", "attacker_ip": "10.0.0.1", "victim_ip": None,
        "first_seen": None, "last_seen": None, "event_count": 1,
        "raw_tps": 0, "latest_technique": None, "predicted_technique": None, "prediction_confidence": None,
    }]))
    resp = client.get("/api/campaigns")
    assert resp.get_json()["campaigns"][0]["victim_ip"] == "Unknown"


@requires_live_neo4j
def test_campaigns_list_live(client):
    resp = client.get("/api/campaigns")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["count"] == len(data["campaigns"])
    if data["campaigns"]:
        assert data["campaigns"][0]["campaign_id"].startswith("CAMP_")


# ============================================================== events / events/<id>

def test_events_list_pagination_fields_present(client, mock_driver):
    mock_driver(_FakeResult([{"total": 0}]), _FakeResult([]))
    resp = client.get("/api/events?page=2&page_size=10")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["page"] == 2
    assert data["page_size"] == 10
    assert data["total_pages"] == 0


def test_events_non_numeric_page_returns_controlled_error_not_html(client, mock_driver):
    mock_driver()
    resp = client.get("/api/events?page=not-a-number")
    assert resp.status_code == 500
    assert resp.is_json


def test_event_detail_not_found_returns_404(client, mock_driver):
    mock_driver(_FakeResult([]))
    resp = client.get("/api/events/nonexistent-id")
    assert resp.status_code == 404
    assert resp.get_json() == {"error": "Not found"}


@requires_live_neo4j
def test_events_list_live(client):
    resp = client.get("/api/events?page=1&page_size=5")
    assert resp.status_code == 200
    data = resp.get_json()
    assert len(data["events"]) <= 5
    for event in data["events"]:
        assert event["severity"] in ("CRITICAL", "HIGH", "MEDIUM", "LOW")


@requires_live_neo4j
def test_event_detail_live_for_a_real_event(client):
    with dashboard_api.driver.session() as session:
        row = session.run(
            "MATCH (:Campaign)-[:HAS_EVENT]->(e:AttackEvent) RETURN elementId(e) AS id LIMIT 1"
        ).single()
    if row is None:
        pytest.skip("no AttackEvent nodes in this live database")
    resp = client.get(f"/api/events/{row['id']}")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["id"] == row["id"]
    assert "recommendations" in data


# ============================================================== investigation (real engine, live-required for real work)

@requires_live_neo4j
def test_investigate_live_end_to_end_real_campaign(client):
    with dashboard_api.driver.session() as session:
        row = session.run("MATCH (c:Campaign) RETURN c.campaign_id AS id LIMIT 1").single()
    if row is None:
        pytest.skip("no Campaign nodes in this live database")

    resp = client.post(f"/api/investigate/{row['id']}", json={"max_steps": 3})
    assert resp.status_code == 200, resp.get_json()
    data = resp.get_json()
    assert data["campaign_id"] == row["id"]
    assert "steps" in data and isinstance(data["steps"], list)
    assert "stopping_reason" in data
    assert "evidence" in data
    json.dumps(data)  # the real end-to-end response must be JSON-serializable as delivered
    if data["steps"]:
        step = data["steps"][0]
        for field in ("why_selected", "candidate_actions", "action_scores",
                      "previous_confidence", "candidate_hypotheses"):
            assert field in step


def test_investigate_nonexistent_campaign_returns_controlled_error(client, mock_driver):
    mock_driver(_FakeResult([]))
    resp = client.post("/api/investigate/does-not-exist")
    assert resp.status_code == 404
    assert "error" in resp.get_json()


# ============================================================== ML severity prediction (live model + live Neo4j)

@requires_live_neo4j
def test_ml_predict_severity_live_end_to_end(client):
    import os
    model_path = os.path.join(os.path.dirname(dashboard_api.__file__), "ml", "models", "xgb_severity.json")
    if not os.path.exists(model_path):
        pytest.skip("no trained ml/models/xgb_severity.json in this environment")

    with dashboard_api.driver.session() as session:
        row = session.run(
            "MATCH (c:Campaign) WHERE c.last_technique IS NOT NULL RETURN c.campaign_id AS id LIMIT 1"
        ).single()
    if row is None:
        pytest.skip("no Campaign with a last_technique in this live database")

    resp = client.post("/api/ml/predict/severity", json={"campaign_id": row["id"]})
    assert resp.status_code == 200, resp.get_json()
    data = resp.get_json()
    assert data["campaign_id"] == row["id"]
    assert "label" in data and "probabilities" in data
    assert "top_k" in data and "model_metadata" in data and "prediction_context" in data
    assert data["prediction_context"]["campaign_id"] == row["id"]
    json.dumps(data)


# ============================================================== attribution (ThreatActor-node mechanism -- see THREAT_ATTRIBUTION.md)

def test_attribution_actors_response_matches_frontend_contract(client, mock_driver):
    mock_driver(_FakeResult([{
        "actor_name": "APT-Test", "confidence": 75.0, "description": "test actor",
        "shared_techniques": ["T1110"], "ta_malware": ["mal1"], "ta_tools": ["tool1"],
    }]))
    resp = client.get("/api/attribution/actors/CAMP_1")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["campaign_id"] == "CAMP_1"
    assert data["attribution"][0]["actor_name"] == "APT-Test"
    assert data["attribution"][0]["confidence"] == 75.0


def test_attribution_actors_no_candidates_returns_empty_list_not_error(client, mock_driver):
    mock_driver(_FakeResult([]))
    resp = client.get("/api/attribution/actors/CAMP_1")
    assert resp.status_code == 200
    assert resp.get_json()["attribution"] == []


@requires_live_neo4j
def test_attribution_actors_live_does_not_error_for_a_real_campaign(client):
    """This is the ThreatActor-node mechanism (distinct from
    ThreatAttributionEngine -- see THREAT_ATTRIBUTION.md), tested live
    for the first time this session. No claim about attribution
    quality: only that the route executes and returns its documented
    shape against real graph data."""
    with dashboard_api.driver.session() as session:
        row = session.run("MATCH (c:Campaign) RETURN c.campaign_id AS id LIMIT 1").single()
    if row is None:
        pytest.skip("no Campaign nodes in this live database")
    resp = client.get(f"/api/attribution/actors/{row['id']}")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["campaign_id"] == row["id"]
    assert isinstance(data["attribution"], list)


# ============================================================== correlation/campaigns (Cypher similarity, distinct from CampaignCorrelationEngine)

def test_correlation_campaigns_response_shape(client, mock_driver):
    mock_driver(_FakeResult([{
        "campaign_id": "CAMP_2", "raw_tps": 40.0, "similarity_score": 55,
        "shared_techniques": ["T1110"], "shared_tactics_raw": ["Credential Access"],
        "shared_attackers": ["10.0.0.1"], "shared_hosts": [],
    }]))
    resp = client.get("/api/correlation/campaigns/CAMP_1")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["campaign_id"] == "CAMP_1"
    assert data["similar_campaigns"][0]["campaign_id"] == "CAMP_2"


def test_correlation_campaigns_nonexistent_campaign_returns_empty_list_not_404(client, mock_driver):
    """Documented current behavior (not changed this session): unlike
    /api/risk/propagation, this route's Cypher MATCH on a nonexistent
    campaign_id simply returns zero rows -- indistinguishable from "no
    similar campaigns found" for a real one. See DASHBOARD_API.md."""
    mock_driver(_FakeResult([]))
    resp = client.get("/api/correlation/campaigns/does-not-exist")
    assert resp.status_code == 200
    assert resp.get_json()["similar_campaigns"] == []


@requires_live_neo4j
def test_correlation_campaigns_live(client):
    with dashboard_api.driver.session() as session:
        row = session.run("MATCH (c:Campaign) RETURN c.campaign_id AS id LIMIT 1").single()
    if row is None:
        pytest.skip("no Campaign nodes in this live database")
    resp = client.get(f"/api/correlation/campaigns/{row['id']}")
    assert resp.status_code == 200
    assert isinstance(resp.get_json()["similar_campaigns"], list)


# ============================================================== risk propagation

def test_risk_propagation_nonexistent_campaign_returns_404(client, mock_driver):
    mock_driver(_FakeResult([]))
    resp = client.get("/api/risk/propagation/does-not-exist")
    assert resp.status_code == 404


def test_risk_propagation_response_shape(client, mock_driver):
    mock_driver(_FakeResult([{
        "campaign_id": "CAMP_1", "campaign_tps": 60.0, "attacker_ip": "10.0.0.1",
        "host_ip": "10.0.0.2",
        "techniques": [{"id": "T1110", "name": "Brute Force", "score": 60.0}],
    }]))
    resp = client.get("/api/risk/propagation/CAMP_1")
    assert resp.status_code == 200
    data = resp.get_json()
    node_types = {n["node_type"] for n in data["propagation"]}
    assert {"Attacker", "Campaign", "Technique", "Host"} <= node_types


@requires_live_neo4j
def test_risk_propagation_live(client):
    with dashboard_api.driver.session() as session:
        row = session.run("MATCH (c:Campaign) RETURN c.campaign_id AS id LIMIT 1").single()
    if row is None:
        pytest.skip("no Campaign nodes in this live database")
    resp = client.get(f"/api/risk/propagation/{row['id']}")
    assert resp.status_code == 200
    assert resp.get_json()["campaign_id"] == row["id"]


# ============================================================== campaign timeline

def test_campaign_timeline_response_shape(client, mock_driver):
    class _FakeTs:
        def iso_format(self):
            return "2026-01-01T00:00:00+00:00"

    mock_driver(_FakeResult([{
        "id": "evt-1", "first_seen": _FakeTs(), "last_seen": _FakeTs(), "occurrences": 2,
        "technique_id": "T1110", "technique_name": "Brute Force", "stage": "Credential Access",
    }]))
    resp = client.get("/api/campaigns/CAMP_1/timeline")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["campaign_id"] == "CAMP_1"
    assert data["events"][0]["technique_id"] == "T1110"


def test_campaign_timeline_nonexistent_campaign_returns_empty_not_error(client, mock_driver):
    mock_driver(_FakeResult([]))
    resp = client.get("/api/campaigns/does-not-exist/timeline")
    assert resp.status_code == 200
    assert resp.get_json()["events"] == []


# ============================================================== attackers

def test_attackers_response_matches_frontend_contract(client, mock_driver):
    mock_driver(_FakeResult([{
        "attacker_ip": "10.0.0.1", "campaign_count": 2, "event_count": 5, "raw_tps": 80.0,
        "first_seen": None, "last_seen": None, "techniques": ["T1110", None], "predicted_technique": "T1078",
    }]))
    resp = client.get("/api/attackers")
    assert resp.status_code == 200
    data = resp.get_json()
    row = data["attackers"][0]
    assert row["techniques_observed"] == ["T1110"]  # None filtered out
    for field in ("attacker_ip", "campaign_count", "event_count", "risk_score", "risk_level", "first_seen", "last_seen"):
        assert field in row


@requires_live_neo4j
def test_attackers_live(client):
    resp = client.get("/api/attackers")
    assert resp.status_code == 200
    assert resp.get_json()["count"] == len(resp.get_json()["attackers"])


# ============================================================== graph/expand

def test_graph_expand_missing_node_id_returns_400(client):
    resp = client.get("/api/graph/expand")
    assert resp.status_code == 400


def test_graph_expand_unknown_node_returns_empty_graph(client, mock_driver):
    mock_driver(_FakeResult([]))
    resp = client.get("/api/graph/expand?node_id=nonexistent")
    assert resp.status_code == 200
    assert resp.get_json() == {"nodes": [], "links": []}


def test_graph_expand_response_shape_with_real_node_objects(client, mock_driver):
    center = _FakeNode("n1", ["Technique"], {"attack_id": "T1110", "name": "Brute Force"})
    neighbor = _FakeNode("n2", ["ThreatActor"], {"name": "APT-X"})
    rel = _FakeRel(center, neighbor, "RESEMBLES")
    mock_driver(
        _FakeResult([{"node_type": "Technique", "attack_id": "T1110"}]),
        _FakeResult([{"all_nodes": [center, neighbor], "all_rels": [rel]}]),
    )
    resp = client.get("/api/graph/expand?node_id=n1&depth=2")
    assert resp.status_code == 200
    data = resp.get_json()
    assert len(data["nodes"]) == 2
    assert data["links"][0] == {"source": "n1", "target": "n2", "label": "RESEMBLES"}


@requires_live_neo4j
def test_graph_expand_live_for_a_real_technique_node(client):
    with dashboard_api.driver.session() as session:
        row = session.run("MATCH (t:Technique) RETURN elementId(t) AS id LIMIT 1").single()
    if row is None:
        pytest.skip("no Technique nodes in this live database")
    resp = client.get(f"/api/graph/expand?node_id={row['id']}&depth=1")
    assert resp.status_code == 200
    data = resp.get_json()
    assert "nodes" in data and "links" in data


# ============================================================== graph (default/investigation/threat_intel layers)

@pytest.mark.parametrize("view_mode", ["campaign", "investigation", "threat_intel"])
def test_graph_all_layers_return_nodes_and_links_keys(client, mock_driver, view_mode):
    mock_driver(*[_FakeResult([]) for _ in range(6)])  # generous: covers every layer's query count
    resp = client.get(f"/api/graph?view_mode={view_mode}")
    assert resp.status_code == 200
    data = resp.get_json()
    assert "nodes" in data and "links" in data


@requires_live_neo4j
def test_graph_default_layer_live(client):
    resp = client.get("/api/graph")
    assert resp.status_code == 200
    data = resp.get_json()
    assert isinstance(data["nodes"], list) and isinstance(data["links"], list)


# ============================================================== query console

def test_query_missing_body_returns_400(client):
    resp = client.post("/api/query", json={})
    assert resp.status_code == 400


def test_query_rejects_mutation_keywords(client):
    resp = client.post("/api/query", json={"query": "CREATE (n:Test) RETURN n"})
    assert resp.status_code == 403
    assert "forbidden" in resp.get_json()["error"].lower()


@pytest.mark.parametrize("mutation", ["MERGE (n:Test)", "DELETE n", "DETACH DELETE n", "SET n.x = 1", "DROP INDEX foo"])
def test_query_rejects_every_mutation_keyword(client, mutation):
    resp = client.post("/api/query", json={"query": f"MATCH (n) {mutation}"})
    assert resp.status_code == 403


def test_query_invalid_cypher_returns_400_not_500(client, monkeypatch):
    class _RaisingSession(_QueueSession):
        def execute_read(self, fn, query):
            raise ValueError("Invalid input 'X': expected ...")

    class _RaisingQueryDriver:
        def session(self):
            return _RaisingSession()

    # monkeypatch.setattr (not a raw `dashboard_api.driver = ...`) so this
    # reverts automatically after the test -- a raw assignment here
    # previously leaked into later tests in the same session, including
    # the live /api/query test below, which then hit this fake driver
    # instead of the real one.
    monkeypatch.setattr(dashboard_api, "driver", _RaisingQueryDriver())
    resp = client.post("/api/query", json={"query": "NOT VALID CYPHER"})
    assert resp.status_code == 400
    assert "error" in resp.get_json()


@requires_live_neo4j
def test_query_live_read_only_count(client):
    resp = client.post("/api/query", json={"query": "MATCH (c:Campaign) RETURN count(c) AS n"})
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["records"][0]["n"] >= 0


# ============================================================== predictions / predict / recommendations / attack-chain / analytics/paths

def test_predict_requires_current_technique(client):
    resp = client.post("/api/predict", json={})
    assert resp.status_code == 400


def test_predict_unknown_technique_returns_null_prediction_not_error(client, monkeypatch):
    import prediction_engine
    monkeypatch.setattr(prediction_engine, "predict_next_readonly", lambda t: None)
    resp = client.post("/api/predict", json={"current_technique": "T9999"})
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["predicted"] is None
    assert data["confidence"] == 0


def test_predict_uses_the_readonly_predictor_not_the_mutating_one(client, monkeypatch):
    """Regression test for the fix: /api/predict has no campaign_id in
    its request contract and must never trigger predict_next's Neo4j
    write side effect. Spies on both functions to prove which one this
    route actually calls."""
    import prediction_engine

    calls = {"readonly": 0, "mutating": 0}
    monkeypatch.setattr(prediction_engine, "predict_next_readonly", lambda t: calls.__setitem__("readonly", calls["readonly"] + 1) or {"current": t, "predicted": "T1078", "confidence": 80.0})
    monkeypatch.setattr(prediction_engine, "predict_next", lambda *a, **k: calls.__setitem__("mutating", calls["mutating"] + 1) or (_ for _ in ()).throw(AssertionError("mutating predict_next must not be called from a stateless API route")))

    resp = client.post("/api/predict", json={"current_technique": "T1110"})

    assert resp.status_code == 200
    assert calls["readonly"] == 1
    assert calls["mutating"] == 0


@requires_live_neo4j
def test_predictions_live(client):
    resp = client.get("/api/predictions")
    assert resp.status_code == 200
    assert "predictions" in resp.get_json()


def test_recommendations_with_technique_param(client, monkeypatch):
    import recommendation_engine
    monkeypatch.setattr(recommendation_engine, "get_recommendations", lambda t: ["Patch system"])
    resp = client.get("/api/recommendations?technique=T1110")
    assert resp.status_code == 200
    assert resp.get_json() == {"technique": "T1110", "recommendations": ["Patch system"]}


def test_attack_chain_without_campaign_returns_transitions(client, mock_driver):
    mock_driver(_FakeResult([]))
    resp = client.get("/api/attack-chain")
    assert resp.status_code == 200
    data = resp.get_json()
    assert "transitions" in data and "chain" in data


def test_attack_chain_with_campaign_returns_chain(client, mock_driver):
    mock_driver(_FakeResult([{
        "technique_id": "T1110", "name": "Brute Force", "stage": "Credential Access",
        "detection_count": 3, "tps": 50.0, "first_seen": None,
    }]))
    resp = client.get("/api/attack-chain?campaign=CAMP_1")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["campaign_id"] == "CAMP_1"
    assert data["chain"][0]["technique_id"] == "T1110"


@requires_live_neo4j
def test_analytics_paths_live(client):
    resp = client.get("/api/analytics/paths")
    assert resp.status_code == 200
    assert "paths" in resp.get_json()


# ============================================================== frontend field-shape cross-check (source of truth: services/api.ts)

def test_all_frontend_consumed_routes_are_registered():
    """frontend/src/services/api.ts is the authoritative list of routes
    the real React app calls -- every one of them must actually be
    registered (guards against a route being renamed/removed on the
    backend without the frontend noticing until runtime)."""
    frontend_consumed = [
        "/api/health", "/api/overview", "/api/events", "/api/events/<event_id>",
        "/api/graph", "/api/graph/expand", "/api/graph/paths", "/api/campaigns",
        "/api/campaigns/<campaign_id>/timeline", "/api/correlation/campaigns/<campaign_id>",
        "/api/analytics/paths", "/api/attribution/actors/<campaign_id>",
        "/api/risk/propagation/<campaign_id>", "/api/attackers", "/api/attack-chain",
        "/api/predictions", "/api/predict", "/api/recommendations", "/api/query",
        "/api/investigate/<campaign_id>", "/api/rag/mitre/search", "/api/ml/predict/severity",
    ]
    registered = {r.rule for r in dashboard_api.app.url_map.iter_rules()}
    missing = [r for r in frontend_consumed if r not in registered]
    assert not missing, f"frontend calls these routes but they are not registered: {missing}"
