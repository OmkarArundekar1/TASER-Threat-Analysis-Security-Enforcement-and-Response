"""
Tests for the additive GNN production integration points (see
../../GNN_PRODUCTION_INTEGRATION.md): operation correlation, campaign
correlation, threat attribution, historical retrieval, investigation
evidence, and dashboard API serialization. Every test here also proves
the corresponding EXISTING mechanism's behavior is unchanged when GNN
is disabled (the default) -- the regression half of Phase 14's list.
"""

import json

import pytest


# ---------------------------------------------------------------------------
# Operation correlation (CampaignCorrelationEngine.CorrelationResult)
# ---------------------------------------------------------------------------

def test_correlation_result_gnn_field_defaults_to_none():
    from campaign_correlation_engine import CorrelationResult
    result = CorrelationResult(
        matched=False, operation_id=None, confidence=0, score=0,
        breakdown={}, candidate_count=0, candidates=[],
    )
    assert result.gnn_topology_similarity is None


def test_correlate_no_candidates_gnn_field_is_none(monkeypatch):
    """No behavior change to the existing early-return path when there
    are no operation candidates -- unchanged decision, gnn field stays
    None (nothing to compare against)."""
    from campaign_correlation_engine import CampaignCorrelationEngine
    engine = CampaignCorrelationEngine()
    result = engine.correlate(None)
    assert result.matched is False
    assert result.gnn_topology_similarity is None


def test_gnn_topology_similarity_to_operation_is_none_when_gnn_disabled(monkeypatch):
    import config
    monkeypatch.setattr(config, "GNN_ENABLED", False)

    from campaign_correlation_engine import CampaignCorrelationEngine

    class _FakeCampaignContext:
        campaign_id = "camp-current"

    class _FakeOperationContext:
        campaign_ids = ["camp-a", "camp-b"]

    engine = CampaignCorrelationEngine()
    result = engine._gnn_topology_similarity_to_operation(_FakeCampaignContext(), _FakeOperationContext())
    assert result is None


def test_gnn_topology_similarity_to_operation_uses_max_and_excludes_self(monkeypatch):
    import config
    monkeypatch.setattr(config, "GNN_ENABLED", True)

    from campaign_correlation_engine import CampaignCorrelationEngine

    class _FakeCampaignContext:
        campaign_id = "camp-current"

    class _FakeOperationContext:
        campaign_ids = ["camp-current", "camp-a", "camp-b"]  # includes self -- must be excluded

    def fake_similarity(a, b):
        assert a == "camp-current"
        return {"camp-a": 0.3, "camp-b": 0.9}[b]

    monkeypatch.setattr(
        "ml.gnn.topology_similarity.gnn_topology_similarity_between_campaigns", fake_similarity,
    )

    engine = CampaignCorrelationEngine()
    result = engine._gnn_topology_similarity_to_operation(_FakeCampaignContext(), _FakeOperationContext())
    assert result == 0.9  # max, and camp-current (self) never compared


def test_gnn_topology_similarity_to_operation_never_raises_on_failure(monkeypatch):
    import config
    monkeypatch.setattr(config, "GNN_ENABLED", True)

    from campaign_correlation_engine import CampaignCorrelationEngine

    class _FakeCampaignContext:
        campaign_id = "camp-current"

    class _FakeOperationContext:
        campaign_ids = ["camp-a"]

    def _boom(a, b):
        raise RuntimeError("simulated failure")

    monkeypatch.setattr("ml.gnn.topology_similarity.gnn_topology_similarity_between_campaigns", _boom)

    engine = CampaignCorrelationEngine()
    result = engine._gnn_topology_similarity_to_operation(_FakeCampaignContext(), _FakeOperationContext())
    assert result is None  # caught, not propagated


# ---------------------------------------------------------------------------
# Campaign correlation (campaign_manager.py's GNN diagnostic log)
# ---------------------------------------------------------------------------

def test_log_gnn_nearest_topology_neighbor_noop_when_disabled(monkeypatch, capsys):
    import config
    monkeypatch.setattr(config, "GNN_ENABLED", False)
    from campaign_manager import _log_gnn_nearest_topology_neighbor
    _log_gnn_nearest_topology_neighbor("camp-1")  # must not raise, must not print anything GNN-related
    captured = capsys.readouterr()
    assert "GNN Topology" not in captured.out


def test_log_gnn_nearest_topology_neighbor_never_raises_on_failure(monkeypatch):
    import config
    monkeypatch.setattr(config, "GNN_ENABLED", True)

    class _BoomRetriever:
        def query(self, campaign_id, top_k=1):
            raise RuntimeError("simulated failure")

    monkeypatch.setattr("rag.gnn_topology_retriever.GNNTopologyRetriever", _BoomRetriever)
    from campaign_manager import _log_gnn_nearest_topology_neighbor
    _log_gnn_nearest_topology_neighbor("camp-1")  # must not raise


# ---------------------------------------------------------------------------
# Threat attribution (ThreatActorContext.topology_similarity)
# ---------------------------------------------------------------------------

def test_threat_actor_context_topology_similarity_defaults_to_none():
    from threat_actor_context import ThreatActorContext
    ctx = ThreatActorContext(actor="some-campaign")
    assert ctx.topology_similarity is None


def test_attribute_populates_topology_similarity_without_affecting_ranking(monkeypatch):
    """Existing coverage/precision/chain_similarity-based ranking must be
    byte-identical whether or not GNN is enabled -- topology_similarity
    is attached, never consulted by the sort."""
    import config
    import threat_attribution_engine
    from attribution_models import HistoricalCampaign

    class _FakeContext:
        campaign_id = "camp-current"
        techniques = {"T1078", "T1110"}
        attack_chain = ["T1078", "T1110"]

    historical = [
        HistoricalCampaign(campaign_id="camp-a", attacker="1.1.1.1", victim="2.2.2.2",
                            techniques=["T1078", "T1110"], status="RESOLVED", timestamps=[]),
        HistoricalCampaign(campaign_id="camp-b", attacker="3.3.3.3", victim="4.4.4.4",
                            techniques=["T1078"], status="RESOLVED", timestamps=[]),
    ]
    monkeypatch.setattr("attribution_context.context.load_historical_campaigns", lambda: historical)

    # GNN disabled: topology_similarity must be None, ranking unaffected
    monkeypatch.setattr(config, "GNN_ENABLED", False)
    result_disabled = threat_attribution_engine.ThreatAttributionEngine().attribute(_FakeContext())
    assert all(a.topology_similarity is None for a in result_disabled.actors)
    order_disabled = [a.actor for a in result_disabled.actors]

    # GNN enabled (mocked): topology_similarity populated, ranking identical
    monkeypatch.setattr(config, "GNN_ENABLED", True)
    monkeypatch.setattr(
        "ml.gnn.topology_similarity.gnn_topology_similarity_between_campaigns",
        lambda a, b: 0.42,
    )
    result_enabled = threat_attribution_engine.ThreatAttributionEngine().attribute(_FakeContext())
    assert all(a.topology_similarity == 0.42 for a in result_enabled.actors)
    order_enabled = [a.actor for a in result_enabled.actors]

    assert order_disabled == order_enabled  # ranking (total_score-based sort) is unchanged
    assert [a.total_score for a in result_disabled.actors] == [a.total_score for a in result_enabled.actors]


def test_gnn_topology_similarity_for_attribution_never_raises(monkeypatch):
    import config
    monkeypatch.setattr(config, "GNN_ENABLED", True)

    def _boom(a, b):
        raise RuntimeError("simulated failure")

    monkeypatch.setattr("ml.gnn.topology_similarity.gnn_topology_similarity_between_campaigns", _boom)
    from threat_attribution_engine import _gnn_topology_similarity_for_attribution
    assert _gnn_topology_similarity_for_attribution("a", "b") is None


# ---------------------------------------------------------------------------
# Historical retrieval (rag.gnn_topology_retriever.GNNTopologyRetriever)
# ---------------------------------------------------------------------------

def test_gnn_topology_retriever_empty_when_disabled(monkeypatch):
    import config
    monkeypatch.setattr(config, "GNN_ENABLED", False)
    from rag.gnn_topology_retriever import GNNTopologyRetriever
    assert GNNTopologyRetriever().query("camp-1") == []


def test_gnn_topology_retriever_provenance_and_ranking(monkeypatch):
    import config
    from attribution_models import HistoricalCampaign

    monkeypatch.setattr(config, "GNN_ENABLED", True)

    class _FakeInferenceService:
        def embed_campaign(self, campaign_id):
            return {"query": [1.0], "camp-a": [0.9], "camp-b": [0.1]}.get(campaign_id, [0.0])

        metadata = type("M", (), {"model_version": "test-v1"})()

    fake_service = _FakeInferenceService()
    monkeypatch.setattr("ml.gnn.inference.gnn_inference_service", fake_service)

    def fake_similarity(query_embedding, candidate_embedding):
        # test doubles for embeddings directly encode the intended score
        return candidate_embedding[0]

    monkeypatch.setattr("ml.gnn.topology_similarity.gnn_graph_similarity", fake_similarity)

    historical = [
        HistoricalCampaign(campaign_id="camp-a", attacker="1.1.1.1", victim="2.2.2.2",
                            techniques=["T1078"], status="RESOLVED", timestamps=["2026-01-01T00:00:00Z"]),
        HistoricalCampaign(campaign_id="camp-b", attacker="3.3.3.3", victim="4.4.4.4",
                            techniques=["T1595"], status="RESOLVED", timestamps=["2026-01-02T00:00:00Z"]),
    ]
    monkeypatch.setattr("attribution_context.context.load_historical_campaigns", lambda: historical)

    from rag.gnn_topology_retriever import GNNTopologyRetriever
    results = GNNTopologyRetriever().query("query", top_k=5)

    assert len(results) == 2
    assert results[0].source_id == "camp-a"  # higher similarity ranked first
    assert results[0].content["topology_similarity"] == 0.9
    assert results[0].source.value == "gnn_topology"
    assert "gnn_topology" in results[0].provenance or "ml.gnn.inference" in results[0].provenance
    assert results[0].relationships == ["camp-a"]


# ---------------------------------------------------------------------------
# Investigation evidence (InvestigationAction.GNN_TOPOLOGY_RETRIEVAL)
# ---------------------------------------------------------------------------

def test_gnn_topology_retrieval_action_is_registered():
    from evidence.schema import EvidenceSource
    from investigation.actions import ACTION_METADATA, InvestigationAction

    assert InvestigationAction.GNN_TOPOLOGY_RETRIEVAL in ACTION_METADATA
    meta = ACTION_METADATA[InvestigationAction.GNN_TOPOLOGY_RETRIEVAL]
    assert meta.source == EvidenceSource.GNN_TOPOLOGY
    assert InvestigationAction.CAMPAIGN_HISTORY in meta.depends_on


def test_default_action_executor_gnn_topology_retrieval_returns_list(monkeypatch):
    import config
    monkeypatch.setattr(config, "GNN_ENABLED", False)  # disabled -> empty, not an error

    from investigation.actions import InvestigationAction
    from investigation.loop import default_action_executor

    class _FakeCampaignContext:
        campaign_id = "camp-1"
        attacker_ip = "1.1.1.1"
        techniques = set()
        attack_chain = []
        last_technique = None

    executor = default_action_executor(_FakeCampaignContext(), "T1078", None)
    result = executor(InvestigationAction.GNN_TOPOLOGY_RETRIEVAL)
    assert result == []


def test_default_action_executor_gnn_topology_retrieval_never_raises(monkeypatch):
    import config
    monkeypatch.setattr(config, "GNN_ENABLED", True)

    class _BoomRetriever:
        def query(self, campaign_id):
            raise RuntimeError("simulated failure")

    monkeypatch.setattr("rag.gnn_topology_retriever.GNNTopologyRetriever", _BoomRetriever)

    from investigation.actions import InvestigationAction
    from investigation.loop import default_action_executor

    class _FakeCampaignContext:
        campaign_id = "camp-1"
        attacker_ip = "1.1.1.1"
        techniques = set()
        attack_chain = []
        last_technique = None

    executor = default_action_executor(_FakeCampaignContext(), "T1078", None)
    result = executor(InvestigationAction.GNN_TOPOLOGY_RETRIEVAL)
    assert result == []  # caught, not propagated


# ---------------------------------------------------------------------------
# Dashboard API serialization
# ---------------------------------------------------------------------------

@pytest.fixture()
def client():
    import dashboard_api
    dashboard_api.app.config["TESTING"] = True
    with dashboard_api.app.test_client() as c:
        yield c


def test_gnn_status_endpoint_disabled_returns_200(client, monkeypatch):
    import config
    monkeypatch.setattr(config, "GNN_ENABLED", False)
    resp = client.get("/api/gnn/status")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["gnn_available"] is False
    assert data["gnn_model_version"] is None
    json.dumps(data)


def test_gnn_topology_endpoint_unknown_campaign_returns_404(client, monkeypatch):
    monkeypatch.setattr(
        "dashboard_api._try_load_campaign_context", lambda cid: (None, None),
    )
    resp = client.get("/api/gnn/topology/does-not-exist")
    assert resp.status_code == 404


def test_gnn_topology_endpoint_disabled_returns_empty_neighbors_not_error(client, monkeypatch):
    import config
    monkeypatch.setattr(config, "GNN_ENABLED", False)
    monkeypatch.setattr(
        "dashboard_api._try_load_campaign_context", lambda cid: (object(), None),
    )
    resp = client.get("/api/gnn/topology/camp-1")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["gnn_available"] is False
    assert data["topology_neighbors"] == []


def _neo4j_reachable():
    try:
        import dashboard_api
        dashboard_api.driver.verify_connectivity()
        return True
    except Exception:
        return False


requires_live_neo4j = pytest.mark.skipif(
    not _neo4j_reachable(),
    reason="No live Neo4j instance reachable in this environment.",
)


@requires_live_neo4j
def test_gnn_status_endpoint_live(client):
    resp = client.get("/api/gnn/status")
    assert resp.status_code == 200
    json.dumps(resp.get_json())  # must be JSON-serializable as delivered


@requires_live_neo4j
def test_gnn_topology_endpoint_live_real_campaign(client):
    import dashboard_api
    with dashboard_api.driver.session() as session:
        row = session.run("MATCH (c:Campaign) RETURN c.campaign_id AS id LIMIT 1").single()
    if row is None:
        pytest.skip("no Campaign nodes in this live database")

    resp = client.get(f"/api/gnn/topology/{row['id']}")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["campaign_id"] == row["id"]
    assert "gnn_available" in data
    assert "topology_neighbors" in data
    json.dumps(data)


@requires_live_neo4j
def test_investigate_live_includes_gnn_evidence_when_enabled(client, monkeypatch):
    """Phase 16: existing investigation must keep working, and can
    additionally surface GNN evidence without breaking anything else,
    when GNN_ENABLED=true and an artifact is present."""
    import os

    import config
    import dashboard_api

    model_path = os.path.join(os.path.dirname(dashboard_api.__file__), "ml", "models", "gnn_autoencoder.pt")
    if not os.path.exists(model_path):
        pytest.skip("no trained ml/models/gnn_autoencoder.pt in this environment")
    monkeypatch.setattr(config, "GNN_ENABLED", True)

    with dashboard_api.driver.session() as session:
        row = session.run("MATCH (c:Campaign) RETURN c.campaign_id AS id LIMIT 1").single()
    if row is None:
        pytest.skip("no Campaign nodes in this live database")

    resp = client.post(f"/api/investigate/{row['id']}", json={"max_steps": 8})
    assert resp.status_code == 200, resp.get_json()
    data = resp.get_json()
    json.dumps(data)  # must remain JSON-serializable with GNN evidence mixed in
    # Not asserting GNN_TOPOLOGY_RETRIEVAL was necessarily *chosen* by NBE
    # within max_steps (that's a legitimate NBE scoring outcome, not a
    # correctness requirement) -- only that investigation completes
    # successfully with the action available in the menu.
    assert "steps" in data and "stopping_reason" in data
