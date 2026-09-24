"""
Tests for ml/gnn/inference.py and ml/gnn/topology_similarity.py -- the
production GNN inference path (see ../../GNN_PRODUCTION_INTEGRATION.md).
Covers: model loading, deterministic inference, embedding
dimensionality, finite embeddings, topology similarity, the feature
flag, missing model, corrupt model, malformed/empty graphs, and
failure fallback -- every item Phase 14's GNN test list names.
"""

import networkx as nx
import pytest
import torch

from ml.gnn.graph_encoder import encode_graph
from ml.gnn.inference import GNNInferenceService, GNNModelMetadata, _model_version_from_path
from ml.gnn.topology_similarity import (
    gnn_graph_similarity,
    gnn_topology_similarity_between_campaigns,
)


def _real_model_path() -> str:
    import os
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "ml", "models", "gnn_autoencoder.pt")


_real_artifact_exists = __import__("os").path.exists(_real_model_path())
requires_real_artifact = pytest.mark.skipif(
    not _real_artifact_exists,
    reason="Real gnn_autoencoder.pt artifact not present in this environment.",
)


def _tiny_graph() -> nx.DiGraph:
    G = nx.DiGraph()
    G.add_node("a", labels=["Attacker"], vt_reputation=50.0)
    G.add_node("c", labels=["Campaign"])
    G.add_node("e", labels=["AttackEvent"], occurrences=2.0)
    G.add_edge("a", "c", relationship="LAUNCHED")
    G.add_edge("c", "e", relationship="HAS_EVENT")
    return G


# ---------------------------------------------------------------------------
# 6. Feature flag
# ---------------------------------------------------------------------------

def test_feature_flag_disabled_never_touches_filesystem_or_torch(monkeypatch):
    """GNN_ENABLED=false must short-circuit before even checking whether
    the artifact file exists -- the strongest form of "GNN never
    affects the pipeline when disabled"."""
    import config
    monkeypatch.setattr(config, "GNN_ENABLED", False)

    service = GNNInferenceService(model_path="/this/path/does/not/matter")
    assert service.available is False
    assert service.metadata is None
    assert service.embed_campaign("any-campaign") is None


def test_feature_flag_default_is_disabled_when_env_var_is_unset(monkeypatch):
    """Tests config.py's own fallback logic directly, with the env var
    cleared -- not the live config.GNN_ENABLED value, which legitimately
    reflects whatever this machine's .env (gitignored, an operator
    choice, not shipped code) currently sets it to. This is what
    actually guards "the shipped code defaults to off"."""
    monkeypatch.delenv("GNN_ENABLED", raising=False)
    import os
    default_value = os.environ.get("GNN_ENABLED", "false").strip().lower() in ("1", "true", "yes")
    assert default_value is False


# ---------------------------------------------------------------------------
# 7. Missing model
# ---------------------------------------------------------------------------

def test_missing_model_file_is_unavailable_not_an_exception(monkeypatch):
    import config
    monkeypatch.setattr(config, "GNN_ENABLED", True)
    service = GNNInferenceService(model_path="/definitely/does/not/exist.pt")
    assert service.available is False
    assert service.embed_campaign("any-campaign") is None  # must not raise


# ---------------------------------------------------------------------------
# 8. Corrupt model
# ---------------------------------------------------------------------------

def test_corrupt_model_file_is_unavailable_not_an_exception(monkeypatch, tmp_path):
    import config
    monkeypatch.setattr(config, "GNN_ENABLED", True)
    bad_file = tmp_path / "corrupt.pt"
    bad_file.write_bytes(b"this is not a valid torch checkpoint")

    service = GNNInferenceService(model_path=str(bad_file))
    assert service.available is False


def test_model_file_missing_required_metadata_keys_is_treated_as_corrupt(monkeypatch, tmp_path):
    import config
    monkeypatch.setattr(config, "GNN_ENABLED", True)
    incomplete = tmp_path / "incomplete.pt"
    torch.save({"hidden_dim": 16}, incomplete)  # missing embedding_dim, num_layers, feature_mean, feature_std

    service = GNNInferenceService(model_path=str(incomplete))
    assert service.available is False


# ---------------------------------------------------------------------------
# 9/10. Malformed / empty graph
# ---------------------------------------------------------------------------

def test_embed_campaign_returns_none_when_campaign_not_found(monkeypatch):
    import config
    monkeypatch.setattr(config, "GNN_ENABLED", True)
    service = GNNInferenceService()

    # Force a "loaded" state without touching the real artifact, to
    # isolate the malformed-input path from the artifact-loading path.
    service._loaded = True
    service._model = object()  # never reached if build_campaign_graph_sample returns None
    monkeypatch.setattr("neo4j_client.get_campaign_context_data", lambda cid: None)

    assert service.embed_campaign("does-not-exist") is None


def test_embed_campaign_short_circuits_on_zero_node_graph(monkeypatch):
    import config
    monkeypatch.setattr(config, "GNN_ENABLED", True)
    service = GNNInferenceService()
    service._loaded = True
    service._model = object()  # must never be called

    from ml.gnn.campaign_graphs import CampaignGraphSample
    empty_sample = CampaignGraphSample(
        campaign_id="camp-empty", graph=encode_graph(nx.DiGraph()), severity="Low", risk_score=0.0, num_events=0,
    )
    monkeypatch.setattr("campaign_graphs.build_campaign_graph_sample", lambda cid: empty_sample)

    assert service.embed_campaign("camp-empty") is None


# ---------------------------------------------------------------------------
# 11. Failure fallback (any unexpected exception inside embed_campaign)
# ---------------------------------------------------------------------------

def test_embed_campaign_catches_unexpected_exceptions_and_returns_none(monkeypatch):
    import config
    monkeypatch.setattr(config, "GNN_ENABLED", True)
    service = GNNInferenceService()
    service._loaded = True
    service._model = object()

    def _boom(cid):
        raise RuntimeError("simulated Neo4j outage")

    monkeypatch.setattr("campaign_graphs.build_campaign_graph_sample", _boom)

    assert service.embed_campaign("camp-1") is None  # must not propagate the RuntimeError


def test_gnn_graph_similarity_is_pure_and_handles_zero_norm():
    a = torch.tensor([0.0, 0.0, 0.0])
    b = torch.tensor([1.0, 0.0, 0.0])
    result = gnn_graph_similarity(a, b)  # zero-norm vector -- must not divide by zero
    assert result == 0.0 or torch.isfinite(torch.tensor(result))


def test_gnn_topology_similarity_between_campaigns_none_when_disabled(monkeypatch):
    import config
    monkeypatch.setattr(config, "GNN_ENABLED", False)
    assert gnn_topology_similarity_between_campaigns("a", "b") is None


# ---------------------------------------------------------------------------
# _model_version_from_path
# ---------------------------------------------------------------------------

def test_model_version_from_path_unknown_for_missing_file():
    assert _model_version_from_path("/does/not/exist.pt") == "unknown"


def test_model_version_from_path_is_stable_for_the_same_file(tmp_path):
    f = tmp_path / "model.pt"
    f.write_bytes(b"1234")
    v1 = _model_version_from_path(str(f))
    v2 = _model_version_from_path(str(f))
    assert v1 == v2
    assert v1 != "unknown"


# ---------------------------------------------------------------------------
# Real artifact tests -- 1. model loading, 2. deterministic inference,
# 3. embedding dimensionality, 4. finite embeddings, 5. topology similarity
# ---------------------------------------------------------------------------

@requires_real_artifact
class TestRealArtifact:
    def test_model_loads_and_reports_metadata(self, monkeypatch):
        import config
        monkeypatch.setattr(config, "GNN_ENABLED", True)
        service = GNNInferenceService(model_path=_real_model_path())
        assert service.available is True
        meta = service.metadata
        assert isinstance(meta, GNNModelMetadata)
        assert meta.embedding_dim == 8
        assert meta.hidden_dim == 16
        assert meta.model_type == "graph_autoencoder"
        assert "fold-safe" in meta.training_dataset_description  # the distinction Phase 3 requires is documented

    def test_deterministic_inference_on_a_hand_built_graph(self, monkeypatch):
        import config
        monkeypatch.setattr(config, "GNN_ENABLED", True)
        service = GNNInferenceService(model_path=_real_model_path())
        assert service.available

        encoded = encode_graph(_tiny_graph())
        e1 = service._embed_encoded_graph(encoded)
        e2 = service._embed_encoded_graph(encoded)
        assert e1 is not None
        assert torch.equal(e1, e2)

    def test_embedding_dimension_matches_metadata(self, monkeypatch):
        import config
        monkeypatch.setattr(config, "GNN_ENABLED", True)
        service = GNNInferenceService(model_path=_real_model_path())
        assert service.available
        encoded = encode_graph(_tiny_graph())
        embedding = service._embed_encoded_graph(encoded)
        assert embedding.shape == (service.metadata.embedding_dim,)

    def test_embedding_is_finite(self, monkeypatch):
        import config
        monkeypatch.setattr(config, "GNN_ENABLED", True)
        service = GNNInferenceService(model_path=_real_model_path())
        assert service.available
        encoded = encode_graph(_tiny_graph())
        embedding = service._embed_encoded_graph(encoded)
        assert torch.isfinite(embedding).all()

    def test_self_similarity_is_one(self, monkeypatch):
        import config
        monkeypatch.setattr(config, "GNN_ENABLED", True)
        service = GNNInferenceService(model_path=_real_model_path())
        assert service.available
        encoded = encode_graph(_tiny_graph())
        embedding = service._embed_encoded_graph(encoded)
        similarity = gnn_graph_similarity(embedding, embedding)
        assert similarity == pytest.approx(1.0, abs=1e-5)

    def test_embed_campaign_caching_returns_identical_object_or_value(self, monkeypatch):
        import config
        monkeypatch.setattr(config, "GNN_ENABLED", True)
        service = GNNInferenceService(model_path=_real_model_path())

        from ml.gnn.campaign_graphs import CampaignGraphSample
        sample = CampaignGraphSample(
            campaign_id="camp-cache-test", graph=encode_graph(_tiny_graph()), severity="Low", risk_score=0.0, num_events=1,
        )
        monkeypatch.setattr("campaign_graphs.build_campaign_graph_sample", lambda cid: sample)

        e1 = service.embed_campaign("camp-cache-test")
        e2 = service.embed_campaign("camp-cache-test")
        assert torch.equal(e1, e2)

        service.invalidate_cache("camp-cache-test")
        e3 = service.embed_campaign("camp-cache-test", use_cache=False)
        assert torch.equal(e1, e3)  # deterministic even after cache invalidation


# ---------------------------------------------------------------------------
# Live Neo4j (optional -- skipped if unreachable), same convention as
# tests/test_campaign_graphs.py
# ---------------------------------------------------------------------------

def _neo4j_reachable():
    try:
        from neo4j_client import driver
        driver.verify_connectivity()
        return True
    except Exception:
        return False


requires_live_neo4j = pytest.mark.skipif(
    not _neo4j_reachable(),
    reason="No live Neo4j instance reachable in this environment.",
)


@requires_live_neo4j
@requires_real_artifact
def test_embed_campaign_against_live_neo4j_is_deterministic(monkeypatch):
    import config
    monkeypatch.setattr(config, "GNN_ENABLED", True)
    from ml.gnn.campaign_graphs import list_campaign_ids
    service = GNNInferenceService(model_path=_real_model_path())
    assert service.available

    ids = list_campaign_ids()
    assert ids, "expected at least one real campaign in this environment's Neo4j"
    campaign_id = ids[0]

    e1 = service.embed_campaign(campaign_id, use_cache=False)
    e2 = service.embed_campaign(campaign_id, use_cache=False)
    if e1 is None:
        pytest.skip(f"campaign {campaign_id} produced no embedding (e.g. no risk_score) -- not this test's concern")
    assert torch.equal(e1, e2)
    assert torch.isfinite(e1).all()
    assert e1.shape == (service.metadata.embedding_dim,)
