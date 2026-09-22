"""
Tests for the graph autoencoder (Option E, GNN_REPRESENTATION_DESIGN.md
Section 6, Objective A): ml/gnn/autoencoder_model.py and
ml/gnn/train_autoencoder.py. Mocks only the database boundary
(neo4j_client.driver), never the model/graph-construction logic itself
-- same rule tests/test_campaign_graphs.py already established.
"""

import networkx as nx
import pytest
import torch

from ml.gnn.autoencoder_model import GraphAutoencoder
from ml.gnn.campaign_graphs import CampaignGraphSample
from ml.gnn.graph_encoder import EDGE_FEATURE_DIM, NUMERIC_PROPS, encode_graph
from ml.gnn.train_autoencoder import (
    _build_targets,
    _feature_stats,
    get_campaign_attacker_ips,
    split_by_attacker_group,
)


def _tiny_graph():
    G = nx.DiGraph()
    G.add_node("a", labels=["Attacker"], vt_reputation=80.0)
    G.add_node("c", labels=["Campaign"])
    G.add_node("e", labels=["AttackEvent"], occurrences=3.0)
    G.add_node("t", labels=["Technique"])
    G.add_edge("a", "c", relationship="LAUNCHED")
    G.add_edge("c", "e", relationship="HAS_EVENT")
    G.add_edge("e", "t", relationship="MATCHES")
    return G


def _sample(campaign_id="camp-1") -> CampaignGraphSample:
    return CampaignGraphSample(
        campaign_id=campaign_id, graph=encode_graph(_tiny_graph()),
        severity="Low", risk_score=10.0, num_events=1,
    )


# ---------------------------------------------------------------------------
# GraphAutoencoder — forward pass / shapes
# ---------------------------------------------------------------------------

def test_forward_pass_output_shapes():
    sample = _sample()
    model = GraphAutoencoder(hidden_dim=16, embedding_dim=8, num_layers=2)
    out = model(sample.graph.x, sample.graph.edge_index)

    n = sample.graph.num_nodes
    assert out.edge_existence_logits.shape == (n, n)
    assert out.edge_type_logits.shape == (n, n, EDGE_FEATURE_DIM)
    assert out.node_feature_recon.shape == (n, len(NUMERIC_PROPS))
    assert out.node_embeddings.shape == (n, 16)


def test_pooled_embedding_shape_matches_embedding_dim():
    sample = _sample()
    model = GraphAutoencoder(hidden_dim=16, embedding_dim=8, num_layers=2)
    z = model.embed_graph(sample.graph.x, sample.graph.edge_index)
    assert z.shape == (8,)


def test_deterministic_inference_same_input_same_output():
    sample = _sample()
    model = GraphAutoencoder(hidden_dim=16, embedding_dim=8, num_layers=2)
    model.eval()
    z1 = model.embed_graph(sample.graph.x, sample.graph.edge_index)
    z2 = model.embed_graph(sample.graph.x, sample.graph.edge_index)
    assert torch.equal(z1, z2)


def test_embedding_values_are_finite():
    sample = _sample()
    model = GraphAutoencoder(hidden_dim=16, embedding_dim=8, num_layers=2)
    z = model.embed_graph(sample.graph.x, sample.graph.edge_index)
    assert torch.isfinite(z).all()


def test_empty_graph_does_not_crash_encoder():
    model = GraphAutoencoder(hidden_dim=16, embedding_dim=8, num_layers=2)
    empty = encode_graph(nx.DiGraph())
    h = model.encode_nodes(empty.x, empty.edge_index)
    assert h.shape == (0, 16)


# ---------------------------------------------------------------------------
# checkpoint save/load
# ---------------------------------------------------------------------------

def test_checkpoint_round_trip_preserves_inference(tmp_path):
    from ml.gnn.train_autoencoder import load_autoencoder

    sample = _sample()
    model = GraphAutoencoder(hidden_dim=16, embedding_dim=8, num_layers=2)
    model.eval()
    z_before = model.embed_graph(sample.graph.x, sample.graph.edge_index)

    model_file = tmp_path / "gnn_autoencoder.pt"
    torch.save({
        "state_dict": model.state_dict(), "hidden_dim": 16, "embedding_dim": 8, "num_layers": 2,
        "feature_mean": torch.zeros(len(NUMERIC_PROPS)), "feature_std": torch.ones(len(NUMERIC_PROPS)),
        "seed": 42, "train_campaign_ids": [], "val_campaign_ids": [], "pos_weight": 1.0,
    }, model_file)

    loaded, checkpoint = load_autoencoder(str(model_file))
    z_after = loaded.embed_graph(sample.graph.x, sample.graph.edge_index)

    assert torch.equal(z_before, z_after)
    assert checkpoint["hidden_dim"] == 16


# ---------------------------------------------------------------------------
# _build_targets
# ---------------------------------------------------------------------------

def test_build_targets_adjacency_matches_edge_index():
    sample = _sample()
    adj, edge_type_idx = _build_targets(sample)
    n = sample.graph.num_nodes
    src, dst = sample.graph.edge_index
    for k in range(src.shape[0]):
        assert adj[int(src[k]), int(dst[k])] == 1.0
    assert int(adj.sum().item()) == sample.graph.edge_index.shape[1]


def test_build_targets_edge_type_idx_is_minus_one_where_no_edge():
    sample = _sample()
    _, edge_type_idx = _build_targets(sample)
    n = sample.graph.num_nodes
    src, dst = sample.graph.edge_index
    edge_pairs = {(int(src[k]), int(dst[k])) for k in range(src.shape[0])}
    for i in range(n):
        for j in range(n):
            if (i, j) not in edge_pairs:
                assert edge_type_idx[i, j] == -1


def test_build_targets_edge_type_idx_matches_true_edge_attr_argmax():
    sample = _sample()
    _, edge_type_idx = _build_targets(sample)
    src, dst = sample.graph.edge_index
    for k in range(src.shape[0]):
        i, j = int(src[k]), int(dst[k])
        expected = int(sample.graph.edge_attr[k].argmax().item())
        assert edge_type_idx[i, j] == expected


# ---------------------------------------------------------------------------
# _feature_stats
# ---------------------------------------------------------------------------

def test_feature_stats_shapes():
    samples = [_sample("camp-1"), _sample("camp-2")]
    mean, std = _feature_stats(samples)
    assert mean.shape == (len(NUMERIC_PROPS),)
    assert std.shape == (len(NUMERIC_PROPS),)
    assert (std > 0).all()  # clamped, never zero (would divide by zero)


def test_feature_stats_empty_samples_does_not_crash():
    mean, std = _feature_stats([])
    assert mean.shape == (len(NUMERIC_PROPS),)
    assert std.shape == (len(NUMERIC_PROPS),)


# ---------------------------------------------------------------------------
# split_by_attacker_group — the leakage-safe split (Phase E)
# ---------------------------------------------------------------------------

def test_split_by_attacker_group_no_attacker_appears_in_both_splits():
    # Mirrors this repository's real, live-verified attacker-group size
    # distribution ([33, 26, 5, 4, 1, 1, 1] across 71 campaigns) at a
    # smaller scale, to exercise the same lumpy-greedy-assignment path.
    attacker_of = {}
    campaign_ids = []
    group_sizes = {"big1": 6, "big2": 5, "small1": 2, "small2": 1, "small3": 1}
    for group, size in group_sizes.items():
        for i in range(size):
            cid = f"{group}-{i}"
            campaign_ids.append(cid)
            attacker_of[cid] = group

    train_ids, val_ids = split_by_attacker_group(campaign_ids, attacker_of, val_fraction=0.2)

    train_attackers = {attacker_of[c] for c in train_ids}
    val_attackers = {attacker_of[c] for c in val_ids}
    assert train_attackers.isdisjoint(val_attackers)
    assert set(train_ids) | set(val_ids) == set(campaign_ids)
    assert set(train_ids).isdisjoint(val_ids)


def test_split_by_attacker_group_is_deterministic():
    attacker_of = {f"c{i}": f"a{i % 4}" for i in range(20)}
    campaign_ids = list(attacker_of.keys())
    first = split_by_attacker_group(campaign_ids, attacker_of, val_fraction=0.2)
    second = split_by_attacker_group(campaign_ids, attacker_of, val_fraction=0.2)
    assert first == second


def test_split_by_attacker_group_handles_missing_attacker():
    campaign_ids = ["c1", "c2", "c3"]
    attacker_of = {"c1": "a1"}  # c2, c3 have no resolvable attacker
    train_ids, val_ids = split_by_attacker_group(campaign_ids, attacker_of, val_fraction=0.3)
    assert set(train_ids) | set(val_ids) == set(campaign_ids)


# ---------------------------------------------------------------------------
# get_campaign_attacker_ips — mocks only the DB boundary
# ---------------------------------------------------------------------------

class _FakeSession:
    def __init__(self, rows):
        self._rows = rows

    def run(self, query, **kwargs):
        assert "LAUNCHED" in query
        return iter(self._rows)

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class _FakeDriver:
    def __init__(self, rows):
        self._rows = rows

    def session(self):
        return _FakeSession(self._rows)


def test_get_campaign_attacker_ips_maps_campaign_to_ip(monkeypatch):
    monkeypatch.setattr(
        "neo4j_client.driver",
        _FakeDriver([{"id": "camp-1", "ip": "1.2.3.4"}, {"id": "camp-2", "ip": "5.6.7.8"}]),
    )
    result = get_campaign_attacker_ips(["camp-1", "camp-2"])
    assert result == {"camp-1": "1.2.3.4", "camp-2": "5.6.7.8"}


def test_get_campaign_attacker_ips_empty_input_makes_no_query():
    assert get_campaign_attacker_ips([]) == {}
