"""
Tests for the GNN pipeline (ml/gnn). Uses hand-built small networkx
graphs for encoder/layer correctness, and the synthetic graph generator
for end-to-end training pipeline validation — same honesty standard as
test_ml_pipeline.py: these confirm the mechanics work, not that the
model is accurate on real campaigns.
"""

import networkx as nx
import pytest
import torch

from ml.gnn.graph_encoder import (
    EDGE_FEATURE_DIM,
    EDGE_TYPE_INDEX,
    FEATURE_DIM,
    NODE_TYPE_INDEX,
    encode_graph,
)
from ml.gnn.layers import SAGEConvLayer
from ml.gnn.model import CampaignGNN, batch_graphs
from ml.gnn.synthetic_graphs import generate_synthetic_campaign_graph, generate_synthetic_dataset
from ml.gnn.train_gnn import MIN_TRAINING_GRAPHS, load_gnn, train_gnn


def _tiny_graph():
    G = nx.DiGraph()
    G.add_node("a", labels=["Attacker"], vt_reputation=80.0)
    G.add_node("c", labels=["Campaign"], risk_score=50.0)
    G.add_node("t", labels=["Technique"])
    # relationship types as GraphBuilder.build actually sets them (see
    # graph_feature_engine.py: G.add_edge(..., relationship=rel["type"], ...))
    G.add_edge("a", "c", relationship="LAUNCHED")
    G.add_edge("c", "t", relationship="HAS_EVENT")
    return G


# ---------------------------------------------------------------------------
# graph_encoder
# ---------------------------------------------------------------------------

def test_encode_graph_shapes_and_type_onehot():
    encoded = encode_graph(_tiny_graph())
    assert encoded.x.shape == (3, FEATURE_DIM)
    assert encoded.num_nodes == 3

    attacker_row = encoded.x[encoded.node_ids.index("a")]
    assert attacker_row[NODE_TYPE_INDEX["Attacker"]] == 1.0
    assert attacker_row[NODE_TYPE_INDEX["Campaign"]] == 0.0


def test_encode_graph_reads_numeric_properties():
    G = _tiny_graph()
    G.nodes["c"]["total_tps"] = 42.0
    encoded = encode_graph(G)
    campaign_row = encoded.x[encoded.node_ids.index("c")]
    # total_tps is at offset len(NODE_TYPES) + index of "total_tps" in NUMERIC_PROPS
    from ml.gnn.graph_encoder import NODE_TYPES, NUMERIC_PROPS
    total_tps_col = len(NODE_TYPES) + NUMERIC_PROPS.index("total_tps")
    assert campaign_row[total_tps_col] == 42.0


def test_encode_graph_excludes_risk_score_as_a_leakage_source():
    """risk_score must never become a node feature: severity (the GNN's
    intended prediction target -- see ../../GNN_FEASIBILITY.md) is a
    deterministic function of a Campaign's risk_score, so encoding it
    would let the model trivially reconstruct the label instead of
    learning from graph structure -- the same leakage
    ml/dataset_utils.py's LEAKAGE_COLUMNS already excludes for XGBoost."""
    from ml.gnn.graph_encoder import NUMERIC_PROPS

    assert "risk_score" not in NUMERIC_PROPS

    G = _tiny_graph()
    G.nodes["c"]["risk_score"] = 999.0  # even if present on the real node, must not leak in
    encoded = encode_graph(G)
    campaign_row = encoded.x[encoded.node_ids.index("c")]
    assert 999.0 not in campaign_row.tolist()


def test_encode_graph_symmetrizes_edges():
    encoded = encode_graph(_tiny_graph())
    # 2 directed edges -> 4 entries after symmetrization
    assert encoded.edge_index.shape[1] == 4


def test_encode_graph_edge_attr_shape_and_alignment():
    encoded = encode_graph(_tiny_graph())
    # edge_attr must have one row per edge_index column, one-hot over EDGE_TYPES
    assert encoded.edge_attr.shape == (encoded.edge_index.shape[1], EDGE_FEATURE_DIM)
    assert torch.allclose(encoded.edge_attr.sum(dim=1), torch.ones(encoded.edge_attr.shape[0]))


def test_encode_graph_edge_attr_reflects_real_relationship_type():
    encoded = encode_graph(_tiny_graph())
    # forward a->c edge is column 0 (insertion order before symmetrization pass)
    launched_row = encoded.edge_attr[0]
    assert launched_row[EDGE_TYPE_INDEX["LAUNCHED"]] == 1.0
    assert launched_row.sum().item() == 1.0


def test_encode_graph_symmetrized_reverse_edge_keeps_same_type():
    encoded = encode_graph(_tiny_graph())
    # a->c (forward, col 0) and c->a (reverse, col 1) represent the same
    # LAUNCHED relationship traversed backward for message passing -- both
    # must carry the same type one-hot, not a distinct "reverse" type.
    assert torch.equal(encoded.edge_attr[0], encoded.edge_attr[1])


def test_encode_graph_unrecognized_relationship_falls_back_to_unknown():
    G = nx.DiGraph()
    G.add_node("x", labels=["Campaign"])
    G.add_node("y", labels=["ThreatActor"])
    G.add_edge("x", "y", relationship="RESEMBLES")  # real type, outside per-campaign scope
    encoded = encode_graph(G)
    assert encoded.edge_attr[0, EDGE_TYPE_INDEX["Unknown"]] == 1.0


def test_encode_graph_missing_relationship_attr_falls_back_to_unknown():
    G = nx.DiGraph()
    G.add_node("x", labels=["Campaign"])
    G.add_node("y", labels=["Host"])
    G.add_edge("x", "y")  # no relationship attr at all
    encoded = encode_graph(G)
    assert encoded.edge_attr[0, EDGE_TYPE_INDEX["Unknown"]] == 1.0


def test_encode_empty_graph_returns_zero_nodes():
    encoded = encode_graph(nx.DiGraph())
    assert encoded.num_nodes == 0
    assert encoded.edge_index.shape == (2, 0)
    assert encoded.edge_attr.shape == (0, EDGE_FEATURE_DIM)


def test_unknown_label_falls_back_to_unknown_type():
    G = nx.DiGraph()
    G.add_node("x", labels=["SomeNewLabel"])
    encoded = encode_graph(G)
    assert encoded.node_types == ["Unknown"]


# ---------------------------------------------------------------------------
# layers / model
# ---------------------------------------------------------------------------

def test_sageconv_mean_aggregation_is_correct():
    # 3 nodes, node 0 has no incoming edges, node 2 aggregates from nodes 0 and 1
    x = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])
    edge_index = torch.tensor([[0, 1], [2, 2]])  # 0->2, 1->2
    layer = SAGEConvLayer(2, 2)
    with torch.no_grad():
        layer.self_lin.weight.copy_(torch.eye(2))
        layer.self_lin.bias.zero_()
        layer.neigh_lin.weight.copy_(torch.eye(2))
        layer.neigh_lin.bias.zero_()

    out = layer(x, edge_index)
    # node 2: self=[0,0], neighbor mean of x[0],x[1] = [0.5, 0.5] -> ReLU(0+0.5, 0+0.5)
    assert torch.allclose(out[2], torch.tensor([0.5, 0.5]), atol=1e-5)
    # node 0: no incoming edges -> neighbor mean is zero -> just self
    assert torch.allclose(out[0], torch.tensor([1.0, 0.0]), atol=1e-5)


def test_model_forward_produces_correct_shapes():
    graphs = [encode_graph(g) for g, _ in generate_synthetic_dataset(4, seed=1)]
    x, edge_index, batch = batch_graphs(graphs)
    model = CampaignGNN(hidden_dim=16, num_classes=4, num_layers=2)
    logits, embedding = model(x, edge_index, batch, num_graphs=4)
    assert logits.shape == (4, 4)
    assert embedding.shape == (4, 16)


def test_model_gradients_flow():
    graphs = [encode_graph(g) for g, _ in generate_synthetic_dataset(3, seed=2)]
    x, edge_index, batch = batch_graphs(graphs)
    model = CampaignGNN(hidden_dim=8, num_classes=2, num_layers=2)
    logits, _ = model(x, edge_index, batch, num_graphs=3)
    loss = logits.sum()
    loss.backward()
    assert model.convs[0].self_lin.weight.grad is not None
    assert not torch.all(model.convs[0].self_lin.weight.grad == 0)


def test_embed_single_matches_batched_embedding():
    graph, _ = generate_synthetic_campaign_graph(seed=5)
    encoded = encode_graph(graph)
    model = CampaignGNN(hidden_dim=8, num_classes=4, num_layers=2)
    model.eval()

    single = model.embed_single(encoded)

    x, edge_index, batch = batch_graphs([encoded])
    with torch.no_grad():
        _, batched = model(x, edge_index, batch, num_graphs=1)

    assert torch.allclose(single, batched.squeeze(0), atol=1e-6)


# ---------------------------------------------------------------------------
# training pipeline (synthetic)
# ---------------------------------------------------------------------------

def test_train_gnn_rejects_too_few_graphs():
    data = generate_synthetic_dataset(MIN_TRAINING_GRAPHS - 1, seed=3)
    with pytest.raises(ValueError, match="Not enough labeled campaign graphs"):
        train_gnn(data)


def test_train_gnn_end_to_end_and_reload(tmp_path):
    data = generate_synthetic_dataset(120, seed=4)
    result = train_gnn(data, save_dir=str(tmp_path), epochs=30)

    assert result.train_graphs + result.val_graphs == 120
    assert 0.0 <= result.val_accuracy <= 1.0
    assert 0.0 <= result.val_macro_f1 <= 1.0
    assert result.final_train_loss == result.final_train_loss  # not NaN

    loaded = load_gnn(result.model_path)
    graph, _ = data[0]
    encoded = encode_graph(graph)
    embedding = loaded.embed_single(encoded)
    assert embedding.shape == (loaded.embedding_dim,)
