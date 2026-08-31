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

from ml.gnn.graph_encoder import FEATURE_DIM, NODE_TYPE_INDEX, encode_graph
from ml.gnn.layers import SAGEConvLayer
from ml.gnn.model import CampaignGNN, batch_graphs
from ml.gnn.synthetic_graphs import generate_synthetic_campaign_graph, generate_synthetic_dataset
from ml.gnn.train_gnn import MIN_TRAINING_GRAPHS, load_gnn, train_gnn


def _tiny_graph():
    G = nx.DiGraph()
    G.add_node("a", labels=["Attacker"], vt_reputation=80.0)
    G.add_node("c", labels=["Campaign"], risk_score=50.0)
    G.add_node("t", labels=["Technique"])
    G.add_edge("a", "c")
    G.add_edge("c", "t")
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
    encoded = encode_graph(_tiny_graph())
    campaign_row = encoded.x[encoded.node_ids.index("c")]
    # risk_score is at offset len(NODE_TYPES) + index of "risk_score" in NUMERIC_PROPS
    from ml.gnn.graph_encoder import NODE_TYPES, NUMERIC_PROPS
    risk_score_col = len(NODE_TYPES) + NUMERIC_PROPS.index("risk_score")
    assert campaign_row[risk_score_col] == 50.0


def test_encode_graph_symmetrizes_edges():
    encoded = encode_graph(_tiny_graph())
    # 2 directed edges -> 4 entries after symmetrization
    assert encoded.edge_index.shape[1] == 4


def test_encode_empty_graph_returns_zero_nodes():
    encoded = encode_graph(nx.DiGraph())
    assert encoded.num_nodes == 0
    assert encoded.edge_index.shape == (2, 0)


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
