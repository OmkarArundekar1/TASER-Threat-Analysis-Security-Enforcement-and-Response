"""
ml/gnn/model.py
==================
CampaignGNN: stacks SAGEConvLayer message-passing over a campaign's
attack subgraph (attacker/campaign/attack-event/technique/host nodes),
mean-pools to a fixed-size graph embedding, and predicts severity from
it — the same target XGBoost predicts from tabular features, so the two
models' predictions can be compared honestly on the same label.

The graph embedding itself (not just the classification head) is the
more important output for the rest of the system: backend/investigation
uses embedding-space similarity between the current campaign and
historical ones as a structural relevance signal that's independent of
(and complements) the pure technique-overlap coverage
threat_attribution_engine.py already computes.

`batch_graphs` implements the standard GNN trick for training on many
small, independently-sized graphs at once: concatenate all node
features into one block-diagonal adjacency, and track which original
graph each node belongs to via a `batch` index vector so pooling knows
where one graph ends and the next begins.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from graph_encoder import FEATURE_DIM, EncodedGraph
from layers import SAGEConvLayer


def batch_graphs(graphs: list[EncodedGraph]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Concatenate multiple EncodedGraphs into one disjoint-union graph.

    Returns (x, edge_index, batch) where `batch[i]` is the index of the
    original graph node i came from.
    """
    xs, edge_indices, batch = [], [], []
    node_offset = 0

    for graph_idx, g in enumerate(graphs):
        xs.append(g.x)
        if g.edge_index.shape[1] > 0:
            edge_indices.append(g.edge_index + node_offset)
        batch.append(torch.full((g.num_nodes,), graph_idx, dtype=torch.long))
        node_offset += g.num_nodes

    x = torch.cat(xs, dim=0) if xs else torch.zeros((0, FEATURE_DIM))
    edge_index = torch.cat(edge_indices, dim=1) if edge_indices else torch.zeros((2, 0), dtype=torch.long)
    batch_vec = torch.cat(batch, dim=0) if batch else torch.zeros((0,), dtype=torch.long)

    return x, edge_index, batch_vec


def _scatter_mean(h: torch.Tensor, batch: torch.Tensor, num_graphs: int) -> torch.Tensor:
    out = torch.zeros((num_graphs, h.shape[1]), device=h.device, dtype=h.dtype)
    out = out.index_add_(0, batch, h)
    counts = torch.zeros(num_graphs, device=h.device, dtype=h.dtype).index_add_(
        0, batch, torch.ones(h.shape[0], device=h.device, dtype=h.dtype)
    )
    return out / counts.clamp(min=1.0).unsqueeze(1)


class CampaignGNN(nn.Module):
    def __init__(
        self,
        in_dim: int = FEATURE_DIM,
        hidden_dim: int = 32,
        num_classes: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        dims = [in_dim] + [hidden_dim] * num_layers
        self.convs = nn.ModuleList([SAGEConvLayer(dims[i], dims[i + 1]) for i in range(num_layers)])
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim, num_classes)
        self.embedding_dim = hidden_dim

    def embed(self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor, num_graphs: int) -> torch.Tensor:
        h = x
        for conv in self.convs:
            h = self.dropout(conv(h, edge_index))
        return _scatter_mean(h, batch, num_graphs)

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor, num_graphs: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        graph_embedding = self.embed(x, edge_index, batch, num_graphs)
        logits = self.classifier(graph_embedding)
        return logits, graph_embedding

    def embed_single(self, graph: EncodedGraph) -> torch.Tensor:
        """Convenience for a single graph at inference time (no batching)."""
        batch = torch.zeros(graph.num_nodes, dtype=torch.long)
        with torch.no_grad():
            return self.embed(graph.x, graph.edge_index, batch, num_graphs=1).squeeze(0)
