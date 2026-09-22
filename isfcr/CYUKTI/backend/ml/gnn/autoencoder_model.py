"""
ml/gnn/autoencoder_model.py
==============================
Graph autoencoder for Option E (cross-campaign graph representation
learning — see ../../GNN_REPRESENTATION_DESIGN.md Section 6, Objective
A). Reuses the existing, already-tested `SAGEConvLayer` message-passing
primitive (`layers.py`) unchanged — this module only adds an encoder
stack that returns per-node embeddings (not pooled, unlike
`model.CampaignGNN`) plus decoder heads, because reconstructing
topology requires per-node identity to survive past the final layer;
a single pooled graph vector cannot answer "which node connects to
which."

Reconstruction targets, chosen deliberately (not "reconstruct every
field" — see `GNN_REPRESENTATION_DESIGN.md` Section 12 for the
rationale this mirrors):

  - Edge existence (every ordered node pair, dense — graphs are tiny,
    at most 24 nodes, so full pair enumeration is cheap and exact,
    no negative-sampling randomness to worry about for reproducibility)
    via a zero-parameter dot-product decoder (Kipf & Welling GAE-style),
    the most standard, least-overfit-prone choice for ~71 training
    graphs.
  - Edge type, conditioned on true edges only (a real edge exists;
    which of the 5 EDGE_TYPES is it) — this is the one target directly
    tied to "topology is the research contribution" (relationship
    *type*, not just existence).
  - Node numeric-feature reconstruction (the 10 curated numeric
    properties, standardized — see train_autoencoder.py) — tests
    whether per-node content survives message passing and pooling.

Deliberately NOT reconstructed: node *type* one-hot. It is a direct
input feature (`NODE_TYPES` slice of `x`), so reconstructing it from an
embedding that was partly built from it is close to an identity task
and would inflate apparent reconstruction quality without evidencing
anything about learned structure.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn

from graph_encoder import EDGE_FEATURE_DIM, FEATURE_DIM, NUMERIC_PROPS
from layers import SAGEConvLayer

# Node-type one-hot occupies the first len(NODE_TYPES) columns of x
# (graph_encoder.py); numeric properties occupy the remaining
# len(NUMERIC_PROPS) columns, in the same order NUMERIC_PROPS lists them.
NUMERIC_FEATURE_OFFSET = FEATURE_DIM - len(NUMERIC_PROPS)


@dataclass
class ReconstructionOutput:
    edge_existence_logits: torch.Tensor   # [N, N] (diagonal excluded by caller)
    edge_type_logits: torch.Tensor        # [N, N, EDGE_FEATURE_DIM]
    node_feature_recon: torch.Tensor      # [N, len(NUMERIC_PROPS)]
    node_embeddings: torch.Tensor = field(repr=False, default=None)  # [N, hidden_dim], pre-bottleneck


class GraphAutoencoder(nn.Module):
    """Encoder: stacked SAGEConvLayer (per-node embeddings, width =
    hidden_dim). Bottleneck: Linear + tanh, applied AFTER mean-pooling,
    projecting down to embedding_dim — the portable "z_G" this whole
    phase is about (see GNN_REPRESENTATION_DESIGN.md Section 2/4 for
    why hidden_dim=FEATURE_DIM and embedding_dim=FEATURE_DIM//2 were
    chosen, documented in train_autoencoder.py alongside the rest of
    the training configuration, not duplicated here)."""

    def __init__(
        self,
        in_dim: int = FEATURE_DIM,
        hidden_dim: int = FEATURE_DIM,
        embedding_dim: int = FEATURE_DIM // 2,
        num_layers: int = 2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        dims = [in_dim] + [hidden_dim] * num_layers
        self.convs = nn.ModuleList([SAGEConvLayer(dims[i], dims[i + 1]) for i in range(num_layers)])
        self.dropout = nn.Dropout(dropout)
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim

        # Decoders operate on the richer pre-bottleneck per-node hidden
        # state, not the compressed z_G -- the bottleneck is for the
        # portable graph-level summary (Section 2), not reconstruction.
        self.edge_type_decoder = nn.Linear(hidden_dim * 2, EDGE_FEATURE_DIM)
        self.node_feature_decoder = nn.Linear(hidden_dim, len(NUMERIC_PROPS))
        self.bottleneck = nn.Linear(hidden_dim, embedding_dim)

    def encode_nodes(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Per-node embeddings, width=hidden_dim. No pooling."""
        h = x
        for conv in self.convs:
            h = self.dropout(conv(h, edge_index))
        return h

    def pooled_embedding(self, h: torch.Tensor) -> torch.Tensor:
        """z_G: mean-pool per-node embeddings, then project through the
        bottleneck. This is the one function anything outside this
        module should call to get "the campaign's embedding" (see
        GNN_REPRESENTATION_DESIGN.md Section 12's future interface)."""
        pooled = h.mean(dim=0, keepdim=True) if h.shape[0] > 0 else torch.zeros((1, self.hidden_dim))
        return torch.tanh(self.bottleneck(pooled)).squeeze(0)

    def reconstruct(self, h: torch.Tensor) -> ReconstructionOutput:
        """All-pairs reconstruction from per-node embeddings h [N, hidden_dim]."""
        n = h.shape[0]
        edge_existence_logits = h @ h.T  # [N, N], dot-product decoder (Kipf & Welling GAE)

        h_i = h.unsqueeze(1).expand(n, n, self.hidden_dim)
        h_j = h.unsqueeze(0).expand(n, n, self.hidden_dim)
        pair_repr = torch.cat([h_i, h_j], dim=-1)  # [N, N, 2*hidden_dim]
        edge_type_logits = self.edge_type_decoder(pair_repr)  # [N, N, EDGE_FEATURE_DIM]

        node_feature_recon = self.node_feature_decoder(h)  # [N, len(NUMERIC_PROPS)]

        return ReconstructionOutput(
            edge_existence_logits=edge_existence_logits,
            edge_type_logits=edge_type_logits,
            node_feature_recon=node_feature_recon,
            node_embeddings=h,
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> ReconstructionOutput:
        h = self.encode_nodes(x, edge_index)
        return self.reconstruct(h)

    def embed_graph(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Convenience: x, edge_index -> z_G in one call, no gradient."""
        with torch.no_grad():
            h = self.encode_nodes(x, edge_index)
            return self.pooled_embedding(h)
