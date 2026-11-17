"""
ml/gnn/topology_similarity.py
=================================
The one place CYUKTI computes "how topologically similar are two
campaign graphs" from GNN embeddings. Named explicitly
`gnn_graph_similarity` / `gnn_topology_similarity_between_campaigns` so
its source is never ambiguous alongside the repository's other, older
similarity signals (technique/attacker/victim/temporal/chain/prediction
similarity, and the two pre-existing `graph_similarity` slots --
`OperationFeatures.graph_similarity` and
`CampaignDecisionEngine.graph_similarity`, both hand-engineered scalar
composites, distinct from this).

Metric: cosine similarity in [-1, 1], the same metric used throughout
GNN_RETRIEVAL_EVALUATION.md, for direct comparability with that
phase's reported numbers.
"""

from __future__ import annotations


def gnn_graph_similarity(embedding_a, embedding_b) -> float:
    """Cosine similarity between two GNN embeddings. Pure function --
    no I/O, no model, just the metric. Returns a value in [-1, 1]."""
    import torch

    denom = (embedding_a.norm() * embedding_b.norm()).clamp(min=1e-9)
    return float(torch.dot(embedding_a, embedding_b) / denom)


def gnn_topology_similarity_between_campaigns(campaign_id_a: str, campaign_id_b: str) -> float | None:
    """Fail-safe end-to-end version: resolves both campaigns' embeddings
    via the shared inference service and returns their cosine
    similarity, or None if either embedding is unavailable for any
    reason (GNN disabled, no artifact, malformed/empty graph, etc.) --
    never raises."""
    from ml.gnn.inference import gnn_inference_service

    embedding_a = gnn_inference_service.embed_campaign(campaign_id_a)
    if embedding_a is None:
        return None
    embedding_b = gnn_inference_service.embed_campaign(campaign_id_b)
    if embedding_b is None:
        return None
    return gnn_graph_similarity(embedding_a, embedding_b)
