"""
ml/gnn/graph_encoder.py
==========================
Converts a networkx.DiGraph — as produced by
graph_feature_engine.GraphBuilder.build(snapshot) from a real
GraphSnapshot, or by any test fixture built the same way — into the
tensors a GNN needs: a node feature matrix, an edge index, and the
node-id / node-type bookkeeping to map results back to real entities.

Deliberately reuses CYUKTI's existing graph-fetching path
(GraphSnapshotLoader -> GraphBuilder) instead of re-querying Neo4j —
this module only knows how to turn an already-built networkx graph into
tensors, so it's fully testable against hand-built graphs with no live
database.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import networkx as nx
import torch

NODE_TYPES = ["Attacker", "Campaign", "AttackEvent", "Technique", "Host", "Unknown"]
NODE_TYPE_INDEX = {t: i for i, t in enumerate(NODE_TYPES)}

# Curated numeric properties CYUKTI's own neo4j_client.py actually writes
# onto these node types (create_attack_event / campaign_manager). Missing
# or non-numeric values default to 0.0 rather than raising, since not
# every node type carries every property.
NUMERIC_PROPS = [
    "risk_score", "occurrences", "total_tps", "tps", "rule_level",
    "vt_reputation", "threat_actor_reputation", "malware_confidence",
    "tool_confidence", "misp_confidence", "ioc_confidence",
]

FEATURE_DIM = len(NODE_TYPES) + len(NUMERIC_PROPS)


@dataclass
class EncodedGraph:
    x: torch.Tensor            # [N, FEATURE_DIM]
    edge_index: torch.Tensor   # [2, E] (symmetrized for message passing)
    node_ids: list[str] = field(default_factory=list)
    node_types: list[str] = field(default_factory=list)

    @property
    def num_nodes(self) -> int:
        return self.x.shape[0]


def _node_type(labels) -> str:
    for label in labels or []:
        if label in NODE_TYPE_INDEX:
            return label
    return "Unknown"


def _numeric(value) -> float:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    return 0.0


def encode_graph(graph: nx.DiGraph) -> EncodedGraph:
    if graph.number_of_nodes() == 0:
        return EncodedGraph(x=torch.zeros((0, FEATURE_DIM)), edge_index=torch.zeros((2, 0), dtype=torch.long))

    node_ids = list(graph.nodes())
    id_to_idx = {node_id: i for i, node_id in enumerate(node_ids)}
    node_types = []

    features = torch.zeros((len(node_ids), FEATURE_DIM), dtype=torch.float32)

    for i, node_id in enumerate(node_ids):
        attrs = graph.nodes[node_id]
        node_type = _node_type(attrs.get("labels"))
        node_types.append(node_type)

        features[i, NODE_TYPE_INDEX[node_type]] = 1.0
        for j, prop in enumerate(NUMERIC_PROPS):
            features[i, len(NODE_TYPES) + j] = _numeric(attrs.get(prop))

    src, dst = [], []
    for u, v in graph.edges():
        src.append(id_to_idx[u])
        dst.append(id_to_idx[v])
        # symmetrize: message passing should flow both directions even
        # though the underlying attack-chain relationship is directional
        src.append(id_to_idx[v])
        dst.append(id_to_idx[u])

    edge_index = torch.tensor([src, dst], dtype=torch.long) if src else torch.zeros((2, 0), dtype=torch.long)

    return EncodedGraph(x=features, edge_index=edge_index, node_ids=node_ids, node_types=node_types)
