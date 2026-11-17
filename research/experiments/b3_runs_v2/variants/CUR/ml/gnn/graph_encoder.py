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
#
# risk_score is deliberately EXCLUDED, not merely omitted: it is a
# Campaign-node property, and the severity label this graph is meant to
# predict (see campaign_graphs.py) is a deterministic thresholded
# function of that exact same risk_score (risk_scoring.risk_level_from_score
# -> label_generator.py). Encoding it as a node feature would let a GNN
# trivially reconstruct the label instead of learning from graph
# structure/MITRE/CTI signals, for the identical reason
# ml/dataset_utils.py already excludes risk_score from XGBoost's
# FEATURE_COLUMNS via its own LEAKAGE_COLUMNS list — this mirrors that
# precedent for the GNN's node features rather than reintroducing the
# same leakage in a different subsystem.
NUMERIC_PROPS = [
    "occurrences", "total_tps", "tps", "rule_level",
    "vt_reputation", "threat_actor_reputation", "malware_confidence",
    "tool_confidence", "misp_confidence", "ioc_confidence",
]

FEATURE_DIM = len(NODE_TYPES) + len(NUMERIC_PROPS)

# Relationship types GraphSnapshotLoader's per-campaign query can actually
# produce (neo4j_client.py: (Attacker)-[LAUNCHED]->(Campaign),
# (Campaign)-[HAS_EVENT]->(AttackEvent), (AttackEvent)-[MATCHES]->(Technique),
# (Campaign)-[TARGETS]->(Host) -- see GNN_FEASIBILITY.md Section 2/13's "not
# captured by this per-campaign extraction" list for the cross-campaign types,
# e.g. SIMILAR_TO, deliberately excluded here because this extraction scope
# doesn't reach them). "Unknown" is a fallback, not a real observed type, kept
# for the same forward-compatibility reason NODE_TYPES has one: if
# GraphSnapshotLoader's scope is ever extended (GNN_FEASIBILITY.md Section 14,
# item 2), an unrecognized relationship degrades gracefully instead of raising.
EDGE_TYPES = ["LAUNCHED", "HAS_EVENT", "MATCHES", "TARGETS", "Unknown"]
EDGE_TYPE_INDEX = {t: i for i, t in enumerate(EDGE_TYPES)}
EDGE_FEATURE_DIM = len(EDGE_TYPES)


@dataclass
class EncodedGraph:
    x: torch.Tensor            # [N, FEATURE_DIM]
    edge_index: torch.Tensor   # [2, E] (symmetrized for message passing)
    edge_attr: torch.Tensor = field(default_factory=lambda: torch.zeros((0, EDGE_FEATURE_DIM)))
    # [E, EDGE_FEATURE_DIM] one-hot relationship type, aligned row-for-row
    # with edge_index's columns. Not yet consumed by SAGEConvLayer/CampaignGNN
    # (the mean aggregator in layers.py is edge-type-agnostic today) --
    # available for a future edge-aware aggregation, per
    # GNN_FEASIBILITY.md Section 7/14's "edge-type encoding design" gap.
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


def _edge_type(attrs: dict) -> str:
    relationship = attrs.get("relationship")
    return relationship if relationship in EDGE_TYPE_INDEX else "Unknown"


def encode_graph(graph: nx.DiGraph) -> EncodedGraph:
    if graph.number_of_nodes() == 0:
        return EncodedGraph(
            x=torch.zeros((0, FEATURE_DIM)),
            edge_index=torch.zeros((2, 0), dtype=torch.long),
            edge_attr=torch.zeros((0, EDGE_FEATURE_DIM)),
        )

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

    src, dst, edge_types = [], [], []
    for u, v, attrs in graph.edges(data=True):
        edge_type = _edge_type(attrs)
        src.append(id_to_idx[u])
        dst.append(id_to_idx[v])
        edge_types.append(edge_type)
        # symmetrize: message passing should flow both directions even
        # though the underlying attack-chain relationship is directional.
        # The reverse edge represents the same real relationship, just
        # traversed backward, so it gets the same type one-hot rather than
        # a distinct "reverse-X" type.
        src.append(id_to_idx[v])
        dst.append(id_to_idx[u])
        edge_types.append(edge_type)

    edge_index = torch.tensor([src, dst], dtype=torch.long) if src else torch.zeros((2, 0), dtype=torch.long)

    edge_attr = torch.zeros((len(edge_types), EDGE_FEATURE_DIM), dtype=torch.float32)
    for i, edge_type in enumerate(edge_types):
        edge_attr[i, EDGE_TYPE_INDEX[edge_type]] = 1.0

    return EncodedGraph(
        x=features,
        edge_index=edge_index,
        edge_attr=edge_attr,
        node_ids=node_ids,
        node_types=node_types,
    )
