"""
ml/gnn/synthetic_graphs.py
=============================
Generates SYNTHETIC campaign attack-subgraphs (same node/edge shape
GraphSnapshotLoader + GraphBuilder produce from real Neo4j data) purely
to validate the GNN training/evaluation/serialization pipeline end to
end when no real campaign graphs exist yet.

As with ml/data_prep/generate_synthetic_dataset.py: this is not real
telemetry. Severity here is generated from a noisy dependence on graph
size/depth and attacker risk properties so the GNN has *something*
learnable to validate the training loop against — it is not a claim
about what real attack graphs look like.
"""

from __future__ import annotations

import random

import networkx as nx

_SEVERITIES = ["Low", "Medium", "High", "Critical"]


def _severity_from_structure(num_techniques: int, attacker_reputation: float, rng: random.Random) -> str:
    composite = num_techniques * 8 + attacker_reputation * 0.5
    composite += rng.gauss(0, 10)
    if composite >= 70:
        return "Critical"
    if composite >= 45:
        return "High"
    if composite >= 20:
        return "Medium"
    return "Low"


def generate_synthetic_campaign_graph(seed: int) -> tuple[nx.DiGraph, str]:
    """Build one synthetic campaign subgraph and its severity label.

    Structure: Attacker -> Campaign -> AttackEvent(s) -> Technique(s),
    Campaign -> Host(s) — the same shape GraphSnapshotLoader's Cypher
    query assembles from real Neo4j data (see graph_feature_engine.py).
    """
    rng = random.Random(seed)
    num_techniques = rng.randint(1, 8)
    attacker_reputation = rng.uniform(0, 100)

    G = nx.DiGraph()
    attacker_id = f"attacker-{seed}"
    campaign_id = f"campaign-{seed}"

    G.add_node(attacker_id, labels=["Attacker"], vt_reputation=attacker_reputation,
               threat_actor_reputation=rng.uniform(0, 100))
    G.add_node(campaign_id, labels=["Campaign"], risk_score=rng.uniform(0, 100),
               occurrences=num_techniques, total_tps=rng.uniform(0, 500))
    G.add_edge(attacker_id, campaign_id)

    for i in range(rng.randint(1, 3)):
        host_id = f"host-{seed}-{i}"
        G.add_node(host_id, labels=["Host"])
        G.add_edge(campaign_id, host_id)

    for i in range(num_techniques):
        event_id = f"event-{seed}-{i}"
        technique_id = f"T{1000 + (seed * 7 + i) % 200}"

        G.add_node(event_id, labels=["AttackEvent"], occurrences=rng.randint(1, 10),
                   tps=rng.uniform(0, 150), rule_level=rng.randint(0, 15))
        G.add_node(technique_id, labels=["Technique"], occurrences=rng.randint(1, 20))

        G.add_edge(campaign_id, event_id)
        G.add_edge(event_id, technique_id)

    severity = _severity_from_structure(num_techniques, attacker_reputation, rng)
    return G, severity


def generate_synthetic_dataset(n: int, seed: int = 42) -> list[tuple[nx.DiGraph, str]]:
    rng = random.Random(seed)
    return [generate_synthetic_campaign_graph(rng.randint(0, 10_000_000)) for _ in range(n)]
