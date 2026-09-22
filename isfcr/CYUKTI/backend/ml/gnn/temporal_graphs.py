"""
ml/gnn/temporal_graphs.py
============================
Temporal snapshot extraction for Objective C (GNN_REPRESENTATION_DESIGN.md
Section 6): for each campaign with 2+ distinctly-timestamped real
AttackEvents, builds a growing sequence of graph snapshots, each
restricted to events with `first_seen <= cutoff`, using only real,
already-stored Neo4j timestamps -- no fabricated data, no SIMILAR_TO/
RESEMBLES, no waiting in real time.

Deliberately a SEPARATE, explicitly-typed query (LAUNCHED/HAS_EVENT/
MATCHES/TARGETS only) rather than reusing GraphSnapshotLoader:

  1. GraphSnapshotLoader has no timestamp-cutoff concept at all -- this
     is genuinely new capability, not a modification of existing
     behavior (GraphSnapshotLoader is not touched, imported, or called
     by this module).
  2. GraphSnapshotLoader's untyped (a)-[r1]->(c) / (c)-[r4]->(h)
     patterns (GNN_OBJECTIVE_DECISION.md Section 8) would pull in
     SIMILAR_TO/HAS_CAMPAIGN/RESEMBLES/LIKELY_NEXT edges that have no
     per-event timestamp of their own to filter by -- explicitly typing
     this new query sidesteps that ambiguity rather than inheriting it.

Reuses `graph_feature_engine.GraphSnapshot`/`GraphBuilder` (already
generic -- they only turn a list of node/relationship dicts into an
nx.DiGraph, regardless of which query produced them) and
`graph_encoder.encode_graph` (unchanged). No production file is
modified by this module.
"""

from __future__ import annotations

from dataclasses import dataclass

from graph_encoder import EncodedGraph, encode_graph

_SNAPSHOT_QUERY = """
MATCH (c:Campaign {campaign_id:$campaign_id})
OPTIONAL MATCH (a:Attacker)-[r1:LAUNCHED]->(c)
OPTIONAL MATCH (c)-[r2:HAS_EVENT]->(e:AttackEvent)
WHERE e.first_seen <= $cutoff
OPTIONAL MATCH (e)-[r3:MATCHES]->(t:Technique)
OPTIONAL MATCH (c)-[r4:TARGETS]->(h:Host)

WITH
    collect(DISTINCT a) +
    collect(DISTINCT c) +
    collect(DISTINCT e) +
    collect(DISTINCT t) +
    collect(DISTINCT h) AS nodes,

    collect(DISTINCT r1) +
    collect(DISTINCT r2) +
    collect(DISTINCT r3) +
    collect(DISTINCT r4) AS rels

RETURN nodes, rels
"""


@dataclass
class TemporalSnapshot:
    campaign_id: str
    snapshot_index: int
    cutoff_timestamp: str
    graph: EncodedGraph
    num_distinct_timestamps_included: int


def list_temporal_campaign_ids() -> list[str]:
    """Live re-query (not a cached/hardcoded count) of campaigns with
    >=2 distinct real AttackEvent timestamps -- the minimum needed for
    even a 2-point temporal pair. Stable (alphabetical) order for
    reproducibility, matching campaign_graphs.list_campaign_ids's
    convention."""
    from neo4j_client import driver

    with driver.session() as session:
        result = session.run("""
            MATCH (c:Campaign)-[:HAS_EVENT]->(e:AttackEvent)
            WITH c, collect(DISTINCT e.first_seen) AS ts
            WHERE size(ts) >= 2
            RETURN c.campaign_id AS id
            ORDER BY id
        """)
        return [record["id"] for record in result]


def _distinct_event_timestamps(campaign_id: str) -> list:
    from neo4j_client import driver

    with driver.session() as session:
        result = session.run("""
            MATCH (:Campaign {campaign_id:$campaign_id})-[:HAS_EVENT]->(e:AttackEvent)
            RETURN DISTINCT e.first_seen AS ts
            ORDER BY ts
        """, campaign_id=campaign_id)
        return [record["ts"] for record in result]


def _build_snapshot_graph(campaign_id: str, cutoff):
    from graph_feature_engine import GraphBuilder, GraphSnapshot
    from neo4j_client import driver

    snapshot = GraphSnapshot()
    with driver.session() as session:
        record = session.run(_SNAPSHOT_QUERY, campaign_id=campaign_id, cutoff=cutoff).single()
        if record is not None:
            for node in record["nodes"]:
                if node is None:
                    continue
                snapshot.nodes.append({
                    "id": node.element_id,
                    "labels": list(node.labels),
                    "properties": dict(node),
                })
            for rel in record["rels"]:
                if rel is None:
                    continue
                snapshot.relationships.append({
                    "id": rel.element_id,
                    "type": rel.type,
                    "start": rel.start_node.element_id,
                    "end": rel.end_node.element_id,
                    "properties": dict(rel),
                })
    return GraphBuilder().build(snapshot)


def build_temporal_snapshots(campaign_id: str) -> list[TemporalSnapshot]:
    """Returns [] if the campaign has fewer than 2 distinct event
    timestamps (checked live, not assumed) -- the documented limitation
    for whoever calls this on a campaign outside list_temporal_campaign_ids().
    Each returned snapshot's node/edge set is a superset of the previous
    one's, by construction (cutoff is non-decreasing) -- later snapshots
    can only gain events, never lose them, which is what "no future
    leakage" (Phase I) means made concrete: snapshot i is built using
    only information with first_seen <= timestamps[i]."""
    timestamps = _distinct_event_timestamps(campaign_id)
    if len(timestamps) < 2:
        return []

    snapshots = []
    for index, cutoff in enumerate(timestamps):
        graph = _build_snapshot_graph(campaign_id, cutoff)
        encoded = encode_graph(graph)
        snapshots.append(TemporalSnapshot(
            campaign_id=campaign_id,
            snapshot_index=index,
            cutoff_timestamp=str(cutoff),
            graph=encoded,
            num_distinct_timestamps_included=index + 1,
        ))
    return snapshots


def build_all_temporal_snapshots() -> dict[str, list[TemporalSnapshot]]:
    return {cid: build_temporal_snapshots(cid) for cid in list_temporal_campaign_ids()}


if __name__ == "__main__":
    import json

    all_snapshots = build_all_temporal_snapshots()
    print(json.dumps({
        "campaigns_with_temporal_data": len(all_snapshots),
        "snapshots_per_campaign": {cid: len(snaps) for cid, snaps in all_snapshots.items()},
    }, indent=2))
