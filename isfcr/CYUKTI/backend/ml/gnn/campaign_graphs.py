"""
ml/gnn/campaign_graphs.py
============================
Real counterpart to synthetic_graphs.py: builds (EncodedGraph,
severity-label) samples from CYUKTI's actual, live Neo4j campaign
graphs — not synthetic fixtures.

Deliberately reuses, rather than re-implements, two already-production
paths so this module can never silently diverge from what XGBoost
already trains on:

  - The graph itself: `graph_feature_engine.graph_analytics.load_graph`
    (GraphSnapshotLoader -> GraphBuilder), the same real-Neo4j extraction
    path `campaign_feature_engine.py` already depends on for XGBoost's
    57 tabular features, then `graph_encoder.encode_graph` (this
    package) to get tensors.
  - The label: `risk_scoring.normalize_risk_score` /
    `risk_level_from_score`, the same two pure functions
    `ml/label_generator.py` already uses to derive XGBoost's `severity`
    label from a campaign's `risk_score`.

This module intentionally stops at dataset assembly — it does not call
train_gnn(). See ../../GNN_FEASIBILITY.md for why: the real severity
label distribution across today's 71 campaigns is severely imbalanced
(one of four classes entirely unobserved), and ~70% of real campaigns
have only a single AttackEvent (a near-trivial subgraph for a GNN's
message passing to learn from) — a genuine dataset-sufficiency
question, not something this module can or should paper over.

`severity` here is a derived/heuristic label (a thresholded function of
`risk_score`), not independently verified ground truth — same caveat
ml/label_generator.py and ml/dataset_utils.py already document for
XGBoost's identical label. Not a new limitation this module introduces.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from graph_encoder import EncodedGraph, encode_graph

# Copied verbatim from ml/label_generator.py's _LEVEL_TO_SEVERITY rather
# than imported, so this module has no import-time dependency on the
# ml/ package's own sys.path bootstrap (ml/gnn/__init__.py only adds
# itself, not ml/, to sys.path). Guarded against silent drift by
# test_campaign_graphs.py's test_severity_mapping_matches_label_generator.
_LEVEL_TO_SEVERITY = {
    "CRITICAL": "Critical",
    "HIGH": "High",
    "MEDIUM": "Medium",
    "LOW": "Low",
}


@dataclass
class CampaignGraphSample:
    campaign_id: str
    graph: EncodedGraph
    severity: str
    risk_score: float
    num_events: int


def list_campaign_ids() -> list[str]:
    """All real Campaign node ids currently in Neo4j, in a stable
    (alphabetical) order — so repeated calls against the same database
    state produce the same dataset order, not an incidental one."""
    from neo4j_client import driver

    with driver.session() as session:
        result = session.run("MATCH (c:Campaign) RETURN c.campaign_id AS id ORDER BY c.campaign_id")
        return [record["id"] for record in result]


def build_campaign_graph_sample(campaign_id: str) -> CampaignGraphSample | None:
    """Extract one real campaign's subgraph + its real severity label.

    Returns None if the campaign has no risk_score on record (should not
    happen for a campaign created via neo4j_client.create_campaign_context,
    but handled rather than crashing a whole batch build over one bad
    row — the same defensive posture GraphSnapshotLoader itself takes on
    a query failure).
    """
    from neo4j_client import get_campaign_context_data
    from risk_scoring import normalize_risk_score, risk_level_from_score

    from graph_feature_engine import graph_analytics

    context = get_campaign_context_data(campaign_id)
    if context is None:
        return None
    risk_score = context["campaign"].get("risk_score")
    if risk_score is None:
        return None

    severity = _LEVEL_TO_SEVERITY[risk_level_from_score(normalize_risk_score(risk_score))]

    nx_graph = graph_analytics.load_graph(campaign_id)
    encoded = encode_graph(nx_graph)
    num_events = sum(1 for node_type in encoded.node_types if node_type == "AttackEvent")

    return CampaignGraphSample(
        campaign_id=campaign_id,
        graph=encoded,
        severity=severity,
        risk_score=float(risk_score),
        num_events=num_events,
    )


def build_real_campaign_dataset() -> list[CampaignGraphSample]:
    """Deterministic given the current database state: same real Neo4j
    data in -> same list out, in the same order, every call."""
    samples = []
    for campaign_id in list_campaign_ids():
        sample = build_campaign_graph_sample(campaign_id)
        if sample is not None:
            samples.append(sample)
    return samples


if __name__ == "__main__":
    import json

    samples = build_real_campaign_dataset()
    print(json.dumps({
        "campaigns_in_neo4j": len(list_campaign_ids()),
        "samples_built": len(samples),
        "severity_distribution": dict(Counter(s.severity for s in samples)),
        "events_per_campaign_distribution": dict(Counter(s.num_events for s in samples)),
    }, indent=2))
