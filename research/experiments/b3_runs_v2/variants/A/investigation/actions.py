"""
investigation/actions.py
===========================
The fixed menu of evidence-gathering actions the investigation loop can
choose from, with static per-action metadata (which EvidenceSource it
targets, how reliable that source generally is, and its approximate
cost/latency). These numbers are declared estimates, not learned or
measured against production telemetry — there is no historical dataset
of "how long did a MISP lookup actually take" to learn them from yet.
They are deliberately conservative and documented so they can be
replaced with measured values once real operational data exists.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from evidence.schema import EvidenceSource


class InvestigationAction(str, Enum):
    MITRE_KNOWLEDGE = "mitre_knowledge"          # mitre_feature_engine (Neo4j-backed ATT&CK lookup)
    MITRE_SEMANTIC_SEARCH = "mitre_semantic_search"  # rag.mitre_retriever (TF-IDF over STIX corpus)
    CTI_LOOKUP = "cti_lookup"                    # threat_intelligence_engine (MISP/VT)
    DETECTION_CHECK = "detection_check"          # detection_confidence_engine (Wazuh/Suricata/Zeek/Sigma/YARA)
    ATTRIBUTION_MATCH = "attribution_match"      # threat_attribution_engine (historical similarity)
    CAMPAIGN_HISTORY = "campaign_history"        # attribution_context (raw historical campaign records)
    CAMPAIGN_NARRATIVE_SEARCH = "campaign_narrative_search"  # rag.campaign_retriever (TF-IDF over historical campaign records)
    GRAPH_STRUCTURE = "graph_structure"          # graph_feature_engine (Neo4j structural analytics)
    XGBOOST_PREDICTION = "xgboost_prediction"    # ml.train_xgboost (severity classifier)
    GNN_TOPOLOGY_RETRIEVAL = "gnn_topology_retrieval"  # rag.gnn_topology_retriever (GNN embedding similarity over historical campaigns)


@dataclass(frozen=True)
class ActionMeta:
    source: EvidenceSource
    reliability: float  # how much this source's evidence is generally trusted, [0, 1]
    cost: float         # relative computational/operational cost, [0, 1]
    latency: float      # relative wall-clock latency, [0, 1] (network calls > local computation)
    depends_on: frozenset[InvestigationAction] = frozenset()
    # Other actions whose underlying data this action's computation already
    # draws on, verified against the real engine code (not guessed) — see
    # investigation/next_best_evidence.py's redundancy_penalty. Declaring a
    # dependency here means: once an action in this set has already been
    # taken, this action's remaining novelty is discounted, because the
    # evidence it would produce substantially overlaps information already
    # in the store rather than being independent corroboration.


ACTION_METADATA: dict[InvestigationAction, ActionMeta] = {
    InvestigationAction.MITRE_KNOWLEDGE: ActionMeta(EvidenceSource.MITRE, reliability=1.0, cost=0.1, latency=0.1),
    # MITRE_SEMANTIC_SEARCH shares EvidenceSource.MITRE with MITRE_KNOWLEDGE,
    # so it is already discounted by the existing same-source novelty
    # mechanism below (REPEAT_QUERY_NOVELTY) -- no separate depends_on
    # declaration needed; declaring one would double-penalize the same fact.
    InvestigationAction.MITRE_SEMANTIC_SEARCH: ActionMeta(EvidenceSource.MITRE, reliability=0.7, cost=0.15, latency=0.15),
    InvestigationAction.CTI_LOOKUP: ActionMeta(EvidenceSource.CTI, reliability=0.75, cost=0.5, latency=0.7),
    InvestigationAction.DETECTION_CHECK: ActionMeta(EvidenceSource.SIEM, reliability=0.85, cost=0.2, latency=0.2),
    # threat_attribution_engine.attribute() calls context.load_historical_campaigns()
    # internally (verified by reading the source), the same data
    # CAMPAIGN_HISTORY's own collector queries directly -- a real,
    # code-verified cross-source overlap the existing same-source discount
    # cannot see (ATTRIBUTION and CAMPAIGN_HISTORY are different
    # EvidenceSource values), which is exactly the gap depends_on exists for.
    InvestigationAction.ATTRIBUTION_MATCH: ActionMeta(
        EvidenceSource.ATTRIBUTION, reliability=0.65, cost=0.4, latency=0.3,
        depends_on=frozenset({InvestigationAction.CAMPAIGN_HISTORY}),
    ),
    InvestigationAction.CAMPAIGN_HISTORY: ActionMeta(EvidenceSource.CAMPAIGN_HISTORY, reliability=0.9, cost=0.3, latency=0.3),
    # Second Multi-RAG source (rag/campaign_retriever.py) -- shares
    # EvidenceSource.CAMPAIGN_HISTORY with the exhaustive CAMPAIGN_HISTORY
    # action above for the same reason MITRE_SEMANTIC_SEARCH shares
    # EvidenceSource.MITRE with MITRE_KNOWLEDGE: both answer questions
    # about the same underlying data (historical campaigns), so the
    # existing same-source novelty discount already prevents double-
    # crediting redundant retrieval -- no separate depends_on needed.
    InvestigationAction.CAMPAIGN_NARRATIVE_SEARCH: ActionMeta(
        EvidenceSource.CAMPAIGN_HISTORY, reliability=0.65, cost=0.3, latency=0.3,
    ),
    InvestigationAction.GRAPH_STRUCTURE: ActionMeta(EvidenceSource.GRAPH, reliability=1.0, cost=0.3, latency=0.3),
    # XGBoost prediction doesn't map to one of the six Evidence sources — it's a
    # model VERDICT, not a retrieved fact. Attributed to GRAPH as the closest
    # existing category (it consumes graph+CTI+MITRE features) purely so the
    # novelty accounting in next_best_evidence.py has something to compare against.
    InvestigationAction.XGBOOST_PREDICTION: ActionMeta(EvidenceSource.GRAPH, reliability=0.6, cost=0.25, latency=0.25),
    # rag.gnn_topology_retriever queries attribution_context.context.load_historical_campaigns()
    # for its candidate pool (same reason as CAMPAIGN_HISTORY/ATTRIBUTION_MATCH's
    # depends_on -- a real, code-verified overlap, declared explicitly since
    # EvidenceSource.GNN_TOPOLOGY is its own distinct source, not shared with
    # CAMPAIGN_HISTORY, so the same-source discount doesn't already catch it).
    # reliability=0.5, deliberately not high: GNN_RETRIEVAL_EVALUATION.md's own
    # findings are "promising but limited" (temporal retrieval well above chance;
    # attacker/host retrieval strong but a mechanistically low bar; no validated
    # comparison establishing GNN retrieval beats CAMPAIGN_NARRATIVE_SEARCH for
    # this specific investigation use case) -- not a research-validated high-trust
    # source yet, and this number should not be read as one.
    InvestigationAction.GNN_TOPOLOGY_RETRIEVAL: ActionMeta(
        EvidenceSource.GNN_TOPOLOGY, reliability=0.5, cost=0.35, latency=0.4,
        depends_on=frozenset({InvestigationAction.CAMPAIGN_HISTORY}),
    ),
}
