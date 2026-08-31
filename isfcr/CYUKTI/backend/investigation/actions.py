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
    GRAPH_STRUCTURE = "graph_structure"          # graph_feature_engine (Neo4j structural analytics)
    XGBOOST_PREDICTION = "xgboost_prediction"    # ml.train_xgboost (severity classifier)


@dataclass(frozen=True)
class ActionMeta:
    source: EvidenceSource
    reliability: float  # how much this source's evidence is generally trusted, [0, 1]
    cost: float         # relative computational/operational cost, [0, 1]
    latency: float      # relative wall-clock latency, [0, 1] (network calls > local computation)


ACTION_METADATA: dict[InvestigationAction, ActionMeta] = {
    InvestigationAction.MITRE_KNOWLEDGE: ActionMeta(EvidenceSource.MITRE, reliability=1.0, cost=0.1, latency=0.1),
    InvestigationAction.MITRE_SEMANTIC_SEARCH: ActionMeta(EvidenceSource.MITRE, reliability=0.7, cost=0.15, latency=0.15),
    InvestigationAction.CTI_LOOKUP: ActionMeta(EvidenceSource.CTI, reliability=0.75, cost=0.5, latency=0.7),
    InvestigationAction.DETECTION_CHECK: ActionMeta(EvidenceSource.SIEM, reliability=0.85, cost=0.2, latency=0.2),
    InvestigationAction.ATTRIBUTION_MATCH: ActionMeta(EvidenceSource.ATTRIBUTION, reliability=0.65, cost=0.4, latency=0.3),
    InvestigationAction.CAMPAIGN_HISTORY: ActionMeta(EvidenceSource.CAMPAIGN_HISTORY, reliability=0.9, cost=0.3, latency=0.3),
    InvestigationAction.GRAPH_STRUCTURE: ActionMeta(EvidenceSource.GRAPH, reliability=1.0, cost=0.3, latency=0.3),
    # XGBoost prediction doesn't map to one of the six Evidence sources — it's a
    # model VERDICT, not a retrieved fact. Attributed to GRAPH as the closest
    # existing category (it consumes graph+CTI+MITRE features) purely so the
    # novelty accounting in next_best_evidence.py has something to compare against.
    InvestigationAction.XGBOOST_PREDICTION: ActionMeta(EvidenceSource.GRAPH, reliability=0.6, cost=0.25, latency=0.25),
}
