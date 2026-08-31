"""
evidence/schema.py
====================
Core Evidence representation shared by every evidence source CYUKTI can
draw on (SIEM detections, MITRE ATT&CK knowledge, CTI/MISP, the Neo4j
graph, historical campaign correlation, and threat attribution matches).

Design intent (see backend/investigation/ and backend/rag/):
    - `confidence` is how much we trust the evidence ITSELF, as reported
      by the system that produced it (e.g. MITRE ATT&CK reference data is
      authoritative -> 1.0; a MISP correlation score is whatever MISP/CTI
      confidence engine computed).
    - `relevance` is how relevant this piece of evidence is to the
      CURRENT investigation query. It starts at 0.0 and is filled in by
      a retriever (backend/rag), never by the collector that produced the
      evidence — a fact does not know in advance what it will be useful
      for.
    - `provenance` is a short human-readable trace of which engine/module
      produced the evidence, so an investigator (or the investigation
      loop) can tell *why* something is being trusted.
    - `relationships` holds ids of related evidence/entities (e.g. a
      historical-campaign match names the campaign_id it matched), so the
      investigation loop can navigate between related facts without a
      second query.

Both `confidence` and `relevance` are normalized to the same [0, 1]
scale, regardless of the 0-100 scale several upstream engines use
internally (ThreatIntelResult, DetectionResult, ThreatActorContext all
report confidence on 0-100 — collectors are responsible for dividing by
100 before constructing Evidence).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class EvidenceSource(str, Enum):
    SIEM = "siem"
    MITRE = "mitre"
    CTI = "cti"
    GRAPH = "graph"
    CAMPAIGN_HISTORY = "campaign_history"
    ATTRIBUTION = "attribution"


class EvidenceType(str, Enum):
    DETECTION = "detection"
    TECHNIQUE_KNOWLEDGE = "technique_knowledge"
    THREAT_INTEL = "threat_intel"
    GRAPH_RELATIONSHIP = "graph_relationship"
    HISTORICAL_MATCH = "historical_match"
    ATTRIBUTION_MATCH = "attribution_match"


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


@dataclass
class Evidence:
    source: EvidenceSource
    source_id: str
    timestamp: str
    type: EvidenceType
    content: dict[str, Any]
    confidence: float
    relevance: float = 0.0
    provenance: str = ""
    relationships: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.confidence = _clamp01(self.confidence)
        self.relevance = _clamp01(self.relevance)
        if isinstance(self.source, str):
            self.source = EvidenceSource(self.source)
        if isinstance(self.type, str):
            self.type = EvidenceType(self.type)

    @property
    def evidence_id(self) -> str:
        """Stable identity for dedup/graph-linking: (source, source_id, type)."""
        return f"{self.source.value}:{self.type.value}:{self.source_id}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "evidence_id": self.evidence_id,
            "source": self.source.value,
            "source_id": self.source_id,
            "timestamp": self.timestamp,
            "type": self.type.value,
            "content": self.content,
            "confidence": round(self.confidence, 4),
            "relevance": round(self.relevance, 4),
            "provenance": self.provenance,
            "relationships": list(self.relationships),
        }
