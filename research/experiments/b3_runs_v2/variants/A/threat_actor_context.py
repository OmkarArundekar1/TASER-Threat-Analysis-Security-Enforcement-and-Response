from dataclasses import dataclass, field

@dataclass
class ThreatActorContext:
    actor: str
    confidence: float = 0.0
    technique_similarity: float = 0.0
    chain_similarity: float = 0.0
    infrastructure_similarity: float = 0.0
    campaign_similarity: float = 0.0
    prediction_similarity: float = 0.0
    total_score: float = 0.0
    matched_techniques: list = field(default_factory=list)
    matched_predictions: list = field(default_factory=list)
    evidence: list = field(default_factory=list)
    coverage: float = 0.0
    precision: float = 0.0
    # Additive, informational only (GNN_PRODUCTION_INTEGRATION.md). GNN
    # embedding cosine similarity between the current campaign and this
    # historical campaign -- None (not 0.0) when unavailable (GNN
    # disabled/no artifact/malformed graph), so "not computed" is never
    # confused with "computed as dissimilar". NEVER used in `total_score`
    # or the ranking `attribute()` sorts by -- ThreatAttributionEngine's
    # existing coverage/precision/chain_similarity computation is
    # completely untouched; this is a distinct, separately-provenanced
    # signal attached alongside it, not a redefinition of attribution.
    topology_similarity: float | None = None
