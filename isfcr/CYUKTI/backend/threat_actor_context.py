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
    