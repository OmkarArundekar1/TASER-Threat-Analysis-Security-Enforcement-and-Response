from dataclasses import dataclass, field
from typing import List

@dataclass
class ThreatAttributionResult:
    actors: List

@dataclass
class HistoricalCampaign:
    campaign_id: str
    attacker: str
    victim: str
    techniques: List[str] = field(default_factory=list)
    timestamps: List[str] = field(default_factory=list)
    status: str = ""
    prediction: str = ""
    prediction_confidence: float = 0.0

@dataclass
class AttributionFeatures:
    coverage: float = 0.0
    precision: float = 0.0
    chain_similarity: float = 0.0

@dataclass
class AttributionResult:
    campaign: HistoricalCampaign
    similarity: float
    features: AttributionFeatures

