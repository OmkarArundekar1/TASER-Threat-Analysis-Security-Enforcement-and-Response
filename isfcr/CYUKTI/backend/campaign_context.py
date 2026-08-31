from dataclasses import dataclass, field
from datetime import datetime
from feature_orchestrator import FeatureVector

@dataclass
class CampaignContext:
    campaign_id: str
    attacker_ip: str
    victim_ip: str
    status: str = "ACTIVE"
    first_seen: datetime | None = None
    last_seen: datetime | None = None
    last_technique: str | None = None
    observed_chain: list = field(default_factory=list)
    predicted_next: str | None = None
    prediction_confidence: float = 0.0
    prediction_hits: int = 0
    prediction_misses: int = 0
    reopened_count: int = 0
    prediction_generated_at: datetime | None = None
    risk_score: float = 0.0
    skip_next_chain_update: bool = False
    techniques: set = field(default_factory=set)
    attack_chain: list = field(default_factory=list)
    features: FeatureVector | None = None