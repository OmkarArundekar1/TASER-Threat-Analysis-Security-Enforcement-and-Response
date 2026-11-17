from dataclasses import dataclass, field
from feature_orchestrator import FeatureVector
from datetime import datetime

@dataclass
class OperationContext:
    operation_id: str
    campaign_ids: list = field(default_factory=list)
    primary_attacker: str = ""
    victims: set = field(default_factory=set)
    techniques: set = field(default_factory=set)
    attack_chain: list = field(default_factory=list)
    features: FeatureVector | None = None
    prediction_profile: dict = field(default_factory=dict)
    risk_profile: dict = field(default_factory=dict)
    first_seen: datetime | None = None
    last_seen: datetime | None = None