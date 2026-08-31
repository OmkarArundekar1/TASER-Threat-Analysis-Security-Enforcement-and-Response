from dataclasses import dataclass
from campaign_feature_engine import CampaignFeatures
from config import CAMPAIGN_WEIGHTS

@dataclass
class CampaignDecision:
    score: float
    confidence: float
    available_features: int
    total_features: int
    breakdown: dict

class CampaignDecisionEngine:
    def evaluate(
        self,
        features: CampaignFeatures
    ):
        breakdown = {
            "prediction_similarity":
                features.prediction_similarity,
            "chain_similarity":
                features.chain_similarity,
            "temporal_similarity":
                features.temporal_similarity,
            "attacker_similarity":
                features.attacker_similarity,
            "duplicate_similarity":
                features.duplicate_similarity,
            "graph_similarity":
                features.graph_similarity,
            "runtime_similarity":
                features.runtime_similarity
        }

        
        weighted_sum = 0.0
        total_weight = 0.0
        available_features = 0
        
        for key, value in breakdown.items():
        
            if value is None:
                continue
        
            weight = CAMPAIGN_WEIGHTS[key]
        
            weighted_sum += value * weight
            total_weight += weight
            available_features += 1
        
        score = (
            weighted_sum / total_weight
            if total_weight > 0
            else 0.0
        )
        
        confidence = round(
            (available_features / len(breakdown)) * 100,
            2
        )
        
        return CampaignDecision(
            score=score,
            confidence=confidence,
            available_features=available_features,
            total_features=len(breakdown),
            breakdown=breakdown
        )
engine = CampaignDecisionEngine()