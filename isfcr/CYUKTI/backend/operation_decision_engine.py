from dataclasses import dataclass
from operation_feature_engine import OperationFeatures
from config import IMPLEMENTED_FEATURES
@dataclass
class OperationDecision:
    score: float
    confidence: float
    decision: str
    breakdown: dict

class OperationDecisionEngine:
    ATTACH_THRESHOLD = 0.70
    weights = {
        "attacker_similarity": 0.20,
        "victim_similarity": 0.15,
        "technique_similarity": 0.30,
        "temporal_similarity": 0.20,
        "chain_similarity": 0.10,
        "prediction_similarity": 0.05,
        "graph_similarity": 0.00,
    }
    def evaluate(
        self,
        features: OperationFeatures
    ):
        score = (
            features.attacker_similarity * self.weights["attacker_similarity"] +
            features.victim_similarity * self.weights["victim_similarity"] +
            features.technique_similarity * self.weights["technique_similarity"] +
            features.temporal_similarity * self.weights["temporal_similarity"] +
            features.chain_similarity * self.weights["chain_similarity"] +
            features.prediction_similarity * self.weights["prediction_similarity"] +
            features.graph_similarity * self.weights["graph_similarity"]
        )
        implemented = len(IMPLEMENTED_FEATURES)
        total = len(OperationFeatures.__dataclass_fields__)
        
        confidence = (implemented / total) * 100
        decision = (
            "ATTACH_TO_OPERATION"
            if score >= self.ATTACH_THRESHOLD
            else "CREATE_NEW_OPERATION"
        )
        return OperationDecision(
            score=round(score, 3),
            confidence=round(confidence, 1),
            decision=decision,
            breakdown = {
                "attacker_similarity": {
                    "value": features.attacker_similarity,
                    "weight": self.weights["attacker_similarity"],
                    "contribution": features.attacker_similarity * self.weights["attacker_similarity"]
                },
            
                "victim_similarity": {
                    "value": features.victim_similarity,
                    "weight": self.weights["victim_similarity"],
                    "contribution": features.victim_similarity * self.weights["victim_similarity"]
                },
            
                "temporal_similarity": {
                    "value": features.temporal_similarity,
                    "weight": self.weights["temporal_similarity"],
                    "contribution": features.temporal_similarity * self.weights["temporal_similarity"]
                },
            
                "technique_similarity": {
                    "value": features.technique_similarity,
                    "weight": self.weights["technique_similarity"],
                    "contribution": features.technique_similarity * self.weights["technique_similarity"]
                },
            
                "chain_similarity": {
                    "value": features.chain_similarity,
                    "weight": self.weights["chain_similarity"],
                    "contribution": features.chain_similarity * self.weights["chain_similarity"]
                },
            
                "prediction_similarity": {
                    "value": features.prediction_similarity,
                    "weight": self.weights["prediction_similarity"],
                    "contribution": features.prediction_similarity * self.weights["prediction_similarity"]
                },
            
                "graph_similarity": {
                    "value": features.graph_similarity,
                    "weight": self.weights["graph_similarity"],
                    "contribution": features.graph_similarity * self.weights["graph_similarity"]
                }
            }
        )