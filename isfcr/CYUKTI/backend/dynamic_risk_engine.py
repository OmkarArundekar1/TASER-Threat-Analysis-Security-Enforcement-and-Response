from dataclasses import dataclass
from datetime import datetime
from config import (
    PREDICTION_WEIGHT,
    DUPLICATE_WEIGHT,
    CAMPAIGN_WEIGHT,
    THREAT_INTEL_RISK_WEIGHT,
    MAX_DYNAMIC_RISK,
    LOW_RISK_THRESHOLD,
    MEDIUM_RISK_THRESHOLD,
    HIGH_RISK_THRESHOLD,
    GRAPH_DENSITY_WEIGHT,
    CHAIN_DEPTH_WEIGHT,
    COMPLEXITY_WEIGHT,
    STRUCTURAL_RISK_WEIGHT,
    EVOLUTION_RATE_WEIGHT,
)


@dataclass
class DynamicRiskResult:
    risk_score: float
    risk_level: str
    risk_color: str
    risk_trend: str
    confidence: float
    calculated_at: str
    breakdown: dict
class DynamicRiskEngine:
    def calculate_confidence(self,features):
        evidence = [
            1.0,
            1.0,
            1.0,
            1.0 if (
                features.threat_intel.ip_reputation != 0 or
                features.threat_intel.threat_actor_reputation != 0
            ) else 0.0,
            1.0 if features.runtime.prediction_frequency > 0 else 0.0,
            1.0 if (
                features.graph.node_count > 0
            ) else 0.0,
            1.0 if (
                features.graph.attack_chain_depth > 0
            ) else 0.0
        ]
    
        return round(sum(evidence) / len(evidence) * 100, 2)
    def calculate(
        self,
        features,
        severity,
        previous_risk=None
    ):
        score = severity.base_severity
        breakdown = dict(severity.breakdown)
        prediction_modifier = min(
            features.runtime.prediction_frequency *
            PREDICTION_WEIGHT,
            10
        )
        breakdown["prediction_modifier"] = prediction_modifier
        score += prediction_modifier
        duplicate_modifier = min(
            features.runtime.duplicate_frequency *
            DUPLICATE_WEIGHT,
            5
        )
        breakdown["duplicate_modifier"] = duplicate_modifier
        score += duplicate_modifier
        campaign_modifier = min(
            features.runtime.attack_event_count *
            CAMPAIGN_WEIGHT,
            15
        )
        graph_modifier = min(
            (
                features.graph.graph_density *
                GRAPH_DENSITY_WEIGHT
                +
                features.graph.attack_chain_depth *
                CHAIN_DEPTH_WEIGHT
                +
                features.graph.campaign_complexity *
                COMPLEXITY_WEIGHT
                +
                features.graph.structural_risk *
                STRUCTURAL_RISK_WEIGHT
                +
                features.graph.evolution_rate *
                EVOLUTION_RATE_WEIGHT
            ),
            25
        )
        breakdown["graph_modifier"] = graph_modifier
        score += graph_modifier
        breakdown["campaign_modifier"] = campaign_modifier
        score += campaign_modifier
        ti_modifier = min(
            (
                features.threat_intel.ip_reputation +
                features.threat_intel.threat_actor_reputation
            ) * THREAT_INTEL_RISK_WEIGHT,
            15
        )
        breakdown["threat_intel_modifier"] = ti_modifier
        score += ti_modifier
        score = min(score, MAX_DYNAMIC_RISK)
        if score < LOW_RISK_THRESHOLD:
            level = "LOW"
            color = "GREEN"
        elif score < MEDIUM_RISK_THRESHOLD:
            level = "MEDIUM"
            color = "YELLOW"
        elif score < HIGH_RISK_THRESHOLD:
            level = "HIGH"
            color = "ORANGE"
        else:
            level = "CRITICAL"
            color = "RED"
        if previous_risk is None:
            trend = "NEW"
        else:
            delta = score - previous_risk
            if delta > 5:
                trend = "RAPIDLY_INCREASING"
            elif delta > 0:
                trend = "INCREASING"
            elif delta < -5:
                trend = "RAPIDLY_DECREASING"
            elif delta < 0:
                trend = "DECREASING"
            else:
                trend = "STABLE"
        
        confidence = self.calculate_confidence(features)
        breakdown["confidence"] = confidence
        breakdown["graph_density"] = features.graph.graph_density
        breakdown["chain_depth"] = features.graph.attack_chain_depth
        breakdown["campaign_complexity"] = (
            features.graph.campaign_complexity
        )
        breakdown["structural_risk"] = (
            features.graph.structural_risk
        )
        return DynamicRiskResult(
            risk_score=round(score, 2),
            risk_level=level,
            risk_color=color,
            risk_trend=trend,
            confidence=confidence,
            calculated_at=datetime.utcnow().isoformat(),
            breakdown=breakdown
        )

engine = DynamicRiskEngine()