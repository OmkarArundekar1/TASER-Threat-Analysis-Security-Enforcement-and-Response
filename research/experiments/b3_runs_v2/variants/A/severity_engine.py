from dataclasses import dataclass
from config import (
    PLATFORM_WEIGHT,
    MITIGATION_WEIGHT,
    THREAT_ACTOR_WEIGHT,
    MALWARE_WEIGHT,
    TOOL_WEIGHT,
    RUNTIME_EVENT_WEIGHT,
    DETECTION_WEIGHT,
    THREAT_INTEL_WEIGHT,
    GRAPH_DENSITY_WEIGHT,
    CHAIN_DEPTH_WEIGHT,
    COMPLEXITY_WEIGHT,
    STRUCTURAL_RISK_WEIGHT,
    EVOLUTION_RATE_WEIGHT,
    MAX_SEVERITY_SCORE
)

@dataclass
class SeverityResult:
    base_severity: float
    breakdown: dict

class SeverityEngine:
    def calculate(self, features):
        score = 0.0
        breakdown = {}
        platform_score = min(
            features.mitre.platform_count * PLATFORM_WEIGHT,
            5
        )
        breakdown["platform_score"] = platform_score
        score += platform_score
        mitigation_score = min(
            features.mitre.mitigation_count * MITIGATION_WEIGHT,
            5
        )
        breakdown["mitigation_score"] = mitigation_score
        score += mitigation_score
        threat_actor_score = min(
            features.mitre.threat_actor_count * THREAT_ACTOR_WEIGHT,
            10
        )
        breakdown["threat_actor_score"] = threat_actor_score
        score += threat_actor_score
        malware_score = min(
            features.mitre.malware_count * MALWARE_WEIGHT,
            5
        )
        breakdown["malware_score"] = malware_score
        score += malware_score
        tool_score = min(
            features.mitre.tool_count * TOOL_WEIGHT,
            5
        )
        breakdown["tool_score"] = tool_score
        score += tool_score
        runtime_score = min(
            features.runtime.attack_event_count *
            RUNTIME_EVENT_WEIGHT,
            5
        )
        graph_score = min(
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
            15
        )
        breakdown["graph_score"] = graph_score
        score += graph_score
        breakdown["runtime_score"] = runtime_score
        score += runtime_score
        ti_score = min(
            features.threat_intel.ip_reputation *
            THREAT_INTEL_WEIGHT,
            3
        )
        breakdown["threat_intel_score"] = ti_score
        score += ti_score
        detection_score = min(
            features.detection.detection_confidence *
            DETECTION_WEIGHT,
            7
        )
        breakdown["detection_score"] = detection_score
        score += detection_score
        score = min(score, MAX_SEVERITY_SCORE)
        breakdown["base_severity"] = score
        return SeverityResult(
            base_severity=score,
            breakdown=breakdown
        )
engine = SeverityEngine()