from dataclasses import dataclass
from neo4j_client import get_threat_intel_features


@dataclass
class ThreatIntelResult:
    confidence: float
    level: str
    breakdown: dict

class ThreatIntelligenceEngine:

    def extract_features(self, attacker_ip):
        return get_threat_intel_features(attacker_ip)

    def calculate(self, attacker_ip):
        f = self.extract_features(attacker_ip)

        confidence = (
            f.ip_reputation +
            f.threat_actor_reputation +
            f.malware_confidence +
            f.tool_confidence +
            f.misp_confidence +
            f.ioc_confidence
        ) / 6

        confidence = round(confidence, 2)

        if confidence >= 85:
            level = "HIGH"

        elif confidence >= 65:
            level = "MEDIUM"

        else:
            level = "LOW"

        return ThreatIntelResult(
            confidence=confidence,
            level=level,
            breakdown={
                "VT": f.ip_reputation,
                "Threat Actor": f.threat_actor_reputation,
                "Malware": f.malware_confidence,
                "Tool": f.tool_confidence,
                "MISP": f.misp_confidence,
                "IOC": f.ioc_confidence,
            }
        )


engine = ThreatIntelligenceEngine()