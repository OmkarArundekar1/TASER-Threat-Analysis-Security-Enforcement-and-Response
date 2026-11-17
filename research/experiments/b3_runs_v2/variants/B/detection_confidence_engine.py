from dataclasses import dataclass
from neo4j_client import get_detection_confidence


@dataclass
class DetectionResult:
    confidence: float
    level: str
    breakdown: dict


class DetectionConfidenceEngine:
    def extract_features(self, event_id):
        return get_detection_confidence(event_id)

    def calculate(self, event_id):
        f = self.extract_features(event_id)

        confidence = min(100, f.detection_confidence * 10)
        if confidence >= 85:
            level = "HIGH"

        elif confidence >= 65:
            level = "MEDIUM"

        else:
            level = "LOW"

        return DetectionResult(
            confidence=confidence,
            level=level,
            breakdown={
                "Wazuh": f.wazuh_level,
                "Suricata": f.suricata_score,
                "Zeek": f.zeek_score,
                "Sigma": f.sigma_score,
                "YARA": f.yara_score,
            }
        )


engine = DetectionConfidenceEngine()