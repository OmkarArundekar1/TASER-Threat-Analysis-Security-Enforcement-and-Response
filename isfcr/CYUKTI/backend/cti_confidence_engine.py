from dataclasses import dataclass

DETECTION_WEIGHT = 0.25
RISK_WEIGHT = 0.20
THREAT_WEIGHT = 0.20
CAMPAIGN_WEIGHT = 0.20
PREDICTION_WEIGHT = 0.15

HIGH_THRESHOLD = 85.0
MEDIUM_THRESHOLD = 65.0

PUBLISH_THRESHOLD = 40.0


@dataclass
class CTIConfidence:
    score: float
    level: str
    publish: bool
    breakdown: dict


# ==========================================================
# Engine
# ==========================================================

class CTIConfidenceEngine:

    @staticmethod
    def _clamp(value):
        """
        Clamp every score into the range [0,100].
        """
        value = float(value)
        return max(0.0, min(100.0, value))

    def calculate(
        self,
        detection_confidence,
        risk_score,
        threat_confidence,
        campaign_confidence,
        prediction_confidence,
    ):

        detection_confidence = self._clamp(detection_confidence)
        risk_score = self._clamp(risk_score)
        threat_confidence = self._clamp(threat_confidence)
        campaign_confidence = self._clamp(campaign_confidence)
        prediction_confidence = self._clamp(prediction_confidence)

        score = (
            detection_confidence * DETECTION_WEIGHT
            + risk_score * RISK_WEIGHT
            + threat_confidence * THREAT_WEIGHT
            + campaign_confidence * CAMPAIGN_WEIGHT
            + prediction_confidence * PREDICTION_WEIGHT
        )

        score = round(score, 2)

        if score >= HIGH_THRESHOLD:
            level = "HIGH"

        elif score >= MEDIUM_THRESHOLD:
            level = "MEDIUM"

        else:
            level = "LOW"

        publish = score >= PUBLISH_THRESHOLD

        breakdown = {

            "Detection Confidence": {
                "value": detection_confidence,
                "weight": DETECTION_WEIGHT,
                "contribution": round(
                    detection_confidence * DETECTION_WEIGHT,
                    2,
                ),
            },

            "Dynamic Risk": {
                "value": risk_score,
                "weight": RISK_WEIGHT,
                "contribution": round(
                    risk_score * RISK_WEIGHT,
                    2,
                ),
            },

            "Threat Intelligence": {
                "value": threat_confidence,
                "weight": THREAT_WEIGHT,
                "contribution": round(
                    threat_confidence * THREAT_WEIGHT,
                    2,
                ),
            },

            "Campaign Intelligence": {
                "value": campaign_confidence,
                "weight": CAMPAIGN_WEIGHT,
                "contribution": round(
                    campaign_confidence * CAMPAIGN_WEIGHT,
                    2,
                ),
            },

            "Prediction Engine": {
                "value": prediction_confidence,
                "weight": PREDICTION_WEIGHT,
                "contribution": round(
                    prediction_confidence * PREDICTION_WEIGHT,
                    2,
                ),
            },
        }

        return CTIConfidence(
            score=score,
            level=level,
            publish=publish,
            breakdown=breakdown,
        )


engine = CTIConfidenceEngine()