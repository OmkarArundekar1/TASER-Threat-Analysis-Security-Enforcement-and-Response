from dataclasses import dataclass
from datetime import datetime, timezone

@dataclass
class MISPEvent:
    info: str
    distribution: int
    threat_level_id: int
    analysis: int
    published: bool
    attributes: list
    tags: list

@dataclass
class IncidentContext:
    campaign_id: str
    operation_id: str
    attacker_ip: str
    victim_ip: str
    event_id: str
    technique: str
    stage: str
    prediction: str
    prediction_confidence: float
    detection: object
    threat: object
    dynamic_risk: object
    cti: object
    recommendations: list
    investigation_payload: str
    timestamp: str
    attribution: object = None

class MISPEventGenerator:
    def _risk_to_threat_level(self, risk):
        if risk >= 70:
            return 1
        elif risk >= 40:
            return 2
        return 3

    def generate(
        self,
        incident: IncidentContext
    ):
        attributes = [
            {
                "type": "ip-src",
                "category": "Network activity",
                "value": incident.attacker_ip,
            },
            {
                "type": "comment",
                "category": "Other",
                "value": f"Victim : {incident.victim_ip}",
            },
            {
                "type": "text",
                "category": "Other",
                "value": f"Campaign : {incident.campaign_id}",
            },
            {
                "type": "text",
                "category": "Other",
                "value": f"Operation : {incident.operation_id}",
            },
            {
                "type": "text",
                "category": "Other",
                "value": f"Technique : {incident.technique}",
            },
            {
                "type": "text",
                "category": "Other",
                "value": f"Predicted : {incident.prediction}",
            },
            {
                "type": "text",
                "category": "Other",
                "value": f"Prediction Confidence : {incident.prediction_confidence}",
            },
            {
                "type": "text",
                "category": "Other",
                "value": f"Dynamic Risk : {incident.dynamic_risk.risk_score}",
            },
            {
                "type": "text",
                "category": "Other",
                "value": f"CTI Confidence : {incident.cti.score}",
            },
            {
                "type": "text",
                "category": "Other",
                "value": f"Attack Event : {incident.event_id}",
            }]
        if incident.attribution and incident.attribution.actors:
        
            best = incident.attribution.actors[0]
        
            attributes.extend([
                {
                    "type": "text",
                    "category": "Other",
                    "value": f"Threat Actor : {best.actor}",
                },
                {
                    "type": "text",
                    "category": "Other",
                    "value": f"Attribution Confidence : {best.total_score:.2f}",
                },
                {
                    "type": "text",
                    "category": "Other",
                    "value": f"Coverage : {best.coverage:.2f}",
                },
                {
                    "type": "text",
                    "category": "Other",
                    "value": f"Precision : {best.precision:.2f}",
                },
                {
                    "type": "text",
                    "category": "Other",
                    "value": f"Chain Similarity : {best.chain_similarity:.2f}",
                },
            ])
        
            for evidence in best.evidence:
                attributes.append(
                    {
                        "type": "comment",
                        "category": "Other",
                        "value": evidence,
                    }
                )
        for recommendation in incident.recommendations:
            attributes.append(
                {
                    "type": "comment",
                    "category": "Other",
                    "value": recommendation,
                }
            )
        tags = [
            f"mitre:{incident.technique}",
            f"stage:{incident.stage}",
            f"campaign:{incident.campaign_id}",
            f"operation:{incident.operation_id}",
            f"prediction:{incident.prediction}",
            f"cti:{incident.cti.level}",
            f"risk:{incident.dynamic_risk.risk_level}",
        ]
        if incident.attribution and incident.attribution.actors:       
            best = incident.attribution.actors[0]
            tags.extend([
                f"actor:{best.actor}",
                f"attribution:{int(best.total_score)}",
            ])
        event = MISPEvent(
            info=f"CYUKTI Campaign {incident.campaign_id}",
            distribution=0,
            threat_level_id=self._risk_to_threat_level(
                incident.dynamic_risk.risk_score
            ),
            analysis=2,
            published=False,
            attributes=attributes,
            tags=tags,
        )
        summary = f"""
        ================ CYUKTI INCIDENT REPORT ================
        
        Campaign ID          : {incident.campaign_id}
        Operation ID         : {incident.operation_id}
        Attack Event         : {incident.event_id}
        
        Timestamp            : {incident.timestamp}
        
        Attacker             : {incident.attacker_ip}
        Victim               : {incident.victim_ip}
        
        Current Technique    : {incident.technique}
        Current Stage        : {incident.stage}
        
        Prediction           : {incident.prediction}
        Prediction Confidence: {incident.prediction_confidence:.2f}
        
        ========================================================
        """
        detection_section = f"""
        ================ DETECTION =================
        
        Detection Confidence : {incident.detection.confidence:.2f}
        Detection Level      : {incident.detection.level}
        
        Wazuh Score          : {incident.detection.breakdown["Wazuh"]}
        Suricata Score       : {incident.detection.breakdown["Suricata"]}
        Zeek Score           : {incident.detection.breakdown["Zeek"]}
        Sigma Score          : {incident.detection.breakdown["Sigma"]}
        YARA Score           : {incident.detection.breakdown["YARA"]}
        
        ===========================================
        """
        risk_section = f"""
        ================ DYNAMIC RISK =================
        
        Risk Score      : {incident.dynamic_risk.risk_score:.2f}
        Risk Level      : {incident.dynamic_risk.risk_level}
        
        Confidence      : {incident.dynamic_risk.confidence:.2f}
        
        ===============================================
        """
        cti_section = f"""
        ================ CTI =================
        
        CTI Score       : {incident.cti.score:.2f}
        
        CTI Level       : {incident.cti.level}
        
        Publish         : {incident.cti.publish}
        
        ======================================
        """
        attribution_section = ""
        if incident.attribution and incident.attribution.actors:
            best = incident.attribution.actors[0]
        
            attribution_section = f"""
        =============== THREAT ATTRIBUTION ===============
        
        Likely Actor      : {best.actor}
        Confidence        : {best.total_score:.2f}
        Coverage          : {best.coverage:.2f}
        Precision         : {best.precision:.2f}
        Chain Similarity  : {best.chain_similarity:.2f}
        
        Evidence:
        {chr(10).join("- " + e for e in best.evidence)}
        
        ==============================================
        """
        recommendation_text = "\n".join(
            f"- {r}" for r in incident.recommendations
        )
        incident_report = (
            summary
            + detection_section
            + risk_section
            + cti_section
            + attribution_section
            + "\nRecommendations\n"
            + recommendation_text
        )
        return {
            "Event": {
                "info": event.info,
                "distribution": event.distribution,
                "threat_level_id": event.threat_level_id,
                "analysis": event.analysis,
                "published": event.published,
                "date": datetime.now(
                    timezone.utc
                ).strftime("%Y-%m-%d"),
                "Attribute": event.attributes,
                "Tag": [
                    {
                        "name": tag
                    }
                    for tag in event.tags
                ]
            }
        }


engine = MISPEventGenerator()