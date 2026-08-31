from dataclasses import dataclass
from neo4j_client import (
    MitreFeatures,
    RuntimeGraphFeatures,
    ThreatIntelFeatures,
    DetectionConfidence
)
from graph_feature_engine import (
    graph_analytics,
    GraphFeatures
)
from mitre_feature_engine import engine as mitre_engine
from runtime_graph_feature_engine import engine as runtime_engine
from threat_intelligence_engine import engine as threat_engine
from detection_confidence_engine import engine as detection_engine

@dataclass
class FeatureVector:
    mitre: MitreFeatures
    runtime: RuntimeGraphFeatures
    graph: GraphFeatures
    threat_intel: ThreatIntelFeatures
    detection: DetectionConfidence

class FeatureOrchestrator:
    def extract_features(
        self,
        attack_id,
        campaign_id,
        attacker_ip,
        event_id
    ):
        mitre = mitre_engine.extract_features(
            attack_id
        )
        runtime = runtime_engine.extract_features(
            campaign_id
        )
        graph_analytics.invalidate_campaign(campaign_id)
        
        graph_features = graph_analytics.extract_features(
            campaign_id,
            force_reload=True
        )
        threat = threat_engine.extract_features(
            attacker_ip
        )
        detection = detection_engine.extract_features(
            event_id
        )
        return FeatureVector(
            mitre=mitre,
            runtime=runtime,
            graph=graph_features,
            threat_intel=threat,
            detection=detection
        )
engine = FeatureOrchestrator()