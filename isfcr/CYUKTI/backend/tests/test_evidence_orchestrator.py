"""
Integration test for evidence/orchestrator.py: verifies the orchestrator
calls every wired engine and assembles their results into one
EvidenceStore, WITHOUT touching a live Neo4j/MISP connection — every
engine call is monkeypatched to return a synthetic, hand-built result.

This tests wiring/assembly correctness, not real detection results.
"""

import attribution_context as attribution_context_module
import detection_confidence_engine
import graph_feature_engine
import mitre_feature_engine
import threat_attribution_engine
import threat_intelligence_engine
from attribution_models import HistoricalCampaign
from campaign_context import CampaignContext
from detection_confidence_engine import DetectionResult
from evidence.orchestrator import EvidenceOrchestrator
from evidence.schema import EvidenceSource
from graph_feature_engine import GraphFeatures
from threat_actor_context import ThreatActorContext
from threat_attribution_engine import ThreatAttributionResult
from threat_intelligence_engine import ThreatIntelResult
from tests.test_evidence import make_mitre_features


def test_orchestrator_assembles_evidence_from_every_source(monkeypatch):
    campaign = CampaignContext(
        campaign_id="camp-1",
        attacker_ip="1.2.3.4",
        victim_ip="5.6.7.8",
    )
    campaign.techniques = {"T1110"}

    monkeypatch.setattr(
        mitre_feature_engine.engine, "extract_features",
        lambda attack_id: make_mitre_features(attack_id=attack_id),
    )
    monkeypatch.setattr(
        threat_intelligence_engine.engine, "calculate",
        lambda ip: ThreatIntelResult(confidence=70.0, level="MEDIUM", breakdown={}),
    )
    monkeypatch.setattr(
        detection_confidence_engine.engine, "calculate",
        lambda event_id: DetectionResult(confidence=88.0, level="HIGH", breakdown={}),
    )
    monkeypatch.setattr(
        threat_attribution_engine.engine, "attribute",
        lambda ctx: ThreatAttributionResult(actors=[
            ThreatActorContext(actor="camp-old", confidence=60.0, evidence=["matched"]),
        ]),
    )
    monkeypatch.setattr(
        attribution_context_module.context, "load_historical_campaigns",
        lambda: [HistoricalCampaign(campaign_id="camp-old", attacker="9.9.9.9",
                                     victim="8.8.8.8", techniques=["T1110"])],
    )

    def fake_graph_features(campaign_id, force_reload=True):
        gf = GraphFeatures()
        gf.node_count = 3
        gf.structural_risk = 10.0
        return gf

    monkeypatch.setattr(graph_feature_engine.graph_analytics, "extract_features", fake_graph_features)

    store = EvidenceOrchestrator().collect_for_campaign(
        campaign, current_attack_id="T1110", event_id="evt-1"
    )

    sources_present = {e.source for e in store.all()}
    assert sources_present == {
        EvidenceSource.MITRE,
        EvidenceSource.CTI,
        EvidenceSource.SIEM,
        EvidenceSource.ATTRIBUTION,
        EvidenceSource.CAMPAIGN_HISTORY,
        EvidenceSource.GRAPH,
    }
    assert len(store) == 6
    assert store.weighted_confidence() > 0.0
