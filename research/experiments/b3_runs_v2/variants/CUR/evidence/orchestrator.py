"""
evidence/orchestrator.py
==========================
Collects Evidence for a campaign from every wired source and returns a
populated EvidenceStore. Mirrors feature_orchestrator.py's structure:
it calls each existing engine, then hands the raw result to the matching
collector to be wrapped as Evidence.

Every dependency is injected through the module-level `engine`/`context`
objects imported below (not hidden inside methods), so tests can
monkeypatch them without a live Neo4j/MISP connection.
"""

from __future__ import annotations

import attribution_context as attribution_context_module
import detection_confidence_engine
import graph_feature_engine
import mitre_feature_engine
import threat_attribution_engine
import threat_intelligence_engine
from evidence.collectors.attribution_collector import collect_attribution_evidence
from evidence.collectors.campaign_history_collector import collect_campaign_history_evidence
from evidence.collectors.cti_collector import collect_cti_evidence
from evidence.collectors.detection_collector import collect_detection_evidence
from evidence.collectors.graph_collector import collect_graph_evidence
from evidence.collectors.mitre_collector import collect_mitre_evidence
from evidence.store import EvidenceStore


class EvidenceOrchestrator:
    def collect_for_campaign(
        self,
        campaign_context,
        current_attack_id: str,
        event_id: str | None = None,
    ) -> EvidenceStore:
        store = EvidenceStore()

        mitre_features = mitre_feature_engine.engine.extract_features(current_attack_id)
        store.add_many(collect_mitre_evidence(mitre_features))

        threat_intel = threat_intelligence_engine.engine.calculate(campaign_context.attacker_ip)
        store.add_many(collect_cti_evidence(campaign_context.attacker_ip, threat_intel))

        if event_id:
            detection_result = detection_confidence_engine.engine.calculate(event_id)
            store.add_many(collect_detection_evidence(event_id, detection_result))

        attribution_result = threat_attribution_engine.engine.attribute(campaign_context)
        store.add_many(collect_attribution_evidence(attribution_result))

        historical_campaigns = attribution_context_module.context.load_historical_campaigns()
        store.add_many(
            collect_campaign_history_evidence(campaign_context.techniques, historical_campaigns)
        )

        graph_features = graph_feature_engine.graph_analytics.extract_features(
            campaign_context.campaign_id, force_reload=True
        )
        store.add_many(collect_graph_evidence(campaign_context.campaign_id, graph_features))

        return store


orchestrator = EvidenceOrchestrator()
