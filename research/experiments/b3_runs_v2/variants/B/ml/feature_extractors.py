from feature_schema import CampaignDatasetRecord
from campaign_feature_engine import engine as campaign_engine
from feature_orchestrator import engine as orchestrator

class DatasetFeatureExtractor:
    def extract(
        self,
        campaign_context,
        current_attack_id,
        attacker_ip,
        event_id,
        severity,
        attributed_actor,
        prediction_correct,
        attribution_correct,
        next_technique
    ):
        features = orchestrator.extract_features(
            attack_id=current_attack_id,
            campaign_id=campaign_context.campaign_id,
            attacker_ip=attacker_ip,
            event_id=event_id
        )
        campaign = campaign_engine.extract(
            campaign_context,
            current_attack_id,
            attacker_ip
        )
        graph = features.graph
        runtime = features.runtime
        mitre = features.mitre
        threat = features.threat_intel
        detection = features.detection

        return CampaignDatasetRecord(
            campaign_id=campaign_context.campaign_id,
            attacker_ip=campaign_context.attacker_ip,
            victim_ip=campaign_context.victim_ip,
            campaign_size=graph.campaign_size,
            unique_techniques=graph.unique_techniques,
            campaign_duration=graph.campaign_duration,
            prediction_similarity=campaign.prediction_similarity or 0.0,
            chain_similarity=campaign.chain_similarity or 0.0,
            temporal_similarity=campaign.temporal_similarity or 0.0,
            attacker_similarity=campaign.attacker_similarity or 0.0,
            duplicate_similarity=campaign.duplicate_similarity or 0.0,
            graph_similarity=campaign.graph_similarity or 0.0,
            runtime_similarity=campaign.runtime_similarity or 0.0,
            node_count=graph.node_count,
            edge_count=graph.edge_count,
            graph_density=graph.graph_density,
            graph_connectivity=graph.graph_connectivity,
            average_degree=graph.average_degree,
            attacker_degree=graph.attacker_degree,
            victim_degree=graph.victim_degree,
            technique_degree=graph.technique_degree,
            attack_chain_depth=graph.attack_chain_depth,
            average_path_length=graph.average_path_length,
            graph_diameter=graph.graph_diameter,
            branching_factor=graph.branching_factor,
            average_clustering=graph.average_clustering,
            average_betweenness=graph.average_betweenness,
            average_closeness=graph.average_closeness,
            community_count=graph.community_count,
            largest_community=graph.largest_community,
            campaign_complexity=graph.campaign_complexity,
            structural_risk=graph.structural_risk,
            evolution_rate=graph.evolution_rate,
            campaign_count=runtime.campaign_count,
            attack_event_count=runtime.attack_event_count,
            graph_degree=runtime.graph_degree,
            incoming_chain_count=runtime.incoming_chain_count,
            outgoing_chain_count=runtime.outgoing_chain_count,
            prediction_frequency=runtime.prediction_frequency,
            duplicate_frequency=runtime.duplicate_frequency,
            platform_count=mitre.platform_count,
            domain_count=mitre.domain_count,
            kill_chain_count=mitre.kill_chain_count,
            threat_actor_count=mitre.threat_actor_count,
            malware_count=mitre.malware_count,
            tool_count=mitre.tool_count,
            mitigation_count=mitre.mitigation_count,
            subtechnique_count=mitre.subtechnique_count,
            ip_reputation=threat.ip_reputation,
            threat_actor_reputation=threat.threat_actor_reputation,
            malware_confidence=threat.malware_confidence,
            tool_confidence=threat.tool_confidence,
            misp_confidence=threat.misp_confidence,
            ioc_confidence=threat.ioc_confidence,
            wazuh_level=detection.wazuh_level,
            suricata_score=detection.suricata_score,
            zeek_score=detection.zeek_score,
            sigma_score=detection.sigma_score,
            yara_score=detection.yara_score,
            detection_confidence=detection.detection_confidence,
            risk_score=campaign_context.risk_score,
            severity=severity,
            attributed_actor=attributed_actor,
            prediction_correct=prediction_correct,
            attribution_correct=attribution_correct,
            next_technique=next_technique
        )

extractor = DatasetFeatureExtractor()