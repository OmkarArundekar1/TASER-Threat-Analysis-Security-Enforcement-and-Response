from dataclasses import dataclass

@dataclass
class CampaignDatasetRecord:

    campaign_id: str
    attacker_ip: str
    victim_ip: str

    campaign_size: int
    unique_techniques: int
    campaign_duration: float

    prediction_similarity: float
    chain_similarity: float
    temporal_similarity: float
    attacker_similarity: float
    duplicate_similarity: float
    graph_similarity: float
    runtime_similarity: float
    node_count: int
    edge_count: int
    graph_density: float
    graph_connectivity: float
    average_degree: float
    attacker_degree: int
    victim_degree: int
    technique_degree: int
    attack_chain_depth: int
    average_path_length: float
    graph_diameter: int
    branching_factor: float
    average_clustering: float
    average_betweenness: float
    average_closeness: float
    community_count: int
    largest_community: int
    campaign_complexity: float
    structural_risk: float
    evolution_rate: float
    campaign_count: int
    attack_event_count: int
    graph_degree: int
    incoming_chain_count: int
    outgoing_chain_count: int
    prediction_frequency: int
    duplicate_frequency: int
    platform_count: int
    domain_count: int
    kill_chain_count: int
    threat_actor_count: int
    malware_count: int
    tool_count: int
    mitigation_count: int
    subtechnique_count: int
    ip_reputation: float
    threat_actor_reputation: float
    malware_confidence: float
    tool_confidence: float
    misp_confidence: float
    ioc_confidence: float
    wazuh_level: int
    suricata_score: float
    zeek_score: float
    sigma_score: float
    yara_score: float
    detection_confidence: float
    risk_score: float
    severity: str
    attributed_actor: str
    prediction_correct: int | None  # None = NOT_APPLICABLE (no observed transition or no learned prediction to check)
    attribution_correct: int
    next_technique: str