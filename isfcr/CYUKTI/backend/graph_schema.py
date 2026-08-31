CAMPAIGN_DEFAULTS = {
    "campaign_count": 1,
    "prediction_frequency": 0,
    "duplicate_frequency": 0,
    "risk_score": 0.0,
    "risk_confidence": 0.0,
    "risk_trend": "NEW",
    "prediction_generated_at": ""
}

ATTACKER_DEFAULTS = {
    "vt_reputation": 0.0,
    "threat_actor_reputation": 0.0,
    "malware_confidence": 0.0,
    "tool_confidence": 0.0,
    "misp_confidence": 0.0,
    "ioc_confidence": 0.0,
}

ATTACK_EVENT_DEFAULTS = {
    "rule_level": 0,
    "suricata_score": 0.0,
    "zeek_score": 0.0,
    "sigma_score": 0.0,
    "yara_score": 0.0,
}

TECHNIQUE_DEFAULTS = {
    "prediction_count": 0,
    "incoming_degree": 0,
    "outgoing_degree": 0,
}

TRANSITION_DEFAULTS = {
    "count": 1,
    "confidence": 1.0,
    "created_at": None,
    "last_seen": None,
}