"""
Regression test for a separate bug found while auditing mitre_mapper.py
consumers: campaign_manager.py had three call sites doing
TPS_MAP.get(technique_id, 0) directly — but TPS_MAP is keyed by stage
name, not technique ID, so context.risk_score in append_technique(),
load_from_database(), and the campaign-reopening search always
silently added/summed zero for every real technique. This is a
different code path from the persisted Neo4j risk_score
(neo4j_client.create_attack_event, driven by realtime_socgraph.py's
correct stage lookup) — this one only affects campaign_manager's
in-memory decisions (reopening, prediction).
"""

from campaign_manager import _tps_for_technique
from config import TPS_MAP


def test_tps_for_technique_routes_through_stage_mapping():
    # T1595 -> Reconnaissance -> TPS_MAP["Reconnaissance"] = 10
    assert _tps_for_technique("T1595") == TPS_MAP["Reconnaissance"]
    assert _tps_for_technique("T1595") != 0


def test_tps_for_technique_unmapped_technique_is_zero():
    assert _tps_for_technique("T9999.999") == 0


def test_tps_for_technique_matches_real_observed_values():
    # from the real dataset audit: T1110.001 (Credential Access, weight 45)
    assert _tps_for_technique("T1110.001") == 45
    assert _tps_for_technique("T1078") == 70  # Initial Access
