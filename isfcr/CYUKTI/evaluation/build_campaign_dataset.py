"""
build_campaign_dataset.py
============================
Builds AUTO_PROPOSED ground truth for threat-qualification, campaign
correlation, and attribution FROM REAL, LIVE Neo4j Campaign nodes --
using only raw, directly-observed fields (attacker_ip, victim_ip,
last_technique used solely to LOCATE the right campaign, never copied
into an expected_* MITRE field) plus each scenario's independently
pre-declared expected_threat_status/expected_attribution.

`campaign.cti_score` and `campaign.campaign_id` ARE read here, but only
as the SYSTEM PREDICTION side (computed at evaluation time in
*_eval.py, never stored inside a GroundTruthRecord's expected_* field).
This script only ever writes campaign_id into `raw_event_id` (a pointer
for the evaluator to re-fetch live state at eval time), never into
`expected_campaign_id` (which always gets an independent SESSION::
key, per ground_truth/builder.py::build_session_boundary).
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
BACKEND_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "backend")
sys.path.insert(0, BACKEND_DIR)

from ground_truth import store
from ground_truth.builder import build_session_boundary
from ground_truth.schema import GroundTruthRecord, ReviewStatus
from scenarios.registry import get_scenario


def _find_real_campaigns(driver):
    """Real Cypher lookups, keyed only on raw/independent fields
    (attacker_ip, victim_ip) plus last_technique used only for
    LOCATING the right real campaign for a known scenario -- not
    copied into any ground-truth MITRE label."""
    queries = {
        "SCN-SESSION-BOUNDARY-002": (  # CAMP_D8605E81: Kali(.106) -> pes1ug23cs411, brute force
            "MATCH (c:Campaign {attacker_ip:'192.168.56.106', victim_ip:'pes1ug23cs411-VirtualBox'}) "
            "RETURN c.campaign_id AS id, c.cti_score AS cti_score, c.risk_score AS risk_score"
        ),
        "SCN-SESSION-BOUNDARY-001": (  # CAMP_1429ADB4: .105 -> pes1ug23cs411, mixed exploitation
            "MATCH (c:Campaign {attacker_ip:'192.168.56.105', victim_ip:'pes1ug23cs411-VirtualBox'}) "
            "RETURN c.campaign_id AS id, c.cti_score AS cti_score, c.risk_score AS risk_score"
        ),
        "SCN-NMAP-001": (
            "MATCH (c:Campaign {attacker_ip:'192.168.56.106', victim_ip:'192.168.56.105', last_technique:'T1595'}) "
            "RETURN c.campaign_id AS id, c.cti_score AS cti_score, c.risk_score AS risk_score"
        ),
    }
    results = {}
    with driver.session() as s:
        for scenario_id, cypher in queries.items():
            results[scenario_id] = [dict(r) for r in s.run(cypher)]
    return results


def build():
    from neo4j_client import driver

    campaigns_by_scenario = _find_real_campaigns(driver)
    for sid, rows in campaigns_by_scenario.items():
        print(f"{sid}: {len(rows)} real campaign(s) found")

    # -------- threat qualification: expected_threat_status vs. real cti_score
    from cti_confidence_engine import PUBLISH_THRESHOLD, NOT_THREAT_THRESHOLD

    tq_records = []
    for sid, rows in campaigns_by_scenario.items():
        scenario = get_scenario(sid)
        for row in rows:
            if row["cti_score"] is None:
                continue  # honestly skip -- not yet scored, not a 0
            sample_id = f"{sid}-{row['id']}"
            tq_records.append(GroundTruthRecord(
                sample_id=sample_id,
                source="neo4j_campaign",
                timestamp="",
                scenario_id=sid,
                raw_event_id=row["id"],  # real Campaign.campaign_id, used only to re-fetch state at eval time
                attacker_identity=scenario.attacker,
                victim_identity=scenario.victim,
                expected_attack=scenario.scenario_name,
                expected_threat_status=scenario.expected_threat_status,
                reviewer="unreviewed",
                review_status=ReviewStatus.AUTO_PROPOSED,
                evidence_reference=f"real Campaign {row['id']}, cti_score={row['cti_score']}; scenario evidence: {scenario.evidence_sources}",
                labeling_method="scenario-declared expected_threat_status (independent analyst judgment) vs. "
                                 "live cti_score read from Neo4j at eval time, classified via "
                                 "cti_confidence_engine.py's own published thresholds "
                                 f"(PUBLISH_THRESHOLD={PUBLISH_THRESHOLD}, NOT_THREAT_THRESHOLD={NOT_THREAT_THRESHOLD})",
                dataset_version="v0-unlocked",
            ))
    if tq_records:
        for r in tq_records:
            r.validate()
        store.save_records("threat_qualification_v0", tq_records)
        print(f"Built {len(tq_records)} threat-qualification ground-truth records.")
    else:
        print("No scored real campaigns found for threat-qualification ground truth -- nothing built.")

    # -------- campaign correlation: independent session boundaries
    cc_records = []
    for sid in ("SCN-SESSION-BOUNDARY-001", "SCN-SESSION-BOUNDARY-002", "SCN-NMAP-001"):
        scenario = get_scenario(sid)
        rows = campaigns_by_scenario.get(sid, [])
        real_ids = [r["id"] for r in rows]
        if not real_ids:
            continue
        record = build_session_boundary(scenario, sample_id=f"CC-{sid}", involved_raw_event_ids=real_ids)
        cc_records.append(record)
    if cc_records:
        store.save_records("campaign_correlation_v0", cc_records)
        print(f"Built {len(cc_records)} campaign-correlation ground-truth records.")
    else:
        print("No real campaigns found for campaign-correlation ground truth -- nothing built.")

    # -------- attribution: known real attacker_ip vs. threat_attribution_engine's ranked candidates
    attr_records = []
    for sid in ("SCN-SESSION-BOUNDARY-001", "SCN-SESSION-BOUNDARY-002", "SCN-NMAP-001"):
        scenario = get_scenario(sid)
        rows = campaigns_by_scenario.get(sid, [])
        for row in rows:
            sample_id = f"ATTR-{sid}-{row['id']}"
            attr_records.append(GroundTruthRecord(
                sample_id=sample_id,
                source="neo4j_campaign",
                timestamp="",
                scenario_id=sid,
                raw_event_id=row["id"],
                attacker_identity=scenario.attacker,
                expected_attribution=scenario.attacker,  # the raw, directly-observed attacker_ip IS the ground truth here
                reviewer="unreviewed",
                review_status=ReviewStatus.AUTO_PROPOSED,
                evidence_reference=f"real Campaign {row['id']}; scenario evidence: {scenario.evidence_sources}",
                labeling_method="the campaign's real, directly-observed attacker_ip (a raw network fact, not an "
                                 "inference) used as the known identity -- compared at eval time against whether "
                                 "threat_attribution_engine's top-ranked similar-campaign candidate for this "
                                 "campaign shares the SAME real attacker_ip",
                dataset_version="v0-unlocked",
            ))
    if attr_records:
        for r in attr_records:
            r.validate()
        store.save_records("attribution_v0", attr_records)
        print(f"Built {len(attr_records)} attribution ground-truth records.")
    else:
        print("No real campaigns found for attribution ground truth -- nothing built.")


if __name__ == "__main__":
    build()
