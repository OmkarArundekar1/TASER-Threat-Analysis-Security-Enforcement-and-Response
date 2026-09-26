"""
ground_truth/builder.py
==========================
GroundTruthBuilder: turns a raw Wazuh alert (or a real Neo4j
AttackEvent/Campaign, read but never re-derived through CYUKTI's own
resolution logic) plus an AttackScenario into an AUTO_PROPOSED
GroundTruthRecord.

CRITICAL: this module never imports mitre_resolver, campaign_manager,
threat_attribution_engine, threat_qualification, or prediction_engine.
The only "system" module it may import is neo4j_client, and only to
READ raw fields (rule_id, agent, timestamps, attacker/victim IPs) that
exist independently of any resolution CYUKTI performed on them.
"""

from __future__ import annotations

from ground_truth.schema import GroundTruthRecord, ReviewStatus
from scenarios.registry import AttackScenario

_LABELING_METHOD_ALERT = (
    "raw Wazuh alert rule.description text cross-referenced against the scenario's "
    "pre-declared expected_techniques (themselves derived from MITRE ATT&CK's own technique "
    "definitions applied to the rule text) -- independent of mitre_resolver.py"
)
_LABELING_METHOD_SESSION = (
    "attacker/victim IP pair + attack_family taken directly from the pre-declared AttackScenario "
    "-- independent of campaign_manager.py's own clustering output"
)


def build_from_raw_alert(
    scenario: AttackScenario,
    alert: dict,
    sample_id: str,
) -> GroundTruthRecord:
    """One ground-truth record per raw alert that plausibly belongs to
    `scenario` (caller is responsible for matching -- see
    matching.match_alert_to_scenario). Only MITRE + threat-status labels
    are populated here; campaign/attribution ground truth for the same
    alert comes from build_session_boundary(), not from this alert-level
    record, to keep the two evaluation axes independent."""
    rule = alert.get("rule") or {}
    rule_id = str(rule.get("id", ""))
    agent = alert.get("agent") or {}

    record = GroundTruthRecord(
        sample_id=sample_id,
        source="wazuh_alert",
        timestamp=alert.get("timestamp", ""),
        scenario_id=scenario.scenario_id,
        raw_event_id=f"rule={rule_id}|agent={agent.get('id', '?')}",
        attacker_identity=scenario.attacker,
        victim_identity=scenario.victim,
        expected_attack=scenario.scenario_name,
        expected_mitre_techniques=list(scenario.expected_techniques),
        expected_threat_status=scenario.expected_threat_status,
        expected_attribution=scenario.expected_attribution,
        reviewer="unreviewed",
        review_status=ReviewStatus.AUTO_PROPOSED,
        evidence_reference=f"raw alert rule.description={rule.get('description', '')!r} rule.id={rule_id}; scenario evidence: {scenario.evidence_sources}",
        labeling_method=_LABELING_METHOD_ALERT,
        dataset_version="v0-unlocked",
    )
    record.validate()
    return record


def build_session_boundary(
    scenario: AttackScenario,
    sample_id: str,
    involved_raw_event_ids: list[str],
) -> GroundTruthRecord:
    """One record representing an independent campaign/session
    boundary: 'these raw events all belong to one real attack session,
    by definition of the scenario' -- expected_campaign_id is an
    INDEPENDENT grouping key built from the scenario itself, never
    CYUKTI's own Campaign.campaign_id."""
    independent_campaign_key = f"SESSION::{scenario.scenario_id}::{scenario.attacker}->{scenario.victim}"
    record = GroundTruthRecord(
        sample_id=sample_id,
        source="scenario_session_boundary",
        timestamp="",
        scenario_id=scenario.scenario_id,
        raw_event_id="|".join(involved_raw_event_ids),
        attacker_identity=scenario.attacker,
        victim_identity=scenario.victim,
        expected_attack=scenario.scenario_name,
        expected_mitre_techniques=[],  # deliberately empty -- see scenarios.json notes
        expected_campaign_id=independent_campaign_key,
        expected_threat_status=scenario.expected_threat_status,
        expected_attribution=scenario.expected_attribution,
        reviewer="unreviewed",
        review_status=ReviewStatus.AUTO_PROPOSED,
        evidence_reference=f"scenario evidence: {scenario.evidence_sources}",
        labeling_method=_LABELING_METHOD_SESSION,
        dataset_version="v0-unlocked",
    )
    record.validate()
    return record
