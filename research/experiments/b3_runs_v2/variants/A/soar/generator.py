"""
soar/generator.py
====================
PlaybookGenerator: turns a real, already-resolved CampaignContext (plus
whatever attribution/prediction/evidence CYUKTI already computed for it)
into a structured Playbook. Every action carries a `reason` string
tracing back to a real signal (a MITRE mitigation record, a field
actually present on the campaign, a risk threshold) -- this generator
never invents evidence it doesn't have.

Deliberately NOT built on response/playbook_generator.py (Generation-1,
proven dead in GENERATION1_DISPOSITION.md -- keyed on synthetic
severity_label/attack_label strings that don't exist anywhere in
CYUKTI's real pipeline, and its sibling Mitigator executes raw shell
commands directly, which is exactly the un-auditable, un-approved
execution model this SOAR layer replaces with Shuffle + an approval
gate).
"""

from __future__ import annotations

from typing import Any

from campaign_context import CampaignContext
from mitre_mapper import MITRE_TO_STAGE
from recommendation_engine import get_recommendations
from risk_scoring import severity_from_tps

from soar.schema import ExecutionPolicy, Playbook, PlaybookAction


class PlaybookGenerator:
    def generate(
        self,
        campaign: CampaignContext,
        attribution: Any | None = None,
        prediction_confidence: float | None = None,
    ) -> Playbook:
        severity = severity_from_tps(campaign.risk_score)
        techniques = sorted(campaign.techniques) if campaign.techniques else (
            [campaign.last_technique] if campaign.last_technique else []
        )
        campaign_type = MITRE_TO_STAGE.get(campaign.last_technique, "Unclassified") if campaign.last_technique else "Unclassified"
        is_high_severity = severity in ("HIGH", "CRITICAL")

        actions: list[PlaybookAction] = []
        order = 1

        actions.append(PlaybookAction(
            action_type="enrich_ip", name="Enrich attacker IP",
            description=f"Look up reputation/context for {campaign.attacker_ip}",
            order=order, inputs={"ip": campaign.attacker_ip},
            expected_output="ip_reputation_report",
            reason="attacker_ip is a real, resolved field on this campaign's CampaignContext.",
        ))
        order += 1

        actions.append(PlaybookAction(
            action_type="threat_intel_lookup", name="Query threat intelligence",
            description=f"Query CTI/MISP for known indicators matching {campaign.attacker_ip}",
            order=order, inputs={"ip": campaign.attacker_ip, "campaign_id": campaign.campaign_id},
            expected_output="cti_confidence_report",
            reason="cti_confidence_engine.py already computes a blended CTI confidence for this campaign.",
        ))
        order += 1

        actions.append(PlaybookAction(
            action_type="historical_campaign_search", name="Check historical campaign matches",
            description="Search prior campaigns for technique/topology similarity",
            order=order, inputs={"campaign_id": campaign.campaign_id},
            expected_output="historical_campaign_matches",
            reason="rag.campaign_retriever (TF-IDF) and, when GNN_ENABLED, "
                   "rag.gnn_topology_retriever both provide real historical-campaign context for this campaign.",
        ))
        order += 1

        if campaign.victim_ip:
            actions.append(PlaybookAction(
                action_type="collect_evidence", name="Collect victim host evidence",
                description=f"Gather investigation evidence for victim host {campaign.victim_ip}",
                order=order, inputs={"ip": campaign.victim_ip, "campaign_id": campaign.campaign_id},
                expected_output="evidence_bundle",
                reason="victim_ip is a real, resolved field on this campaign's CampaignContext.",
            ))
            order += 1

        for technique in techniques:
            if not technique:
                continue
            try:
                mitigations = get_recommendations(technique)
            except Exception:
                mitigations = []
            for m in mitigations[:2]:  # cap per-technique noise; real, not arbitrary padding
                actions.append(PlaybookAction(
                    action_type="apply_mitigation", name=m.get("recommendation", "Apply mitigation"),
                    description=m.get("reason") or f"MITRE-recommended mitigation for {technique}",
                    order=order,
                    inputs={"technique": technique, "mitigation_id": m.get("mitre_mitigation")},
                    expected_output="mitigation_applied",
                    requires_approval=True,
                    reason=f"MITRE ATT&CK mitigation {m.get('mitre_mitigation')} "
                           f"for technique {technique} (recommendation_engine.py, live Neo4j CourseOfAction match).",
                ))
                order += 1

        if is_high_severity and campaign.attacker_ip:
            actions.append(PlaybookAction(
                action_type="block_ip", name="Block attacker IP",
                description=f"Block {campaign.attacker_ip} at the perimeter firewall",
                order=order, inputs={"ip": campaign.attacker_ip},
                expected_output="block_confirmation",
                destructive=True, requires_approval=True,
                reason=f"Campaign severity is {severity} (risk_score={campaign.risk_score}) "
                       "-- perimeter blocking is only proposed at HIGH/CRITICAL severity.",
            ))
            order += 1

        if is_high_severity and campaign.victim_ip:
            actions.append(PlaybookAction(
                action_type="isolate_host", name="Isolate victim host",
                description=f"Isolate {campaign.victim_ip} from the network pending investigation",
                order=order, inputs={"ip": campaign.victim_ip},
                expected_output="isolation_confirmation",
                destructive=True, requires_approval=True,
                reason=f"Campaign severity is {severity} (risk_score={campaign.risk_score}) "
                       "-- host isolation is only proposed at HIGH/CRITICAL severity.",
            ))
            order += 1

        actions.append(PlaybookAction(
            action_type="create_incident", name="Create incident ticket",
            description=f"Open a tracked incident for campaign {campaign.campaign_id}",
            order=order, inputs={"campaign_id": campaign.campaign_id, "severity": severity},
            expected_output="ticket_id",
            reason="A resolved campaign with real evidence is a real, ticket-worthy incident.",
        ))
        order += 1

        actions.append(PlaybookAction(
            action_type="notify_soc", name="Notify SOC",
            description=f"Notify the SOC channel about campaign {campaign.campaign_id} ({severity})",
            order=order, inputs={"campaign_id": campaign.campaign_id, "severity": severity},
            expected_output="notification_sent",
            reason="Every generated playbook ends with a SOC notification so a human always sees it.",
        ))

        required_evidence = ["campaign_context"]
        if attribution is not None:
            required_evidence.append("attribution")
        if prediction_confidence is not None:
            required_evidence.append("prediction")

        policy = ExecutionPolicy.ANALYST_APPROVAL if any(a.requires_approval for a in actions) else ExecutionPolicy.RECOMMEND_ONLY

        return Playbook(
            name=f"{campaign_type.upper().replace(' ', '_')}_RESPONSE",
            description=f"Generated response playbook for campaign {campaign.campaign_id} "
                        f"(stage={campaign_type}, severity={severity}).",
            trigger_conditions={"campaign_type": campaign_type, "min_severity": severity},
            campaign_type=campaign_type,
            mitre_techniques=techniques,
            severity=severity,
            risk=campaign.risk_score,
            required_evidence=required_evidence,
            actions=actions,
            execution_policy=policy,
            source_campaign_id=campaign.campaign_id,
        )
