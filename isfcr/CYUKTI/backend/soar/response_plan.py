"""
soar/response_plan.py
========================
ResponsePlanGenerator: composes the analyst-facing "threat detected,
here is what should happen next" narrative (SHUFFLE_RESPONSE_MODEL.md)
from CYUKTI's already-real pieces -- never generates a new opinion of
its own, only narrates what the other engines already computed:

- threat_qualification.ThreatQualificationResult -- why this is/isn't a threat
- campaign_selection.SelectionResult -- which historical campaign matched, and why
- soar.schema.Playbook -- what should happen next (soar.generator.PlaybookGenerator)

Any of the three may be None (e.g. no CTI confidence computed yet for
this campaign, or no historical candidates found) -- the plan degrades
honestly rather than fabricating the missing section.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from campaign_context import CampaignContext
from campaign_selection import SelectionResult
from risk_scoring import severity_from_tps
from soar.schema import Playbook
from threat_qualification import ThreatQualificationResult


@dataclass
class ResponsePlan:
    campaign_id: str
    threat_summary: str
    why_threat: list[str]
    mitre_techniques: list[str]
    severity: str
    risk_score: float
    selected_historical_campaign: str | None
    why_selected: str | None
    selection_confidence: str | None
    playbook: Playbook | None
    historical_playbook_success_rate: float | None
    historical_playbook_executions: int
    misp_status: str  # "READY" | "BLOCKED" | "NOT_APPLICABLE"
    misp_reason: str
    # Added for the active-containment layer (backend/active_response/) --
    # all optional/defaulted so every pre-existing caller (test_soar_response_plan.py,
    # test_determinism.py, test_final_trace.py, test_failure_injection.py) is unaffected.
    correlation_id: str | None = None
    incident_id: str | None = None
    attack_event_id: str | None = None
    operation_id: str | None = None
    investigation_id: str | None = None
    evidence_references: list[str] = field(default_factory=list)
    selected_containment_action: str | None = None   # a ContainmentAction value, narrated only -- never decided here
    approval_required: bool | None = None
    priority: str | None = None                        # derived from severity, informational
    expected_outcome: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "campaign_id": self.campaign_id,
            "threat_summary": self.threat_summary,
            "why_threat": self.why_threat,
            "mitre_techniques": self.mitre_techniques,
            "severity": self.severity,
            "risk_score": self.risk_score,
            "selected_historical_campaign": self.selected_historical_campaign,
            "why_selected": self.why_selected,
            "selection_confidence": self.selection_confidence,
            "playbook": self.playbook.to_dict() if self.playbook else None,
            "historical_playbook_success_rate": self.historical_playbook_success_rate,
            "historical_playbook_executions": self.historical_playbook_executions,
            "misp_status": self.misp_status,
            "misp_reason": self.misp_reason,
            "correlation_id": self.correlation_id,
            "incident_id": self.incident_id,
            "attack_event_id": self.attack_event_id,
            "operation_id": self.operation_id,
            "investigation_id": self.investigation_id,
            "evidence_references": self.evidence_references,
            "selected_containment_action": self.selected_containment_action,
            "approval_required": self.approval_required,
            "priority": self.priority,
            "expected_outcome": self.expected_outcome,
        }


class ResponsePlanGenerator:
    def generate(
        self,
        campaign: CampaignContext,
        playbook: Playbook | None = None,
        qualification: ThreatQualificationResult | None = None,
        selection: SelectionResult | None = None,
        correlation_id: str | None = None,
        incident_id: str | None = None,
        attack_event_id: str | None = None,
        operation_id: str | None = None,
        investigation_id: str | None = None,
        evidence_references: list[str] | None = None,
        response_decision: Any | None = None,  # an active_response.decision.ResponseDecision, narrated only
    ) -> ResponsePlan:
        severity = severity_from_tps(campaign.risk_score)
        techniques = sorted(campaign.techniques) if campaign.techniques else (
            [campaign.last_technique] if campaign.last_technique else []
        )

        threat_summary = (
            f"Campaign {campaign.campaign_id}: attacker {campaign.attacker_ip} -> "
            f"victim {campaign.victim_ip}, technique(s) {', '.join(techniques) or 'none resolved'}, "
            f"severity {severity}."
        )

        why_threat: list[str] = []
        if qualification is not None:
            why_threat.append(
                f"Threat classification: {qualification.classification} "
                f"(CTI confidence score {qualification.cti_score})."
            )
            for check in qualification.checks:
                if check.passed:
                    why_threat.append(f"✓ {check.detail}")
        else:
            why_threat.append(
                "No CTI confidence has been computed for this campaign yet -- "
                "threat classification is not available."
            )

        selected_campaign_id = None
        why_selected = None
        selection_confidence = None
        historical_success_rate = None
        historical_executions = 0
        if selection is not None and selection.selected is not None:
            selected_campaign_id = selection.selected.campaign_id
            why_selected = selection.explanation
            selection_confidence = selection.confidence
            historical_success_rate = selection.selected.historical_playbook_success_rate
            historical_executions = selection.selected.historical_playbook_executions

        if qualification is None:
            misp_status, misp_reason = "NOT_APPLICABLE", "No CTI confidence computed for this campaign yet."
        elif qualification.may_publish_to_misp:
            misp_status, misp_reason = "READY", qualification.reason
        else:
            misp_status, misp_reason = "BLOCKED", qualification.reason

        priority = {"CRITICAL": "P1", "HIGH": "P2", "MEDIUM": "P3", "LOW": "P4"}.get(severity, "P4")

        selected_containment_action = None
        approval_required = None
        expected_outcome = None
        if response_decision is not None:
            selected_containment_action = (
                response_decision.selected_action.value if response_decision.selected_action else None
            )
            approval_required = response_decision.approval_required
            expected_outcome = (
                f"Policy result {response_decision.policy_result.value}: {response_decision.policy_reason}"
            )

        return ResponsePlan(
            campaign_id=campaign.campaign_id,
            threat_summary=threat_summary,
            why_threat=why_threat,
            mitre_techniques=techniques,
            severity=severity,
            risk_score=campaign.risk_score,
            selected_historical_campaign=selected_campaign_id,
            why_selected=why_selected,
            selection_confidence=selection_confidence,
            playbook=playbook,
            historical_playbook_success_rate=historical_success_rate,
            historical_playbook_executions=historical_executions,
            misp_status=misp_status,
            misp_reason=misp_reason,
            correlation_id=correlation_id,
            incident_id=incident_id,
            attack_event_id=attack_event_id,
            operation_id=operation_id,
            investigation_id=investigation_id,
            evidence_references=evidence_references or [],
            selected_containment_action=selected_containment_action,
            approval_required=approval_required,
            priority=priority,
            expected_outcome=expected_outcome,
        )


response_plan_generator = ResponsePlanGenerator()
