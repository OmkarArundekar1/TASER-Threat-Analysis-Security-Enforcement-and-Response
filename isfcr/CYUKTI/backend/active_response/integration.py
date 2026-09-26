"""
active_response/integration.py
=================================
The single real integration point between CYUKTI's existing
investigation/threat-qualification output and the policy/decision/plan
layer built in Phase X. This module is the ONLY place that maps a real
`ThreatQualificationResult` + a real `CampaignContext` (+ optionally a
real investigation `ConfidenceEstimate`) into a `PolicyInput` --
nothing upstream of this (threat_qualification.py, investigation/*.py,
campaign_manager.py) is modified. This is strictly additive glue.

Flow (matches Phase Z Section 4 exactly):
    Investigation -> Threat Qualification -> ResponsePolicyEngine
    -> ResponseDecision -> ResponsePlan
"""

from __future__ import annotations

from campaign_context import CampaignContext
from risk_scoring import severity_from_tps
from threat_qualification import ThreatQualificationResult

from active_response.containment_actions import ContainmentAction
from active_response.correlation import derive_correlation_id
from active_response.decision import ResponseDecision
from active_response.policy import PolicyInput, ResponsePolicyEngine

_policy_engine = ResponsePolicyEngine()

# Evidence-coverage gate reused from investigation/confidence.py's own
# MIN_COVERAGE_FOR_MODEL_TRUST concept -- NOT re-derived from scratch,
# just referenced here so "evidence_sufficient" means the same thing
# investigation/confidence.py already means by it. Kept as a local
# constant (rather than importing the investigation module's internal
# name) so this integration layer has no import-time dependency on
# investigation/*.py internals changing shape.
MIN_EVIDENCE_COVERAGE_FOR_RESPONSE = 0.15


def decide_for_campaign(
    campaign: CampaignContext,
    qualification: ThreatQualificationResult | None,
    investigation_confidence: float | None = None,
    evidence_coverage: float | None = None,
    requested_action: ContainmentAction | None = ContainmentAction.BLOCK_SOURCE_IP,
    auto_contain_config: bool = False,
    never_block_ips: frozenset[str] = frozenset(),
    environment: str = "production",
    operation_id: str | None = None,
    investigation_id: str | None = None,
    incident_id: str | None = None,
    attack_event_id: str | None = None,
) -> ResponseDecision:
    """The real integration call site. `qualification` is None exactly
    when threat_qualification.py hasn't computed anything yet for this
    campaign (an honest, common state, per ResponsePlanGenerator's own
    existing handling) -- this maps to threat_class=None, which
    ResponsePolicyEngine already treats as OBSERVE, never CONTAIN."""
    correlation_id = derive_correlation_id(campaign_id=campaign.campaign_id)

    threat_class = qualification.classification if qualification is not None else None
    evidence_sufficient = (
        evidence_coverage is not None and evidence_coverage >= MIN_EVIDENCE_COVERAGE_FOR_RESPONSE
    )
    severity = severity_from_tps(campaign.risk_score)

    policy_input = PolicyInput(
        correlation_id=correlation_id,
        threat_class=threat_class,
        investigation_confidence=investigation_confidence,
        evidence_sufficient=evidence_sufficient,
        attack_progression=campaign.last_technique,
        attacker_ip=campaign.attacker_ip,
        victim_ip=campaign.victim_ip,
        severity=severity,
        requested_action=requested_action,
        environment=environment,
        auto_contain_config=auto_contain_config,
        never_block_ips=never_block_ips,
        incident_id=incident_id,
        attack_event_id=attack_event_id,
        campaign_id=campaign.campaign_id,
        operation_id=operation_id,
        investigation_id=investigation_id,
        risk_level=severity,
    )
    return _policy_engine.decide(policy_input)
