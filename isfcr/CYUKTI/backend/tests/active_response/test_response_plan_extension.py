from campaign_context import CampaignContext
from soar.response_plan import ResponsePlanGenerator

from active_response.containment_actions import ContainmentAction
from active_response.decision import PolicyOutcome
from active_response.policy import PolicyInput, ResponsePolicyEngine


def _campaign():
    return CampaignContext(campaign_id="CAMP_X", attacker_ip="10.0.0.5", victim_ip="10.0.0.10",
                            risk_score=5000, last_technique="T1110", techniques={"T1110"})


def test_response_plan_carries_correlation_and_incident_ids():
    plan = ResponsePlanGenerator().generate(
        _campaign(), correlation_id="corr-1", incident_id="INC-1",
        attack_event_id="AE-1", operation_id="OP-1", investigation_id="INV-1",
        evidence_references=["evidence://alert/100510"],
    )
    d = plan.to_dict()
    assert d["correlation_id"] == "corr-1"
    assert d["incident_id"] == "INC-1"
    assert d["attack_event_id"] == "AE-1"
    assert d["operation_id"] == "OP-1"
    assert d["investigation_id"] == "INV-1"
    assert d["evidence_references"] == ["evidence://alert/100510"]


def test_response_plan_without_new_fields_defaults_safely():
    plan = ResponsePlanGenerator().generate(_campaign())
    d = plan.to_dict()
    assert d["correlation_id"] is None
    assert d["evidence_references"] == []
    assert d["selected_containment_action"] is None


def test_response_plan_narrates_a_real_policy_decision_without_deciding_itself():
    decision = ResponsePolicyEngine().decide(PolicyInput(
        correlation_id="corr-1", threat_class="QUALIFIED_THREAT", investigation_confidence=0.9,
        evidence_sufficient=True, attack_progression=None, attacker_ip="10.0.0.5", victim_ip="10.0.0.10",
        severity="HIGH", requested_action=ContainmentAction.BLOCK_SOURCE_IP, environment="lab",
        auto_contain_config=True, never_block_ips=frozenset(),
    ))
    assert decision.policy_result == PolicyOutcome.CONTAIN

    plan = ResponsePlanGenerator().generate(_campaign(), correlation_id="corr-1", response_decision=decision)
    d = plan.to_dict()
    assert d["selected_containment_action"] == "BLOCK_SOURCE_IP"
    assert d["approval_required"] is False
    assert "CONTAIN" in d["expected_outcome"]


def test_priority_derives_from_severity():
    plan = ResponsePlanGenerator().generate(_campaign())
    assert plan.priority in ("P1", "P2", "P3", "P4")
