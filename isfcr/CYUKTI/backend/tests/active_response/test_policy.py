from active_response.containment_actions import ContainmentAction
from active_response.decision import PolicyOutcome
from active_response.policy import ResponsePolicyEngine, PolicyInput


def _base_input(**overrides) -> PolicyInput:
    defaults = dict(
        correlation_id="corr-1", threat_class="QUALIFIED_THREAT",
        investigation_confidence=0.8, evidence_sufficient=True,
        attack_progression="active exploitation", attacker_ip="10.0.0.5",
        victim_ip="10.0.0.10", severity="HIGH",
        requested_action=ContainmentAction.BLOCK_SOURCE_IP,
        environment="lab", auto_contain_config=True,
        never_block_ips=frozenset(), risk_level="HIGH",
    )
    defaults.update(overrides)
    return PolicyInput(**defaults)


def test_not_threat_results_in_observe():
    engine = ResponsePolicyEngine()
    decision = engine.decide(_base_input(threat_class="NOT_THREAT"))
    assert decision.policy_result == PolicyOutcome.OBSERVE
    assert decision.selected_action is None


def test_missing_threat_class_results_in_observe():
    engine = ResponsePolicyEngine()
    decision = engine.decide(_base_input(threat_class=None))
    assert decision.policy_result == PolicyOutcome.OBSERVE


def test_insufficient_evidence_results_in_investigate():
    engine = ResponsePolicyEngine()
    decision = engine.decide(_base_input(evidence_sufficient=False))
    assert decision.policy_result == PolicyOutcome.INVESTIGATE
    assert decision.selected_action is None


def test_low_confidence_results_in_recommend_with_approval_required():
    engine = ResponsePolicyEngine()
    decision = engine.decide(_base_input(investigation_confidence=0.1))
    assert decision.policy_result == PolicyOutcome.RECOMMEND
    assert decision.approval_required is True


def test_suspicious_never_auto_contains_even_at_high_confidence():
    engine = ResponsePolicyEngine()
    decision = engine.decide(_base_input(threat_class="SUSPICIOUS", investigation_confidence=0.99))
    assert decision.policy_result == PolicyOutcome.RECOMMEND
    assert decision.approval_required is True


def test_severity_alone_is_not_sufficient_for_containment():
    """CRITICAL severity but low confidence and insufficient evidence
    must NOT auto-contain -- severity is informational only."""
    engine = ResponsePolicyEngine()
    decision = engine.decide(_base_input(
        severity="CRITICAL", risk_level="CRITICAL",
        investigation_confidence=0.05, evidence_sufficient=False,
    ))
    assert decision.policy_result != PolicyOutcome.CONTAIN


def test_unsupported_action_is_never_auto_selected_for_containment():
    engine = ResponsePolicyEngine()
    decision = engine.decide(_base_input(requested_action=ContainmentAction.ISOLATE_HOST))
    assert decision.policy_result == PolicyOutcome.RECOMMEND


def test_no_requested_action_results_in_recommend():
    engine = ResponsePolicyEngine()
    decision = engine.decide(_base_input(requested_action=None))
    assert decision.policy_result == PolicyOutcome.RECOMMEND


def test_allowlisted_source_ip_is_never_auto_contained():
    engine = ResponsePolicyEngine()
    decision = engine.decide(_base_input(never_block_ips=frozenset({"10.0.0.5"})))
    assert decision.policy_result == PolicyOutcome.RECOMMEND
    assert "never-block" in decision.policy_reason.lower() or "allowlist" in decision.policy_reason.lower()


def test_auto_contain_false_never_auto_contains_regardless_of_everything_else():
    engine = ResponsePolicyEngine()
    decision = engine.decide(_base_input(auto_contain_config=False))
    assert decision.policy_result == PolicyOutcome.RECOMMEND
    assert decision.auto_contain_allowed is False


def test_all_gates_clear_results_in_contain():
    engine = ResponsePolicyEngine()
    decision = engine.decide(_base_input())
    assert decision.policy_result == PolicyOutcome.CONTAIN
    assert decision.selected_action == ContainmentAction.BLOCK_SOURCE_IP
    assert decision.approval_required is False


def test_decision_is_fully_traceable():
    engine = ResponsePolicyEngine()
    decision = engine.decide(_base_input(
        incident_id="INC-1", attack_event_id="AE-1", campaign_id="CAMP_X",
        operation_id="OP-1", investigation_id="INV-1",
    ))
    d = decision.to_dict()
    assert d["correlation_id"] == "corr-1"
    assert d["incident_id"] == "INC-1"
    assert d["campaign_id"] == "CAMP_X"
    assert "decision_id" in d and d["decision_id"]
    assert "created_at" in d and d["created_at"]
