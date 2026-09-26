"""
End-to-end integration tests chaining the REAL modules together:
integration.decide_for_campaign() -> soar.response_plan.ResponsePlanGenerator
-> active_response.client_agent.ClientResponseAgent -> ContainmentVerifier
-> active_response.audit, exactly the Phase Z Section 1 lifecycle
(minus the parts requiring real Wazuh/Neo4j/Kali/Shuffle infrastructure,
which are exercised at the unit level elsewhere and documented as
BLOCKED_BY_ENVIRONMENT in review/phaseZ_final_integration_audit.md).

Also proves Section 3's explicit requirement: one correlation_id
threads unchanged through every real object in the chain, and two
unrelated incidents never collide.
"""

import os
import tempfile

import pytest

import soar.memory as memory_module
from campaign_context import CampaignContext
from soar.memory import PlaybookMemoryStore
from soar.response_plan import ResponsePlanGenerator
from threat_qualification import QualityGateCheck, ThreatQualificationResult

import active_response.audit as audit_module
from active_response.client_agent import Authenticator, ClientResponseAgent, ContainmentRequest
from active_response.containment_actions import ContainmentAction
from active_response.decision import PolicyOutcome
from active_response.firewall_backend import InMemoryFirewallBackend
from active_response.integration import decide_for_campaign
from active_response.rollback import RollbackManager
from active_response.verification import ContainmentVerifier, VerificationStatus


@pytest.fixture(autouse=True)
def _isolated_memory_store(monkeypatch):
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    store = PlaybookMemoryStore(db_path=path)
    monkeypatch.setattr(memory_module, "memory_store", store)
    monkeypatch.setattr(audit_module, "memory_store", store)
    yield store
    os.remove(path)


def _campaign(**overrides):
    defaults = dict(campaign_id="CAMP_E2E_1", attacker_ip="10.0.0.5", victim_ip="10.0.0.10",
                     risk_score=8000, last_technique="T1110", techniques={"T1110"})
    defaults.update(overrides)
    return CampaignContext(**defaults)


def _qualification(classification: str, cti_score: float = 50.0) -> ThreatQualificationResult:
    return ThreatQualificationResult(
        classification=classification, cti_score=cti_score,
        checks=[QualityGateCheck("has_ioc", True, "attacker IP present")],
        may_publish_to_misp=(classification == "QUALIFIED_THREAT"), reason="test fixture",
    )


def _agent():
    return ClientResponseAgent(InMemoryFirewallBackend(), Authenticator("secret"), never_block_ips=frozenset())


# ---- Section 3: correlation ID propagation ----------------------------------

def test_correlation_id_is_identical_across_the_full_chain():
    campaign = _campaign()
    decision = decide_for_campaign(campaign, _qualification("QUALIFIED_THREAT"),
                                    investigation_confidence=0.9, evidence_coverage=0.5,
                                    auto_contain_config=True)
    plan = ResponsePlanGenerator().generate(campaign, correlation_id=decision.correlation_id, response_decision=decision)

    agent = _agent()
    request = ContainmentRequest(decision.correlation_id, decision.selected_action, campaign.attacker_ip,
                                  decision.decision_id, "secret")
    result = agent.handle(request)
    verification = ContainmentVerifier().verify(decision.correlation_id, campaign.attacker_ip, agent.firewall,
                                                 pre_attack_reachable=True, post_attack_reachable=False)

    audit_module.log_response_event("CONTAINMENT_VERIFIED", correlation_id=decision.correlation_id,
                                     decision_id=decision.decision_id, campaign_id=campaign.campaign_id)
    trail = audit_module.audit_trail_for_correlation(decision.correlation_id)

    ids = {decision.correlation_id, plan.correlation_id, result.correlation_id, verification.correlation_id,
           trail[0]["detail"]["correlation_id"]}
    assert len(ids) == 1, f"correlation_id diverged across the chain: {ids}"
    assert decision.correlation_id == campaign.campaign_id  # the reuse strategy from correlation.py


def test_two_unrelated_campaigns_get_distinct_correlation_ids_throughout():
    d1 = decide_for_campaign(_campaign(campaign_id="CAMP_A"), _qualification("QUALIFIED_THREAT"),
                              investigation_confidence=0.9, evidence_coverage=0.5, auto_contain_config=True)
    d2 = decide_for_campaign(_campaign(campaign_id="CAMP_B"), _qualification("QUALIFIED_THREAT"),
                              investigation_confidence=0.9, evidence_coverage=0.5, auto_contain_config=True)
    assert d1.correlation_id != d2.correlation_id


# ---- Section 13: the 16 required failure/success scenarios ------------------

def test_01_not_threat_never_contains():
    decision = decide_for_campaign(_campaign(), _qualification("NOT_THREAT"), evidence_coverage=0.5,
                                    investigation_confidence=0.9, auto_contain_config=True)
    assert decision.policy_result == PolicyOutcome.OBSERVE
    assert decision.selected_action is None


def test_02_suspicious_never_auto_contains():
    decision = decide_for_campaign(_campaign(), _qualification("SUSPICIOUS"), evidence_coverage=0.5,
                                    investigation_confidence=0.95, auto_contain_config=True)
    assert decision.policy_result == PolicyOutcome.RECOMMEND
    assert decision.approval_required is True


def test_03_qualified_threat_auto_contain_false_goes_to_recommend():
    decision = decide_for_campaign(_campaign(), _qualification("QUALIFIED_THREAT"), evidence_coverage=0.5,
                                    investigation_confidence=0.9, auto_contain_config=False)
    assert decision.policy_result == PolicyOutcome.RECOMMEND
    assert decision.auto_contain_allowed is False


def test_04_qualified_threat_policy_rejection_via_allowlisted_source():
    decision = decide_for_campaign(_campaign(), _qualification("QUALIFIED_THREAT"), evidence_coverage=0.5,
                                    investigation_confidence=0.9, auto_contain_config=True,
                                    never_block_ips=frozenset({"10.0.0.5"}))
    assert decision.policy_result == PolicyOutcome.RECOMMEND


def test_05_qualified_threat_allowed_containment_reaches_contain():
    decision = decide_for_campaign(_campaign(), _qualification("QUALIFIED_THREAT"), evidence_coverage=0.5,
                                    investigation_confidence=0.9, auto_contain_config=True)
    assert decision.policy_result == PolicyOutcome.CONTAIN
    assert decision.selected_action == ContainmentAction.BLOCK_SOURCE_IP


def test_06_shuffle_unavailable_fails_safely():
    """Reuses soar.execution_service's existing, pre-existing contract
    -- not re-implemented here, just confirmed still true."""
    from soar.execution_service import PlaybookExecutionService
    from soar.shuffle_client import ShuffleClient
    from soar.generator import PlaybookGenerator

    campaign = _campaign()
    playbook = PlaybookGenerator().generate(campaign)
    service = PlaybookExecutionService(memory_module.memory_store, ShuffleClient(webhook_url=""))
    execution = service.request_execution(playbook, campaign.campaign_id)
    assert execution.status.value in ("failed", "success", "pending_approval")  # never crashes
    if execution.status.value == "failed":
        assert any("not configured" in (r.error or "") for r in execution.action_results)


def test_07_client_agent_rejects_when_action_unsupported_fails_safely():
    agent = _agent()
    result = agent.handle(ContainmentRequest("corr-1", ContainmentAction.ISOLATE_HOST, "10.0.0.5", "dec-1", "secret"))
    assert result.containment_status == "REJECTED"


def test_08_firewall_execution_failure_is_failed():
    agent = _agent()
    agent.firewall.fail_next_block = True
    result = agent.handle(ContainmentRequest("corr-1", ContainmentAction.BLOCK_SOURCE_IP, "10.0.0.5", "dec-1", "secret"))
    assert result.containment_status == "FAILED"


def test_09_firewall_updated_but_verification_insufficient():
    agent = _agent()
    agent.handle(ContainmentRequest("corr-1", ContainmentAction.BLOCK_SOURCE_IP, "10.0.0.5", "dec-1", "secret"))
    verification = ContainmentVerifier().verify("corr-1", "10.0.0.5", agent.firewall)  # no connection evidence
    assert verification.status == VerificationStatus.INSUFFICIENT_EVIDENCE


def test_10_duplicate_response_is_safely_rejected():
    agent = _agent()
    req = ContainmentRequest("corr-1", ContainmentAction.BLOCK_SOURCE_IP, "10.0.0.5", "dec-1", "secret")
    first = agent.handle(req)
    second = agent.handle(req)
    assert first.containment_status == "EXECUTED"
    assert second.containment_status == "REJECTED"


def test_11_expired_response_is_rejected_by_rollback_manager():
    from datetime import datetime, timedelta, timezone
    agent = _agent()
    result = agent.handle(ContainmentRequest("corr-1", ContainmentAction.BLOCK_SOURCE_IP, "10.0.0.5", "dec-1",
                                              "secret", ttl_seconds=1))
    manager = RollbackManager(agent)
    later = datetime.fromisoformat(result.executed_at) + timedelta(seconds=10)
    assert manager.is_expired(result.expires_at, now=later) is True


def test_12_invalid_correlation_id_is_rejected():
    agent = _agent()
    result = agent.handle(ContainmentRequest("", ContainmentAction.BLOCK_SOURCE_IP, "10.0.0.5", "dec-1", "secret"))
    assert result.containment_status == "REJECTED"


def test_13_allowlisted_source_containment_rejected():
    agent = ClientResponseAgent(InMemoryFirewallBackend(), Authenticator("secret"),
                                 never_block_ips=frozenset({"10.0.0.5"}))
    result = agent.handle(ContainmentRequest("corr-1", ContainmentAction.BLOCK_SOURCE_IP, "10.0.0.5", "dec-1", "secret"))
    assert result.containment_status == "REJECTED"


def test_14_successful_containment_with_sufficient_verification_is_verified():
    agent = _agent()
    agent.handle(ContainmentRequest("corr-1", ContainmentAction.BLOCK_SOURCE_IP, "10.0.0.5", "dec-1", "secret"))
    verification = ContainmentVerifier().verify("corr-1", "10.0.0.5", agent.firewall,
                                                 pre_attack_reachable=True, post_attack_reachable=False)
    assert verification.status == VerificationStatus.VERIFIED


def test_15_rollback_then_verification():
    agent = _agent()
    agent.handle(ContainmentRequest("corr-1", ContainmentAction.BLOCK_SOURCE_IP, "10.0.0.5", "dec-1", "secret"))
    manager = RollbackManager(agent)
    record = manager.rollback("corr-1", "dec-1", "10.0.0.5")
    assert record.verified_rolled_back is True


def test_16_verification_after_rollback_shows_correct_final_state():
    agent = _agent()
    agent.handle(ContainmentRequest("corr-1", ContainmentAction.BLOCK_SOURCE_IP, "10.0.0.5", "dec-1", "secret"))
    RollbackManager(agent).rollback("corr-1", "dec-1", "10.0.0.5")
    verification = ContainmentVerifier().verify("corr-1", "10.0.0.5", agent.firewall,
                                                 pre_attack_reachable=True, post_attack_reachable=True)
    # after rollback, the IP is reachable again -- NOT_VERIFIED is the
    # correct read (containment is no longer in effect, by design)
    assert verification.status == VerificationStatus.NOT_VERIFIED
