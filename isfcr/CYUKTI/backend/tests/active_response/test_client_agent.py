import pytest

from active_response.client_agent import Authenticator, ClientResponseAgent, ContainmentRequest
from active_response.containment_actions import ContainmentAction
from active_response.firewall_backend import InMemoryFirewallBackend


def _agent(never_block_ips=frozenset()):
    return ClientResponseAgent(
        firewall=InMemoryFirewallBackend(),
        authenticator=Authenticator(expected_token="real-secret"),
        never_block_ips=never_block_ips,
    )


def _request(**overrides):
    defaults = dict(
        correlation_id="corr-1", action=ContainmentAction.BLOCK_SOURCE_IP,
        target_ip="10.0.0.5", decision_id="dec-1", auth_token="real-secret",
    )
    defaults.update(overrides)
    return ContainmentRequest(**defaults)


def test_successful_containment_is_independently_reconfirmed():
    agent = _agent()
    result = agent.handle(_request())
    assert result.containment_status == "EXECUTED"
    assert result.independently_reconfirmed is True
    assert result.rule_reference is not None


def test_unauthorized_request_is_rejected_not_executed():
    agent = _agent()
    result = agent.handle(_request(auth_token="wrong-token"))
    assert result.containment_status == "REJECTED"
    assert "auth" in result.rejection_reason.lower()
    assert agent.firewall.is_blocked("10.0.0.5") is False


def test_missing_auth_token_is_rejected():
    agent = _agent()
    result = agent.handle(_request(auth_token=None))
    assert result.containment_status == "REJECTED"


def test_missing_correlation_id_is_rejected():
    agent = _agent()
    result = agent.handle(_request(correlation_id=""))
    assert result.containment_status == "REJECTED"
    assert "correlation_id" in result.rejection_reason.lower()


def test_unsupported_action_is_rejected_never_executed():
    agent = _agent()
    result = agent.handle(_request(action=ContainmentAction.ISOLATE_HOST))
    assert result.containment_status == "REJECTED"
    assert "allowlist" in result.rejection_reason.lower()


def test_allowlisted_source_ip_is_rejected():
    agent = _agent(never_block_ips=frozenset({"10.0.0.5"}))
    result = agent.handle(_request())
    assert result.containment_status == "REJECTED"
    assert "never-block" in result.rejection_reason.lower()


def test_invalid_ip_is_rejected():
    agent = _agent()
    result = agent.handle(_request(target_ip="not-an-ip"))
    assert result.containment_status == "REJECTED"


def test_duplicate_correlation_id_is_rejected_on_second_attempt():
    agent = _agent()
    first = agent.handle(_request())
    assert first.containment_status == "EXECUTED"
    second = agent.handle(_request())
    assert second.containment_status == "REJECTED"
    assert "duplicate" in second.rejection_reason.lower()


def test_firewall_failure_results_in_failed_not_executed():
    agent = _agent()
    agent.firewall.fail_next_block = True
    result = agent.handle(_request())
    assert result.containment_status == "FAILED"
    assert result.independently_reconfirmed is False


def test_rollback_is_verified_independently():
    agent = _agent()
    agent.handle(_request())
    rollback_result = agent.rollback("corr-1", "dec-1", ContainmentAction.BLOCK_SOURCE_IP, "10.0.0.5")
    assert rollback_result.containment_status == "EXECUTED"
    assert rollback_result.independently_reconfirmed is True
    assert agent.firewall.is_blocked("10.0.0.5") is False


def test_ttl_sets_a_real_expiry_timestamp():
    agent = _agent()
    result = agent.handle(_request(ttl_seconds=3600))
    assert result.expires_at is not None


def test_no_ttl_means_no_expiry():
    agent = _agent()
    result = agent.handle(_request())
    assert result.expires_at is None
