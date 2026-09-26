from active_response.client_agent import Authenticator, ClientResponseAgent, ContainmentRequest
from active_response.containment_actions import ContainmentAction
from active_response.firewall_backend import InMemoryFirewallBackend
from active_response.rollback import RollbackManager


def _agent():
    return ClientResponseAgent(InMemoryFirewallBackend(), Authenticator("secret"))


def test_rollback_after_real_containment_is_verified():
    agent = _agent()
    agent.handle(ContainmentRequest("corr-1", ContainmentAction.BLOCK_SOURCE_IP, "10.0.0.5", "dec-1", "secret"))
    manager = RollbackManager(agent)
    record = manager.rollback("corr-1", "dec-1", "10.0.0.5")
    assert record.verified_rolled_back is True
    assert agent.firewall.is_blocked("10.0.0.5") is False


def test_is_expired_true_when_ttl_elapsed():
    from datetime import datetime, timedelta, timezone
    manager = RollbackManager(_agent())
    past = (datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat()
    assert manager.is_expired(past) is True


def test_is_expired_false_when_no_ttl_set():
    manager = RollbackManager(_agent())
    assert manager.is_expired(None) is False


def test_is_expired_false_when_ttl_in_future():
    from datetime import datetime, timedelta, timezone
    manager = RollbackManager(_agent())
    future = (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat()
    assert manager.is_expired(future) is False
