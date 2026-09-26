import pytest

from active_response.firewall_backend import InMemoryFirewallBackend, InvalidSourceIP


def test_block_then_is_blocked_is_true():
    fw = InMemoryFirewallBackend()
    result = fw.block("10.0.0.5")
    assert result.success
    assert fw.is_blocked("10.0.0.5") is True


def test_is_blocked_false_for_never_blocked_ip():
    fw = InMemoryFirewallBackend()
    assert fw.is_blocked("10.0.0.9") is False


def test_unblock_removes_the_block():
    fw = InMemoryFirewallBackend()
    fw.block("10.0.0.5")
    result = fw.unblock("10.0.0.5")
    assert result.success
    assert fw.is_blocked("10.0.0.5") is False


def test_unblock_of_never_blocked_ip_is_still_a_success_noop():
    fw = InMemoryFirewallBackend()
    result = fw.unblock("10.0.0.9")
    assert result.success
    assert "was not blocked" in result.detail


def test_invalid_ip_is_rejected_not_silently_accepted():
    fw = InMemoryFirewallBackend()
    with pytest.raises(InvalidSourceIP):
        fw.block("not-an-ip")
    with pytest.raises(InvalidSourceIP):
        fw.is_blocked("also-not-an-ip")


def test_simulated_failure_injection_reports_failure_not_success():
    fw = InMemoryFirewallBackend()
    fw.fail_next_block = True
    result = fw.block("10.0.0.5")
    assert result.success is False
    assert fw.is_blocked("10.0.0.5") is False


def test_failure_injection_is_one_shot():
    fw = InMemoryFirewallBackend()
    fw.fail_next_block = True
    fw.block("10.0.0.5")
    result = fw.block("10.0.0.5")
    assert result.success is True
