from active_response.firewall_backend import InMemoryFirewallBackend
from active_response.verification import ContainmentVerifier, VerificationStatus


def test_full_before_after_evidence_yields_verified():
    fw = InMemoryFirewallBackend()
    fw.block("10.0.0.5")
    result = ContainmentVerifier().verify(
        "corr-1", "10.0.0.5", fw, pre_attack_reachable=True, post_attack_reachable=False,
    )
    assert result.status == VerificationStatus.VERIFIED


def test_still_reachable_after_containment_is_not_verified():
    fw = InMemoryFirewallBackend()
    fw.block("10.0.0.5")
    result = ContainmentVerifier().verify(
        "corr-1", "10.0.0.5", fw, pre_attack_reachable=True, post_attack_reachable=True,
    )
    assert result.status == VerificationStatus.NOT_VERIFIED


def test_no_connection_evidence_but_firewall_confirms_is_insufficient_not_verified():
    """This is the core anti-false-positive guard: an HTTP-200-style
    'it worked' signal (here, just the firewall backend's own state)
    must never alone produce VERIFIED."""
    fw = InMemoryFirewallBackend()
    fw.block("10.0.0.5")
    result = ContainmentVerifier().verify("corr-1", "10.0.0.5", fw)
    assert result.status == VerificationStatus.INSUFFICIENT_EVIDENCE
    assert result.status != VerificationStatus.VERIFIED


def test_no_firewall_confirmation_and_no_connection_evidence_is_not_verified():
    fw = InMemoryFirewallBackend()  # never blocked
    result = ContainmentVerifier().verify("corr-1", "10.0.0.5", fw)
    assert result.status == VerificationStatus.NOT_VERIFIED


def test_partial_connection_evidence_is_insufficient():
    fw = InMemoryFirewallBackend()
    fw.block("10.0.0.5")
    result = ContainmentVerifier().verify("corr-1", "10.0.0.5", fw, pre_attack_reachable=True)
    assert result.status == VerificationStatus.INSUFFICIENT_EVIDENCE


def test_verification_result_is_fully_serializable():
    fw = InMemoryFirewallBackend()
    fw.block("10.0.0.5")
    result = ContainmentVerifier().verify(
        "corr-1", "10.0.0.5", fw, pre_attack_reachable=True, post_attack_reachable=False,
        wazuh_telemetry_recurrence=False,
    )
    d = result.to_dict()
    assert d["status"] == "VERIFIED"
    assert d["correlation_id"] == "corr-1"
