"""
Phase Z Section 7's exact required boundary tests: none of these
individual signals, alone, may ever produce CONTAINMENT_VERIFIED.
"""

import inspect

from active_response.firewall_backend import InMemoryFirewallBackend
from active_response.verification import ContainmentVerifier, VerificationStatus


def test_firewall_state_only_is_insufficient_evidence():
    fw = InMemoryFirewallBackend()
    fw.block("10.0.0.5")
    result = ContainmentVerifier().verify("corr-1", "10.0.0.5", fw)  # no pre/post connection evidence at all
    assert result.status == VerificationStatus.INSUFFICIENT_EVIDENCE


def test_response_agent_success_only_is_insufficient_evidence():
    """A ClientResponseAgent reporting containment_status='EXECUTED' is
    exactly the firewall-state-only case above from the verifier's
    point of view -- ContainmentVerifier.verify() never even takes a
    ContainmentResult as an argument, so an agent's own success report
    cannot, structurally, be the thing that produces VERIFIED."""
    from active_response.client_agent import Authenticator, ClientResponseAgent, ContainmentRequest
    from active_response.containment_actions import ContainmentAction

    agent = ClientResponseAgent(InMemoryFirewallBackend(), Authenticator("secret"))
    result = agent.handle(ContainmentRequest("corr-1", ContainmentAction.BLOCK_SOURCE_IP, "10.0.0.5", "dec-1", "secret"))
    assert result.containment_status == "EXECUTED"  # the agent itself reports success

    verification = ContainmentVerifier().verify("corr-1", "10.0.0.5", agent.firewall)  # no connection evidence given
    assert verification.status == VerificationStatus.INSUFFICIENT_EVIDENCE


def test_verify_signature_has_no_shuffle_or_http_status_parameter():
    """Structural proof, not just a behavioral test: 'Shuffle success'
    and 'HTTP 200' are not even representable inputs to verify() --
    there is no parameter for either, so no caller can pass one in and
    have it influence the result at all, let alone produce VERIFIED."""
    params = set(inspect.signature(ContainmentVerifier.verify).parameters)
    for banned in ("shuffle_success", "http_status", "api_response", "shuffle_result", "acknowledgment"):
        assert banned not in params


def test_shuffle_success_concept_cannot_reach_verified_even_if_smuggled_via_wazuh_recurrence_field():
    """Belt-and-suspenders: even using the one loosely-typed evidence
    field that exists (wazuh_telemetry_recurrence) to represent 'Shuffle
    said it worked' still cannot produce VERIFIED without real
    pre/post connection evidence."""
    fw = InMemoryFirewallBackend()
    fw.block("10.0.0.5")
    result = ContainmentVerifier().verify(
        "corr-1", "10.0.0.5", fw, wazuh_telemetry_recurrence=False,
    )
    assert result.status != VerificationStatus.VERIFIED
    assert result.status == VerificationStatus.INSUFFICIENT_EVIDENCE
