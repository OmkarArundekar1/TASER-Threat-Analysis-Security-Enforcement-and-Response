"""
active_response/verification.py
==================================
ContainmentVerification: independent evidence, checked AFTER a
containment action executes, that the action actually took effect.
`ContainmentVerifier.verify()` is the ONLY function in this package
allowed to return `verified=True`, and it requires REAL evidence
objects, not a boolean the caller asserts.

The strongest evidence (Section 9's before/after connection test) is
supported via `pre_attack_reachable` / `post_attack_reachable` --
but this session has no way to actually drive a real connection
attempt from the Kali attacker VM (no shell access to it), so any
live use of that path is BLOCKED_BY_ENVIRONMENT; the weaker
firewall-state-only check (`firewall_state_confirms_block`) is what
this session's own tests exercise against the real
InMemoryFirewallBackend.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum


class VerificationStatus(str, Enum):
    VERIFIED = "VERIFIED"
    NOT_VERIFIED = "NOT_VERIFIED"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
    BLOCKED_BY_ENVIRONMENT = "BLOCKED_BY_ENVIRONMENT"


@dataclass
class ContainmentVerification:
    correlation_id: str
    ip: str
    firewall_state_confirms_block: bool | None       # independent is_blocked() re-check
    pre_attack_reachable: bool | None                 # before containment: attacker->client reached?
    post_attack_reachable: bool | None                # after containment: attacker->client reached?
    wazuh_telemetry_recurrence: bool | None            # did the same attack pattern recur post-containment?
    status: VerificationStatus
    reason: str
    checked_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict:
        return {
            "correlation_id": self.correlation_id, "ip": self.ip,
            "firewall_state_confirms_block": self.firewall_state_confirms_block,
            "pre_attack_reachable": self.pre_attack_reachable,
            "post_attack_reachable": self.post_attack_reachable,
            "wazuh_telemetry_recurrence": self.wazuh_telemetry_recurrence,
            "status": self.status.value, "reason": self.reason, "checked_at": self.checked_at,
        }


class ContainmentVerifier:
    def verify(
        self,
        correlation_id: str,
        ip: str,
        firewall_backend,
        pre_attack_reachable: bool | None = None,
        post_attack_reachable: bool | None = None,
        wazuh_telemetry_recurrence: bool | None = None,
    ) -> ContainmentVerification:
        firewall_confirms = firewall_backend.is_blocked(ip)

        # Strongest possible evidence: a real, independently-observed
        # before/after connectivity change.
        if pre_attack_reachable is True and post_attack_reachable is False and firewall_confirms:
            return ContainmentVerification(
                correlation_id, ip, firewall_confirms, pre_attack_reachable, post_attack_reachable,
                wazuh_telemetry_recurrence, VerificationStatus.VERIFIED,
                "Independent connection test: reachable before containment, unreachable after, "
                "and the firewall backend independently confirms the block rule is present.",
            )

        if pre_attack_reachable is True and post_attack_reachable is True:
            return ContainmentVerification(
                correlation_id, ip, firewall_confirms, pre_attack_reachable, post_attack_reachable,
                wazuh_telemetry_recurrence, VerificationStatus.NOT_VERIFIED,
                "Connection still succeeded after containment was requested -- containment did not take effect, "
                "regardless of what the firewall backend or any API call reported.",
            )

        # No live connection-test evidence available (the common case in
        # this environment -- no access to drive a real attacker-side
        # connection attempt) -- fall back to the weaker, still-real
        # firewall-state check, but never call it VERIFIED at the same
        # confidence as a real connection test.
        if pre_attack_reachable is None and post_attack_reachable is None:
            if firewall_confirms:
                return ContainmentVerification(
                    correlation_id, ip, firewall_confirms, None, None, wazuh_telemetry_recurrence,
                    VerificationStatus.INSUFFICIENT_EVIDENCE,
                    "Firewall backend independently confirms a block rule for this IP, but no real "
                    "attacker-side connection test was available to confirm the block is actually "
                    "effective end-to-end -- reported as INSUFFICIENT_EVIDENCE, not VERIFIED.",
                )
            return ContainmentVerification(
                correlation_id, ip, firewall_confirms, None, None, wazuh_telemetry_recurrence,
                VerificationStatus.NOT_VERIFIED,
                "Firewall backend does not show a block rule for this IP -- containment did not take effect.",
            )

        return ContainmentVerification(
            correlation_id, ip, firewall_confirms, pre_attack_reachable, post_attack_reachable,
            wazuh_telemetry_recurrence, VerificationStatus.INSUFFICIENT_EVIDENCE,
            "Partial connection-test evidence supplied (only one of pre/post) -- not enough to verify.",
        )


verifier = ContainmentVerifier()
