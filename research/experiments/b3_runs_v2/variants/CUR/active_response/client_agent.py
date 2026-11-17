"""
active_response/client_agent.py
==================================
ClientResponseAgent: the controlled agent that would run on a
protected endpoint. Takes a `ContainmentRequest` built ONLY from a
real `ResponseDecision` (never raw campaign/alert data, never a
free-form command) and:

  1. authenticates the request (pluggable Authenticator -- fails closed
     if none/invalid)
  2. validates the request schema (dataclass construction itself
     enforces this)
  3. validates the correlation ID is present
  4. validates the requested action against the executable allowlist
  5. checks the local never-block allowlist
  6. applies BLOCK_SOURCE_IP via a FirewallBackend
  7. independently re-checks firewall state (never trusts block()'s own
     return value as sufficient enforcement evidence)
  8. records the result
  9. returns structured ContainmentResult telemetry -- never a bare
     "command executed successfully" string

Never accepts an arbitrary command or executable path -- there is no
such field anywhere in ContainmentRequest.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from active_response.containment_actions import ContainmentAction, is_executable
from active_response.firewall_backend import FirewallBackend, InvalidSourceIP


class Authenticator:
    """Minimal, real, pluggable auth check -- a shared-secret token
    compared with hmac.compare_digest (constant-time), not `==`, so
    this isn't a timing side-channel toy. A production deployment would
    swap in mTLS/real service-to-service auth; this is the seam."""

    def __init__(self, expected_token: str):
        self._expected_token = expected_token

    def authenticate(self, provided_token: str | None) -> bool:
        import hmac
        if not provided_token or not self._expected_token:
            return False
        return hmac.compare_digest(provided_token, self._expected_token)


class ContainmentRequestError(Exception):
    """Raised for any request that fails auth/schema/allowlist checks
    -- the agent fails closed, it never partially executes."""


@dataclass
class ContainmentRequest:
    correlation_id: str
    action: ContainmentAction
    target_ip: str
    decision_id: str
    auth_token: str | None
    ttl_seconds: int | None = None   # None = permanent until explicit rollback


@dataclass
class ContainmentResult:
    correlation_id: str
    decision_id: str
    action: ContainmentAction
    target_ip: str
    containment_status: str          # "EXECUTED" | "FAILED" | "REJECTED"
    firewall_result_detail: str
    rule_reference: str | None
    independently_reconfirmed: bool  # re-read is_blocked() after block(), not just block()'s own return
    executed_at: str
    expires_at: str | None
    rejection_reason: str | None = None

    def to_dict(self) -> dict:
        return {
            "correlation_id": self.correlation_id, "decision_id": self.decision_id,
            "action": self.action.value, "target_ip": self.target_ip,
            "containment_status": self.containment_status,
            "firewall_result_detail": self.firewall_result_detail,
            "rule_reference": self.rule_reference,
            "independently_reconfirmed": self.independently_reconfirmed,
            "executed_at": self.executed_at, "expires_at": self.expires_at,
            "rejection_reason": self.rejection_reason,
        }


class ClientResponseAgent:
    def __init__(self, firewall: FirewallBackend, authenticator: Authenticator,
                 never_block_ips: frozenset[str] = frozenset()):
        self.firewall = firewall
        self.authenticator = authenticator
        self.never_block_ips = never_block_ips
        self._seen_correlation_ids: set[str] = set()   # duplicate-request guard
        self._seen_lock = threading.Lock()              # see _reserve_correlation_id

    def _reserve_correlation_id(self, correlation_id: str) -> bool:
        """Atomically check-and-reserve. A plain 'if in set: ... else:
        set.add(...)' is a check-then-act race: two concurrent calls
        for the same correlation_id could both observe 'not seen' before
        either reserves it, and both would go on to execute a real
        firewall block -- exactly the kind of duplicate-execution bug
        Section 14's 'race conditions in state transitions' asks to be
        checked for. Returns True iff THIS call is the one that reserved it."""
        with self._seen_lock:
            if correlation_id in self._seen_correlation_ids:
                return False
            self._seen_correlation_ids.add(correlation_id)
            return True

    def handle(self, request: ContainmentRequest) -> ContainmentResult:
        now = datetime.now(timezone.utc)

        if not self.authenticator.authenticate(request.auth_token):
            return self._rejected(request, now, "Authentication failed.")

        if not request.correlation_id:
            return self._rejected(request, now, "Missing correlation_id.")

        # Reserve BEFORE any other check so two concurrent requests can
        # never both pass this gate for the same correlation_id, even if
        # a later check (allowlist, IP validation) would reject one of
        # them anyway -- the reservation itself, not just the eventual
        # containment call, is the race-sensitive resource.
        if not self._reserve_correlation_id(request.correlation_id):
            return self._rejected(request, now, f"Duplicate request for correlation_id={request.correlation_id}.")

        if not is_executable(request.action):
            return self._rejected(request, now, f"{request.action.value} is not on the executable allowlist.")

        if request.target_ip in self.never_block_ips:
            return self._rejected(request, now, f"{request.target_ip} is on the never-block allowlist.")

        try:
            from active_response.firewall_backend import validate_ip
            validate_ip(request.target_ip)
        except InvalidSourceIP as e:
            return self._rejected(request, now, str(e))

        block_result = self.firewall.block(request.target_ip)
        # Never trust block()'s own success flag as final evidence --
        # independently re-read state via a SEPARATE call.
        reconfirmed = self.firewall.is_blocked(request.target_ip) if block_result.success else False

        status = "EXECUTED" if (block_result.success and reconfirmed) else "FAILED"
        expires_at = (
            (now + timedelta(seconds=request.ttl_seconds)).isoformat() if request.ttl_seconds else None
        )

        return ContainmentResult(
            correlation_id=request.correlation_id, decision_id=request.decision_id,
            action=request.action, target_ip=request.target_ip,
            containment_status=status,
            firewall_result_detail=block_result.detail,
            rule_reference=block_result.rule_reference if status == "EXECUTED" else None,
            independently_reconfirmed=reconfirmed,
            executed_at=now.isoformat(), expires_at=expires_at,
        )

    def rollback(self, correlation_id: str, decision_id: str, action: ContainmentAction, target_ip: str) -> ContainmentResult:
        now = datetime.now(timezone.utc)
        unblock_result = self.firewall.unblock(target_ip)
        still_blocked = self.firewall.is_blocked(target_ip)
        status = "EXECUTED" if (unblock_result.success and not still_blocked) else "FAILED"
        return ContainmentResult(
            correlation_id=correlation_id, decision_id=decision_id, action=action, target_ip=target_ip,
            containment_status=status, firewall_result_detail=unblock_result.detail,
            rule_reference=None, independently_reconfirmed=not still_blocked,
            executed_at=now.isoformat(), expires_at=None,
        )

    def _rejected(self, request: ContainmentRequest, now: datetime, reason: str) -> ContainmentResult:
        return ContainmentResult(
            correlation_id=request.correlation_id, decision_id=request.decision_id,
            action=request.action, target_ip=request.target_ip,
            containment_status="REJECTED", firewall_result_detail="",
            rule_reference=None, independently_reconfirmed=False,
            executed_at=now.isoformat(), expires_at=None, rejection_reason=reason,
        )
