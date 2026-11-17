"""
active_response/rollback.py
==============================
RollbackManager: safe expiry/rollback for BLOCK_SOURCE_IP. A rollback
is never reported as successful merely because unblock() was called --
it goes through the same ContainmentVerifier re-check pattern as the
original containment (independently re-read firewall state after
acting).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

from active_response.client_agent import ClientResponseAgent, ContainmentResult
from active_response.containment_actions import ContainmentAction


@dataclass
class RollbackRecord:
    correlation_id: str
    decision_id: str
    target_ip: str
    requested_at: str
    result: ContainmentResult
    verified_rolled_back: bool

    def to_dict(self) -> dict:
        return {
            "correlation_id": self.correlation_id, "decision_id": self.decision_id,
            "target_ip": self.target_ip, "requested_at": self.requested_at,
            "result": self.result.to_dict(), "verified_rolled_back": self.verified_rolled_back,
        }


class RollbackManager:
    def __init__(self, agent: ClientResponseAgent):
        self.agent = agent

    def is_expired(self, expires_at_iso: str | None, now: datetime | None = None) -> bool:
        if expires_at_iso is None:
            return False
        now = now or datetime.now(timezone.utc)
        return now >= datetime.fromisoformat(expires_at_iso)

    def rollback(self, correlation_id: str, decision_id: str, target_ip: str) -> RollbackRecord:
        requested_at = datetime.now(timezone.utc).isoformat()
        result = self.agent.rollback(correlation_id, decision_id, ContainmentAction.BLOCK_SOURCE_IP, target_ip)
        # Rollback is verified only if the agent's own independent
        # re-check (is_blocked() after unblock()) confirms it -- never
        # inferred from unblock()'s bare success flag alone.
        verified = result.containment_status == "EXECUTED" and result.independently_reconfirmed
        return RollbackRecord(
            correlation_id=correlation_id, decision_id=decision_id, target_ip=target_ip,
            requested_at=requested_at, result=result, verified_rolled_back=verified,
        )
