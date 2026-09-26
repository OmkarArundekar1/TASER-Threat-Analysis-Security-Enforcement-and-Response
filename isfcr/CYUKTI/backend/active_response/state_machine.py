"""
active_response/state_machine.py
===================================
Explicit response states and the ONLY valid transitions between them.
Structurally impossible to jump straight to VERIFIED: the graph below
has no edge into VERIFIED except from VERIFY, and no edge into VERIFY
except from CONTAIN. A caller cannot route around this by calling
transition() with an unlisted edge -- InvalidStateTransition is raised,
not silently ignored.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum


class ResponseState(str, Enum):
    OBSERVE = "OBSERVE"
    INVESTIGATE = "INVESTIGATE"
    RECOMMEND = "RECOMMEND"
    CONTAIN = "CONTAIN"          # containment has been requested/executed, not yet verified
    VERIFY = "VERIFY"            # verification is in progress
    VERIFIED = "VERIFIED"        # independently confirmed effective
    FAILED = "FAILED"
    ROLLED_BACK = "ROLLED_BACK"
    EXPIRED = "EXPIRED"


# Adjacency list of the ONLY legal transitions. Every edge is one this
# module's own tests exercise; every non-edge has a corresponding
# "rejects" test (test_active_response_state_machine.py).
_VALID_TRANSITIONS: dict[ResponseState, frozenset[ResponseState]] = {
    ResponseState.OBSERVE: frozenset({ResponseState.INVESTIGATE}),
    ResponseState.INVESTIGATE: frozenset({ResponseState.RECOMMEND, ResponseState.OBSERVE}),
    ResponseState.RECOMMEND: frozenset({ResponseState.CONTAIN, ResponseState.OBSERVE}),
    ResponseState.CONTAIN: frozenset({ResponseState.VERIFY, ResponseState.FAILED}),
    ResponseState.VERIFY: frozenset({ResponseState.VERIFIED, ResponseState.FAILED}),
    ResponseState.VERIFIED: frozenset({ResponseState.ROLLED_BACK, ResponseState.EXPIRED}),
    ResponseState.FAILED: frozenset(),      # terminal
    ResponseState.ROLLED_BACK: frozenset(), # terminal
    ResponseState.EXPIRED: frozenset(),     # terminal
}


class InvalidStateTransition(Exception):
    def __init__(self, current: ResponseState, target: ResponseState):
        super().__init__(
            f"Illegal response-state transition {current.value} -> {target.value}. "
            f"Valid targets from {current.value}: "
            f"{sorted(s.value for s in _VALID_TRANSITIONS.get(current, frozenset()))}"
        )
        self.current = current
        self.target = target


@dataclass
class StateTransitionRecord:
    from_state: ResponseState
    to_state: ResponseState
    timestamp: str
    evidence: str = ""

    def to_dict(self) -> dict:
        return {"from_state": self.from_state.value, "to_state": self.to_state.value,
                "timestamp": self.timestamp, "evidence": self.evidence}


@dataclass
class ResponseStateMachine:
    """One instance per response lifecycle (one per ResponseDecision).
    `history` is the full, ordered, auditable transition log -- never
    mutated in place, only appended to."""
    current_state: ResponseState = ResponseState.OBSERVE
    history: list[StateTransitionRecord] = field(default_factory=list)

    def transition(self, target: ResponseState, evidence: str = "") -> StateTransitionRecord:
        """Requiring a non-empty `evidence` string for the two
        highest-stakes transitions (into VERIFIED, and CONTAIN->VERIFY)
        is a deliberate, structural guard against exactly the failure
        mode Section 10 describes: converting an HTTP 200 into a
        printed "ATTACK BLOCKED" with nothing behind it."""
        allowed = _VALID_TRANSITIONS.get(self.current_state, frozenset())
        if target not in allowed:
            raise InvalidStateTransition(self.current_state, target)
        if target in (ResponseState.VERIFY, ResponseState.VERIFIED) and not evidence:
            raise ValueError(
                f"Transition to {target.value} requires non-empty evidence -- "
                f"a state change into verification territory can never be evidence-free."
            )
        record = StateTransitionRecord(
            from_state=self.current_state, to_state=target,
            timestamp=datetime.now(timezone.utc).isoformat(), evidence=evidence,
        )
        self.history.append(record)
        self.current_state = target
        return record

    def is_terminal(self) -> bool:
        return not _VALID_TRANSITIONS.get(self.current_state, frozenset())
