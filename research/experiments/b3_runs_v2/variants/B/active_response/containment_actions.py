"""
active_response/containment_actions.py
==========================================
The strict containment-action allowlist. Appearing in the
`ContainmentAction` enum does NOT make an action executable -- only
membership in `EXECUTABLE_ACTIONS` does, and that set has exactly one
member for this phase. No arbitrary shell/PowerShell/SSH execution
path exists anywhere in this package (verified by
tests/test_active_response_safety.py, which greps this package's own
source for banned patterns).
"""

from __future__ import annotations

from enum import Enum


class ContainmentAction(str, Enum):
    BLOCK_SOURCE_IP = "BLOCK_SOURCE_IP"

    # Declared, NOT executable this phase -- appearing here is
    # documentation of future scope, never a code path. See
    # is_executable() and EXECUTABLE_ACTIONS below, which are the only
    # things any caller may actually branch on before executing.
    ISOLATE_HOST = "ISOLATE_HOST"
    BLOCK_NETWORK_FLOW = "BLOCK_NETWORK_FLOW"
    DISABLE_COMPROMISED_ACCOUNT = "DISABLE_COMPROMISED_ACCOUNT"
    TERMINATE_MALICIOUS_SESSION = "TERMINATE_MALICIOUS_SESSION"


# The ONLY set any execution path may consult to decide "can I actually
# run this." Everything else in ContainmentAction is planned/future and
# must be rejected before it reaches a firewall backend.
EXECUTABLE_ACTIONS: frozenset[ContainmentAction] = frozenset({ContainmentAction.BLOCK_SOURCE_IP})

DESTRUCTIVE_ACTIONS: frozenset[ContainmentAction] = frozenset({
    ContainmentAction.BLOCK_SOURCE_IP,
    ContainmentAction.ISOLATE_HOST,
    ContainmentAction.BLOCK_NETWORK_FLOW,
    ContainmentAction.DISABLE_COMPROMISED_ACCOUNT,
    ContainmentAction.TERMINATE_MALICIOUS_SESSION,
})


def is_executable(action: ContainmentAction) -> bool:
    return action in EXECUTABLE_ACTIONS


def is_destructive(action: ContainmentAction) -> bool:
    return action in DESTRUCTIVE_ACTIONS


class UnsupportedContainmentAction(Exception):
    """Raised whenever any code in this package is asked to execute an
    action outside EXECUTABLE_ACTIONS -- fails closed, never silently
    no-ops or falls back to a broader action."""
