"""
active_response/correlation.py
=================================
The single cross-layer correlation_id for one incident's active-response
lifecycle. Per Phase Z's explicit instruction, this does NOT replace
attack_event_id/campaign_id/operation_id/investigation_id -- those
already-stable identifiers are preserved as-is everywhere they're
used. correlation_id is an ADDITIONAL identifier for tracing the
response-side lifecycle (decision -> plan -> containment -> verification
-> audit), reusing the most stable ID CYUKTI already generates rather
than minting a redundant one.

Strategy (deterministic, tested):
  1. If a real campaign_id exists, correlation_id = that campaign_id
     verbatim. campaign_id is already CYUKTI's own generated-once,
     immutable, per-incident identifier (campaign_manager.py) -- reusing
     it directly means every existing tool that already knows a
     campaign_id (the dashboard, soar/api.py's existing routes,
     experiments/) can look up its response lifecycle with zero new
     lookup table.
  2. Else if only an attack_event_id exists (pre-campaign-resolution),
     correlation_id = f"PRECAMPAIGN:{attack_event_id}".
  3. Else (neither exists yet -- pure detection-time call), a fresh
     ID is minted and MUST be persisted by the caller before it can be
     reused; derive_correlation_id() never remembers past calls itself
     (no hidden global state), so callers own propagation.
"""

from __future__ import annotations

from active_response.decision import new_id


def derive_correlation_id(campaign_id: str | None = None, attack_event_id: str | None = None) -> str:
    if campaign_id:
        return campaign_id
    if attack_event_id:
        return f"PRECAMPAIGN:{attack_event_id}"
    return new_id("corr")


class MissingCorrelationID(Exception):
    """Raised by any active_response entry point that requires a
    correlation_id and received none/empty -- response execution must
    fail safely rather than proceed with an untraceable request."""


def require_correlation_id(correlation_id: str | None) -> str:
    if not correlation_id:
        raise MissingCorrelationID(
            "No correlation_id supplied -- response execution cannot proceed untraceably. "
            "Derive one first via derive_correlation_id(campaign_id=..., attack_event_id=...)."
        )
    return correlation_id
