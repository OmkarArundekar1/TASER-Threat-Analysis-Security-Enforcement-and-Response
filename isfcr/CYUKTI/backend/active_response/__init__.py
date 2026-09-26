"""
active_response
==================
CYUKTI's closed-loop active containment layer: DECIDE -> CONTAIN ->
VERIFY -> AUDIT -> LEARN, sitting upstream and downstream of the
existing soar/ playbook-execution layer rather than duplicating it.

- Upstream of soar/: ResponsePolicyEngine decides IF containment is
  even permissible (threat qualification alone is never sufficient),
  before a Playbook/PlaybookExecution is ever requested.
- Downstream of soar/: soar.execution_service marks a Shuffle trigger
  SUCCESS as soon as Shuffle acknowledges the request -- it never
  independently checks whether containment actually took effect in the
  real world. ContainmentVerifier closes exactly that gap. A response
  may only be reported as CONTAINMENT_VERIFIED after independent
  evidence is checked; it is never inferred from a 200-OK API response
  alone.

See review/phaseX_active_response_report.md for what is
implemented/tested/live-verified vs BLOCKED_BY_ENVIRONMENT.
"""
