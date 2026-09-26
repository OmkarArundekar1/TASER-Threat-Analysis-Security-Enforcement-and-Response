"""
evaluators/active_response_eval.py
=====================================
Phase Z Section 11: schema/import-hook ONLY. This module does not
evaluate anything and never will until real, independent human/lab
ground truth exists for a real containment action -- none exists in
this repository (zero real containment actions have ever executed
against a live host; see review/phaseX_active_response_report.md and
review/phaseZ_final_integration_audit.md).

The six fields Phase Z asks whether can be independently labeled:
  - detection_success: YES, in principle -- an independent reviewer
    watching the real Wazuh/IDS telemetry can judge whether the attack
    was actually detected, same methodology as evaluation/'s existing
    MITRE-mapping ground truth (raw alert text, independent of CYUKTI).
  - response_decision_correct: YES, in principle -- a reviewer reading
    the real evidence CYUKTI's ResponseDecision was based on can judge
    whether OBSERVE/RECOMMEND/CONTAIN was the right call, independent
    of what CYUKTI itself decided.
  - containment_executed: YES -- this is a directly observable fact
    (did the firewall rule / client-agent report EXECUTED), not really
    a "ground truth" question at all, more a real-world log fact.
  - containment_verified: YES, in principle -- an independent reviewer
    with real before/after connection-test access can confirm this
    independently of ContainmentVerifier's own output.
  - attack_recurrence: YES, in principle -- requires a real subsequent
    attack attempt to observe, independent of CYUKTI.
  - false_containment: YES, in principle -- requires an independent
    reviewer confirming a contained IP was never actually malicious.

All six are HUMAN_REVIEW_REQUIRED: no ground truth exists for any of
them because zero real containment actions have ever been executed
against a live host in this environment. This module exists so the
schema is ready the moment a real lab run produces at least one real
containment event to label -- it is not run, and reports so honestly.
"""

from __future__ import annotations

from dataclasses import dataclass

from evaluators.base import EvaluationResult, MetricStatus

REQUIRED_LABEL_FIELDS = (
    "detection_success",
    "response_decision_correct",
    "containment_executed",
    "attack_recurrence",
    "false_containment",
)
# containment_verified is deliberately NOT independently re-labelable from
# the same evidence a reviewer would use for the others -- it requires the
# SAME real before/after connection-test evidence ContainmentVerifier
# itself needs, so an "independent" label here would really just be a
# second execution of the identical verification method, not a distinct
# ground-truth source. Documented as a real methodological note, not an
# oversight.


@dataclass
class ActiveResponseGroundTruthSchema:
    """The record shape a human reviewer would fill in per real
    containment event, once at least one exists. Mirrors
    ground_truth.schema.GroundTruthRecord's provenance/review-state
    conventions exactly (AUTO_PROPOSED/HUMAN_REVIEWED/LOCKED) rather
    than inventing a parallel one."""
    sample_id: str
    correlation_id: str
    scenario_id: str
    detection_success: bool | None = None
    response_decision_correct: bool | None = None
    containment_executed: bool | None = None
    attack_recurrence: bool | None = None
    false_containment: bool | None = None
    evidence_reference: str = ""
    reviewer: str = "unreviewed"


def evaluate() -> EvaluationResult:
    return EvaluationResult(
        task="active_response_accuracy",
        status=MetricStatus.BLOCKED_BY_ENVIRONMENT,
        n=0,
        reason=(
            "No real containment action has ever executed against a live host in this environment "
            "(review/phaseX_active_response_report.md, Section N: BLOCKED_BY_ENVIRONMENT). There is "
            "zero real data to build ground truth from, human-reviewed or otherwise. This evaluator's "
            "schema (ActiveResponseGroundTruthSchema) and the required label fields are ready; running "
            "it requires a real lab demonstration first, then independent human review of that real "
            "event -- neither has happened. Reported as BLOCKED_BY_ENVIRONMENT, not NOT_MEASURED, "
            "because no code gap blocks this -- only the absence of a real event to evaluate."
        ),
        method="Schema/import-hook only, per Phase Z Section 11 -- never fabricates labels or reports "
               "an accuracy number.",
        limitations=[
            "containment_verified is not independently re-labelable from different evidence than "
            "ContainmentVerifier itself already requires -- see module docstring.",
        ],
    )
