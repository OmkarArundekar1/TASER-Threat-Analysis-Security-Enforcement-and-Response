"""
evaluators/investigation_eval.py
===================================
Evaluates investigation-loop conclusion correctness against independent
analyst verdicts. CYUKTI's real "conclusion" concept (per Phase 0's
audit) is the investigation's final ConfidenceEstimate plus whichever
hypothesis in InvestigationState.candidate_hypotheses carries the
highest weight at stop time -- there is no separately-named "verdict"
field, so this evaluator does not invent one.

Needs evaluation/labels/investigation_verdicts.json: one row per
completed real investigation, with an independent analyst verdict
(CORRECT / INCORRECT / PARTIALLY_CORRECT / INSUFFICIENT_EVIDENCE) on
whether the investigation's top hypothesis was actually right. No such
file exists yet -- building it requires a human to read each
investigation's evidence trail and top hypothesis and independently
judge it, which this session cannot fabricate. Real, tested, ready to
run the moment that file exists.
"""

from __future__ import annotations

import json
import os

from evaluators.base import EvaluationResult, MetricStatus

_VERDICTS_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "labels", "investigation_verdicts.json"
)

_CORRECT_LABELS = {"CORRECT"}
_INCORRECT_LABELS = {"INCORRECT"}
# PARTIALLY_CORRECT and INSUFFICIENT_EVIDENCE are reported separately,
# never silently folded into either bucket (per the task's own
# "do not silently remove ambiguous samples" instruction).


def load_verdicts() -> list[dict] | None:
    if not os.path.exists(_VERDICTS_PATH):
        return None
    with open(_VERDICTS_PATH) as f:
        return json.load(f)


def evaluate() -> EvaluationResult:
    verdicts = load_verdicts()
    if not verdicts:
        return EvaluationResult(
            task="investigation_confidence_accuracy",
            status=MetricStatus.NOT_MEASURED,
            reason="No independent analyst verdicts exist at evaluation/labels/investigation_verdicts.json. "
                   "CYUKTI's investigation loop has been run against real campaigns (Phase 21/22, "
                   "review/phase21_real_investigation_validation.md), but no human has independently judged "
                   "whether the resulting top hypothesis was actually correct for any of those runs. Building "
                   "this requires a reviewer to read the real evidence trail and judge it -- see "
                   "evaluation/review/export_review_queue.py.",
        )

    unreviewed = [v for v in verdicts if v.get("reviewer", "unreviewed") == "unreviewed"]
    if unreviewed:
        return EvaluationResult(
            task="investigation_confidence_accuracy",
            status=MetricStatus.GROUND_TRUTH_REVIEW_REQUIRED,
            n=len(verdicts),
            reason=f"{len(unreviewed)}/{len(verdicts)} verdicts are unreviewed.",
        )

    from evaluation_metrics import brier_score, expected_calibration_error

    n = len(verdicts)
    correct = sum(1 for v in verdicts if v["verdict"] in _CORRECT_LABELS)
    incorrect = sum(1 for v in verdicts if v["verdict"] in _INCORRECT_LABELS)
    partial = sum(1 for v in verdicts if v["verdict"] == "PARTIALLY_CORRECT")
    insufficient = sum(1 for v in verdicts if v["verdict"] == "INSUFFICIENT_EVIDENCE")

    binary_subset = [v for v in verdicts if v["verdict"] in (_CORRECT_LABELS | _INCORRECT_LABELS)]
    metrics = {
        "n": n, "correct": correct, "incorrect": incorrect,
        "partially_correct": partial, "insufficient_evidence": insufficient,
        "accuracy_on_binary_subset": (correct / len(binary_subset)) if binary_subset else None,
        "binary_subset_n": len(binary_subset),
    }

    confidences = [v["investigation_confidence"] for v in binary_subset if v.get("investigation_confidence") is not None]
    outcomes = [v["verdict"] in _CORRECT_LABELS for v in binary_subset if v.get("investigation_confidence") is not None]
    if confidences:
        metrics["brier_score"] = brier_score(confidences, outcomes)
        metrics["calibration"] = expected_calibration_error(confidences, outcomes)

    return EvaluationResult(
        task="investigation_confidence_accuracy",
        status=MetricStatus.MEASURED,
        n=n,
        metrics=metrics,
        dataset_name="investigation_verdicts",
        method="Independent analyst verdict (CORRECT/INCORRECT/PARTIALLY_CORRECT/INSUFFICIENT_EVIDENCE) per real "
               "investigation vs. CYUKTI's own top candidate_hypothesis at stop time. Brier score / calibration "
               "computed on the binary (CORRECT vs INCORRECT) subset against the stated investigation_confidence.",
        limitations=[
            "PARTIALLY_CORRECT and INSUFFICIENT_EVIDENCE cases are excluded from the binary accuracy/calibration "
            "numbers (reported separately, never silently dropped) since they are not resolvable to a clean "
            "correct/incorrect binary outcome.",
        ],
    )
