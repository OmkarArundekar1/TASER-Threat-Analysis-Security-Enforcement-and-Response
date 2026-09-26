"""
evaluators/threat_qualification_eval.py
==========================================
Independent threat-qualification accuracy: EXPECTED status (from
scenario-declared, analyst-judgment ground truth) vs. CYUKTI's ACTUAL
classification, computed HERE at evaluation time by reading the real,
live `Campaign.cti_score` from Neo4j and classifying it via
cti_confidence_engine.py's own published thresholds -- the same logic
ThreatQualificationEngine.qualify() would apply, without needing to
reconstruct a full IncidentContext object.
"""

from __future__ import annotations

from evaluators.base import EvaluationResult, MetricStatus, classify_review_status, GroundTruthReviewRequired
from ground_truth import store


def _classify(cti_score: float, publish_threshold: float, not_threat_threshold: float) -> str:
    if cti_score >= publish_threshold:
        return "QUALIFIED_THREAT"
    if cti_score >= not_threat_threshold:
        return "SUSPICIOUS"
    return "NOT_THREAT"


def evaluate(dataset_name: str = "threat_qualification_v0", allow_preliminary: bool = True) -> EvaluationResult:
    records, review_status = store.load_best_available(dataset_name)
    if not records:
        return EvaluationResult(
            task="threat_qualification_accuracy", status=MetricStatus.NOT_MEASURED,
            reason=f"No ground-truth dataset '{dataset_name}' has been built yet.",
        )

    try:
        status = classify_review_status(review_status, allow_preliminary)
    except GroundTruthReviewRequired as e:
        return EvaluationResult(
            task="threat_qualification_accuracy", status=MetricStatus.GROUND_TRUTH_REVIEW_REQUIRED,
            n=len(records), dataset_name=dataset_name, reason=str(e),
        )

    from cti_confidence_engine import PUBLISH_THRESHOLD, NOT_THREAT_THRESHOLD
    from neo4j_client import driver
    from evaluation_metrics import classification_report, confusion_matrix

    y_true, y_pred, per_record, skipped = [], [], [], 0
    with driver.session() as s:
        for r in records:
            row = s.run(
                "MATCH (c:Campaign {campaign_id:$id}) RETURN c.cti_score AS cti_score",
                id=r.raw_event_id,
            ).single()
            if row is None or row["cti_score"] is None:
                skipped += 1
                continue
            actual = _classify(row["cti_score"], PUBLISH_THRESHOLD, NOT_THREAT_THRESHOLD)
            y_true.append(r.expected_threat_status)
            y_pred.append(actual)
            per_record.append({
                "sample_id": r.sample_id, "campaign_id": r.raw_event_id,
                "cti_score": row["cti_score"], "expected": r.expected_threat_status, "actual": actual,
            })

    n = len(y_true)
    if n == 0:
        return EvaluationResult(
            task="threat_qualification_accuracy", status=MetricStatus.UNMEASURABLE,
            dataset_name=dataset_name, ground_truth_review_status=review_status.value,
            reason=f"{skipped} ground-truth record(s) found but none had a live, re-fetchable cti_score in Neo4j.",
        )

    report = classification_report(y_true, y_pred)
    matrix = confusion_matrix(y_true, y_pred, labels=["NOT_THREAT", "SUSPICIOUS", "QUALIFIED_THREAT"])

    return EvaluationResult(
        task="threat_qualification_accuracy",
        status=status,
        n=n,
        metrics={
            "n": n, "skipped_no_live_score": skipped,
            "accuracy": report["accuracy"], "macro_f1": report["macro_f1"],
            "weighted_f1": report["weighted_f1"], "per_label": report["per_label"],
            "confusion_matrix": matrix, "per_record": per_record,
        },
        ground_truth_review_status=review_status.value,
        dataset_name=dataset_name,
        method="Scenario-declared expected_threat_status (AI-assisted analyst judgment, not CYUKTI's own "
               "classification) vs. live Campaign.cti_score re-fetched from Neo4j at evaluation time, classified "
               "via cti_confidence_engine.py's own published PUBLISH_THRESHOLD/NOT_THREAT_THRESHOLD constants.",
        limitations=[
            "Ground truth is AUTO_PROPOSED (AI-assisted, not human-reviewed) -- see review/paper_metrics_source_of_truth.md.",
            "expected_threat_status was judged per-SCENARIO (one label for all campaigns matching that attacker/victim "
            "pair), not per-individual-campaign, so disagreement may reflect legitimate campaign-to-campaign variation "
            "within a scenario, not necessarily a CYUKTI error.",
        ],
    )
