"""
evaluators/mitre_eval.py
===========================
Independent MITRE mapping accuracy: EXPECTED technique (from
ground-truth records, built without ever calling mitre_resolver) vs.
CYUKTI's ACTUAL technique (computed HERE, at evaluation time, by
calling the real backend.mitre_resolver.resolve_mitre() against the
frozen raw alert). The comparison, not either side alone, is what this
module owns.
"""

from __future__ import annotations

import sys
from collections import Counter

from evaluators.base import EvaluationResult, MetricStatus, classify_review_status, GroundTruthReviewRequired
from ground_truth import raw_alerts, store
from ground_truth.schema import ReviewStatus

BACKEND_DIR = None  # set by run_all.py / caller before first use, via ensure_backend_on_path()


def ensure_backend_on_path(backend_dir: str) -> None:
    if backend_dir not in sys.path:
        sys.path.insert(0, backend_dir)


def evaluate(dataset_name: str = "mitre_mapping_v0", allow_preliminary: bool = True) -> EvaluationResult:
    records, review_status = store.load_best_available(dataset_name)
    if not records:
        return EvaluationResult(
            task="mitre_mapping_accuracy", status=MetricStatus.NOT_MEASURED,
            reason=f"No ground-truth dataset '{dataset_name}' has been built yet.",
        )

    try:
        status = classify_review_status(review_status, allow_preliminary)
    except GroundTruthReviewRequired as e:
        return EvaluationResult(
            task="mitre_mapping_accuracy", status=MetricStatus.GROUND_TRUTH_REVIEW_REQUIRED,
            n=len(records), dataset_name=dataset_name, reason=str(e),
        )

    from mitre_resolver import resolve_mitre  # imported here, only at eval time, never by the builder

    raw = raw_alerts.load_raw_alerts(dataset_name)

    y_true_sets: list[set] = []
    y_pred_sets: list[set] = []
    provenance_counts: Counter = Counter()
    skipped = 0
    per_record = []

    for r in records:
        alert = raw.get(r.sample_id)
        if alert is None:
            skipped += 1
            continue
        resolution = resolve_mitre(alert)
        expected = set(r.expected_mitre_techniques)
        actual = set(resolution.technique_ids or [])
        y_true_sets.append(expected)
        y_pred_sets.append(actual)
        provenance_counts[resolution.provenance] += 1
        per_record.append({
            "sample_id": r.sample_id,
            "expected": sorted(expected),
            "actual": sorted(actual),
            "provenance": resolution.provenance,
            "exact_match": expected == actual,
        })

    n = len(y_true_sets)
    if n == 0:
        return EvaluationResult(
            task="mitre_mapping_accuracy", status=MetricStatus.UNMEASURABLE,
            dataset_name=dataset_name, ground_truth_review_status=review_status.value,
            reason=f"{skipped} ground-truth record(s) found but no matching raw alert was stored for any of them.",
        )

    exact_matches = sum(1 for t, p in zip(y_true_sets, y_pred_sets) if t == p)
    # micro precision/recall/F1 over the technique-set membership (multi-label, alert-level)
    tp = sum(len(t & p) for t, p in zip(y_true_sets, y_pred_sets))
    fp = sum(len(p - t) for t, p in zip(y_true_sets, y_pred_sets))
    fn = sum(len(t - p) for t, p in zip(y_true_sets, y_pred_sets))
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    unknown_rate = sum(1 for p in y_pred_sets if not p) / n

    metrics = {
        "n": n,
        "skipped_no_raw_alert": skipped,
        "exact_match_ratio": exact_matches / n,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "unknown_rate": unknown_rate,
        "provenance_distribution": dict(provenance_counts),
        "per_record": per_record,
    }

    return EvaluationResult(
        task="mitre_mapping_accuracy",
        status=status,
        n=n,
        metrics=metrics,
        ground_truth_review_status=review_status.value,
        dataset_name=dataset_name,
        method="Independent ground truth (see scenarios/scenarios.json) vs. live mitre_resolver.resolve_mitre() "
               "called at evaluation time against the frozen raw alert. Multi-label precision/recall/F1 computed "
               "over technique-set membership; exact_match_ratio requires the full predicted set to equal the "
               "full expected set.",
        limitations=[
            "Ground truth is AUTO_PROPOSED (AI-assisted, not human-reviewed) unless ground_truth_review_status "
            "says otherwise -- see review/paper_metrics_source_of_truth.md.",
            f"n={n} is small; see the Wilson confidence interval in the summary report before citing this as precise.",
        ],
    )
