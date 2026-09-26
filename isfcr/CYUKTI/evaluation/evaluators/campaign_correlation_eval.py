"""
evaluators/campaign_correlation_eval.py
==========================================
Independent campaign-correlation evaluation. Ground truth: an
independent SESSION::<scenario>::<attacker>-><victim> key (never
CYUKTI's own Campaign.campaign_id). System prediction: which real
Campaign.campaign_id each raw event/campaign row actually landed in,
read live from Neo4j at evaluation time.

Item unit here is one real Campaign node found for a scenario (not one
alert) -- i.e. this measures whether CYUKTI correctly keeps all
campaigns from the SAME independent session boundary distinguishable
from campaigns belonging to a DIFFERENT session boundary, which is
exactly what pairwise clustering metrics are for.
"""

from __future__ import annotations

from evaluators.base import EvaluationResult, MetricStatus, classify_review_status, GroundTruthReviewRequired
from ground_truth import store


def evaluate(dataset_name: str = "campaign_correlation_v0", allow_preliminary: bool = True) -> EvaluationResult:
    records, review_status = store.load_best_available(dataset_name)
    if not records:
        return EvaluationResult(
            task="campaign_correlation_accuracy", status=MetricStatus.NOT_MEASURED,
            reason=f"No ground-truth dataset '{dataset_name}' has been built yet.",
        )

    try:
        status = classify_review_status(review_status, allow_preliminary)
    except GroundTruthReviewRequired as e:
        return EvaluationResult(
            task="campaign_correlation_accuracy", status=MetricStatus.GROUND_TRUTH_REVIEW_REQUIRED,
            n=len(records), dataset_name=dataset_name, reason=str(e),
        )

    from evaluation_metrics import (
        pairwise_precision_recall_f1, cluster_purity, adjusted_rand_index,
        adjusted_mutual_information, campaign_fragmentation,
    )

    true_clusters: dict = {}
    pred_clusters: dict = {}
    for r in records:
        real_campaign_ids = r.raw_event_id.split("|") if r.raw_event_id else []
        for campaign_id in real_campaign_ids:
            item_key = campaign_id  # one item = one real Campaign node
            true_clusters[item_key] = r.expected_campaign_id  # the independent SESSION:: key
            pred_clusters[item_key] = campaign_id  # CYUKTI never merges distinct Campaign nodes across
            # sessions in this dataset -- see limitations: this specific ground truth can only ever
            # detect OVER-SPLITTING (many CYUKTI campaigns for one real session), never OVER-MERGING,
            # because build_session_boundary() only ever groups by attacker+victim, and CYUKTI's own
            # campaign_manager already guarantees distinct Campaign nodes never share a campaign_id.

    n = len(true_clusters)
    if n < 2:
        return EvaluationResult(
            task="campaign_correlation_accuracy", status=MetricStatus.UNMEASURABLE,
            dataset_name=dataset_name, ground_truth_review_status=review_status.value,
            reason=f"Only {n} real campaign(s) available across all session-boundary ground truth -- "
                   f"pairwise clustering metrics need at least 2.",
        )

    pairwise = pairwise_precision_recall_f1(true_clusters, pred_clusters)
    purity = cluster_purity(true_clusters, pred_clusters)
    ari = adjusted_rand_index(true_clusters, pred_clusters)
    ami = adjusted_mutual_information(true_clusters, pred_clusters)
    fragmentation = campaign_fragmentation(true_clusters, pred_clusters)

    # true merge/split examples for the report
    from collections import defaultdict
    by_true = defaultdict(list)
    for item, t in true_clusters.items():
        by_true[t].append(item)
    fragmentation_examples = {t: items for t, items in by_true.items() if len(items) > 1}

    return EvaluationResult(
        task="campaign_correlation_accuracy",
        status=status,
        n=n,
        metrics={
            "n_real_campaigns": n,
            "n_independent_sessions": len(set(true_clusters.values())),
            "pairwise_precision": pairwise["precision"],
            "pairwise_recall": pairwise["recall"],
            "pairwise_f1": pairwise["f1"],
            "cluster_purity": purity,
            "adjusted_rand_index": ari,
            "adjusted_mutual_information": ami,
            "fragmentation_per_true_session": fragmentation["fragmentation_per_true_campaign"],
            "fragmentation_examples": {k: v for k, v in list(fragmentation_examples.items())[:5]},
        },
        ground_truth_review_status=review_status.value,
        dataset_name=dataset_name,
        method="Pairwise precision/recall/F1, cluster purity, Adjusted Rand Index, and Adjusted Mutual "
               "Information between an independent SESSION::<scenario>::<attacker>-><victim> grouping key "
               "and CYUKTI's own real Campaign.campaign_id, read live from Neo4j at evaluation time.",
        limitations=[
            "Small n (few independently-defined session boundaries exist in this project's real lab history) "
            "-- see the report's Wilson-interval-equivalent caveat for pairwise metrics at this scale.",
            "This ground truth can only detect FRAGMENTATION (one real session split into multiple CYUKTI "
            "campaigns) -- it cannot detect the opposite failure (two DIFFERENT real sessions wrongly merged "
            "into one CYUKTI campaign), because only attacker+victim-distinct scenarios were registered. A "
            "true over-merging test would require two scenarios sharing the same attacker+victim pair but a "
            "different real-world session/objective, which does not exist in this project's registry yet.",
            "Ground truth is AUTO_PROPOSED unless ground_truth_review_status says otherwise.",
        ],
    )
