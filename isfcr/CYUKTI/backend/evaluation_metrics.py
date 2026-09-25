"""
evaluation_metrics.py
========================
Generic, ground-truth-agnostic metric functions for CYUKTI's evaluation
framework (ACCURACY_EVALUATION.md). Every function here takes real
predicted/actual labels supplied by the caller -- none of them compute,
assume, or fabricate ground truth themselves. Where CYUKTI currently
has no independently-verified ground truth for a task (see
ACCURACY_EVALUATION.md), these functions exist ready to use once such
labels are available; they are not run against fabricated data in this
repository.

Two families:
  - Classification metrics (precision/recall/F1/confusion matrix,
    macro/micro/weighted averaging) -- for MITRE mapping, threat
    classification, severity prediction.
  - Clustering/retrieval metrics (pairwise precision/recall/F1,
    cluster purity, Recall@K, MRR) -- for campaign/operation
    correlation and GNN/RAG retrieval.
"""

from __future__ import annotations

from collections import defaultdict
from itertools import combinations


# ---------------------------------------------------------------- classification

def confusion_counts(y_true: list, y_pred: list, label) -> tuple[int, int, int, int]:
    """(tp, fp, fn, tn) for one label, one-vs-rest."""
    tp = fp = fn = tn = 0
    for actual, predicted in zip(y_true, y_pred):
        if actual == label and predicted == label:
            tp += 1
        elif actual != label and predicted == label:
            fp += 1
        elif actual == label and predicted != label:
            fn += 1
        else:
            tn += 1
    return tp, fp, fn, tn


def precision_recall_f1(y_true: list, y_pred: list, label) -> dict:
    tp, fp, fn, _ = confusion_counts(y_true, y_pred, label)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "support": tp + fn}


def classification_report(y_true: list, y_pred: list) -> dict:
    """Per-label precision/recall/F1/support, plus macro-F1, micro-F1,
    weighted-F1, balanced accuracy, and exact-match accuracy. Labels
    are every distinct value seen in y_true or y_pred -- nothing is
    assumed about the label set in advance."""
    labels = sorted(set(y_true) | set(y_pred), key=str)
    per_label = {label: precision_recall_f1(y_true, y_pred, label) for label in labels}

    n = len(y_true)
    accuracy = sum(1 for a, p in zip(y_true, y_pred) if a == p) / n if n else 0.0

    macro_f1 = sum(m["f1"] for m in per_label.values()) / len(labels) if labels else 0.0
    total_support = sum(m["support"] for m in per_label.values())
    weighted_f1 = (
        sum(m["f1"] * m["support"] for m in per_label.values()) / total_support
        if total_support else 0.0
    )

    total_tp = sum(confusion_counts(y_true, y_pred, label)[0] for label in labels)
    total_fp = sum(confusion_counts(y_true, y_pred, label)[1] for label in labels)
    total_fn = sum(confusion_counts(y_true, y_pred, label)[2] for label in labels)
    micro_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) else 0.0
    micro_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) else 0.0
    micro_f1 = (
        2 * micro_precision * micro_recall / (micro_precision + micro_recall)
        if (micro_precision + micro_recall) else 0.0
    )

    recalls = [m["recall"] for m in per_label.values() if m["support"] > 0]
    balanced_accuracy = sum(recalls) / len(recalls) if recalls else 0.0

    return {
        "per_label": per_label,
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "macro_f1": macro_f1,
        "micro_f1": micro_f1,
        "weighted_f1": weighted_f1,
    }


def false_positive_negative_rates(y_true: list, y_pred: list, positive_label) -> dict:
    """For a binary/one-vs-rest threat classification (e.g. positive_label='THREAT')."""
    tp, fp, fn, tn = confusion_counts(y_true, y_pred, positive_label)
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    fnr = fn / (fn + tp) if (fn + tp) else 0.0
    return {"false_positive_rate": fpr, "false_negative_rate": fnr, "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def multilabel_exact_match_ratio(y_true: list[set], y_pred: list[set]) -> float:
    """For multi-label MITRE technique sets per alert: fraction where
    the predicted technique set is EXACTLY the true set."""
    if not y_true:
        return 0.0
    return sum(1 for t, p in zip(y_true, y_pred) if set(t) == set(p)) / len(y_true)


# ---------------------------------------------------------------- clustering / correlation

def pairwise_precision_recall_f1(true_clusters: dict, pred_clusters: dict) -> dict:
    """true_clusters/pred_clusters: item_id -> cluster_id. Standard
    pairwise clustering evaluation: for every pair of items, do they
    agree on same-cluster/different-cluster between the two labelings?
    Used for campaign/operation correlation -- never against CYUKTI's
    own similarity score as ground truth (that would be circular)."""
    items = sorted(set(true_clusters) & set(pred_clusters))
    if len(items) < 2:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "pairs_evaluated": 0}

    tp = fp = fn = 0
    for a, b in combinations(items, 2):
        same_true = true_clusters[a] == true_clusters[b]
        same_pred = pred_clusters[a] == pred_clusters[b]
        if same_true and same_pred:
            tp += 1
        elif same_pred and not same_true:
            fp += 1
        elif same_true and not same_pred:
            fn += 1

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "pairs_evaluated": tp + fp + fn}


def cluster_purity(true_clusters: dict, pred_clusters: dict) -> float:
    """Fraction of items whose predicted cluster's majority true-label
    matches their own true label -- a measure of over-merging (a
    predicted cluster mixing multiple true campaigns pulls purity down)."""
    items = sorted(set(true_clusters) & set(pred_clusters))
    if not items:
        return 0.0

    by_pred_cluster = defaultdict(list)
    for item in items:
        by_pred_cluster[pred_clusters[item]].append(item)

    correct = 0
    for members in by_pred_cluster.values():
        true_label_counts = defaultdict(int)
        for m in members:
            true_label_counts[true_clusters[m]] += 1
        majority_count = max(true_label_counts.values())
        correct += majority_count

    return correct / len(items)


def campaign_fragmentation(true_clusters: dict, pred_clusters: dict) -> dict:
    """How many predicted clusters a single true campaign got split
    across (over-fragmentation), and vice versa (over-merging).
    Returns per-true-campaign predicted-cluster counts and per-
    predicted-cluster true-campaign counts."""
    items = sorted(set(true_clusters) & set(pred_clusters))
    true_to_pred = defaultdict(set)
    pred_to_true = defaultdict(set)
    for item in items:
        true_to_pred[true_clusters[item]].add(pred_clusters[item])
        pred_to_true[pred_clusters[item]].add(true_clusters[item])

    return {
        "fragmentation_per_true_campaign": {k: len(v) for k, v in true_to_pred.items()},
        "over_merging_per_predicted_cluster": {k: len(v) for k, v in pred_to_true.items()},
    }


# ---------------------------------------------------------------- retrieval

def recall_at_k(relevant_ids: set, ranked_ids: list, k: int) -> float:
    if not relevant_ids:
        return 0.0
    retrieved_top_k = set(ranked_ids[:k])
    return len(retrieved_top_k & relevant_ids) / len(relevant_ids)


def mean_reciprocal_rank(relevant_ids_per_query: list[set], ranked_ids_per_query: list[list]) -> float:
    reciprocal_ranks = []
    for relevant, ranked in zip(relevant_ids_per_query, ranked_ids_per_query):
        rank = next((i + 1 for i, item in enumerate(ranked) if item in relevant), None)
        reciprocal_ranks.append(1.0 / rank if rank else 0.0)
    return sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else 0.0
