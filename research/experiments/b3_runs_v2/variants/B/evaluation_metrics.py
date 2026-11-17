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


def confusion_matrix(y_true: list, y_pred: list, labels: list | None = None) -> dict:
    """Full N x N confusion matrix as a dict of dicts:
    matrix[actual_label][predicted_label] -> count. `labels` fixes the
    row/column order (and can include a label with zero support, e.g.
    a MITRE technique that never appeared); defaults to every distinct
    value seen in y_true or y_pred."""
    if labels is None:
        labels = sorted(set(y_true) | set(y_pred), key=str)
    matrix = {actual: {predicted: 0 for predicted in labels} for actual in labels}
    for actual, predicted in zip(y_true, y_pred):
        if actual in matrix and predicted in matrix[actual]:
            matrix[actual][predicted] += 1
    return {"labels": labels, "matrix": matrix}


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


def precision_at_k(relevant_ids: set, ranked_ids: list, k: int) -> float:
    """For playbook recommendation: of the top-K recommended
    playbooks/actions, what fraction were actually relevant (e.g.
    actually used/approved by an analyst, or actually successful)?
    0.0 if k <= 0 rather than dividing by zero."""
    if k <= 0:
        return 0.0
    retrieved_top_k = ranked_ids[:k]
    if not retrieved_top_k:
        return 0.0
    return sum(1 for item in retrieved_top_k if item in relevant_ids) / len(retrieved_top_k)


def mean_reciprocal_rank(relevant_ids_per_query: list[set], ranked_ids_per_query: list[list]) -> float:
    reciprocal_ranks = []
    for relevant, ranked in zip(relevant_ids_per_query, ranked_ids_per_query):
        rank = next((i + 1 for i, item in enumerate(ranked) if item in relevant), None)
        reciprocal_ranks.append(1.0 / rank if rank else 0.0)
    return sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else 0.0


def ndcg_at_k(graded_relevance: dict, ranked_ids: list, k: int) -> float:
    """Normalized Discounted Cumulative Gain at K, for graded relevance
    (e.g. 0=irrelevant .. 3=highly relevant), used for RAG evaluation
    where a query set has graded rather than binary judgments.
    `graded_relevance`: item_id -> relevance grade (missing = 0)."""
    import math

    def _dcg(ids: list) -> float:
        return sum(
            graded_relevance.get(item, 0) / math.log2(i + 2)
            for i, item in enumerate(ids[:k])
        )

    dcg = _dcg(ranked_ids)
    ideal_order = sorted(graded_relevance, key=lambda i: graded_relevance[i], reverse=True)
    idcg = _dcg(ideal_order)
    return dcg / idcg if idcg > 0 else 0.0


# ---------------------------------------------------------------- clustering agreement (chance-corrected)

def _pair_confusion(true_clusters: dict, pred_clusters: dict) -> tuple[int, int, int, int]:
    """(n11, n10, n01, n00) pair counts shared between two labelings,
    over every pair of items present in both. n11: same cluster in both.
    n10: same in true, different in pred. n01: different in true, same
    in pred. n00: different in both."""
    items = sorted(set(true_clusters) & set(pred_clusters))
    n11 = n10 = n01 = n00 = 0
    for a, b in combinations(items, 2):
        same_true = true_clusters[a] == true_clusters[b]
        same_pred = pred_clusters[a] == pred_clusters[b]
        if same_true and same_pred:
            n11 += 1
        elif same_true and not same_pred:
            n10 += 1
        elif not same_true and same_pred:
            n01 += 1
        else:
            n00 += 1
    return n11, n10, n01, n00


def adjusted_rand_index(true_clusters: dict, pred_clusters: dict) -> float:
    """Chance-corrected clustering agreement, -1..1 (1 = perfect
    agreement, ~0 = agreement expected by random chance). Computed
    directly from the contingency table (no scipy/sklearn dependency),
    equivalent to sklearn.metrics.adjusted_rand_score."""
    items = sorted(set(true_clusters) & set(pred_clusters))
    if len(items) < 2:
        return 0.0

    contingency = defaultdict(lambda: defaultdict(int))
    true_totals = defaultdict(int)
    pred_totals = defaultdict(int)
    for item in items:
        t, p = true_clusters[item], pred_clusters[item]
        contingency[t][p] += 1
        true_totals[t] += 1
        pred_totals[p] += 1

    def _comb2(n: int) -> float:
        return n * (n - 1) / 2.0

    sum_comb_c = sum(_comb2(n) for row in contingency.values() for n in row.values())
    sum_comb_true = sum(_comb2(n) for n in true_totals.values())
    sum_comb_pred = sum(_comb2(n) for n in pred_totals.values())
    total = len(items)
    total_comb = _comb2(total)

    expected_index = (sum_comb_true * sum_comb_pred) / total_comb if total_comb else 0.0
    max_index = (sum_comb_true + sum_comb_pred) / 2.0
    denom = max_index - expected_index
    if denom == 0:
        return 1.0 if sum_comb_c == expected_index else 0.0
    return (sum_comb_c - expected_index) / denom


def adjusted_mutual_information(true_clusters: dict, pred_clusters: dict) -> float:
    """Chance-corrected mutual information between two labelings, using
    the permutation-model expected MI (exact hypergeometric form),
    equivalent to sklearn.metrics.adjusted_mutual_info_score with
    average_method='arithmetic'. Pure Python, no scipy dependency."""
    import math
    from math import lgamma

    items = sorted(set(true_clusters) & set(pred_clusters))
    n = len(items)
    if n == 0:
        return 0.0

    contingency = defaultdict(lambda: defaultdict(int))
    true_totals = defaultdict(int)
    pred_totals = defaultdict(int)
    for item in items:
        t, p = true_clusters[item], pred_clusters[item]
        contingency[t][p] += 1
        true_totals[t] += 1
        pred_totals[p] += 1

    def _entropy(totals: dict) -> float:
        return -sum((c / n) * math.log(c / n) for c in totals.values() if c > 0)

    h_true = _entropy(true_totals)
    h_pred = _entropy(pred_totals)

    mi = 0.0
    for t, row in contingency.items():
        for p, nij in row.items():
            if nij == 0:
                continue
            mi += (nij / n) * math.log((n * nij) / (true_totals[t] * pred_totals[p]))

    if h_true == 0.0 or h_pred == 0.0:
        return 1.0 if mi == 0.0 else 0.0

    def _log_comb(a: int, b: int) -> float:
        if b < 0 or b > a:
            return float("-inf")
        return lgamma(a + 1) - lgamma(b + 1) - lgamma(a - b + 1)

    emi = 0.0
    for a in true_totals.values():
        for b in pred_totals.values():
            for nij in range(max(1, a + b - n), min(a, b) + 1):
                log_term = (
                    _log_comb(a, nij) + _log_comb(n - a, b - nij) - _log_comb(n, b)
                )
                if log_term == float("-inf"):
                    continue
                term_prob = math.exp(log_term)
                if term_prob <= 0:
                    continue
                emi += term_prob * (nij / n) * math.log((n * nij) / (a * b))

    mean_h = (h_true + h_pred) / 2.0
    denom = mean_h - emi
    if denom == 0:
        return 1.0 if (mi - emi) == 0 else 0.0
    return (mi - emi) / denom


# ---------------------------------------------------------------- calibration

def brier_score(confidences: list[float], outcomes: list[bool]) -> float:
    """Mean squared error between a stated confidence (0-1) and the
    binary outcome (1 if correct/true, 0 otherwise). Lower is better;
    0 = perfect calibration and discrimination."""
    if not confidences:
        return 0.0
    return sum((c - (1.0 if o else 0.0)) ** 2 for c, o in zip(confidences, outcomes)) / len(confidences)


def expected_calibration_error(confidences: list[float], outcomes: list[bool], n_bins: int = 10) -> dict:
    """ECE: bins predictions by stated confidence, compares each bin's
    mean confidence to its actual accuracy, weights by bin size.
    Returns the scalar ECE plus the per-bin breakdown for transparency."""
    if not confidences:
        return {"ece": 0.0, "bins": []}

    bins = [[] for _ in range(n_bins)]
    for c, o in zip(confidences, outcomes):
        idx = min(int(c * n_bins), n_bins - 1)
        bins[idx].append((c, o))

    n = len(confidences)
    ece = 0.0
    bin_report = []
    for i, bucket in enumerate(bins):
        if not bucket:
            continue
        mean_conf = sum(c for c, _ in bucket) / len(bucket)
        accuracy = sum(1 for _, o in bucket if o) / len(bucket)
        weight = len(bucket) / n
        ece += weight * abs(mean_conf - accuracy)
        bin_report.append({
            "bin_range": (i / n_bins, (i + 1) / n_bins),
            "n": len(bucket),
            "mean_confidence": mean_conf,
            "accuracy": accuracy,
        })
    return {"ece": ece, "bins": bin_report}


# ---------------------------------------------------------------- confidence intervals

def wilson_confidence_interval(successes: int, n: int, z: float = 1.96) -> dict:
    """Wilson score interval for a binary proportion (e.g. accuracy on
    n samples) -- more defensible than a normal-approximation interval
    at small n, which is exactly the regime most of this project's
    evaluation samples fall into. z=1.96 -> ~95% CI."""
    if n == 0:
        return {"point": 0.0, "low": 0.0, "high": 0.0, "n": 0}
    p_hat = successes / n
    denom = 1 + z ** 2 / n
    center = p_hat + z ** 2 / (2 * n)
    margin = z * ((p_hat * (1 - p_hat) / n + z ** 2 / (4 * n ** 2)) ** 0.5)
    low = (center - margin) / denom
    high = (center + margin) / denom
    return {"point": p_hat, "low": max(0.0, low), "high": min(1.0, high), "n": n}
