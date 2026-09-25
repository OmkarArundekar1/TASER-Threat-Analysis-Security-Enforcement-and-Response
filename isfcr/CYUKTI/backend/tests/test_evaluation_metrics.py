import pytest

from evaluation_metrics import (
    confusion_counts,
    precision_recall_f1,
    classification_report,
    false_positive_negative_rates,
    multilabel_exact_match_ratio,
    pairwise_precision_recall_f1,
    cluster_purity,
    campaign_fragmentation,
    recall_at_k,
    mean_reciprocal_rank,
)


def test_confusion_counts_basic():
    y_true = ["A", "A", "B", "B"]
    y_pred = ["A", "B", "B", "B"]
    tp, fp, fn, tn = confusion_counts(y_true, y_pred, "A")
    assert (tp, fp, fn, tn) == (1, 0, 1, 2)


def test_precision_recall_f1_perfect_prediction():
    y_true = ["A", "B", "A", "B"]
    y_pred = ["A", "B", "A", "B"]
    result = precision_recall_f1(y_true, y_pred, "A")
    assert result == {"precision": 1.0, "recall": 1.0, "f1": 1.0, "support": 2}


def test_precision_recall_f1_no_predictions_for_label_is_zero_not_crash():
    y_true = ["A", "A"]
    y_pred = ["B", "B"]
    result = precision_recall_f1(y_true, y_pred, "A")
    assert result["precision"] == 0.0
    assert result["recall"] == 0.0
    assert result["f1"] == 0.0


def test_classification_report_accuracy_and_macro_f1():
    y_true = ["A", "A", "B", "B"]
    y_pred = ["A", "B", "B", "B"]
    report = classification_report(y_true, y_pred)
    assert report["accuracy"] == 0.75
    assert set(report["per_label"].keys()) == {"A", "B"}
    assert 0.0 <= report["macro_f1"] <= 1.0
    assert 0.0 <= report["weighted_f1"] <= 1.0
    assert 0.0 <= report["micro_f1"] <= 1.0


def test_classification_report_perfect_prediction_all_metrics_are_one():
    y_true = ["A", "B", "A", "B"]
    y_pred = ["A", "B", "A", "B"]
    report = classification_report(y_true, y_pred)
    assert report["accuracy"] == 1.0
    assert report["macro_f1"] == 1.0
    assert report["balanced_accuracy"] == 1.0


def test_false_positive_negative_rates():
    y_true = ["THREAT", "THREAT", "BENIGN", "BENIGN"]
    y_pred = ["THREAT", "BENIGN", "THREAT", "BENIGN"]
    rates = false_positive_negative_rates(y_true, y_pred, "THREAT")
    assert rates["false_negative_rate"] == 0.5  # one THREAT missed
    assert rates["false_positive_rate"] == 0.5  # one BENIGN misclassified


def test_multilabel_exact_match_ratio():
    y_true = [{"T1110"}, {"T1110", "T1078"}]
    y_pred = [{"T1110"}, {"T1110"}]
    assert multilabel_exact_match_ratio(y_true, y_pred) == 0.5


def test_pairwise_precision_recall_f1_perfect_clustering():
    true_clusters = {"e1": "camp_a", "e2": "camp_a", "e3": "camp_b"}
    pred_clusters = {"e1": "camp_a", "e2": "camp_a", "e3": "camp_b"}
    result = pairwise_precision_recall_f1(true_clusters, pred_clusters)
    assert result["precision"] == 1.0
    assert result["recall"] == 1.0


def test_pairwise_precision_recall_f1_over_merging_hurts_precision():
    true_clusters = {"e1": "camp_a", "e2": "camp_a", "e3": "camp_b"}
    pred_clusters = {"e1": "camp_x", "e2": "camp_x", "e3": "camp_x"}  # everything merged
    result = pairwise_precision_recall_f1(true_clusters, pred_clusters)
    assert result["recall"] == 1.0  # the true-same pair is still predicted same
    assert result["precision"] < 1.0  # but false same-cluster pairs were introduced


def test_pairwise_precision_recall_f1_over_fragmenting_hurts_recall():
    true_clusters = {"e1": "camp_a", "e2": "camp_a", "e3": "camp_b"}
    pred_clusters = {"e1": "camp_a", "e2": "camp_b_split", "e3": "camp_b"}  # camp_a split apart
    result = pairwise_precision_recall_f1(true_clusters, pred_clusters)
    assert result["recall"] < 1.0


def test_cluster_purity_perfect_is_one():
    true_clusters = {"e1": "a", "e2": "a", "e3": "b"}
    pred_clusters = {"e1": "a", "e2": "a", "e3": "b"}
    assert cluster_purity(true_clusters, pred_clusters) == 1.0


def test_cluster_purity_mixed_cluster_reduces_purity():
    true_clusters = {"e1": "a", "e2": "a", "e3": "b"}
    pred_clusters = {"e1": "x", "e2": "x", "e3": "x"}  # one predicted cluster mixes a's and b
    assert cluster_purity(true_clusters, pred_clusters) == pytest.approx(2 / 3)


def test_campaign_fragmentation_reports_split_and_merge():
    true_clusters = {"e1": "a", "e2": "a", "e3": "b"}
    pred_clusters = {"e1": "p1", "e2": "p2", "e3": "p2"}
    result = campaign_fragmentation(true_clusters, pred_clusters)
    assert result["fragmentation_per_true_campaign"]["a"] == 2  # split into p1 and p2
    assert result["over_merging_per_predicted_cluster"]["p2"] == 2  # p2 mixes a and b


def test_recall_at_k():
    relevant = {"camp_1", "camp_2"}
    ranked = ["camp_5", "camp_1", "camp_3", "camp_2"]
    assert recall_at_k(relevant, ranked, k=2) == 0.5
    assert recall_at_k(relevant, ranked, k=4) == 1.0


def test_recall_at_k_no_relevant_items_is_zero_not_crash():
    assert recall_at_k(set(), ["a", "b"], k=2) == 0.0


def test_mean_reciprocal_rank():
    relevant = [{"camp_1"}, {"camp_2"}]
    ranked = [["camp_5", "camp_1"], ["camp_2", "camp_5"]]
    # query 1: hit at rank 2 -> 1/2; query 2: hit at rank 1 -> 1/1
    assert mean_reciprocal_rank(relevant, ranked) == pytest.approx((0.5 + 1.0) / 2)


def test_mean_reciprocal_rank_no_hit_scores_zero_for_that_query():
    relevant = [{"camp_9"}]
    ranked = [["camp_1", "camp_2"]]
    assert mean_reciprocal_rank(relevant, ranked) == 0.0
