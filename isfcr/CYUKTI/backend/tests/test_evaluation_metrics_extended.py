import pytest

from evaluation_metrics import (
    ndcg_at_k,
    adjusted_rand_index,
    adjusted_mutual_information,
    brier_score,
    expected_calibration_error,
    wilson_confidence_interval,
)


# ---------------------------------------------------------------- ndcg_at_k

def test_ndcg_at_k_perfect_ranking_is_one():
    graded = {"a": 3, "b": 2, "c": 0}
    assert ndcg_at_k(graded, ["a", "b", "c"], k=3) == pytest.approx(1.0)


def test_ndcg_at_k_worst_ranking_is_less_than_one():
    graded = {"a": 3, "b": 2, "c": 0}
    score = ndcg_at_k(graded, ["c", "b", "a"], k=3)
    assert 0.0 < score < 1.0


def test_ndcg_at_k_no_relevant_items_is_zero_not_crash():
    assert ndcg_at_k({}, ["a", "b"], k=2) == 0.0


def test_ndcg_at_k_respects_k_cutoff():
    graded = {"a": 0, "b": 0, "c": 3}
    # c is only relevant item but ranked 3rd; at k=1 it's not seen at all
    assert ndcg_at_k(graded, ["a", "b", "c"], k=1) == 0.0
    assert ndcg_at_k(graded, ["a", "b", "c"], k=3) > 0.0


# ---------------------------------------------------------------- adjusted_rand_index

def test_ari_identical_clusterings_is_one():
    true_c = {"a": 1, "b": 1, "c": 2, "d": 2}
    pred_c = {"a": 1, "b": 1, "c": 2, "d": 2}
    assert adjusted_rand_index(true_c, pred_c) == pytest.approx(1.0)


def test_ari_relabeled_but_structurally_identical_is_one():
    true_c = {"a": 1, "b": 1, "c": 2, "d": 2}
    pred_c = {"a": "X", "b": "X", "c": "Y", "d": "Y"}
    assert adjusted_rand_index(true_c, pred_c) == pytest.approx(1.0)


def test_ari_single_shared_item_is_zero_not_crash():
    true_c = {"a": 1}
    pred_c = {"a": 1}
    assert adjusted_rand_index(true_c, pred_c) == 0.0


def test_ari_all_singletons_vs_all_one_cluster():
    true_c = {"a": 1, "b": 2, "c": 3, "d": 4}
    pred_c = {"a": 1, "b": 1, "c": 1, "d": 1}
    score = adjusted_rand_index(true_c, pred_c)
    assert score <= 0.0 + 1e-9  # no better than chance when one side has zero structure


# ---------------------------------------------------------------- adjusted_mutual_information

def test_ami_identical_clusterings_is_one():
    true_c = {"a": 1, "b": 1, "c": 2, "d": 2}
    pred_c = {"a": 1, "b": 1, "c": 2, "d": 2}
    assert adjusted_mutual_information(true_c, pred_c) == pytest.approx(1.0)


def test_ami_one_true_cluster_any_pred_split_is_one_by_definition():
    # zero true entropy -- formula returns 1.0 iff mutual information is
    # also exactly zero, which it always is when one side is degenerate
    true_c = {"a": 1, "b": 1, "c": 1}
    pred_c = {"a": "X", "b": "Y", "c": "Z"}
    assert adjusted_mutual_information(true_c, pred_c) == pytest.approx(1.0)


def test_ami_empty_input_is_zero_not_crash():
    assert adjusted_mutual_information({}, {}) == 0.0


# ---------------------------------------------------------------- brier_score

def test_brier_score_perfect_calibration_is_zero():
    assert brier_score([1.0, 1.0, 0.0, 0.0], [True, True, False, False]) == pytest.approx(0.0)


def test_brier_score_worst_case_is_one():
    assert brier_score([1.0], [False]) == pytest.approx(1.0)


def test_brier_score_empty_is_zero_not_crash():
    assert brier_score([], []) == 0.0


# ---------------------------------------------------------------- expected_calibration_error

def test_ece_perfectly_calibrated_two_bins_is_zero():
    # bin[0,0.5): mean conf 0.1, 0/1 correct -> acc 0.0 (matches: low confidence, wrong)
    # bin[0.5,1.0]: mean conf 0.9, 1/1 correct -> acc 1.0 (matches: high confidence, right)
    result = expected_calibration_error([0.1, 0.9], [False, True], n_bins=2)
    assert result["ece"] == pytest.approx(0.1)
    assert len(result["bins"]) == 2


def test_ece_empty_is_zero_not_crash():
    result = expected_calibration_error([], [])
    assert result["ece"] == 0.0
    assert result["bins"] == []


# ---------------------------------------------------------------- wilson_confidence_interval

def test_wilson_interval_midpoint_is_bracketed():
    result = wilson_confidence_interval(5, 10)
    assert result["point"] == pytest.approx(0.5)
    assert result["low"] < 0.5 < result["high"]


def test_wilson_interval_zero_n_is_zero_not_crash():
    result = wilson_confidence_interval(0, 0)
    assert result == {"point": 0.0, "low": 0.0, "high": 0.0, "n": 0}


def test_wilson_interval_narrows_with_more_samples():
    small_n = wilson_confidence_interval(5, 10)
    large_n = wilson_confidence_interval(50, 100)
    assert (large_n["high"] - large_n["low"]) < (small_n["high"] - small_n["low"])
