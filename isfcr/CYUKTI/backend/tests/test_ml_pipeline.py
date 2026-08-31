"""
Tests the dataset -> XGBoost training -> evaluation -> serialization ->
inference round trip using the SYNTHETIC fixture generator
(ml/data_prep/generate_synthetic_dataset.py). These tests validate that
the pipeline mechanics work end to end; they say nothing about real
detection accuracy, since no real campaign telemetry exists yet.
"""

import shutil

import pandas as pd
import pytest
from dataclasses import asdict

from ml.data_prep.generate_synthetic_dataset import generate_synthetic_records
from ml.dataset_utils import FEATURE_COLUMNS, validate_dataset_size
from ml.train_xgboost import XGBoostCampaignClassifier
from ml.evaluate_model import evaluate


@pytest.fixture(scope="module")
def synthetic_df():
    records = generate_synthetic_records(300, seed=7)
    return pd.DataFrame([asdict(r) for r in records])


def test_validate_dataset_size_rejects_too_few_rows():
    with pytest.raises(ValueError, match="Not enough labeled campaign records"):
        validate_dataset_size(pd.DataFrame([{"severity": "Low"}]))


def test_synthetic_dataset_has_all_expected_columns(synthetic_df):
    for col in FEATURE_COLUMNS:
        assert col in synthetic_df.columns
    assert set(synthetic_df["severity"].unique()) <= {"Low", "Medium", "High", "Critical"}


def test_train_produces_multiclass_model(tmp_path, synthetic_df):
    clf = XGBoostCampaignClassifier()
    result = clf.train(synthetic_df, target_column="severity", save_dir=str(tmp_path))

    assert result.target_column == "severity"
    assert set(result.classes) <= {"Low", "Medium", "High", "Critical"}
    assert result.train_rows + result.val_rows == len(synthetic_df)
    assert 0.0 <= result.val_accuracy <= 1.0
    assert 0.0 <= result.val_macro_f1 <= 1.0
    assert len(result.feature_importance) == len(FEATURE_COLUMNS)
    # feature_importance should be sorted descending
    values = list(result.feature_importance.values())
    assert values == sorted(values, reverse=True)


def test_train_rejects_single_class_target(synthetic_df):
    df = synthetic_df.copy()
    df["severity"] = "Low"  # collapse to one class
    clf = XGBoostCampaignClassifier()
    with pytest.raises(ValueError, match="only one class"):
        clf.train(df, target_column="severity")


def test_save_load_roundtrip_predicts_consistently(tmp_path, synthetic_df):
    clf = XGBoostCampaignClassifier()
    clf.train(synthetic_df, target_column="severity", save_dir=str(tmp_path))

    sample = synthetic_df[FEATURE_COLUMNS].iloc[0].values
    direct_prediction = clf.predict(sample)

    loaded = XGBoostCampaignClassifier.load(tmp_path / "xgb_severity.json")
    loaded_prediction = loaded.predict(sample)

    assert direct_prediction["label"] == loaded_prediction["label"]
    assert direct_prediction["confidence"] == pytest.approx(loaded_prediction["confidence"], abs=1e-6)


def test_evaluate_returns_well_formed_report(tmp_path, synthetic_df):
    clf = XGBoostCampaignClassifier()
    clf.train(synthetic_df, target_column="severity", save_dir=str(tmp_path))

    holdout = synthetic_df.sample(20, random_state=1)
    report = evaluate(clf, holdout, target_column="severity")

    assert report["n_samples"] == 20
    assert len(report["confusion_matrix"]) == len(report["labels"])
    assert "macro avg" in report["classification_report"]
    assert 0.0 <= report["mean_confidence"] <= 1.0


@pytest.fixture(autouse=True, scope="module")
def _cleanup_synthetic_artifacts():
    yield
    shutil.rmtree("ml/datasets_synthetic", ignore_errors=True)
