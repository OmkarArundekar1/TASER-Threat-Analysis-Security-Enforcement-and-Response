"""
ml/evaluate_model.py
=======================
Evaluates a trained XGBoostCampaignClassifier against a held-out
DataFrame slice and produces a JSON-serialisable report: classification
report, confusion matrix, and macro/weighted F1. Does not train
anything — this is strictly measurement.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from dataset_utils import FEATURE_COLUMNS
from train_xgboost import DEFAULT_MODEL_DIR, XGBoostCampaignClassifier


def evaluate(
    model: XGBoostCampaignClassifier,
    df: pd.DataFrame,
    target_column: str,
) -> dict[str, Any]:
    from sklearn.metrics import classification_report, confusion_matrix

    if df.empty:
        raise ValueError("Cannot evaluate on an empty dataset.")

    X = df[FEATURE_COLUMNS].apply(pd.to_numeric, errors="coerce").fillna(0.0).values.astype(np.float32)
    y_true = df[target_column].astype(str).tolist()

    predictions = [model.predict(row) for row in X]
    y_pred = [p["label"] for p in predictions]

    labels = sorted(set(y_true) | set(y_pred))
    report = classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0)
    matrix = confusion_matrix(y_true, y_pred, labels=labels).tolist()

    return {
        "target_column": target_column,
        "n_samples": len(df),
        "labels": labels,
        "classification_report": report,
        "confusion_matrix": matrix,
        "mean_confidence": round(float(np.mean([p["confidence"] for p in predictions])), 4),
    }


def evaluate_and_save(model_path: str, df: pd.DataFrame, target_column: str, out_path: str) -> dict[str, Any]:
    model = XGBoostCampaignClassifier.load(model_path)
    report = evaluate(model, df, target_column)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    from dataset_utils import load_dataset

    df = load_dataset()
    model_path = os.path.join(DEFAULT_MODEL_DIR, "xgb_severity.json")
    out_path = os.path.join(DEFAULT_MODEL_DIR, "eval_severity.json")
    report = evaluate_and_save(model_path, df, "severity", out_path)
    print(json.dumps({k: v for k, v in report.items() if k != "confusion_matrix"}, indent=2))
