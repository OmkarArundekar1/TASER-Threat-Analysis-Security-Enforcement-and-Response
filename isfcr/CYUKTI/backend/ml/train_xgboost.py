"""
ml/train_xgboost.py
======================
Trains an XGBoost classifier on the accumulated campaign dataset
(feature_schema.CampaignDatasetRecord rows collected via
ml.dataset_builder as real campaigns resolve).

Default target: `severity` (Low/Medium/High/Critical) — the clearest
campaign-level outcome to predict from the graph/MITRE/CTI/detection
feature vector. `prediction_correct` / `attribution_correct` are also
supported as binary targets via `target_column`, since the same
pipeline (imbalance handling, calibration, evaluation) applies.

Refuses to train on too few rows (see dataset_utils.MIN_TRAINING_ROWS)
rather than silently producing a model with no statistical basis.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from dataset_utils import FEATURE_COLUMNS, validate_dataset_size

logger = logging.getLogger(__name__)

# Anchored to ml/'s own directory (matching ssl_pipeline.py/ssft.py's
# convention) rather than a bare relative "models" — a bare relative
# path resolves differently depending on the caller's cwd, which caused
# a real bug: dashboard_api.py always looks for the trained model at
# backend/ml/models/, but training run with cwd=backend/ saved to
# backend/models/ instead, so the API kept reporting "no model trained"
# even after a real training run had actually succeeded.
DEFAULT_MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")


@dataclass
class TrainingResult:
    target_column: str
    classes: list[str]
    train_rows: int
    val_rows: int
    val_macro_f1: float
    val_accuracy: float
    feature_importance: dict[str, float]
    model_path: str


class XGBoostCampaignClassifier:
    """
    Usage (training)::
        clf = XGBoostCampaignClassifier()
        result = clf.train(df, target_column="severity")

    Usage (inference)::
        clf = XGBoostCampaignClassifier.load("models/xgb_severity.json")
        clf.predict(feature_vector)  # -> {"label": ..., "confidence": ..., "probabilities": {...}}
    """

    def __init__(self) -> None:
        self._model: Any = None
        self._calibrated: Any = None
        self._label_encoder: Any = None
        self._classes: list[str] = []
        self._target_column: str = ""

    def train(
        self,
        df: pd.DataFrame,
        target_column: str = "severity",
        save_dir: str = DEFAULT_MODEL_DIR,
        n_estimators: int = 200,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        calibrate: bool = True,
        min_rows: int | None = None,
    ) -> TrainingResult:
        import xgboost as xgb
        from sklearn.calibration import CalibratedClassifierCV
        from sklearn.metrics import accuracy_score, classification_report
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import LabelEncoder
        from sklearn.utils.class_weight import compute_sample_weight

        validate_dataset_size(df, min_rows) if min_rows is not None else validate_dataset_size(df)

        if target_column not in df.columns:
            raise ValueError(f"target_column '{target_column}' not found in dataset columns")

        missing_features = [c for c in FEATURE_COLUMNS if c not in df.columns]
        if missing_features:
            raise ValueError(f"Dataset is missing expected feature columns: {missing_features}")

        X = df[FEATURE_COLUMNS].apply(pd.to_numeric, errors="coerce").fillna(0.0).values.astype(np.float32)
        y_raw = df[target_column].astype(str)

        le = LabelEncoder()
        y = le.fit_transform(y_raw)

        if len(le.classes_) < 2:
            raise ValueError(
                f"'{target_column}' has only one class ({le.classes_}) in this dataset — "
                "nothing to classify. Need at least 2 distinct label values."
            )

        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42,
            stratify=y if min(np.bincount(y)) >= 2 else None,
        )

        sample_weight = compute_sample_weight("balanced", y_train)

        model = xgb.XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            objective="multi:softprob" if len(le.classes_) > 2 else "binary:logistic",
            eval_metric="mlogloss" if len(le.classes_) > 2 else "logloss",
            random_state=42,
            n_jobs=-1,
        )
        model.fit(X_train, y_train, sample_weight=sample_weight)

        if calibrate:
            calibrated = CalibratedClassifierCV(model, method="sigmoid", cv="prefit")
            calibrated.fit(X_val, y_val)
        else:
            calibrated = model

        y_pred = calibrated.predict(X_val)
        report = classification_report(
            y_val, y_pred, labels=list(range(len(le.classes_))),
            target_names=le.classes_, output_dict=True, zero_division=0,
        )

        importances = model.feature_importances_.tolist()
        feature_importance = dict(
            sorted(zip(FEATURE_COLUMNS, importances), key=lambda kv: kv[1], reverse=True)
        )

        self._model = model
        self._calibrated = calibrated
        self._label_encoder = le
        self._classes = list(le.classes_)
        self._target_column = target_column

        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        model_file = save_path / f"xgb_{target_column}.json"
        self.save(model_file)

        return TrainingResult(
            target_column=target_column,
            classes=self._classes,
            train_rows=len(X_train),
            val_rows=len(X_val),
            val_macro_f1=report["macro avg"]["f1-score"],
            val_accuracy=accuracy_score(y_val, y_pred),
            feature_importance=feature_importance,
            model_path=str(model_file),
        )

    def predict(self, feature_vector) -> dict[str, Any]:
        if self._calibrated is None:
            raise RuntimeError("Model not loaded/trained. Call train() or load().")

        x = np.asarray(feature_vector, dtype=np.float32).reshape(1, -1)
        x = np.nan_to_num(x, nan=0.0, posinf=1e6, neginf=-1e6)

        proba = self._calibrated.predict_proba(x)[0]
        pred_idx = int(np.argmax(proba))

        return {
            "label": self._classes[pred_idx],
            "confidence": round(float(proba[pred_idx]), 4),
            "probabilities": {cls: round(float(p), 4) for cls, p in zip(self._classes, proba)},
        }

    def save(self, path: str | Path) -> None:
        import joblib

        path = Path(path)
        self._model.save_model(str(path))
        meta_path = path.with_suffix(".meta.joblib")
        joblib.dump(
            {"classes": self._classes, "target_column": self._target_column, "calibrated": self._calibrated},
            meta_path,
        )

    @classmethod
    def load(cls, path: str | Path) -> "XGBoostCampaignClassifier":
        import joblib
        import xgboost as xgb

        path = Path(path)
        meta_path = path.with_suffix(".meta.joblib")
        meta = joblib.load(meta_path)

        obj = cls()
        model = xgb.XGBClassifier()
        model.load_model(str(path))
        obj._model = model
        obj._calibrated = meta["calibrated"]
        obj._classes = meta["classes"]
        obj._target_column = meta["target_column"]
        return obj


if __name__ == "__main__":
    from dataset_utils import load_dataset

    df = load_dataset()
    clf = XGBoostCampaignClassifier()
    result = clf.train(df, target_column="severity")
    print(json.dumps(result.__dict__, indent=2, default=str))
