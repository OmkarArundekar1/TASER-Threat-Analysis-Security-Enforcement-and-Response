"""
classifiers/lightgbm_classifier.py
====================================
LightGBM multi-class attack classifier trained on CICIDS2017 labels.

Design:
    - Trains on ALL day-CSVs in datasets/ directory for full label coverage
    - 7 classes: BENIGN, DDoS, PortScan, WebAttack, BruteForce, Infiltration, Bot
    - class_weight='balanced' → handles severe class imbalance
    - Inference: <1ms per sample via LightGBM native
    - NO IP features — trained on same statistical features as autoencoder
    - Saves model + label encoder to models/
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np

logger = logging.getLogger(__name__)

_LABEL_MAP = {
    "benign": "BENIGN",
    "ddos": "DDoS",
    "portscan": "PortScan",
    "web attack": "WebAttack",
    "web attack – brute force": "WebAttack",
    "web attack – xss": "WebAttack",
    "web attack – sql injection": "WebAttack",
    "brute force": "BruteForce",
    "ssh-patator": "BruteForce",
    "ftp-patator": "BruteForce",
    "infiltration": "Infiltration",
    "bot": "Bot",
    "heartbleed": "Exploit",
    "dos hulk": "DoS",
    "dos goldeneye": "DoS",
    "dos slowloris": "DoS",
    "dos slowhttptest": "DoS",
}


class AttackClassifier:
    """
    LightGBM-based multi-class network attack classifier.

    Usage (training)::
        clf = AttackClassifier()
        clf.train(datasets_dir="datasets/", save_dir="models/")

    Usage (inference)::
        clf = AttackClassifier.load("models/lgbm_classifier.pkl")
        result = clf.classify(feature_vector)
        # → {"label": "DDoS", "confidence": 0.97, "probabilities": {...}}
    """

    def __init__(self) -> None:
        self._model: Any = None
        self._label_encoder: Any = None
        self._classes: list[str] = []

    # ── Training ──────────────────────────────────────────────────────────────
    def train(
        self,
        datasets_dir: str = "datasets/",
        save_dir: str = "models/",
        n_estimators: int = 300,
        learning_rate: float = 0.05,
        max_depth: int = 8,
        num_leaves: int = 63,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
    ) -> dict[str, Any]:
        """
        Load CICIDS2017 CSVs, extract features + labels, train LightGBM.
        Drops ALL IP/identifier columns before training.
        """
        try:
            import lightgbm as lgb
            from sklearn.preprocessing import LabelEncoder
            from sklearn.model_selection import train_test_split
            from sklearn.metrics import classification_report
            import pandas as pd
        except ImportError as e:
            raise RuntimeError(f"Missing dependency: {e}. Run: pip install lightgbm scikit-learn")

        datasets_path = Path(datasets_dir)
        csv_files = list(datasets_path.glob("*.csv"))
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in {datasets_dir}")

        logger.info("Loading %d CICIDS2017 files...", len(csv_files))

        dfs = []
        for f in csv_files:
            try:
                df = pd.read_csv(f, low_memory=False)
                df.columns = df.columns.str.strip()
                if "Label" in df.columns:
                    dfs.append(df)
            except Exception as e:
                logger.warning("Skipping %s: %s", f.name, e)

        if not dfs:
            raise RuntimeError("No valid CSVs with 'Label' column found.")

        data = pd.concat(dfs, ignore_index=True)
        logger.info("Total rows: %d", len(data))

        # Drop IP / identifier columns
        drop_cols = ["Flow ID", "Source IP", "Destination IP", "Timestamp", "Label"]
        labels_raw = data["Label"].str.lower().str.strip()
        labels = labels_raw.map(lambda x: _LABEL_MAP.get(x, "Other"))

        data = data.drop(columns=[c for c in drop_cols if c in data.columns])
        data = data.apply(pd.to_numeric, errors="coerce")
        data.replace([np.inf, -np.inf], np.nan, inplace=True)
        data.fillna(0.0, inplace=True)

        X = data.values.astype(np.float32)
        le = LabelEncoder()
        y = le.fit_transform(labels)

        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42, stratify=y)

        model = lgb.LGBMClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            num_leaves=num_leaves,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            class_weight="balanced",
            n_jobs=-1,
            random_state=42,
            verbose=-1,
        )
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)])

        y_pred = model.predict(X_val)
        report = classification_report(y_val, y_pred, target_names=le.classes_, output_dict=True)
        logger.info("Validation macro F1: %.4f", report["macro avg"]["f1-score"])

        self._model = model
        self._label_encoder = le
        self._classes = list(le.classes_)

        # Save
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        joblib.dump((model, le), save_path / "lgbm_classifier.pkl")
        logger.info("Model saved to %s", save_path / "lgbm_classifier.pkl")

        return {
            "classes": self._classes,
            "val_macro_f1": report["macro avg"]["f1-score"],
            "train_samples": len(X_train),
            "val_samples": len(X_val),
        }

    # ── Inference ─────────────────────────────────────────────────────────────
    def classify(self, feature_vector: list[float] | np.ndarray) -> dict[str, Any]:
        """Classify a single feature vector. Returns JSON-serialisable dict."""
        if self._model is None:
            raise RuntimeError("Model not loaded. Call train() or load().")

        x = np.asarray(feature_vector, dtype=np.float32).reshape(1, -1)
        x = np.nan_to_num(x, nan=0.0, posinf=1e6)

        pred_idx = int(self._model.predict(x)[0])
        proba = self._model.predict_proba(x)[0].tolist()

        label = self._classes[pred_idx]
        confidence = round(float(proba[pred_idx]), 4)

        return {
            "label": label,
            "confidence": confidence,
            "probabilities": {cls: round(float(p), 4) for cls, p in zip(self._classes, proba)},
        }

    def classify_batch(self, X: np.ndarray) -> list[dict[str, Any]]:
        """Classify multiple feature vectors at once."""
        if self._model is None:
            raise RuntimeError("Model not loaded.")
        X = np.nan_to_num(np.asarray(X, dtype=np.float32), nan=0.0, posinf=1e6)
        pred_idx = self._model.predict(X).tolist()
        probas = self._model.predict_proba(X).tolist()

        results = []
        for i, (p_idx, proba) in enumerate(zip(pred_idx, probas)):
            label = self._classes[p_idx]
            results.append({
                "label": label,
                "confidence": round(float(proba[p_idx]), 4),
                "probabilities": {cls: round(float(p), 4) for cls, p in zip(self._classes, proba)},
            })
        return results

    # ── Persistence ───────────────────────────────────────────────────────────
    @classmethod
    def load(cls, path: str = "models/lgbm_classifier.pkl") -> "AttackClassifier":
        obj = cls()
        model, le = joblib.load(path)
        obj._model = model
        obj._label_encoder = le
        obj._classes = list(le.classes_)
        logger.info("AttackClassifier loaded from %s | classes: %s", path, obj._classes)
        return obj
