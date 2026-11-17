"""
ml/gnn/xgboost_ablation.py
==============================
Ablation: does adding the GNN's learned embedding z_G to CYUKTI's
existing 57 XGBoost campaign features change severity-prediction
performance? See ../../GNN_XGBOOST_ABLATION.md for the full write-up;
this module is the experiment itself.

Leakage design (the single most important property of this module):
evaluation is Leave-One-Attacker-Group-Out (LOGO) cross-validation. For
each fold, a FRESH GraphAutoencoder is fit (ml.gnn.train_autoencoder.fit_autoencoder)
using ONLY that fold's training campaigns -- feature standardization
statistics, pos_weight, and every model parameter are fit exclusively
on the training split. The held-out fold's campaigns are embedded by
passing their graphs through the already-trained encoder
(ml.gnn.train_autoencoder.embed_samples, inference only, no gradient) --
never used to fit the encoder, the standardization, or the XGBoost
model for that fold. This mirrors, at the embedding-generation level,
exactly what "held-out test data" already means for the XGBoost side.

Never reads SIMILAR_TO/RESEMBLES. Never modifies
ml/models/xgb_severity.json or any production artifact -- every model
trained here is in-memory only, for this experiment's metrics.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from campaign_graphs import CampaignGraphSample, build_real_campaign_dataset
from dataset_utils import FEATURE_COLUMNS
from train_autoencoder import embed_samples, fit_autoencoder, get_campaign_attacker_ips

CSV_PATH = "ml/datasets/campaign_dataset.csv"
EMBEDDING_DIM = 8


@dataclass
class DatasetMapping:
    csv_campaigns: int
    live_gnn_campaigns: int
    usable_campaigns: int
    csv_only_excluded: list[str]
    gnn_only_excluded: list[str]


def load_ablation_dataset(csv_path: str = CSV_PATH) -> tuple[pd.DataFrame, dict[str, CampaignGraphSample], DatasetMapping]:
    """Traces the real mapping between the persisted XGBoost dataset
    (ml/datasets/campaign_dataset.csv, ~60 rows -- the only source of
    the 7 similarity features that require live alert-time context and
    cannot be recomputed after the fact) and the live 71-campaign GNN
    extraction. Does not assume 1:1 -- reports and excludes mismatches
    explicitly rather than silently dropping them."""
    df = pd.read_csv(csv_path)
    gnn_samples = {s.campaign_id: s for s in build_real_campaign_dataset()}

    csv_ids = set(df["campaign_id"])
    gnn_ids = set(gnn_samples)
    usable_ids = sorted(csv_ids & gnn_ids)

    mapping = DatasetMapping(
        csv_campaigns=len(csv_ids),
        live_gnn_campaigns=len(gnn_ids),
        usable_campaigns=len(usable_ids),
        csv_only_excluded=sorted(csv_ids - gnn_ids),
        gnn_only_excluded=sorted(gnn_ids - csv_ids),
    )

    df = df[df["campaign_id"].isin(usable_ids)].reset_index(drop=True)
    samples = {cid: gnn_samples[cid] for cid in usable_ids}
    return df, samples, mapping


def build_attacker_groups(campaign_ids: list[str]) -> dict[str, list[str]]:
    attacker_of = get_campaign_attacker_ips(campaign_ids)
    groups: dict[str, list[str]] = defaultdict(list)
    for cid in campaign_ids:
        groups[attacker_of.get(cid, f"__no_attacker__{cid}")].append(cid)
    return dict(groups)


def _fit_and_eval_xgb(X_train: np.ndarray, y_train_str: list[str], X_test: np.ndarray,
                       y_test_str: list[str], all_labels: list[str]) -> dict:
    """Same XGBoost hyperparameters as production (ml/train_xgboost.py's
    XGBoostCampaignClassifier.train defaults: n_estimators=200,
    max_depth=6, learning_rate=0.1, random_state=42, n_jobs=-1,
    "balanced" sample weighting) -- reused identically for BASELINE and
    AUGMENTED conditions, the only difference between them is X's
    column count. Deviates from production in one disclosed way: no
    CalibratedClassifierCV step, since that requires yet another
    internal split on data already this scarce for some folds -- this
    experiment compares raw predict_proba/predict, identically for
    both conditions, so the comparison stays fair even though it
    differs from the deployed artifact's own calibration step.

    Per-fold label handling: a fold's TRAINING labels may not contain
    every class in `all_labels` (this repository's real data has at
    least one fold where Medium never appears in training -- see
    GNN_XGBOOST_ABLATION.md Section 6). XGBoost's sklearn wrapper
    requires contiguous 0..k-1 labels PRESENT IN y_train, so a
    per-fold LabelEncoder (fit on y_train only) is used for the
    model itself, then predictions are mapped back to string labels
    before scoring against `all_labels` -- a class absent from training
    correctly scores 0 precision/recall/F1 on that fold's test set
    (an honest result, not suppressed or worked around)."""
    import xgboost as xgb
    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        classification_report,
        confusion_matrix,
        f1_score,
    )
    from sklearn.preprocessing import LabelEncoder
    from sklearn.utils.class_weight import compute_sample_weight

    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train_str)
    n_classes_present = len(le.classes_)
    sample_weight = compute_sample_weight("balanced", y_train_enc)

    model = xgb.XGBClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.1,
        objective="multi:softprob" if n_classes_present > 2 else "binary:logistic",
        eval_metric="mlogloss" if n_classes_present > 2 else "logloss",
        random_state=42, n_jobs=-1,
    )
    model.fit(X_train, y_train_enc, sample_weight=sample_weight)
    y_pred_enc = model.predict(X_test)
    y_pred_str = list(le.inverse_transform(y_pred_enc))

    report = classification_report(
        y_test_str, y_pred_str, labels=all_labels, output_dict=True, zero_division=0,
    )
    cm = confusion_matrix(y_test_str, y_pred_str, labels=all_labels)

    return {
        "accuracy": float(accuracy_score(y_test_str, y_pred_str)),
        "macro_f1": float(f1_score(y_test_str, y_pred_str, labels=all_labels, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_test_str, y_pred_str, labels=all_labels, average="weighted", zero_division=0)),
        "balanced_accuracy": (
            float(balanced_accuracy_score(y_test_str, y_pred_str)) if len(set(y_test_str)) > 1 else float("nan")
        ),
        "per_class": {
            label: {
                "precision": report[label]["precision"],
                "recall": report[label]["recall"],
                "f1": report[label]["f1-score"],
                "support": report[label]["support"],
            }
            for label in all_labels
        },
        "confusion_matrix": cm.tolist(),
        "classes_present_in_train": list(le.classes_),
        "predictions": dict(zip(range(len(y_pred_str)), y_pred_str)),  # positional; caller re-attaches campaign_id
    }


@dataclass
class FoldResult:
    held_out_group: str
    test_campaign_ids: list[str]
    train_size: int
    test_size: int
    train_label_counts: dict
    test_label_counts: dict
    baseline: dict
    augmented: dict


@dataclass
class AblationResult:
    mapping: DatasetMapping
    all_labels: list[str]
    fold_results: list[FoldResult] = field(default_factory=list)
    pooled_baseline: dict | None = None
    pooled_augmented: dict | None = None


def run_logo_ablation(seed: int = 42) -> AblationResult:
    df, samples, mapping = load_ablation_dataset()
    campaign_ids = df["campaign_id"].tolist()
    groups = build_attacker_groups(campaign_ids)
    all_labels = sorted(df["severity"].unique())

    df_by_id = df.set_index("campaign_id")

    fold_results: list[FoldResult] = []
    oof_true: dict[str, str] = {}
    oof_baseline_pred: dict[str, str] = {}
    oof_augmented_pred: dict[str, str] = {}

    for held_out_group in sorted(groups):
        test_ids = groups[held_out_group]
        train_ids = [c for c in campaign_ids if c not in test_ids]

        train_samples = [samples[c] for c in train_ids]
        test_samples = [samples[c] for c in test_ids]

        # --- fold-safe GNN embedding: fit encoder on TRAIN graphs only ---
        model, feature_mean, feature_std, _pos_weight, _history = fit_autoencoder(
            train_samples, hidden_dim=16, embedding_dim=EMBEDDING_DIM, num_layers=2,
            epochs=200, lr=0.01, seed=seed,
        )
        train_embeddings = embed_samples(model, train_samples, feature_mean, feature_std)
        test_embeddings = embed_samples(model, test_samples, feature_mean, feature_std)  # inference only

        # --- feature matrices ---
        X_train_base = df_by_id.loc[train_ids, FEATURE_COLUMNS].apply(
            pd.to_numeric, errors="coerce",
        ).fillna(0.0).values.astype(np.float32)
        X_test_base = df_by_id.loc[test_ids, FEATURE_COLUMNS].apply(
            pd.to_numeric, errors="coerce",
        ).fillna(0.0).values.astype(np.float32)

        train_emb_mat = np.stack([train_embeddings[c].detach().numpy() for c in train_ids]).astype(np.float32)
        test_emb_mat = np.stack([test_embeddings[c].detach().numpy() for c in test_ids]).astype(np.float32)

        X_train_aug = np.concatenate([X_train_base, train_emb_mat], axis=1)
        X_test_aug = np.concatenate([X_test_base, test_emb_mat], axis=1)

        y_train = df_by_id.loc[train_ids, "severity"].tolist()
        y_test = df_by_id.loc[test_ids, "severity"].tolist()

        baseline = _fit_and_eval_xgb(X_train_base, y_train, X_test_base, y_test, all_labels)
        augmented = _fit_and_eval_xgb(X_train_aug, y_train, X_test_aug, y_test, all_labels)

        for i, cid in enumerate(test_ids):
            oof_true[cid] = y_test[i]
            oof_baseline_pred[cid] = baseline["predictions"][i]
            oof_augmented_pred[cid] = augmented["predictions"][i]

        from collections import Counter
        fold_results.append(FoldResult(
            held_out_group=held_out_group,
            test_campaign_ids=test_ids,
            train_size=len(train_ids),
            test_size=len(test_ids),
            train_label_counts=dict(Counter(y_train)),
            test_label_counts=dict(Counter(y_test)),
            baseline=baseline,
            augmented=augmented,
        ))

    pooled_ids = list(oof_true.keys())
    pooled_baseline = _score_predictions(
        [oof_true[c] for c in pooled_ids], [oof_baseline_pred[c] for c in pooled_ids], all_labels,
    )
    pooled_augmented = _score_predictions(
        [oof_true[c] for c in pooled_ids], [oof_augmented_pred[c] for c in pooled_ids], all_labels,
    )

    return AblationResult(
        mapping=mapping, all_labels=all_labels, fold_results=fold_results,
        pooled_baseline=pooled_baseline, pooled_augmented=pooled_augmented,
    )


def _score_predictions(y_true: list[str], y_pred: list[str], all_labels: list[str]) -> dict:
    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        classification_report,
        confusion_matrix,
        f1_score,
    )

    report = classification_report(y_true, y_pred, labels=all_labels, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=all_labels)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=all_labels, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=all_labels, average="weighted", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "per_class": {
            label: {
                "precision": report[label]["precision"],
                "recall": report[label]["recall"],
                "f1": report[label]["f1-score"],
                "support": report[label]["support"],
            }
            for label in all_labels
        },
        "confusion_matrix": cm.tolist(),
        "n_samples": len(y_true),
    }


if __name__ == "__main__":
    result = run_logo_ablation()
    report = {
        "mapping": result.mapping.__dict__,
        "all_labels": result.all_labels,
        "folds": [
            {
                "held_out_group": f.held_out_group,
                "train_size": f.train_size,
                "test_size": f.test_size,
                "train_label_counts": f.train_label_counts,
                "test_label_counts": f.test_label_counts,
                "baseline": {k: v for k, v in f.baseline.items() if k != "predictions"},
                "augmented": {k: v for k, v in f.augmented.items() if k != "predictions"},
            }
            for f in result.fold_results
        ],
        "pooled_baseline": result.pooled_baseline,
        "pooled_augmented": result.pooled_augmented,
    }
    print(json.dumps(report, indent=2, default=str))
