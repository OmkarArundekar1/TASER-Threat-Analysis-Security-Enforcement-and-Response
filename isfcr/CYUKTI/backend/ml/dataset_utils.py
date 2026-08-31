"""
ml/dataset_utils.py
======================
Shared column bookkeeping for the campaign dataset (feature_schema.py's
CampaignDatasetRecord), so train_xgboost.py / evaluate_model.py /
runtime_predictor.py agree on exactly what's an identifier, a label, and
a model feature without hand-duplicating the list in three places.
"""

from __future__ import annotations

from dataclasses import fields

import pandas as pd

from feature_schema import CampaignDatasetRecord

IDENTIFIER_COLUMNS = ["campaign_id", "attacker_ip", "victim_ip"]
LABEL_COLUMNS = ["severity", "attributed_actor", "prediction_correct", "attribution_correct", "next_technique"]

# risk_score is not a label, but `severity` is a deterministic function of
# it (risk_scoring.severity_from_tps -> label_generator.py), discovered
# while auditing real campaign data: training on risk_score let the model
# trivially reconstruct the label instead of learning anything from the
# MITRE/graph/CTI signals it's actually meant to reason over. Excluded as
# a leakage source, not treated as a legitimate feature.
LEAKAGE_COLUMNS = ["risk_score"]

FEATURE_COLUMNS = [
    f.name for f in fields(CampaignDatasetRecord)
    if f.name not in IDENTIFIER_COLUMNS and f.name not in LABEL_COLUMNS and f.name not in LEAKAGE_COLUMNS
]

MIN_TRAINING_ROWS = 30  # below this, a train/val split + cross-validated metrics are not meaningful


def load_dataset():
    """Load the accumulated campaign dataset as a DataFrame.

    Import is local to avoid a hard dependency on the `ml` package's own
    __init__ sys.path bootstrap having already run when this module is
    imported directly (e.g. from a script run outside backend/).
    """
    from ml.dataset_writer import writer
    return writer.load()


def validate_dataset_size(df: pd.DataFrame, min_rows: int = MIN_TRAINING_ROWS) -> None:
    if df.empty or len(df) < min_rows:
        raise ValueError(
            f"Not enough labeled campaign records to train a meaningful model: "
            f"found {len(df)}, need at least {min_rows}. "
            "This dataset accumulates from real investigations via "
            "ml.dataset_builder.builder.build(...) as campaigns resolve — "
            "it cannot be backfilled without live data. For pipeline "
            "development/testing use ml/data_prep/generate_synthetic_dataset.py "
            "to produce a clearly-labeled synthetic fixture instead."
        )
