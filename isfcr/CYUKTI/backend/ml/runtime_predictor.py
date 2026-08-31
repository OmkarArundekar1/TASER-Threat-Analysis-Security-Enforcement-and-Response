"""
ml/runtime_predictor.py
==========================
Bridges a live, still-unresolved campaign to a trained
XGBoostCampaignClassifier: extracts the same feature vector
feature_extractors.py builds for the training dataset (via
feature_orchestrator + campaign_feature_engine), then runs inference.

The label fields (severity/attributed_actor/prediction_correct/
attribution_correct/next_technique) are not knowable yet at prediction time — they're
passed as placeholders into DatasetFeatureExtractor.extract() because
that method threads them straight into the output record without using
them to compute any feature, and they are excluded from
dataset_utils.FEATURE_COLUMNS, so they cannot leak into the model input.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from dataset_utils import FEATURE_COLUMNS


class RuntimeCampaignPredictor:
    def __init__(self, model):
        self._model = model

    @classmethod
    def from_model_path(cls, model_path: str) -> "RuntimeCampaignPredictor":
        from train_xgboost import XGBoostCampaignClassifier
        return cls(XGBoostCampaignClassifier.load(model_path))

    def predict_for_campaign(
        self,
        campaign_context,
        current_attack_id: str,
        event_id: str,
    ) -> dict[str, Any]:
        from feature_extractors import extractor

        record = extractor.extract(
            campaign_context=campaign_context,
            current_attack_id=current_attack_id,
            attacker_ip=campaign_context.attacker_ip,
            event_id=event_id,
            severity="",
            attributed_actor="",
            prediction_correct=None,
            attribution_correct=0,
            next_technique="",
        )

        row = pd.DataFrame([{c: getattr(record, c) for c in FEATURE_COLUMNS}])
        feature_vector = row[FEATURE_COLUMNS].apply(pd.to_numeric, errors="coerce").fillna(0.0).values[0]

        return self._model.predict(feature_vector)
