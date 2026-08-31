"""
Regression test for a real path-mismatch bug: train_xgboost.py's default
save location didn't match where dashboard_api.py looks for the trained
model (backend/ml/models/), so the API kept reporting "no model trained"
even after a real training run had succeeded.
"""

import os

from ml.train_xgboost import DEFAULT_MODEL_DIR


def test_default_model_dir_matches_api_expectation():
    # dashboard_api.py: os.path.join(os.path.dirname(__file__), "ml", "models", "xgb_severity.json")
    expected = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ml", "models")
    assert os.path.normpath(DEFAULT_MODEL_DIR) == os.path.normpath(expected)
