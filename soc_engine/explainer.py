"""
soc_engine/explainer.py
========================
Feature attribution via per-feature reconstruction error.

Why this approach:
    SHAP/LIME are too slow for real-time SOC inference.
    Per-feature abs(original - reconstructed) is:
        - O(1) per sample (already computed in forward pass)
        - Directly interpretable: which network features made the autoencoder fail
        - Compatible with feature names from CICIDS2017 / Zeek / Suricata

Output includes:
    - feature_index: position in the 312-dim window vector
    - feature_name:  human-readable name (if feature names provided)
    - attribution:   absolute reconstruction error for that feature
    - contribution_pct: % of total reconstruction error
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch

logger = logging.getLogger(__name__)


class FeatureExplainer:
    """
    Explains anomalies by identifying which features drove high reconstruction error.

    Args:
        feature_names: Optional list of feature names matching the input dimension.
                       If None, features are labelled by index.
        top_k:         Number of top contributing features to return (default 5).
    """

    def __init__(
        self,
        feature_names: list[str] | None = None,
        top_k: int = 5,
    ) -> None:
        self.feature_names = feature_names
        self.top_k = top_k

    def explain(
        self,
        model: "torch.nn.Module",  # Autoencoder
        x: np.ndarray | torch.Tensor,
    ) -> list[dict[str, Any]]:
        """
        Compute per-feature attribution for each sample in x.

        Args:
            model: Autoencoder with .per_feature_errors() method.
            x:     Input tensor/array of shape (N, input_dim).

        Returns:
            List of N attribution dicts, each with 'top_features' key.
        """
        if isinstance(x, np.ndarray):
            x_tensor = torch.tensor(x, dtype=torch.float32)
        else:
            x_tensor = x.float()

        # Move to model device
        device = next(model.parameters()).device
        x_tensor = x_tensor.to(device)

        per_feat_err = model.per_feature_errors(x_tensor).cpu().numpy()  # (N, D)

        results = []
        for i in range(per_feat_err.shape[0]):
            results.append(self._top_features(per_feat_err[i]))
        return results

    def explain_single(
        self,
        model: "torch.nn.Module",
        x: np.ndarray | torch.Tensor,
    ) -> dict[str, Any]:
        """
        Explain a SINGLE sample. x must be shape (D,) or (1, D).
        """
        if isinstance(x, np.ndarray):
            x = x.ravel()[np.newaxis, :]  # (1, D)
        else:
            x = x.view(1, -1)
        return self.explain(model, x)[0]

    def _top_features(self, per_feat_err: np.ndarray) -> dict[str, Any]:
        """Extract top-k features from a per-feature error vector."""
        total_err = float(per_feat_err.sum()) + 1e-9
        top_indices = np.argsort(per_feat_err)[::-1][:self.top_k]

        top_features = []
        for idx in top_indices:
            name = (
                self.feature_names[idx]
                if self.feature_names and idx < len(self.feature_names)
                else f"feature_{idx}"
            )
            attribution = float(per_feat_err[idx])
            top_features.append({
                "feature_index": int(idx),
                "feature_name": name,
                "attribution": round(attribution, 6),
                "contribution_pct": round(100.0 * attribution / total_err, 2),
            })

        return {"top_features": top_features}


# ── Standalone batch explainer (no model needed if errors pre-computed) ────────

def explain_from_per_feature_errors(
    per_feat_errors: np.ndarray,
    feature_names: list[str] | None = None,
    top_k: int = 5,
) -> list[dict[str, Any]]:
    """
    Explain from pre-computed per-feature errors array of shape (N, D).
    Useful when you already have per_feature_errors from a separate step.
    """
    explainer = FeatureExplainer(feature_names=feature_names, top_k=top_k)
    results = []
    for i in range(per_feat_errors.shape[0]):
        results.append(explainer._top_features(per_feat_errors[i]))
    return results
