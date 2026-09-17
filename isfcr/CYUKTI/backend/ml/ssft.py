"""
ml/ssft.py
============
Self-Supervised Feature Transformation.

The repo's own config.yaml (paths.ssft_dataset: "models/ssft_dataset.npz")
already anticipated this exact step before this audit began — this
module is what actually implements it: take the encoder trained purely
by reconstruction loss in ssl_pipeline.py (no attack/severity labels
involved anywhere in that training), and use it to TRANSFORM raw
312-dim CICIDS2017 window features into their 32-dim learned latent
representation.

This is "self-supervised feature transformation" in the literal sense —
a feature transform derived from a self-supervised model — and it has a
concrete downstream purpose: the 32-dim SSFT vector is a compact,
learned summary of a traffic window that is cheaper to feed into
downstream models (XGBoost/LightGBM/GNN node features) than the full
312-dim raw window, and captures nonlinear structure a raw feature
vector doesn't expose directly.

Does not retrain anything — this only transforms data through an
already-trained encoder from ssl_pipeline.py.

Runtime integration status (see ../SSFT_RUNTIME.md for the full
write-up): this transform operates in the CICIDS2017 network-flow
*window* feature space (312 dims: 78 raw flow columns x 4 sliding-
window statistics — see ml/data_prep/sliding_window_preprocess.py).
CYUKTI's live, production severity model
(ml.train_xgboost.XGBoostCampaignClassifier, via
ml.runtime_predictor.RuntimeCampaignPredictor) operates in a
completely different feature space: 57 CAMPAIGN-GRAPH features
(ml/feature_schema.py — graph topology, MITRE counts, CTI/detection
scores) computed per Neo4j Campaign, not per network-flow window.
There is no live data source in this repository that produces a
network-flow window for a given Campaign, so there is no honest way to
run a real campaign's data through the trained encoder here — doing so
would mean feeding 57 campaign-topology numbers through a scaler/model
fit on 312 unrelated flow-statistics columns, which is not a
meaningful transformation, just numbers going through matrix
multiplications. That would be exactly the kind of superficial
"SSL/SSFT wrapper" CYUKTI's own engineering discipline exists to avoid
-- so it was not built. What THIS module lacked, and what this session
added, was a genuine single-window *runtime* inference path (as
opposed to only a whole-file *batch* one) within its own correct
domain -- see SSFTRuntimeTransformer below, which both
transform_to_ssft_dataset() (batch) and any future window-level caller
(single instance) now share as the one authoritative transform path.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np

_ML_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL_DIR = os.path.join(_ML_DIR, "models")
DEFAULT_WINDOW_FEATURES_PATH = os.path.join(_ML_DIR, "artifacts", "window_features.npy")


class SSFTRuntimeTransformer:
    """Loads the trained SSL scaler + encoder once, then transforms any
    number of raw CICIDS2017-schema window feature vectors into their
    learned latent representation + reconstruction error.

    This is the ONE authoritative transform code path: whether called
    once per window (a genuine runtime/inference use) or once for an
    entire pre-built window_features.npy (transform_to_ssft_dataset's
    batch/offline use, unchanged in behavior after this refactor), both
    go through identical scaler.transform -> model.encode /
    model.reconstruction_errors calls. Mirrors the existing
    ml.runtime_predictor.RuntimeCampaignPredictor pattern (load once at
    construction, transform/predict cheaply per call afterward) so
    train-time batch processing and runtime single-instance inference
    can never silently diverge into two separate implementations.
    """

    def __init__(self, model_dir: str = DEFAULT_MODEL_DIR) -> None:
        import joblib
        import torch

        from soc_engine.model import Autoencoder

        model_dir_path = Path(model_dir)
        model_path = model_dir_path / "autoencoder_best.pth"
        scaler_path = model_dir_path / "ssl_scaler.pkl"

        if not model_path.exists():
            raise FileNotFoundError(
                f"No trained SSL autoencoder at {model_path}. Run "
                "ml/ssl_pipeline.py (train_ssl_autoencoder) first — SSFT "
                "transforms data through an already-trained self-supervised "
                "model, it does not train one."
            )

        self._scaler = joblib.load(scaler_path)
        self._model = Autoencoder(input_dim=self._scaler.n_features_in_)
        self._model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
        self._model.eval()

    @property
    def input_dim(self) -> int:
        return int(self._scaler.n_features_in_)

    def transform(self, windows: np.ndarray) -> dict[str, Any]:
        """windows: shape (D,) for a single window, or (N, D) for a
        batch — D must equal self.input_dim (the dimensionality the
        scaler/encoder were actually trained on).

        Returns {"latent": ..., "reconstruction_error": ...}. For a
        single window (1D input), "latent" is shape (latent_dim,) and
        "reconstruction_error" is a plain float — matching the
        single-instance contract ml.runtime_predictor.RuntimeCampaignPredictor
        already established for XGBoost, so callers don't need to know
        which transform produced a result to consume it consistently.
        For a batch (2D input), both are arrays with one row per window
        — the exact shape transform_to_ssft_dataset already persisted
        to ssft_dataset.npz before this refactor, unchanged.
        """
        import torch

        arr = np.asarray(windows, dtype=np.float64)
        single = arr.ndim == 1
        arr = np.atleast_2d(arr)

        if arr.shape[1] != self.input_dim:
            raise ValueError(
                f"SSFTRuntimeTransformer expects {self.input_dim}-dimensional "
                f"window features (the shape ssl_scaler.pkl was fit on), got {arr.shape[1]}."
            )

        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        scaled = self._scaler.transform(arr).astype(np.float32)

        x = torch.tensor(scaled, dtype=torch.float32)
        with torch.no_grad():
            latent = self._model.encode(x).numpy()
        errors = self._model.reconstruction_errors(x).numpy()

        if single:
            return {"latent": latent[0], "reconstruction_error": float(errors[0])}
        return {"latent": latent, "reconstruction_error": errors}


def transform_to_ssft_dataset(
    window_features_path: str = DEFAULT_WINDOW_FEATURES_PATH,
    model_dir: str = DEFAULT_MODEL_DIR,
    output_path: str | None = None,
) -> str:
    """Transform raw window features through the trained SSL encoder.

    Saves a compressed .npz with:
        - latent:       (N, latent_dim) encoder output
        - reconstruction_error: (N,) per-window MSE, for reuse as an
          anomaly/severity signal without re-running the encoder

    Batch/offline entry point — delegates to SSFTRuntimeTransformer,
    the same class a single-window runtime caller would use, so this
    function's behavior (and any bug fix to it) can never silently
    diverge from single-instance inference.
    """
    model_dir_path = Path(model_dir)
    transformer = SSFTRuntimeTransformer(model_dir_path)  # raises FileNotFoundError if untrained

    raw = np.load(window_features_path)
    result = transformer.transform(raw)

    output_path = output_path or str(model_dir_path.parent / "artifacts" / "ssft_dataset.npz")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, latent=result["latent"], reconstruction_error=result["reconstruction_error"])

    return output_path


if __name__ == "__main__":
    path = transform_to_ssft_dataset()
    print(f"SSFT dataset written to {path}")
