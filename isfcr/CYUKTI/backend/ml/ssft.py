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
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

_ML_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL_DIR = os.path.join(_ML_DIR, "models")
DEFAULT_WINDOW_FEATURES_PATH = os.path.join(_ML_DIR, "artifacts", "window_features.npy")


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
    """
    import joblib
    import torch

    from soc_engine.model import Autoencoder

    model_dir = Path(model_dir)
    model_path = model_dir / "autoencoder_best.pth"
    scaler_path = model_dir / "ssl_scaler.pkl"

    if not model_path.exists():
        raise FileNotFoundError(
            f"No trained SSL autoencoder at {model_path}. Run "
            "ml/ssl_pipeline.py (train_ssl_autoencoder) first — SSFT "
            "transforms data through an already-trained self-supervised "
            "model, it does not train one."
        )

    raw = np.load(window_features_path)
    raw = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)

    scaler = joblib.load(scaler_path)
    scaled = scaler.transform(raw).astype(np.float32)

    model = Autoencoder(input_dim=scaled.shape[1])
    model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
    model.eval()

    x = torch.tensor(scaled, dtype=torch.float32)
    with torch.no_grad():
        latent = model.encode(x).numpy()
    errors = model.reconstruction_errors(x).numpy()

    output_path = output_path or str(model_dir.parent / "artifacts" / "ssft_dataset.npz")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, latent=latent, reconstruction_error=errors)

    return output_path


if __name__ == "__main__":
    path = transform_to_ssft_dataset()
    print(f"SSFT dataset written to {path}")
