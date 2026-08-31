"""
ml/ssl_pipeline.py
=====================
Orchestrates the actual self-supervised learning step CYUKTI's spec
calls for: soc_engine's Autoencoder, trained with a pure reconstruction
objective (no labels) on real CICIDS2017 sliding-window features. This
was already implemented (soc_engine/model.py, trainer.py, scorer.py,
threshold.py) but had never actually been run — no models/ directory
existed. This script is what makes it real:

    1. Load real windowed CICIDS2017 features (ml/artifacts/window_features.npy
       — 21,176 windows x 312 features, produced by
       ml/data_prep/sliding_window_preprocess.py from Monday-WorkingHours).
    2. Fit a StandardScaler fresh (the old ml/artifacts/scaler.pkl was
       pickled with scikit-learn 1.8.0; this environment has 1.3.2, and
       sklearn's own unpickling warning says cross-version scaler state
       can silently produce wrong results — refitting is cheap and
       removes that risk entirely).
    3. Train the Autoencoder with AutoencoderTrainer (early stopping,
       LR scheduling) — the self-supervised objective is reconstruction:
       minimize ||x - decode(encode(x))||^2 with no severity/attack
       labels used anywhere in this step.
    4. Calibrate SeverityScorer + AdaptiveThreshold on the TRAIN split's
       own reconstruction errors (their intended calibration data — see
       soc_engine/scorer.py's docstring: "fit on benign baseline").
    5. Persist scaler + model + calibration so ssft.py and the runtime
       inference path can reuse them without retraining.

Downstream purpose of the learned representation (per your requirement
that SSL not be a buzzword): the encoder's 32-dim latent vector is the
SSFT step's transformed feature representation (ml/ssft.py), and
reconstruction-error-derived severity/anomaly scores are meant to become
additional Evidence (SIEM-sourced) for the investigation loop, alongside
the rule-based detection_confidence_engine signal that already exists.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

MIN_WINDOWS_TO_TRAIN = 100
_ML_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_WINDOW_FEATURES_PATH = os.path.join(_ML_DIR, "artifacts", "window_features.npy")
DEFAULT_SAVE_DIR = os.path.join(_ML_DIR, "models")


@dataclass
class SSLTrainingResult:
    model_path: str
    scaler_path: str
    scorer_path: str
    threshold_path: str
    n_windows: int
    n_train: int
    n_val: int
    best_val_loss: float
    epochs_trained: int


def train_ssl_autoencoder(
    window_features_path: str = DEFAULT_WINDOW_FEATURES_PATH,
    save_dir: str = DEFAULT_SAVE_DIR,
    epochs: int = 30,
    seed: int = 42,
) -> SSLTrainingResult:
    from sklearn.preprocessing import StandardScaler
    import joblib
    import torch

    from soc_engine.model import Autoencoder, get_device
    from soc_engine.trainer import AutoencoderTrainer
    from soc_engine.scorer import SeverityScorer
    from soc_engine.threshold import AdaptiveThreshold

    raw = np.load(window_features_path)
    if raw.shape[0] < MIN_WINDOWS_TO_TRAIN:
        raise ValueError(
            f"Not enough windows to train the SSL autoencoder: found {raw.shape[0]}, "
            f"need at least {MIN_WINDOWS_TO_TRAIN}. Run "
            "ml/data_prep/sliding_window_preprocess.py against more raw CICIDS2017 data first."
        )
    raw = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)

    np.random.seed(seed)
    scaler = StandardScaler()
    scaled = scaler.fit_transform(raw).astype(np.float32)

    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    scaler_path = save_path / "ssl_scaler.pkl"
    joblib.dump(scaler, scaler_path)

    torch.manual_seed(seed)
    model = Autoencoder(input_dim=scaled.shape[1])
    model_path = save_path / "autoencoder_best.pth"
    trainer = AutoencoderTrainer(model, epochs=epochs, save_path=model_path)
    summary = trainer.fit(scaled, val_fraction=0.15)

    # Calibrate severity scoring / adaptive threshold on the model's own
    # reconstruction errors over the full (train+val) set it was fit on —
    # this IS the "benign baseline" scorer.py's docstring calls for, since
    # this dataset has no attack labels attached at the SSL stage.
    trained = Autoencoder(input_dim=scaled.shape[1])
    trained.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
    errors = trained.reconstruction_errors(torch.tensor(scaled, dtype=torch.float32)).numpy()

    scorer = SeverityScorer().fit(errors)
    scorer_path = save_path / "severity_scorer.json"
    scorer_path.write_text(json.dumps(scorer.get_params(), indent=2))

    threshold = AdaptiveThreshold().fit(errors)
    threshold_path = save_path / "adaptive_threshold.json"
    threshold.save(threshold_path)

    n_val = max(1, int(len(scaled) * 0.15))
    return SSLTrainingResult(
        model_path=str(model_path),
        scaler_path=str(scaler_path),
        scorer_path=str(scorer_path),
        threshold_path=str(threshold_path),
        n_windows=len(scaled),
        n_train=len(scaled) - n_val,
        n_val=n_val,
        best_val_loss=summary["best_val_loss"],
        epochs_trained=summary["epochs_trained"],
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    result = train_ssl_autoencoder()
    print(json.dumps(result.__dict__, indent=2))
