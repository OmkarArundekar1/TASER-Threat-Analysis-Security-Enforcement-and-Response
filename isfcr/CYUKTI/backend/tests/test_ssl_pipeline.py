"""
Tests for the SSL (ml/ssl_pipeline.py) and SSFT (ml/ssft.py) steps.

Uses a small synthetic window-feature array purely to keep the test
fast (a full run already trained on the real 21,176-window CICIDS2017
Monday dataset in ml/models/ and ml/artifacts/ssft_dataset.npz — see
ml/ssl_pipeline.py's docstring). These tests validate the mechanics
(training converges without diverging, calibration/persistence and the
SSFT transform round-trip correctly) rather than re-asserting real-world
anomaly-detection accuracy.
"""

import numpy as np
import pytest

from ml.ssl_pipeline import MIN_WINDOWS_TO_TRAIN, train_ssl_autoencoder
from ml.ssft import transform_to_ssft_dataset


def _synthetic_windows(n=200, dim=312, seed=0):
    rng = np.random.RandomState(seed)
    return rng.normal(loc=0.0, scale=1.0, size=(n, dim)).astype(np.float64)


def test_train_ssl_autoencoder_rejects_too_few_windows(tmp_path):
    windows = _synthetic_windows(n=MIN_WINDOWS_TO_TRAIN - 1)
    path = tmp_path / "tiny.npy"
    np.save(path, windows)

    with pytest.raises(ValueError, match="Not enough windows"):
        train_ssl_autoencoder(window_features_path=str(path), save_dir=str(tmp_path))


def test_train_ssl_autoencoder_produces_artifacts_and_converges(tmp_path):
    windows = _synthetic_windows(n=300)
    features_path = tmp_path / "windows.npy"
    np.save(features_path, windows)

    result = train_ssl_autoencoder(
        window_features_path=str(features_path), save_dir=str(tmp_path), epochs=10,
    )

    assert result.n_windows == 300
    assert result.n_train + result.n_val == 300
    assert result.best_val_loss < 2.0  # should learn something on Gaussian-ish data, not diverge
    from pathlib import Path
    assert Path(result.model_path).exists()
    assert Path(result.scaler_path).exists()
    assert Path(result.scorer_path).exists()
    assert Path(result.threshold_path).exists()


def test_ssft_transform_requires_trained_model(tmp_path):
    windows = _synthetic_windows(n=50)
    features_path = tmp_path / "windows.npy"
    np.save(features_path, windows)

    with pytest.raises(FileNotFoundError, match="No trained SSL autoencoder"):
        transform_to_ssft_dataset(window_features_path=str(features_path), model_dir=str(tmp_path))


def test_ssft_transform_round_trip(tmp_path):
    windows = _synthetic_windows(n=250)
    features_path = tmp_path / "windows.npy"
    np.save(features_path, windows)

    result = train_ssl_autoencoder(window_features_path=str(features_path), save_dir=str(tmp_path), epochs=8)

    out_path = tmp_path / "ssft_out.npz"
    written_path = transform_to_ssft_dataset(
        window_features_path=str(features_path), model_dir=str(tmp_path), output_path=str(out_path),
    )
    assert written_path == str(out_path)

    data = np.load(out_path)
    assert data["latent"].shape == (250, 32)
    assert data["reconstruction_error"].shape == (250,)
    assert np.all(data["reconstruction_error"] >= 0)
