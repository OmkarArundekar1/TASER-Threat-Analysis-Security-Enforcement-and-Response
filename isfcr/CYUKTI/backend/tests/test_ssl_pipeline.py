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
from ml.ssft import SSFTRuntimeTransformer, transform_to_ssft_dataset


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


# ---------------------------------------------------------------------------
# SSFTRuntimeTransformer -- the single-window runtime path added this
# session, sharing one transform code path with transform_to_ssft_dataset's
# existing batch/offline path (see ml/ssft.py's module docstring and
# ../SSFT_RUNTIME.md for why this, and not a campaign-feature bridge, is
# the genuine "runtime" gap that existed).
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def trained_model_dir(tmp_path_factory):
    windows = _synthetic_windows(n=300, seed=7)
    save_dir = tmp_path_factory.mktemp("ssft_runtime_model")
    features_path = save_dir / "windows.npy"
    np.save(features_path, windows)
    train_ssl_autoencoder(window_features_path=str(features_path), save_dir=str(save_dir), epochs=8)
    return str(save_dir)


def test_runtime_transformer_requires_a_trained_model(tmp_path):
    with pytest.raises(FileNotFoundError, match="No trained SSL autoencoder"):
        SSFTRuntimeTransformer(model_dir=str(tmp_path))


def test_runtime_transformer_single_window_shape_and_type(trained_model_dir):
    transformer = SSFTRuntimeTransformer(model_dir=trained_model_dir)
    window = _synthetic_windows(n=1, seed=1)[0]  # shape (312,) -- a genuine single instance, not a batch of 1

    result = transformer.transform(window)

    assert result["latent"].shape == (32,)
    assert isinstance(result["reconstruction_error"], float)
    assert result["reconstruction_error"] >= 0


def test_runtime_transformer_rejects_wrong_dimensional_input(trained_model_dir):
    transformer = SSFTRuntimeTransformer(model_dir=trained_model_dir)
    wrong_shape_window = np.zeros(10)  # not 312-dim

    with pytest.raises(ValueError, match="312-dimensional"):
        transformer.transform(wrong_shape_window)


def test_runtime_transformer_handles_nan_and_inf_input_without_crashing(trained_model_dir):
    transformer = SSFTRuntimeTransformer(model_dir=trained_model_dir)
    window = _synthetic_windows(n=1, seed=2)[0]
    window[0] = np.nan
    window[1] = np.inf
    window[2] = -np.inf

    result = transformer.transform(window)

    assert result["latent"].shape == (32,)
    assert np.isfinite(result["reconstruction_error"])


def test_runtime_transformer_is_deterministic(trained_model_dir):
    """No randomness (dropout, etc.) may leak into inference -- the same
    window must produce bit-identical output across repeated calls,
    otherwise "deterministic transformation" (an explicit requirement
    for this integration) would not actually hold."""
    transformer = SSFTRuntimeTransformer(model_dir=trained_model_dir)
    window = _synthetic_windows(n=1, seed=3)[0]

    first = transformer.transform(window)
    second = transformer.transform(window)

    np.testing.assert_array_equal(first["latent"], second["latent"])
    assert first["reconstruction_error"] == second["reconstruction_error"]


def test_runtime_transformer_batch_call_matches_single_window_calls(trained_model_dir):
    """The batch code path (N windows at once) and calling the SAME
    class once per window must agree exactly -- proving there is
    genuinely one transform implementation underneath both, not two
    that happen to look similar."""
    transformer = SSFTRuntimeTransformer(model_dir=trained_model_dir)
    windows = _synthetic_windows(n=5, seed=4)

    batch_result = transformer.transform(windows)
    per_window_results = [transformer.transform(w) for w in windows]

    for i, single in enumerate(per_window_results):
        np.testing.assert_allclose(batch_result["latent"][i], single["latent"], rtol=1e-5, atol=1e-6)
        assert batch_result["reconstruction_error"][i] == pytest.approx(single["reconstruction_error"], rel=1e-5)


def test_transform_to_ssft_dataset_and_runtime_transformer_agree_on_the_same_data(trained_model_dir):
    """Training/runtime parity, end to end: the batch entry point
    (transform_to_ssft_dataset, used to build ssft_dataset.npz) and the
    runtime entry point (SSFTRuntimeTransformer, used for single-window
    inference) must produce identical results for the identical input
    -- this is the actual parity guarantee this integration exists to
    establish, verified directly rather than assumed from shared code."""
    windows = _synthetic_windows(n=20, seed=5)
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp:
        features_path = Path(tmp) / "windows.npy"
        np.save(features_path, windows)
        out_path = Path(tmp) / "out.npz"
        transform_to_ssft_dataset(
            window_features_path=str(features_path), model_dir=trained_model_dir, output_path=str(out_path),
        )
        # np.load() on a .npz keeps the file handle open until closed or
        # garbage-collected -- on Windows that blocks the enclosing
        # TemporaryDirectory's own cleanup with a PermissionError, so
        # this reads the arrays out into plain ndarrays and closes the
        # handle explicitly before the `with` block exits.
        with np.load(out_path) as npz:
            batch_via_function_latent = np.array(npz["latent"])
            batch_via_function_error = np.array(npz["reconstruction_error"])

        transformer = SSFTRuntimeTransformer(model_dir=trained_model_dir)
        batch_via_class = transformer.transform(windows)

        np.testing.assert_array_equal(batch_via_function_latent, batch_via_class["latent"])
        np.testing.assert_array_equal(batch_via_function_error, batch_via_class["reconstruction_error"])
