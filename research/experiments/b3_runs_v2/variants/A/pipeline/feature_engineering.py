"""
pipeline/feature_engineering.py
=================================
Sliding window feature extraction from network flow data.

Design:
    Refactored from ml_pipeline/sliding_window_preprocess.py.
    Adds p25/p75 quantile features for richer temporal characterisation.
    NO IP columns — pure statistical flow features.

    Window feature vector per window:
        mean × F + std × F + max × F + min × F [+ p25 × F + p75 × F]
    where F = number of numeric features (78 for CICIDS2017)
    → default 312-dim (or 468-dim with percentiles enabled)
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class SlidingWindowExtractor:
    """
    Extracts statistical feature vectors over sliding windows.

    Args:
        window_size:        Number of rows per window (default 100).
        stride:             Step size between windows (default 25).
        include_percentiles: Also extract p25/p75 per feature (default True).
    """

    def __init__(
        self,
        window_size: int = 100,
        stride: int = 25,
        include_percentiles: bool = True,
    ) -> None:
        self.window_size = window_size
        self.stride = stride
        self.include_percentiles = include_percentiles

    def extract(
        self,
        data: np.ndarray,
        metadata_df: pd.DataFrame | None = None,
    ) -> tuple[np.ndarray, list[dict[str, Any]]]:
        """
        Extract window feature vectors and metadata from a data array.

        Args:
            data:        Clean numeric array of shape (N, F). NO IP columns.
            metadata_df: Optional DataFrame with 'Source IP', 'Destination IP',
                         'Timestamp' for metadata linkage (NOT fed into model).

        Returns:
            (windows, window_metadata)
            windows:          np.ndarray of shape (W, D) — D=312 or 468
            window_metadata:  list of dicts with temporal linkage info
        """
        N, F = data.shape
        num_windows = max(0, (N - self.window_size) // self.stride)

        if num_windows == 0:
            logger.warning(
                "Data too short (%d rows) for window_size=%d stride=%d",
                N, self.window_size, self.stride,
            )
            return np.empty((0, F * (6 if self.include_percentiles else 4))), []

        expected_dim = F * (6 if self.include_percentiles else 4)
        windows = np.empty((num_windows, expected_dim), dtype=np.float32)
        window_meta: list[dict[str, Any]] = []

        for i, start in enumerate(range(0, num_windows * self.stride, self.stride)):
            end = start + self.window_size
            window = data[start:end]  # (window_size, F)

            feat = self._compute_features(window)  # (D,)
            windows[i] = feat

            meta: dict[str, Any] = {"window_index": i, "start_idx": start, "end_idx": end}
            if metadata_df is not None:
                sl = metadata_df.iloc[start:end]
                # Store ONLY for tracing — NOT fed into model
                meta["timestamp_start"] = str(sl["Timestamp"].iloc[0]) if "Timestamp" in sl else None
                meta["timestamp_end"] = str(sl["Timestamp"].iloc[-1]) if "Timestamp" in sl else None
            window_meta.append(meta)

        logger.info("Extracted %d windows of dim %d from %d samples", num_windows, expected_dim, N)
        return windows, window_meta

    def _compute_features(self, window: np.ndarray) -> np.ndarray:
        """
        Compute statistical aggregates over a single window.
        Returns concatenated [mean, std, max, min] + optional [p25, p75].
        """
        mean = np.mean(window, axis=0)
        std  = np.std(window, axis=0)
        maxv = np.max(window, axis=0)
        minv = np.min(window, axis=0)

        parts = [mean, std, maxv, minv]
        if self.include_percentiles:
            p25 = np.percentile(window, 25, axis=0)
            p75 = np.percentile(window, 75, axis=0)
            parts += [p25, p75]

        feat = np.concatenate(parts).astype(np.float32)
        # Guard against NaN/Inf in features
        feat = np.nan_to_num(feat, nan=0.0, posinf=1e6, neginf=-1e6)
        return feat

    @property
    def output_dim(self) -> int:
        """Returns output dimension per window (needs F from first data call)."""
        return -1  # known after first call; use windows.shape[1]


def extract_windows_from_csv(
    csv_path: str,
    window_size: int = 100,
    stride: int = 25,
    include_percentiles: bool = True,
    drop_cols: list[str] | None = None,
) -> tuple[np.ndarray, list[dict[str, Any]], list[str]]:
    """
    Convenience function: load CICIDS CSV → clean → extract windows.

    Returns:
        (windows, window_metadata, feature_names)
    """
    import pandas as pd

    default_drop = ["Flow ID", "Source IP", "Destination IP", "Label"]
    if drop_cols:
        default_drop.extend(drop_cols)

    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()

    # Save metadata before dropping
    meta_cols = [c for c in ["Source IP", "Destination IP", "Timestamp"] if c in df.columns]
    metadata_df = df[meta_cols].copy() if meta_cols else None

    # Drop IP / identifier columns
    drop = [c for c in default_drop if c in df.columns]
    df = df.drop(columns=drop, errors="ignore")

    # Numeric coercion
    df = df.apply(pd.to_numeric, errors="coerce")
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)

    if metadata_df is not None:
        metadata_df = metadata_df.loc[df.index]

    feature_names = list(df.columns)
    extractor = SlidingWindowExtractor(window_size=window_size, stride=stride,
                                       include_percentiles=include_percentiles)
    windows, window_meta = extractor.extract(df.values, metadata_df)

    return windows, window_meta, feature_names
