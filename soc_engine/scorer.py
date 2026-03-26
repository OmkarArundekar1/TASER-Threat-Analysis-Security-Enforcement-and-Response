"""
soc_engine/scorer.py
=====================
Robust anomaly severity scoring using log1p → soft compression → Median/MAD.

Pipeline:
    raw MSE errors
        → log1p transform        (log-normalises heavy tail)
        → soft clip at 99.9 pct  (compresses extreme outliers, no hard clipping)
        → Median/MAD normalise   (robust z-score, immune to skew/outliers)
        → multi-level label      (NORMAL / LOW / MEDIUM / HIGH)

Critical features:
    - NO hard clipping → preserves outlier information via soft log compression
    - MAD × 1.4826 = consistency constant (equivalent to σ for Gaussian)
    - Ensures no NaN, stable distribution, no identical flattened max values
    - Returns JSON-serialisable Python dicts
"""

from __future__ import annotations

import logging
import math
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# ── Severity thresholds (configurable via config.yaml) ────────────────────────
DEFAULT_THRESHOLDS = {"HIGH": 8.0, "MEDIUM": 4.0, "LOW": 2.0}
MAD_CONSISTENCY = 1.4826  # makes MAD consistent estimator of σ for Gaussian


class SeverityScorer:
    """
    Fits a robust scorer on calibration errors, then scores new errors.

    Usage::
        scorer = SeverityScorer()
        scorer.fit(calibration_errors)          # fit on benign baseline
        results = scorer.score(new_errors)      # score live window errors
    """

    def __init__(self, thresholds: dict[str, float] | None = None, clip_percentile: float = 99.9) -> None:
        self.thresholds = thresholds or DEFAULT_THRESHOLDS
        self.clip_percentile = clip_percentile

        # Calibration parameters (set after .fit())
        self.median_log: float = 0.0
        self.mad_log: float = 1.0
        self.upper_bound: float = float("inf")
        self._fitted: bool = False

    # ── Calibration ──────────────────────────────────────────────────────
    def fit(self, errors: np.ndarray) -> "SeverityScorer":
        """
        Fit scorer on benign baseline errors (e.g. training set reconstruction errors).

        Args:
            errors: 1-D array of raw MSE reconstruction errors.

        Returns:
            self (for chaining)
        """
        errors = self._validate(errors)
        errors_log = np.log1p(errors)

        self.upper_bound = float(np.percentile(errors_log, self.clip_percentile))
        compressed = self._soft_compress(errors_log, self.upper_bound)

        self.median_log = float(np.median(compressed))
        mad_raw = float(np.median(np.abs(compressed - self.median_log)))
        # Apply MAD consistency constant + epsilon guard against zero-MAD
        self.mad_log = max(mad_raw * MAD_CONSISTENCY, 1e-6)
        self._fitted = True

        logger.info(
            "SeverityScorer fitted — median=%.4f  MAD=%.4f  upper_bound=%.4f",
            self.median_log, self.mad_log, self.upper_bound,
        )
        return self

    # ── Scoring ──────────────────────────────────────────────────────────
    def score(self, errors: np.ndarray) -> list[dict[str, Any]]:
        """
        Score an array of reconstruction errors.

        Args:
            errors: 1-D array of raw MSE errors (one per window).

        Returns:
            List of dicts with keys: severity_score, severity_label, is_anomaly, confidence.
        """
        if not self._fitted:
            raise RuntimeError("SeverityScorer must be .fit() before scoring.")

        errors = self._validate(errors)
        errors_log = np.log1p(errors)
        compressed = self._soft_compress(errors_log, self.upper_bound)
        severity_scores = (compressed - self.median_log) / self.mad_log

        results = []
        for i,s in enumerate(severity_scores):
            label = self._label(float(s))
            is_anomaly = int(label != "NORMAL")
            confidence = float(self._confidence(s))
            results.append({
                "window_index": i,
                "severity_score": round(float(s), 6),
                "severity_label": label,
                "is_anomaly": is_anomaly,
                "confidence": confidence,
            })
        return results

    def score_single(self, error: float) -> dict[str, Any]:
        """Score a single reconstruction error."""
        return self.score(np.array([error]))[0]

    # ── Label mapping ─────────────────────────────────────────────────────
    def _label(self, severity: float) -> str:
        if severity > self.thresholds["HIGH"]:
            return "HIGH"
        elif severity > self.thresholds["MEDIUM"]:
            return "MEDIUM"
        elif severity > self.thresholds["LOW"]:
            return "LOW"
        return "NORMAL"

    # ── Soft compression ─────────────────────────────────────────────────
    @staticmethod
    def _soft_compress(x: np.ndarray, upper_bound: float) -> np.ndarray:
        """
        Soft-compress values above upper_bound using log1p(excess).
        Values below upper_bound are unchanged.
        Guarantees: no NaN, no identical-max-flattening.
        """
        excess = x - upper_bound
        compressed = np.where(
            x > upper_bound,
            upper_bound + np.log1p(np.maximum(excess, 0.0)),
            x,
        )
        # Guard against any residual NaN / Inf from upstream
        compressed = np.nan_to_num(compressed, nan=0.0, posinf=upper_bound * 2, neginf=0.0)
        return compressed

    # ── Confidence ───────────────────────────────────────────────────────
    @staticmethod
    def _confidence(severity: float) -> float:
        """
        Sigmoid-based confidence in [0, 1].
        Severity ≈ 8 → ~0.997, severity ≈ 2 → ~0.5.
        """
        return round(1.0 / (1.0 + math.exp(-0.5 * (severity - 2.0))), 4)

    @staticmethod
    def _validate(errors: np.ndarray) -> np.ndarray:
        arr = np.asarray(errors, dtype=np.float64).ravel()
        arr = np.nan_to_num(arr, nan=0.0, posinf=1e6, neginf=0.0)
        arr = np.clip(arr, 0.0, None)  # errors are non-negative
        return arr

    # ── Persistence helpers ──────────────────────────────────────────────
    def get_params(self) -> dict[str, float]:
        return {
            "median_log": self.median_log,
            "mad_log": self.mad_log,
            "upper_bound": self.upper_bound,
        }

    def set_params(self, params: dict[str, float]) -> None:
        self.median_log = params["median_log"]
        self.mad_log = params["mad_log"]
        self.upper_bound = params["upper_bound"]
        self._fitted = True


# ── Convenience function ───────────────────────────────────────────────────────

def compute_scores(
    errors: np.ndarray,
    thresholds: dict[str, float] | None = None,
    clip_percentile: float = 99.9,
) -> list[dict[str, Any]]:
    """
    One-shot scoring (fits and scores on same data — useful for analysis).
    For production, use SeverityScorer.fit() on held-out benign data.
    """
    scorer = SeverityScorer(thresholds=thresholds, clip_percentile=clip_percentile)
    scorer.fit(errors)
    return scorer.score(errors)
