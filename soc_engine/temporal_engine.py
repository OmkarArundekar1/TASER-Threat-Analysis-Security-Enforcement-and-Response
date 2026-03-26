"""
soc_engine/temporal_engine.py
==============================
Temporal smoothing over severity labels for detecting slow / multi-stage attacks.

Problem being solved:
    A single anomalous window does NOT mean an attack.
    Real SOC analysts look for patterns over time.
    Low-and-slow brute force, distributed scanning, and stealth exfiltration
    appear as repeated LOW/MEDIUM signals, not sudden HIGH spikes.

Algorithm:
    Maintain a rolling window of N severity labels.
    Apply majority-progression rules:
        ≥1 HIGH  in window → escalate to HIGH
        ≥2 MEDIUM in window → escalate to MEDIUM
        ≥3 LOW  in window → escalate to LOW
        else → NORMAL

    This suppresses single-window noise while catching slow-burn attacks.

Additional feature:
    Exponential Moving Average (EMA) on raw severity scores →
    smoothed_score for continuous signal monitoring.
"""

from __future__ import annotations

import logging
from collections import deque
from typing import Any

logger = logging.getLogger(__name__)

LABEL_RANK = {"NORMAL": 0, "LOW": 1, "MEDIUM": 2, "HIGH": 3}
RANK_LABEL = {v: k for k, v in LABEL_RANK.items()}


class TemporalSmoother:
    """
    Maintains a rolling window of severity scores and labels.
    Escalates temporal label based on configurable majority rules.

    Args:
        smooth_window:      Rolling window size (default 5).
        ema_alpha:          EMA decay factor for smoothed score (default 0.3).
        high_min_count:     Min HIGH labels in window to declare HIGH (default 1).
        medium_min_count:   Min MEDIUM labels in window to declare MEDIUM (default 2).
        low_min_count:      Min LOW labels in window to declare LOW (default 3).
    """

    def __init__(
        self,
        smooth_window: int = 5,
        ema_alpha: float = 0.3,
        high_min_count: int = 1,
        medium_min_count: int = 2,
        low_min_count: int = 3,
    ) -> None:
        self.smooth_window = smooth_window
        self.ema_alpha = ema_alpha
        self.rules = {
            "HIGH": high_min_count,
            "MEDIUM": medium_min_count,
            "LOW": low_min_count,
        }
        self._label_buffer: deque[str] = deque(maxlen=smooth_window)
        self._score_buffer: deque[float] = deque(maxlen=smooth_window)
        self._ema_score: float = 0.0
        self._window_count: int = 0

    def update(self, severity_score: float, severity_label: str) -> dict[str, Any]:
        """
        Feed one new window's severity into the smoother.

        Returns:
            dict with temporal_label and smoothed_score.
        """
        self._label_buffer.append(severity_label)
        self._score_buffer.append(severity_score)
        self._window_count += 1

        # EMA on severity score
        if self._window_count == 1:
            self._ema_score = severity_score
        else:
            self._ema_score = (
                self.ema_alpha * severity_score + (1 - self.ema_alpha) * self._ema_score
            )

        temporal_label = self._compute_temporal_label()

        return {
            "temporal_label": temporal_label,
            "smoothed_score": round(self._ema_score, 6),
            "buffer_size": len(self._label_buffer),
            "window_counts": self._count_labels(),
        }

    def _compute_temporal_label(self) -> str:
        """Apply escalation rules on current rolling buffer."""
        counts = self._count_labels()

        if counts.get("HIGH", 0) >= self.rules["HIGH"]:
            return "HIGH"
        if counts.get("MEDIUM", 0) >= self.rules["MEDIUM"]:
            return "MEDIUM"
        if counts.get("LOW", 0) >= self.rules["LOW"]:
            return "LOW"
        return "NORMAL"

    def _count_labels(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for label in self._label_buffer:
            counts[label] = counts.get(label, 0) + 1
        return counts

    def reset(self) -> None:
        """Reset state (e.g. when switching to a new IP or time segment)."""
        self._label_buffer.clear()
        self._score_buffer.clear()
        self._ema_score = 0.0
        self._window_count = 0

    @property
    def current_ema(self) -> float:
        return round(self._ema_score, 6)


def apply_temporal_smoothing(
    window_results: list[dict[str, Any]],
    smooth_window: int = 5,
    high_min_count: int = 1,
    medium_min_count: int = 2,
    low_min_count: int = 3,
) -> list[dict[str, Any]]:
    """
    Batch-apply temporal smoothing over a list of scored window results.

    Args:
        window_results: List of scorer output dicts (must have severity_score, severity_label).
        smooth_window:  Rolling window size.

    Returns:
        Updated list with temporal_label and smoothed_score added to each dict.
    """
    smoother = TemporalSmoother(
        smooth_window=smooth_window,
        high_min_count=high_min_count,
        medium_min_count=medium_min_count,
        low_min_count=low_min_count,
    )
    enriched = []
    for entry in window_results:
        temporal_info = smoother.update(
            severity_score=entry["severity_score"],
            severity_label=entry["severity_label"],
        )
        enriched.append({**entry, **temporal_info})
    return enriched
