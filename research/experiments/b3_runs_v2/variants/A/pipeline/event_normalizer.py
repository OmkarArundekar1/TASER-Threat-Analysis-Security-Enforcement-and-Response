"""
pipeline/event_normalizer.py
=============================
Converts raw Suricata / Zeek / CICIDS events into a unified SecurityEvent schema.

Output is always JSON-serialisable (no torch tensors, no numpy).

Schema::

    {
        "event_id":       str,          # uuid4
        "timestamp":      str,          # ISO 8601
        "src_port":       int | None,
        "dst_port":       int | None,
        "proto":          str,
        "sensor":         str,          # "suricata" | "zeek" | "cicids"
        "severity":       float,        # raw severity if available
        "feature_vector": [float, ...], # model-ready feature vector
        "window_index":   int | None,   # links back to sliding window
        "raw":            {}            # original event (for audit trail)
    }

Note: src_ip / dst_ip are stored in raw{} for audit ONLY — NEVER in feature_vector.
"""

from __future__ import annotations

import uuid
from typing import Any

import numpy as np


class SecurityEvent:
    """Lightweight dataclass for a normalised security event."""

    __slots__ = [
        "event_id", "timestamp", "src_port", "dst_port", "proto",
        "sensor", "severity", "feature_vector", "window_index", "raw",
    ]

    def __init__(
        self,
        timestamp: str,
        sensor: str,
        feature_vector: list[float],
        src_port: int | None = None,
        dst_port: int | None = None,
        proto: str = "unknown",
        severity: float = 0.0,
        window_index: int | None = None,
        raw: dict[str, Any] | None = None,
    ) -> None:
        self.event_id = str(uuid.uuid4())
        self.timestamp = timestamp
        self.src_port = src_port
        self.dst_port = dst_port
        self.proto = proto
        self.sensor = sensor
        self.severity = severity
        self.feature_vector = [float(v) for v in feature_vector]
        self.window_index = window_index
        self.raw = raw or {}

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "timestamp": self.timestamp,
            "src_port": self.src_port,
            "dst_port": self.dst_port,
            "proto": self.proto,
            "sensor": self.sensor,
            "severity": round(self.severity, 6),
            "feature_vector": [round(v, 6) for v in self.feature_vector],
            "window_index": self.window_index,
            "raw": self.raw,
        }


class EventNormalizer:
    """
    Normalises events from different sensors into SecurityEvent.

    Usage::
        normalizer = EventNormalizer()
        event = normalizer.normalize_suricata(suricata_dict, feature_vector)
        event = normalizer.normalize_zeek(zeek_dict, feature_vector)
        event = normalizer.normalize_cicids(cicids_row, feature_vector, window_index)
    """

    def normalize_suricata(
        self,
        raw: dict[str, Any],
        feature_vector: list[float] | np.ndarray,
    ) -> SecurityEvent:
        """Normalise a Suricata alert dict."""
        if isinstance(feature_vector, np.ndarray):
            feature_vector = feature_vector.tolist()

        return SecurityEvent(
            timestamp=raw.get("timestamp", ""),
            sensor="suricata",
            feature_vector=feature_vector,
            src_port=self._int(raw.get("src_port")),
            dst_port=self._int(raw.get("dest_port") or raw.get("dst_port")),
            proto=str(raw.get("proto", "unknown")).lower(),
            severity=float(raw.get("severity", 0)),
            raw={
                "signature": raw.get("signature", ""),
                "stage": raw.get("stage", "unknown"),
                "src_ip": raw.get("src_ip"),    # audit only
                "dst_ip": raw.get("dst_ip"),    # audit only
            },
        )

    def normalize_zeek(
        self,
        raw: dict[str, Any],
        feature_vector: list[float] | np.ndarray,
    ) -> SecurityEvent:
        """Normalise a Zeek connection log dict."""
        if isinstance(feature_vector, np.ndarray):
            feature_vector = feature_vector.tolist()

        return SecurityEvent(
            timestamp=raw.get("timestamp", ""),
            sensor="zeek",
            feature_vector=feature_vector,
            src_port=self._int(raw.get("src_port")),
            dst_port=self._int(raw.get("dst_port")),
            proto=str(raw.get("proto", "unknown")).lower(),
            raw={
                "src_ip": raw.get("src_ip"),    # audit only
                "dst_ip": raw.get("dst_ip"),    # audit only
                "bytes": raw.get("bytes"),
                "packets": raw.get("packets"),
                "duration": raw.get("duration"),
            },
        )

    def normalize_cicids(
        self,
        raw: dict[str, Any],
        feature_vector: list[float] | np.ndarray,
        window_index: int | None = None,
    ) -> SecurityEvent:
        """Normalise a CICIDS2017 row dict."""
        if isinstance(feature_vector, np.ndarray):
            feature_vector = feature_vector.tolist()

        return SecurityEvent(
            timestamp=str(raw.get("Timestamp", "")),
            sensor="cicids",
            feature_vector=feature_vector,
            src_port=self._int(raw.get("Source Port") or raw.get("src_port")),
            dst_port=self._int(raw.get("Destination Port") or raw.get("dst_port")),
            proto=str(raw.get("Protocol", "unknown")),
            severity=0.0,
            window_index=window_index,
            raw={
                "label": raw.get("Label", "UNKNOWN"),
                "src_ip": raw.get("Source IP"),   # audit only
                "dst_ip": raw.get("Destination IP"),  # audit only
            },
        )

    @staticmethod
    def _int(val: Any) -> int | None:
        try:
            return int(val)
        except (TypeError, ValueError):
            return None
