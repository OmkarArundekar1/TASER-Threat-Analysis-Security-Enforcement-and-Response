"""
Wraps a raw-sensor detection result (detection_confidence_engine.py) as
Evidence — the "did a sensor actually fire, and how loudly" signal.
"""

from __future__ import annotations

from datetime import datetime, timezone

from evidence.schema import Evidence, EvidenceSource, EvidenceType


def collect_detection_evidence(event_id: str, detection_result) -> list[Evidence]:
    if detection_result is None:
        return []

    return [
        Evidence(
            source=EvidenceSource.SIEM,
            source_id=event_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
            type=EvidenceType.DETECTION,
            content={
                "event_id": event_id,
                "level": detection_result.level,
                "breakdown": detection_result.breakdown,
            },
            confidence=detection_result.confidence / 100.0,
            provenance="detection_confidence_engine (Wazuh/Suricata/Zeek/Sigma/YARA)",
        )
    ]
