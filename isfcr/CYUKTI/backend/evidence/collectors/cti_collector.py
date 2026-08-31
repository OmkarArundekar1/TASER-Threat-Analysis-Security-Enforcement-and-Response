"""
Wraps CTI/MISP threat-intelligence results (threat_intelligence_engine.py)
as Evidence. The engine's own confidence is already a calibrated 0-100
blend of VT/threat-actor/malware/tool/MISP/IOC signals — we carry that
through as-is (normalized to 0-1), rather than re-deriving it, since it
already IS the source's stated confidence.
"""

from __future__ import annotations

from datetime import datetime, timezone

from evidence.schema import Evidence, EvidenceSource, EvidenceType


def collect_cti_evidence(attacker_ip: str, threat_intel_result) -> list[Evidence]:
    if threat_intel_result is None:
        return []

    return [
        Evidence(
            source=EvidenceSource.CTI,
            source_id=attacker_ip,
            timestamp=datetime.now(timezone.utc).isoformat(),
            type=EvidenceType.THREAT_INTEL,
            content={
                "attacker_ip": attacker_ip,
                "level": threat_intel_result.level,
                "breakdown": threat_intel_result.breakdown,
            },
            confidence=threat_intel_result.confidence / 100.0,
            provenance="threat_intelligence_engine (MISP/CTI + VT reputation)",
        )
    ]
