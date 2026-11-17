"""
Wraps a MitreFeatures result (backend/neo4j_client.py, populated by
mitre_feature_engine) as Evidence. MITRE ATT&CK reference data is treated
as authoritative (confidence 1.0) unless the technique has been
deprecated/revoked upstream, in which case it's still valid evidence but
flagged as stale.
"""

from __future__ import annotations

from datetime import datetime, timezone

from evidence.schema import Evidence, EvidenceSource, EvidenceType


def collect_mitre_evidence(mitre_features) -> list[Evidence]:
    if mitre_features is None:
        return []

    confidence = 0.3 if (mitre_features.deprecated or mitre_features.revoked) else 1.0

    return [
        Evidence(
            source=EvidenceSource.MITRE,
            source_id=mitre_features.attack_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
            type=EvidenceType.TECHNIQUE_KNOWLEDGE,
            content={
                "attack_id": mitre_features.attack_id,
                "technique_name": mitre_features.technique_name,
                "description": mitre_features.description,
                "platforms": mitre_features.platforms,
                "kill_chain_phases": mitre_features.kill_chain_phases,
                "is_subtechnique": mitre_features.is_subtechnique,
                "parent_technique": mitre_features.parent_technique,
                "threat_actor_count": mitre_features.threat_actor_count,
                "malware_count": mitre_features.malware_count,
                "tool_count": mitre_features.tool_count,
                "mitigation_count": mitre_features.mitigation_count,
                "deprecated": mitre_features.deprecated,
                "revoked": mitre_features.revoked,
            },
            confidence=confidence,
            provenance="mitre_feature_engine (ATT&CK STIX corpus)",
            relationships=[mitre_features.parent_technique] if mitre_features.parent_technique else [],
        )
    ]
