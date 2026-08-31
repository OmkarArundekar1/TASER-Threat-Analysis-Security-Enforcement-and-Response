"""
mitre_rule_registry.py
=========================
CYUKTI-reviewed Wazuh rule -> ATT&CK technique mappings (Phase 20
REVIEWED_RULE_MAPPING tier). Consulted by mitre_resolver.resolve_mitre()
only when a given alert has no native Wazuh rule.mitre mapping.

This is a manually-curated registry, not a generated one. Every entry
must be independently justifiable from the Wazuh rule's actual detection
semantics (what specifically the rule matches), not from a rule's
description text or a guess at what its author intended. Do not add an
entry here because a rule fires in a security context "generally" — the
bar is the same one applied to a native Wazuh MITRE mapping: could you
defend this specific technique_id to a reviewer who knows ATT&CK.

Every technique_id listed here is validated against the existing
Neo4j Technique nodes (imported from the vendored MITRE ATT&CK STIX
corpus) at resolution time -- a technique that does not exist, or is
revoked/deprecated, causes the whole entry to be rejected and the
resolution downgrades to UNKNOWN. See mitre_resolver.py.

Keyed by Wazuh rule ID as a string, matching alert["rule"]["id"]'s type
in the raw Wazuh alert JSON.

Intentionally starts empty. Do NOT populate this with mappings for
currently-unmapped lab rules (40704, 52002, 2904, 86601, 2902, 531,
210020) without an explicit, independent review -- most of those are
routine OS/package-manager/service telemetry, not attacker behavior,
and forcing a technique onto them would fabricate ground truth.
"""

from __future__ import annotations

REVIEWED_RULE_MAPPINGS: dict[str, dict] = {
    # "5401": {
    #     "technique_ids": ("T1548.003",),
    #     "rationale": "Example only -- 5401 already has a native Wazuh "
    #                  "mitre mapping, so it would never reach this tier.",
    #     "reviewer": "<name>",
    #     "enabled": True,
    # },
}
