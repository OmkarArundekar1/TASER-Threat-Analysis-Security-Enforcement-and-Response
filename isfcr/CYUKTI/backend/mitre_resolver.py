"""
mitre_resolver.py
=====================
Phase 20 MITRE ATT&CK resolution layer.

Problem this solves: realtime_socgraph.process_alert() used to require
alert["rule"]["mitre"] to be present, discarding every Wazuh alert
without a native MITRE mapping -- most legitimate telemetry (systemd
failures, AppArmor denials, package-manager activity, disk usage) has no
native mapping and was previously invisible to CYUKTI entirely.

This module separates "does this alert carry a defensible ATT&CK
technique" from "should CYUKTI ingest this alert at all" -- ingestion is
now unconditional (see realtime_socgraph.py); MITRE attribution is
enrichment, resolved here with an explicit precedence and provenance
trail, and UNKNOWN is a first-class, valid outcome rather than a reason
to drop the event.

Resolution precedence (first match wins; earlier tiers are never
reinterpreted or overridden by later ones):

    1. NATIVE_WAZUH           rule.mitre.id, exactly as Wazuh supplied it.
    2. REVIEWED_RULE_MAPPING  mitre_rule_registry.REVIEWED_RULE_MAPPINGS,
                              keyed by Wazuh rule ID.
    3. DETERMINISTIC_INFERENCE  explicit structural rules only -- no fuzzy
                              matching, no semantic similarity, no LLM
                              inference. Ships with zero live rules; see
                              DETERMINISTIC_INFERENCE_RULES below.
    4. UNKNOWN                technique_ids=() -- never fabricated.

REVIEWED_RULE_MAPPING and DETERMINISTIC_INFERENCE results are validated
against the existing Neo4j Technique nodes (imported from the vendored
MITRE ATT&CK STIX corpus by mitre_import/) before being accepted -- a
technique that doesn't exist, or is revoked/deprecated, causes the whole
result to downgrade to UNKNOWN. NATIVE_WAZUH results are NOT subject to
this veto: a native Wazuh mapping is never reinterpreted by CYUKTI.

Confidence is an ordinal category, not a fabricated statistical
probability -- nothing in this deterministic design produces a real
probability distribution over techniques.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

from neo4j_client import driver
from mitre_rule_registry import REVIEWED_RULE_MAPPINGS

RESOLVER_VERSION = "1.0.0"

PROVENANCE_NATIVE_WAZUH = "NATIVE_WAZUH"
PROVENANCE_REVIEWED_RULE_MAPPING = "REVIEWED_RULE_MAPPING"
PROVENANCE_DETERMINISTIC_INFERENCE = "DETERMINISTIC_INFERENCE"
PROVENANCE_AMBIGUOUS = "AMBIGUOUS"
PROVENANCE_UNKNOWN = "UNKNOWN"

CONFIDENCE_CONFIRMED = "CONFIRMED"
CONFIDENCE_REVIEWED = "REVIEWED"
CONFIDENCE_CANDIDATE = "CANDIDATE"
CONFIDENCE_NONE = "NONE"


@dataclass(frozen=True)
class MitreResolution:
    technique_ids: tuple[str, ...]
    provenance: str
    confidence: str
    reason: str
    resolver_version: str = RESOLVER_VERSION


def _unknown(reason: str) -> MitreResolution:
    return MitreResolution(
        technique_ids=(),
        provenance=PROVENANCE_UNKNOWN,
        confidence=CONFIDENCE_NONE,
        reason=reason,
    )


def _validate_technique_ids(technique_ids: tuple[str, ...]) -> tuple[bool, str]:
    """Checks every technique_id against the existing Neo4j Technique
    nodes (imported from the vendored MITRE ATT&CK STIX corpus -- see
    mitre_import/). Returns (all_valid, reason_if_not). A technique_id
    that doesn't exist, or exists but is revoked/deprecated, fails
    validation -- the caller downgrades the whole result to UNKNOWN
    rather than accepting a partially-valid technique list, to avoid
    ambiguity about which half of a rejected mapping is trustworthy.
    """
    with driver.session() as session:
        for technique_id in technique_ids:
            record = session.run(
                "MATCH (t:Technique {attack_id: $id}) RETURN t.revoked AS revoked, t.deprecated AS deprecated",
                id=technique_id,
            ).single()
            if record is None:
                return False, f"{technique_id} does not exist in the imported ATT&CK STIX corpus"
            if record["revoked"]:
                return False, f"{technique_id} is revoked in ATT&CK"
            if record["deprecated"]:
                return False, f"{technique_id} is deprecated in ATT&CK"
    return True, ""


def _resolve_native(alert: dict) -> Optional[MitreResolution]:
    rule = alert.get("rule") or {}
    mitre = rule.get("mitre") or {}
    technique_ids = tuple(mitre.get("id") or [])

    if not technique_ids:
        return None

    return MitreResolution(
        technique_ids=technique_ids,
        provenance=PROVENANCE_NATIVE_WAZUH,
        confidence=CONFIDENCE_CONFIRMED,
        reason="Wazuh rule.mitre.id",
    )


def _resolve_reviewed(alert: dict) -> Optional[MitreResolution]:
    rule = alert.get("rule") or {}
    rule_id = rule.get("id")
    if rule_id is None:
        return None

    entry = REVIEWED_RULE_MAPPINGS.get(str(rule_id))
    if entry is None or not entry.get("enabled", True):
        return None

    technique_ids = tuple(entry.get("technique_ids") or ())
    if not technique_ids:
        return None

    valid, invalid_reason = _validate_technique_ids(technique_ids)
    if not valid:
        return _unknown(
            f"Reviewed mapping for rule {rule_id} rejected: {invalid_reason}"
        )

    rationale = entry.get("rationale", "")
    return MitreResolution(
        technique_ids=technique_ids,
        provenance=PROVENANCE_REVIEWED_RULE_MAPPING,
        confidence=CONFIDENCE_REVIEWED,
        reason=f"Reviewed mapping for rule {rule_id}" + (f": {rationale}" if rationale else ""),
    )


# Deterministic inference rule contract: a callable taking the raw alert
# dict and returning either
#   - None                    (this rule does not apply to this alert)
#   - a 1-tuple of technique_ids (a confident, unambiguous match)
#   - a tuple of >1 technique_ids (multiple equally-plausible candidates
#     -- treated as AMBIGUOUS, never resolved to an arbitrary one of them)
# No fuzzy matching, semantic similarity, or LLM inference is permitted
# here -- only explicit structural checks against alert/rule fields
# (rule ID, groups, decoder, description) that are independently
# justifiable from Wazuh's own semantics.
#
# Intentionally empty: no alert observed in the current lab data has a
# structural signal strong enough to justify a deterministic rule here.
# Add entries only when a specific, defensible rule can be written and
# tested -- do not populate this to raise coverage numbers.
DETERMINISTIC_INFERENCE_RULES: list[Callable[[dict], Optional[tuple[str, ...]]]] = []


def _resolve_inference(alert: dict) -> Optional[MitreResolution]:
    for rule_fn in DETERMINISTIC_INFERENCE_RULES:
        candidates = rule_fn(alert)
        if not candidates:
            continue

        if len(candidates) > 1:
            return MitreResolution(
                technique_ids=(),
                provenance=PROVENANCE_AMBIGUOUS,
                confidence=CONFIDENCE_NONE,
                reason=(
                    f"Deterministic inference rule {rule_fn!r} produced "
                    f"multiple candidates {candidates} with no defensible "
                    "way to choose one"
                ),
            )

        valid, invalid_reason = _validate_technique_ids(candidates)
        if not valid:
            return _unknown(
                f"Deterministic inference candidate from {rule_fn!r} rejected: {invalid_reason}"
            )

        return MitreResolution(
            technique_ids=candidates,
            provenance=PROVENANCE_DETERMINISTIC_INFERENCE,
            confidence=CONFIDENCE_CANDIDATE,
            reason=f"Deterministic inference rule {rule_fn!r} matched",
        )

    return None


def resolve_mitre(alert: dict) -> MitreResolution:
    native = _resolve_native(alert)
    if native is not None:
        return native

    reviewed = _resolve_reviewed(alert)
    if reviewed is not None:
        return reviewed

    inferred = _resolve_inference(alert)
    if inferred is not None:
        return inferred

    return _unknown("No native, reviewed, or deterministically-inferred ATT&CK mapping")
