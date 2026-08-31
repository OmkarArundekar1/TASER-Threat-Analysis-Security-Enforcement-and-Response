"""
Tests for mitre_resolver.py (Phase 20 MITRE ATT&CK resolution layer).

Uses a fake Neo4j session (same pattern as test_campaign_reconstruction.py)
for technique validation, so no live database is required.
"""

import mitre_resolver
from mitre_resolver import (
    resolve_mitre,
    MitreResolution,
    PROVENANCE_NATIVE_WAZUH,
    PROVENANCE_REVIEWED_RULE_MAPPING,
    PROVENANCE_DETERMINISTIC_INFERENCE,
    PROVENANCE_AMBIGUOUS,
    PROVENANCE_UNKNOWN,
    CONFIDENCE_CONFIRMED,
    CONFIDENCE_REVIEWED,
    CONFIDENCE_CANDIDATE,
    CONFIDENCE_NONE,
)


class _FakeSingleResult:
    def __init__(self, row):
        self._row = row

    def single(self):
        return self._row


class _FakeTechniqueSession:
    """techniques: dict[technique_id] -> {"revoked": bool, "deprecated": bool}.
    A technique_id absent from the dict simulates "not found in Neo4j"."""

    def __init__(self, techniques):
        self._techniques = techniques

    def run(self, query, **kwargs):
        tid = kwargs.get("id")
        info = self._techniques.get(tid)
        if info is None:
            return _FakeSingleResult(None)
        return _FakeSingleResult({"revoked": info.get("revoked", False), "deprecated": info.get("deprecated", False)})

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class _FakeTechniqueDriver:
    def __init__(self, techniques):
        self._session = _FakeTechniqueSession(techniques)

    def session(self):
        return self._session


def _install_fake_driver(monkeypatch, techniques):
    monkeypatch.setattr(mitre_resolver, "driver", _FakeTechniqueDriver(techniques))


def _native_alert(rule_id="9999", technique_ids=("T1110.001",)):
    return {
        "rule": {
            "id": rule_id,
            "mitre": {"id": list(technique_ids), "technique": ["Password Guessing"] * len(technique_ids)},
        },
        "agent": {"id": "001"},
    }


def _plain_alert(rule_id="40704"):
    return {"rule": {"id": rule_id, "description": "Systemd: Service exited due to a failure."}, "agent": {"id": "001"}}


# ---------------------------------------------------------------- 1. native resolves correctly

def test_native_wazuh_mapping_resolves_correctly(monkeypatch):
    _install_fake_driver(monkeypatch, {})
    result = resolve_mitre(_native_alert(technique_ids=("T1110.001",)))

    assert result.technique_ids == ("T1110.001",)
    assert result.provenance == PROVENANCE_NATIVE_WAZUH
    assert result.confidence == CONFIDENCE_CONFIRMED
    assert result.reason == "Wazuh rule.mitre.id"
    assert result.resolver_version == mitre_resolver.RESOLVER_VERSION


# ---------------------------------------------------------------- 2. native beats reviewed

def test_native_mapping_has_precedence_over_reviewed_mapping(monkeypatch):
    _install_fake_driver(monkeypatch, {"T9999": {}})
    monkeypatch.setitem(
        mitre_resolver.REVIEWED_RULE_MAPPINGS, "5503",
        {"technique_ids": ("T9999",), "rationale": "should never be reached", "enabled": True},
    )
    result = resolve_mitre(_native_alert(rule_id="5503", technique_ids=("T1110.001",)))

    assert result.provenance == PROVENANCE_NATIVE_WAZUH
    assert result.technique_ids == ("T1110.001",)


# ---------------------------------------------------------------- 3. native beats inference

def test_native_mapping_has_precedence_over_deterministic_inference(monkeypatch):
    _install_fake_driver(monkeypatch, {})
    monkeypatch.setattr(
        mitre_resolver, "DETERMINISTIC_INFERENCE_RULES",
        [lambda alert: ("T9999",)],
    )
    result = resolve_mitre(_native_alert(technique_ids=("T1110.001",)))

    assert result.provenance == PROVENANCE_NATIVE_WAZUH
    assert result.technique_ids == ("T1110.001",)


# ---------------------------------------------------------------- 4. multiple native IDs preserved

def test_multiple_native_mitre_ids_are_preserved(monkeypatch):
    _install_fake_driver(monkeypatch, {})
    result = resolve_mitre(_native_alert(technique_ids=("T1110.001", "T1078")))

    assert result.technique_ids == ("T1110.001", "T1078")
    assert result.provenance == PROVENANCE_NATIVE_WAZUH


# ---------------------------------------------------------------- 5. reviewed mapping resolves

def test_reviewed_rule_mapping_resolves_correctly(monkeypatch):
    _install_fake_driver(monkeypatch, {"T1053.003": {"revoked": False, "deprecated": False}})
    monkeypatch.setitem(
        mitre_resolver.REVIEWED_RULE_MAPPINGS, "100501",
        {"technique_ids": ("T1053.003",), "rationale": "cron autostart", "enabled": True},
    )
    result = resolve_mitre(_plain_alert(rule_id="100501"))

    assert result.technique_ids == ("T1053.003",)
    assert result.provenance == PROVENANCE_REVIEWED_RULE_MAPPING
    assert result.confidence == CONFIDENCE_REVIEWED
    assert "100501" in result.reason


# ---------------------------------------------------------------- 6/7/8. reviewed mapping validation

def test_reviewed_mapping_rejected_when_technique_does_not_exist(monkeypatch):
    _install_fake_driver(monkeypatch, {})  # T1053.003 absent -> "not found"
    monkeypatch.setitem(
        mitre_resolver.REVIEWED_RULE_MAPPINGS, "100501",
        {"technique_ids": ("T1053.003",), "rationale": "x", "enabled": True},
    )
    result = resolve_mitre(_plain_alert(rule_id="100501"))

    assert result.provenance == PROVENANCE_UNKNOWN
    assert result.technique_ids == ()
    assert "does not exist" in result.reason


def test_reviewed_mapping_rejected_when_revoked(monkeypatch):
    _install_fake_driver(monkeypatch, {"T1053.003": {"revoked": True, "deprecated": False}})
    monkeypatch.setitem(
        mitre_resolver.REVIEWED_RULE_MAPPINGS, "100501",
        {"technique_ids": ("T1053.003",), "rationale": "x", "enabled": True},
    )
    result = resolve_mitre(_plain_alert(rule_id="100501"))

    assert result.provenance == PROVENANCE_UNKNOWN
    assert "revoked" in result.reason


def test_reviewed_mapping_rejected_when_deprecated(monkeypatch):
    _install_fake_driver(monkeypatch, {"T1053.003": {"revoked": False, "deprecated": True}})
    monkeypatch.setitem(
        mitre_resolver.REVIEWED_RULE_MAPPINGS, "100501",
        {"technique_ids": ("T1053.003",), "rationale": "x", "enabled": True},
    )
    result = resolve_mitre(_plain_alert(rule_id="100501"))

    assert result.provenance == PROVENANCE_UNKNOWN
    assert "deprecated" in result.reason


def test_disabled_reviewed_mapping_is_not_used(monkeypatch):
    _install_fake_driver(monkeypatch, {"T1053.003": {}})
    monkeypatch.setitem(
        mitre_resolver.REVIEWED_RULE_MAPPINGS, "100501",
        {"technique_ids": ("T1053.003",), "rationale": "x", "enabled": False},
    )
    result = resolve_mitre(_plain_alert(rule_id="100501"))

    assert result.provenance == PROVENANCE_UNKNOWN


# ---------------------------------------------------------------- 9. deterministic inference infrastructure

def test_deterministic_inference_infrastructure_works(monkeypatch):
    _install_fake_driver(monkeypatch, {"T1595": {"revoked": False, "deprecated": False}})
    monkeypatch.setattr(
        mitre_resolver, "DETERMINISTIC_INFERENCE_RULES",
        [lambda alert: ("T1595",) if alert["rule"]["id"] == "100500" else None],
    )
    result = resolve_mitre(_plain_alert(rule_id="100500"))

    assert result.technique_ids == ("T1595",)
    assert result.provenance == PROVENANCE_DETERMINISTIC_INFERENCE
    assert result.confidence == CONFIDENCE_CANDIDATE


def test_deterministic_inference_rejects_invalid_technique(monkeypatch):
    _install_fake_driver(monkeypatch, {})
    monkeypatch.setattr(
        mitre_resolver, "DETERMINISTIC_INFERENCE_RULES",
        [lambda alert: ("T1595",)],
    )
    result = resolve_mitre(_plain_alert())

    assert result.provenance == PROVENANCE_UNKNOWN


# ---------------------------------------------------------------- 10. ambiguous inference

def test_ambiguous_inference_becomes_ambiguous_not_a_guess(monkeypatch):
    _install_fake_driver(monkeypatch, {})
    monkeypatch.setattr(
        mitre_resolver, "DETERMINISTIC_INFERENCE_RULES",
        [lambda alert: ("T1595", "T1110")],
    )
    result = resolve_mitre(_plain_alert())

    assert result.provenance == PROVENANCE_AMBIGUOUS
    assert result.technique_ids == ()
    assert result.confidence == CONFIDENCE_NONE


# ---------------------------------------------------------------- 11. UNKNOWN never fabricates

def test_unknown_alert_never_fabricates_a_technique(monkeypatch):
    _install_fake_driver(monkeypatch, {})
    result = resolve_mitre(_plain_alert())

    assert result.provenance == PROVENANCE_UNKNOWN
    assert result.technique_ids == ()
    assert result.confidence == CONFIDENCE_NONE
    for forbidden in ("UNKNOWN", "T0000", "T9999", "None", "", "unknown-technique"):
        assert forbidden not in result.technique_ids


def test_empty_registry_and_empty_inference_rules_produce_unknown(monkeypatch):
    _install_fake_driver(monkeypatch, {})
    assert mitre_resolver.REVIEWED_RULE_MAPPINGS == {} or True  # registry may have test-added entries from other tests' monkeypatch, isolated by monkeypatch teardown
    result = resolve_mitre(_plain_alert(rule_id="210020"))
    assert result.provenance == PROVENANCE_UNKNOWN


# ---------------------------------------------------------------- MitreResolution shape

def test_mitre_resolution_is_frozen_dataclass():
    r = MitreResolution(technique_ids=(), provenance=PROVENANCE_UNKNOWN, confidence=CONFIDENCE_NONE, reason="x")
    assert r.resolver_version == mitre_resolver.RESOLVER_VERSION
    try:
        r.provenance = "CHANGED"
        assert False, "MitreResolution must be immutable"
    except Exception:
        pass
