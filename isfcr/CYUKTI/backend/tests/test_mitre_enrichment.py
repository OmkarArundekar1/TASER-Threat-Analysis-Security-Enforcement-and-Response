"""
Tests for mitre_resolver.enrich_technique_metadata() -- the per-technique
display/evaluation data model (mitre_id, technique_name, tactic,
mapping_source, mapping_confidence, mapping_reason) requested in
MITRE_MAPPING.md. Separate from resolve_mitre()'s own resolution logic
(already covered by test_mitre_resolver.py) -- this only tests the
additive enrichment layer.
"""

import mitre_resolver
from mitre_resolver import (
    MitreResolution,
    PROVENANCE_NATIVE_WAZUH,
    CONFIDENCE_CONFIRMED,
    PROVENANCE_UNKNOWN,
    CONFIDENCE_NONE,
    enrich_technique_metadata,
    _tactic_labels_from_kill_chain_phases,
)


class _FakeResult:
    def __init__(self, row):
        self._row = row

    def single(self):
        return self._row


class _FakeSession:
    def __init__(self, rows_by_id):
        self._rows_by_id = rows_by_id

    def run(self, query, **kwargs):
        return _FakeResult(self._rows_by_id.get(kwargs.get("id")))

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class _FakeDriver:
    def __init__(self, rows_by_id):
        self._session = _FakeSession(rows_by_id)

    def session(self):
        return self._session


class _RaisingDriver:
    def session(self):
        raise ConnectionError("Neo4j unreachable")


def test_tactic_labels_strips_kill_chain_name_prefix():
    assert _tactic_labels_from_kill_chain_phases(["mitre-attack:credential-access"]) == ["Credential Access"]


def test_tactic_labels_handles_bare_tactic_names():
    assert _tactic_labels_from_kill_chain_phases(["initial-access"]) == ["Initial Access"]


def test_tactic_labels_empty_for_none_or_empty_input():
    assert _tactic_labels_from_kill_chain_phases(None) == []
    assert _tactic_labels_from_kill_chain_phases([]) == []


def test_enrich_returns_empty_list_for_unknown_resolution():
    resolution = MitreResolution(
        technique_ids=(), provenance=PROVENANCE_UNKNOWN, confidence=CONFIDENCE_NONE, reason="no mapping",
    )
    assert enrich_technique_metadata(resolution) == []


def test_enrich_populates_name_and_tactic_from_neo4j(monkeypatch):
    monkeypatch.setattr(mitre_resolver, "driver", _FakeDriver({
        "T1110": {"name": "Brute Force", "phases": ["mitre-attack:credential-access"]},
    }))
    resolution = MitreResolution(
        technique_ids=("T1110",), provenance=PROVENANCE_NATIVE_WAZUH,
        confidence=CONFIDENCE_CONFIRMED, reason="Wazuh rule.mitre.id",
    )
    records = enrich_technique_metadata(resolution)
    assert len(records) == 1
    assert records[0] == {
        "mitre_id": "T1110",
        "technique_name": "Brute Force",
        "tactic": ["Credential Access"],
        "mapping_source": PROVENANCE_NATIVE_WAZUH,
        "mapping_confidence": CONFIDENCE_CONFIRMED,
        "mapping_reason": "Wazuh rule.mitre.id",
    }


def test_enrich_handles_multiple_technique_ids_independently(monkeypatch):
    monkeypatch.setattr(mitre_resolver, "driver", _FakeDriver({
        "T1110.001": {"name": "Password Guessing", "phases": ["mitre-attack:credential-access"]},
        "T1021.004": {"name": "SSH", "phases": ["mitre-attack:lateral-movement"]},
    }))
    resolution = MitreResolution(
        technique_ids=("T1110.001", "T1021.004"), provenance=PROVENANCE_NATIVE_WAZUH,
        confidence=CONFIDENCE_CONFIRMED, reason="Wazuh rule.mitre.id",
    )
    records = enrich_technique_metadata(resolution)
    assert [r["mitre_id"] for r in records] == ["T1110.001", "T1021.004"]
    assert records[0]["technique_name"] == "Password Guessing"
    assert records[1]["tactic"] == ["Lateral Movement"]


def test_enrich_reports_none_name_when_technique_not_found(monkeypatch):
    monkeypatch.setattr(mitre_resolver, "driver", _FakeDriver({}))
    resolution = MitreResolution(
        technique_ids=("T9999",), provenance=PROVENANCE_NATIVE_WAZUH,
        confidence=CONFIDENCE_CONFIRMED, reason="Wazuh rule.mitre.id",
    )
    records = enrich_technique_metadata(resolution)
    assert records[0]["technique_name"] is None
    assert records[0]["tactic"] == []


def test_enrich_never_raises_when_neo4j_unavailable(monkeypatch):
    monkeypatch.setattr(mitre_resolver, "driver", _RaisingDriver())
    resolution = MitreResolution(
        technique_ids=("T1110",), provenance=PROVENANCE_NATIVE_WAZUH,
        confidence=CONFIDENCE_CONFIRMED, reason="Wazuh rule.mitre.id",
    )
    records = enrich_technique_metadata(resolution)
    assert records[0]["technique_name"] is None
    assert records[0]["mapping_source"] == PROVENANCE_NATIVE_WAZUH
