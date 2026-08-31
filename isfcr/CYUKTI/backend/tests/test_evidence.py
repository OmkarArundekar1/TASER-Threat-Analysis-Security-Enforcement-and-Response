"""
Unit tests for the evidence layer (backend/evidence). All fixtures are
synthetic hand-built dataclass instances — no live Neo4j/MISP connection
required, and nothing here claims to represent real detections.
"""

from attribution_models import HistoricalCampaign
from detection_confidence_engine import DetectionResult
from evidence.collectors.attribution_collector import collect_attribution_evidence
from evidence.collectors.campaign_history_collector import collect_campaign_history_evidence
from evidence.collectors.cti_collector import collect_cti_evidence
from evidence.collectors.detection_collector import collect_detection_evidence
from evidence.collectors.graph_collector import collect_graph_evidence
from evidence.collectors.mitre_collector import collect_mitre_evidence
from evidence.schema import Evidence, EvidenceSource, EvidenceType
from evidence.store import EvidenceStore
from graph_feature_engine import GraphFeatures
from neo4j_client import MitreFeatures
from threat_actor_context import ThreatActorContext
from threat_attribution_engine import ThreatAttributionResult
from threat_intelligence_engine import ThreatIntelResult


def make_mitre_features(**overrides):
    defaults = dict(
        attack_id="T1110",
        technique_name="Brute Force",
        description="Adversaries may use brute force techniques.",
        platforms=["Windows", "Linux"],
        platform_count=2,
        domains=["enterprise-attack"],
        domain_count=1,
        kill_chain_phases=["credential-access"],
        kill_chain_count=1,
        is_subtechnique=False,
        deprecated=False,
        revoked=False,
        object_version="1.0",
        threat_actor_count=3,
        malware_count=1,
        tool_count=2,
        mitigation_count=4,
        subtechnique_count=4,
        parent_technique=None,
    )
    defaults.update(overrides)
    return MitreFeatures(**defaults)


# ---------------------------------------------------------------------------
# Evidence / EvidenceStore
# ---------------------------------------------------------------------------

def test_evidence_clamps_confidence_and_relevance():
    e = Evidence(
        source=EvidenceSource.MITRE,
        source_id="T1110",
        timestamp="2026-01-01T00:00:00+00:00",
        type=EvidenceType.TECHNIQUE_KNOWLEDGE,
        content={},
        confidence=1.5,
        relevance=-0.2,
    )
    assert e.confidence == 1.0
    assert e.relevance == 0.0


def test_evidence_id_is_stable_and_dedups_in_store():
    e1 = Evidence(
        source=EvidenceSource.MITRE, source_id="T1110", timestamp="t",
        type=EvidenceType.TECHNIQUE_KNOWLEDGE, content={"a": 1}, confidence=1.0,
    )
    e2 = Evidence(
        source=EvidenceSource.MITRE, source_id="T1110", timestamp="t2",
        type=EvidenceType.TECHNIQUE_KNOWLEDGE, content={"a": 2}, confidence=0.5,
    )
    store = EvidenceStore()
    assert store.add(e1) is True
    assert store.add(e2) is False  # same evidence_id (source, type, source_id) -> dedup
    assert len(store) == 1


def test_weighted_confidence_uses_relevance_as_weight():
    store = EvidenceStore()
    store.add(Evidence(
        source=EvidenceSource.CTI, source_id="1.2.3.4", timestamp="t",
        type=EvidenceType.THREAT_INTEL, content={}, confidence=1.0, relevance=1.0,
    ))
    store.add(Evidence(
        source=EvidenceSource.SIEM, source_id="evt1", timestamp="t",
        type=EvidenceType.DETECTION, content={}, confidence=0.0, relevance=0.5,
    ))
    # weighted mean = (1.0*1.0 + 0.0*0.5) / (1.0 + 0.5) = 0.667
    assert store.weighted_confidence() == round(1.0 / 1.5, 4)


def test_weighted_confidence_ignores_zero_relevance_evidence():
    store = EvidenceStore()
    store.add(Evidence(
        source=EvidenceSource.MITRE, source_id="T1595", timestamp="t",
        type=EvidenceType.TECHNIQUE_KNOWLEDGE, content={}, confidence=1.0, relevance=0.0,
    ))
    # a single authoritative-but-irrelevant fact must NOT read as high
    # confidence — this exact scenario made the real investigation loop
    # stop after one MITRE lookup on every real campaign tested.
    assert store.weighted_confidence() == 0.0


def test_detect_conflicts_flags_divergent_confidence_same_entity():
    store = EvidenceStore()
    store.add(Evidence(
        source=EvidenceSource.CTI, source_id="1.2.3.4", timestamp="t",
        type=EvidenceType.THREAT_INTEL, content={}, confidence=0.9,
        relationships=["1.2.3.4"],
    ))
    store.add(Evidence(
        source=EvidenceSource.SIEM, source_id="evt1", timestamp="t",
        type=EvidenceType.DETECTION, content={}, confidence=0.1,
        relationships=["1.2.3.4"],
    ))
    conflicts = store.detect_conflicts()
    assert len(conflicts) == 1
    assert conflicts[0].entity_id == "1.2.3.4"
    assert conflicts[0].confidence_gap == 0.8


def test_detect_conflicts_ignores_same_source_disagreement():
    store = EvidenceStore()
    store.add(Evidence(
        source=EvidenceSource.CTI, source_id="a", timestamp="t",
        type=EvidenceType.THREAT_INTEL, content={}, confidence=0.9, relationships=["x"],
    ))
    store.add(Evidence(
        source=EvidenceSource.CTI, source_id="b", timestamp="t",
        type=EvidenceType.THREAT_INTEL, content={}, confidence=0.1, relationships=["x"],
    ))
    assert store.detect_conflicts() == []


# ---------------------------------------------------------------------------
# Collectors
# ---------------------------------------------------------------------------

def test_mitre_collector_authoritative_confidence():
    ev = collect_mitre_evidence(make_mitre_features())
    assert len(ev) == 1
    assert ev[0].confidence == 1.0
    assert ev[0].source == EvidenceSource.MITRE
    assert ev[0].content["attack_id"] == "T1110"


def test_mitre_collector_deprecated_lowers_confidence():
    ev = collect_mitre_evidence(make_mitre_features(deprecated=True))
    assert ev[0].confidence == 0.3


def test_mitre_collector_none_input_returns_empty():
    assert collect_mitre_evidence(None) == []


def test_cti_collector_normalizes_0_100_scale():
    result = ThreatIntelResult(confidence=82.0, level="MEDIUM", breakdown={"VT": 80})
    ev = collect_cti_evidence("1.2.3.4", result)
    assert ev[0].confidence == 0.82
    assert ev[0].content["level"] == "MEDIUM"


def test_detection_collector_normalizes_0_100_scale():
    result = DetectionResult(confidence=95.0, level="HIGH", breakdown={"Wazuh": 10})
    ev = collect_detection_evidence("evt-1", result)
    assert ev[0].confidence == 0.95
    assert ev[0].source == EvidenceSource.SIEM


def test_attribution_collector_ranks_relevance_by_decay():
    result = ThreatAttributionResult(actors=[
        ThreatActorContext(actor="camp-A", confidence=90.0, evidence=["match"]),
        ThreatActorContext(actor="camp-B", confidence=90.0, evidence=["match"]),
    ])
    ev = collect_attribution_evidence(result)
    assert len(ev) == 2
    assert ev[0].relevance > ev[1].relevance  # rank 0 decays less than rank 1
    assert ev[0].confidence == ev[1].confidence == 0.9  # confidence itself is unaffected by rank


def test_attribution_collector_empty_actors_returns_empty():
    assert collect_attribution_evidence(ThreatAttributionResult(actors=[])) == []


def test_campaign_history_collector_relevance_is_technique_coverage():
    campaigns = [
        HistoricalCampaign(campaign_id="c1", attacker="1.1.1.1", victim="2.2.2.2",
                            techniques=["T1110", "T1078"]),
        HistoricalCampaign(campaign_id="c2", attacker="3.3.3.3", victim="4.4.4.4",
                            techniques=["T9999"]),
    ]
    ev = collect_campaign_history_evidence(observed_techniques=["T1110", "T1078"], historical_campaigns=campaigns)
    by_id = {e.source_id: e for e in ev}
    assert by_id["c1"].relevance == 1.0  # full coverage
    assert by_id["c2"].relevance == 0.0  # no overlap
    assert by_id["c1"].confidence == 1.0  # historical fact, not an estimate


def test_graph_collector_relevance_from_structural_risk():
    features = GraphFeatures()
    features.node_count = 5
    features.edge_count = 4
    features.structural_risk = 20.0  # half of the 40.0 normalizer
    ev = collect_graph_evidence("camp-1", features)
    assert ev[0].relevance == 0.5
    assert ev[0].confidence == 1.0
