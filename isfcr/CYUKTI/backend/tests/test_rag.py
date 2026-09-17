"""
Tests for the Multi-RAG semantic retrieval layer (backend/rag).

test_semantic_retriever_* use a small hand-built corpus for fast,
deterministic unit testing. test_mitre_retriever_* run against the REAL
858-technique MITRE ATT&CK corpus (mitre_import/mitredata) — these
confirm the retriever surfaces genuinely relevant techniques for a
plain-language query, not just that it runs without crashing.
"""

import pytest

from attribution_models import HistoricalCampaign
from evidence.schema import EvidenceSource, EvidenceType
from rag.campaign_retriever import CampaignNarrativeRetriever
from rag.mitre_retriever import MitreSemanticRetriever
from rag.retriever import SemanticRetriever

_CORPUS = [
    {"doc_id": "d1", "text": "Adversaries may guess or brute force passwords to gain access to accounts."},
    {"doc_id": "d2", "text": "Adversaries may scan networks to discover open ports and running services."},
    {"doc_id": "d3", "text": "Adversaries may encrypt data on target systems to disrupt availability."},
]


def test_index_requires_documents():
    retriever = SemanticRetriever()
    with pytest.raises(ValueError, match="empty document set"):
        retriever.index([])


def test_query_before_index_raises():
    retriever = SemanticRetriever()
    with pytest.raises(RuntimeError, match="not been indexed"):
        retriever.query("anything")


def test_query_ranks_most_similar_document_first():
    retriever = SemanticRetriever()
    retriever.index(_CORPUS)

    results = retriever.query("attacker is guessing account passwords repeatedly", top_k=3)
    assert results[0].doc_id == "d1"
    assert results[0].relevance > 0


def test_min_relevance_filters_unrelated_documents():
    retriever = SemanticRetriever()
    retriever.index(_CORPUS)

    results = retriever.query("brute force password guessing", top_k=3, min_relevance=0.3)
    doc_ids = {r.doc_id for r in results}
    assert "d1" in doc_ids
    assert "d3" not in doc_ids  # ransomware-style doc shares no vocabulary with the query


# ---------------------------------------------------------------------------
# Real MITRE corpus (mitre_import/mitredata/attack-stix-data)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def mitre_retriever():
    return MitreSemanticRetriever()


def test_mitre_retriever_indexes_hundreds_of_real_techniques(mitre_retriever):
    retrieved = mitre_retriever.query("password", top_k=1)
    internal = mitre_retriever._ensure_indexed()
    assert len(internal._documents) > 500  # real corpus has 858 non-deprecated techniques


def test_mitre_retriever_surfaces_relevant_technique_for_brute_force_query(mitre_retriever):
    results = mitre_retriever.query(
        "attacker repeatedly guessing user account passwords to log in", top_k=5,
    )
    assert len(results) > 0
    names = [r.content["name"] for r in results]
    assert any("brute force" in n.lower() or "password" in n.lower() or "credential" in n.lower() for n in names), names


def test_mitre_retriever_returns_evidence_with_correct_provenance(mitre_retriever):
    results = mitre_retriever.query("phishing email with malicious attachment", top_k=3)
    for e in results:
        assert e.source == EvidenceSource.MITRE
        assert e.type == EvidenceType.TECHNIQUE_KNOWLEDGE
        assert e.confidence == 1.0
        assert 0.0 <= e.relevance <= 1.0
        assert "rag.mitre_retriever" in e.provenance


# ---------------------------------------------------------------------------
# Second Multi-RAG source: real historical campaign records
# (attribution_context.py's HistoricalCampaign -- the same data
# threat_attribution_engine.py and campaign_history_collector.py already
# consume). Neo4j is the only external boundary, mocked here the same
# way tests/test_threat_attribution.py mocks it for the same function.
# ---------------------------------------------------------------------------

def _historical(campaign_id, techniques, attacker="203.0.113.9", victim="198.51.100.9", status="ARCHIVED"):
    return HistoricalCampaign(
        campaign_id=campaign_id, attacker=attacker, victim=victim,
        techniques=list(techniques), timestamps=["2026-01-01T00:00:00+00:00"] * len(techniques),
        status=status,
    )


@pytest.fixture()
def mock_historical_campaigns(monkeypatch):
    import attribution_context as attribution_context_module

    state = {"campaigns": []}
    monkeypatch.setattr(
        attribution_context_module.context, "load_historical_campaigns",
        lambda: list(state["campaigns"]),
    )
    return state


def test_campaign_retriever_raises_a_controlled_error_when_no_history_exists_yet(mock_historical_campaigns):
    mock_historical_campaigns["campaigns"] = []
    retriever = CampaignNarrativeRetriever()
    with pytest.raises(RuntimeError, match="No historical campaigns"):
        retriever.query("brute force followed by lateral movement")


def test_campaign_retriever_excludes_campaigns_with_no_resolved_techniques(mock_historical_campaigns):
    mock_historical_campaigns["campaigns"] = [_historical("CAMP_EMPTY", [])]
    retriever = CampaignNarrativeRetriever()
    with pytest.raises(RuntimeError, match="No historical campaigns"):
        retriever.query("anything")


def test_campaign_retriever_ranks_the_real_matching_campaign_first(mock_historical_campaigns):
    mock_historical_campaigns["campaigns"] = [
        _historical("CAMP_BRUTE_FORCE", ["T1110", "T1110.001", "T1078"]),
        _historical("CAMP_RANSOMWARE", ["T1486", "T1490"]),
    ]
    retriever = CampaignNarrativeRetriever()

    results = retriever.query("T1110 T1110.001 T1078", top_k=2)

    assert results[0].source_id == "CAMP_BRUTE_FORCE"
    assert results[0].relevance > 0


def test_campaign_retriever_returns_evidence_with_correct_contract(mock_historical_campaigns):
    mock_historical_campaigns["campaigns"] = [_historical("CAMP_OLD_1", ["T1110", "T1078"])]
    retriever = CampaignNarrativeRetriever()

    results = retriever.query("T1110 T1078", top_k=3)

    assert results
    for e in results:
        assert e.source == EvidenceSource.CAMPAIGN_HISTORY
        assert e.type == EvidenceType.HISTORICAL_MATCH
        assert e.confidence == 1.0
        assert 0.0 <= e.relevance <= 1.0
        assert "rag.campaign_retriever" in e.provenance
        assert e.content["campaign_id"] == "CAMP_OLD_1"
        assert e.content["techniques"] == ["T1110", "T1078"]
