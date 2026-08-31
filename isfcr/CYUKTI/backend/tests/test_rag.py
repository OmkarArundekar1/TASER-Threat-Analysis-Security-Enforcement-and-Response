"""
Tests for the Multi-RAG semantic retrieval layer (backend/rag).

test_semantic_retriever_* use a small hand-built corpus for fast,
deterministic unit testing. test_mitre_retriever_* run against the REAL
858-technique MITRE ATT&CK corpus (mitre_import/mitredata) — these
confirm the retriever surfaces genuinely relevant techniques for a
plain-language query, not just that it runs without crashing.
"""

import pytest

from evidence.schema import EvidenceSource, EvidenceType
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
