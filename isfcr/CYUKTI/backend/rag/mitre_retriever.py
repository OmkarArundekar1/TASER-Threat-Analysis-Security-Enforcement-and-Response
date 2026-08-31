"""
rag/mitre_retriever.py
=========================
Concrete Multi-RAG source: semantic search over the real MITRE ATT&CK
technique corpus (858 attack-pattern objects, loaded via
mitre_import.parser from the restored enterprise-attack.json STIX
file — see mitre_import/config.py:STIX_FILE), indexed with the generic
TF-IDF retriever in rag/retriever.py.

This is complementary to mitre_mapper.py's keyword-lookup approach and
mitre_feature_engine.py's id-based Neo4j lookup: those work when you
already know the technique id or an exact keyword match; this retriever
answers "which ATT&CK techniques best match this free-text description
of observed behavior," which is exactly the kind of query the keyword
table can miss.
"""

from __future__ import annotations

from datetime import datetime, timezone

from evidence.schema import Evidence, EvidenceSource, EvidenceType
from rag.retriever import SemanticRetriever


def _technique_documents() -> list[dict]:
    from mitre_import.parser import get_by_type

    documents = []
    for obj in get_by_type("attack-pattern"):
        if obj.get("revoked") or obj.get("x_mitre_deprecated"):
            continue

        attack_id = None
        for ref in obj.get("external_references", []):
            if ref.get("source_name") == "mitre-attack":
                attack_id = ref.get("external_id")
                break
        if not attack_id:
            continue

        name = obj.get("name", "")
        description = obj.get("description", "")
        documents.append({
            "doc_id": attack_id,
            "text": f"{name}. {description}",
            "name": name,
            "kill_chain_phases": [p.get("phase_name") for p in obj.get("kill_chain_phases", [])],
        })

    return documents


class MitreSemanticRetriever:
    """Lazily builds and caches the TF-IDF index over the real ATT&CK corpus."""

    def __init__(self) -> None:
        self._retriever: SemanticRetriever | None = None

    def _ensure_indexed(self) -> SemanticRetriever:
        if self._retriever is None:
            documents = _technique_documents()
            retriever = SemanticRetriever()
            retriever.index(documents)
            self._retriever = retriever
        return self._retriever

    def query(self, query_text: str, top_k: int = 5, min_relevance: float = 0.05) -> list[Evidence]:
        retriever = self._ensure_indexed()
        results = retriever.query(query_text, top_k=top_k, min_relevance=min_relevance)

        return [
            Evidence(
                source=EvidenceSource.MITRE,
                source_id=doc.doc_id,
                timestamp=datetime.now(timezone.utc).isoformat(),
                type=EvidenceType.TECHNIQUE_KNOWLEDGE,
                content={
                    "attack_id": doc.doc_id,
                    "name": doc.metadata.get("name"),
                    "matched_text": doc.text,
                    "kill_chain_phases": doc.metadata.get("kill_chain_phases"),
                },
                confidence=1.0,  # authoritative ATT&CK reference text
                relevance=doc.relevance,  # query-specific semantic match strength
                provenance="rag.mitre_retriever (TF-IDF over ATT&CK STIX corpus)",
                relationships=[doc.doc_id],
            )
            for doc in results
        ]


mitre_retriever = MitreSemanticRetriever()
