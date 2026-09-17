"""
rag/campaign_retriever.py
============================
Second concrete Multi-RAG source (see rag/retriever.py's own docstring:
"the same class can index CTI/MISP event descriptions or
historical-campaign narratives" — this is that second instance).
Semantic search over real, resolved historical campaign records
(attribution_context.py's HistoricalCampaign, the same source
threat_attribution_engine.py and evidence/collectors/campaign_history_collector.py
already consume), indexed with the generic TF-IDF retriever in
rag/retriever.py.

This is complementary to, not a duplicate of, the existing
CAMPAIGN_HISTORY investigation action
(evidence/collectors/campaign_history_collector.py): that action
returns every historical campaign that shares at least one technique
with the current one (exhaustive, structured, set-overlap). This
retriever answers a genuinely different question — "which past
campaigns best match this free-text description of observed
behavior" — exactly the same complementary relationship
rag/mitre_retriever.py already has with mitre_mapper.py's exact
keyword lookup (see that module's own docstring). No new provenance or
evidence mechanism is introduced: results become ordinary Evidence,
same as every other collector.

Document text is built entirely from real HistoricalCampaign fields
(attacker, victim, technique IDs) — nothing is fabricated or invented;
this is a real, deterministic formatting of real structured data into
searchable text, the same pattern mitre_retriever.py already uses for
real STIX fields.
"""

from __future__ import annotations

from datetime import datetime, timezone

from evidence.schema import Evidence, EvidenceSource, EvidenceType
from rag.retriever import SemanticRetriever


def _campaign_documents() -> list[dict]:
    from attribution_context import context

    documents = []
    for campaign in context.load_historical_campaigns():
        if not campaign.techniques:
            continue
        text = (
            f"Campaign {campaign.campaign_id}: attacker {campaign.attacker} "
            f"targeting {campaign.victim}. Techniques observed: {' '.join(campaign.techniques)}."
        )
        documents.append({
            "doc_id": campaign.campaign_id,
            "text": text,
            "attacker": campaign.attacker,
            "victim": campaign.victim,
            "techniques": campaign.techniques,
            "status": campaign.status,
        })

    return documents


class CampaignNarrativeRetriever:
    """Builds a TF-IDF index over real historical campaign records at
    construction time (via _ensure_indexed on first query) and caches it
    for this instance's lifetime. Deliberately NOT exposed as a
    module-level singleton the way mitre_retriever.py's MitreSemanticRetriever
    is: that corpus (the vendored STIX file) never changes at runtime, so
    caching it for the whole process's lifetime is correct; the
    historical-campaign corpus grows as real campaigns resolve, so
    investigation/loop.py's default_action_executor constructs a fresh
    instance per investigation instead, trading a small amount of
    redundant indexing for not silently searching a stale corpus.
    """

    def __init__(self) -> None:
        self._retriever: SemanticRetriever | None = None

    def _ensure_indexed(self) -> SemanticRetriever:
        if self._retriever is None:
            documents = _campaign_documents()
            if not documents:
                raise RuntimeError(
                    "No historical campaigns with resolved techniques are "
                    "available yet to index — the campaign-narrative RAG "
                    "source has nothing to search until at least one real "
                    "campaign has resolved (see attribution_context.py)."
                )
            retriever = SemanticRetriever()
            retriever.index(documents)
            self._retriever = retriever
        return self._retriever

    def query(self, query_text: str, top_k: int = 5, min_relevance: float = 0.05) -> list[Evidence]:
        retriever = self._ensure_indexed()
        results = retriever.query(query_text, top_k=top_k, min_relevance=min_relevance)

        return [
            Evidence(
                source=EvidenceSource.CAMPAIGN_HISTORY,
                source_id=doc.doc_id,
                timestamp=datetime.now(timezone.utc).isoformat(),
                type=EvidenceType.HISTORICAL_MATCH,
                content={
                    "campaign_id": doc.doc_id,
                    "attacker": doc.metadata.get("attacker"),
                    "victim": doc.metadata.get("victim"),
                    "techniques": doc.metadata.get("techniques"),
                    "status": doc.metadata.get("status"),
                    "matched_text": doc.text,
                },
                confidence=1.0,  # a real, resolved historical campaign record
                relevance=doc.relevance,  # query-specific semantic match strength
                provenance="rag.campaign_retriever (TF-IDF over historical campaign records)",
                relationships=[doc.doc_id],
            )
            for doc in results
        ]
