"""
Wraps threat_attribution_engine.py's ThreatAttributionResult (a ranked
list of ThreatActorContext candidates) as Evidence. Each candidate
already carries its own free-text `evidence` list (coverage/precision/
matched-technique strings) — that text becomes the Evidence content, and
the candidate's own similarity score becomes both confidence (how sound
the match is) and relevance (a candidate ranked #1 is presumptively more
relevant to the current investigation than one ranked #5).
"""

from __future__ import annotations

from datetime import datetime, timezone

from evidence.schema import Evidence, EvidenceSource, EvidenceType


def collect_attribution_evidence(attribution_result) -> list[Evidence]:
    if attribution_result is None or not attribution_result.actors:
        return []

    evidence_items = []
    n = len(attribution_result.actors)

    for rank, actor in enumerate(attribution_result.actors):
        score = actor.confidence / 100.0
        # rank-decayed relevance: top match relevance == score, decays toward 0 by the last candidate
        rank_decay = 1.0 - (rank / n) if n > 1 else 1.0

        evidence_items.append(
            Evidence(
                source=EvidenceSource.ATTRIBUTION,
                source_id=actor.actor,
                timestamp=datetime.now(timezone.utc).isoformat(),
                type=EvidenceType.ATTRIBUTION_MATCH,
                content={
                    "candidate_campaign_id": actor.actor,
                    "matched_techniques": actor.matched_techniques,
                    "technique_similarity": actor.technique_similarity,
                    "chain_similarity": actor.chain_similarity,
                    "coverage": actor.coverage,
                    "precision": actor.precision,
                    "notes": actor.evidence,
                    "rank": rank,
                },
                confidence=score,
                relevance=score * rank_decay,
                provenance="threat_attribution_engine (historical campaign similarity)",
                relationships=[actor.actor],
            )
        )

    return evidence_items
