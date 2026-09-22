from dataclasses import dataclass
from typing import List

from attribution_context import context
from attribution_similarity import AttributionSimilarity
from attribution_models import ThreatAttributionResult
from threat_actor_context import ThreatActorContext


TOP_K = 5


def _gnn_topology_similarity_for_attribution(campaign_id_a: str, campaign_id_b: str) -> float | None:
    """Additive, informational only -- see ThreatActorContext.topology_similarity's
    docstring. Fails safe to None for any reason, never raises into
    attribute()."""
    try:
        from ml.gnn.topology_similarity import gnn_topology_similarity_between_campaigns
        return gnn_topology_similarity_between_campaigns(campaign_id_a, campaign_id_b)
    except Exception:
        import logging
        logging.getLogger(__name__).exception(
            "GNN topology similarity computation failed for attribution (%s vs %s) -- "
            "continuing without it.", campaign_id_a, campaign_id_b,
        )
        return None


class ThreatAttributionEngine:
    def attribute(
        self,
        campaign_context,
    ) -> ThreatAttributionResult:
        observed = set(
            campaign_context.techniques
        )
        candidates = []
        historical_campaigns = context.load_historical_campaigns()
        for campaign in historical_campaigns:
            historical = set(
                campaign.techniques
            )
            
            coverage = AttributionSimilarity.coverage(
                observed,
                historical,
            )
                        
            precision = AttributionSimilarity.precision(
                observed,
                historical,
            )
            campaign_chain = list(
                campaign_context.attack_chain
            )
            
            historical_chain = campaign.techniques
            chain_similarity = AttributionSimilarity.chain_similarity(
                campaign_chain,
                historical_chain,
            )
            similarity = (
                coverage * 0.50
                +
                precision * 0.20
                +
                chain_similarity * 0.30
            )
            if similarity == 0:
                continue
            matched = sorted(
                observed &
                historical
            )
            # Additive, informational only -- see ThreatActorContext's own
            # field docstring. Computed from `similarity` above's already-
            # decided candidate, never influences it or the eventual sort
            # (candidates.sort(key=lambda x: x.total_score, ...) below is
            # unchanged and never reads topology_similarity).
            topology_similarity = _gnn_topology_similarity_for_attribution(
                campaign_context.campaign_id, campaign.campaign_id,
            )
            result = ThreatActorContext(
                actor=campaign.campaign_id,
                confidence=round(
                    similarity * 100,
                    2,
                ),
                
                technique_similarity=round(
                    coverage * 100,
                    2,
                ),
                matched_techniques=matched,
                chain_similarity=round(chain_similarity * 100, 2),
                evidence=[
                    f"Coverage : {coverage * 100:.2f}%",
                    f"Precision: {precision * 100:.2f}%",
                    f"Matched {len(matched)} ATT&CK techniques",
                ],

                total_score=round(
                    similarity * 100,
                    2,
                ),
                coverage=round(
                    coverage * 100,
                    2,
                ),
                
                precision=round(
                    precision * 100,
                    2,
                ),
                topology_similarity=topology_similarity,
            )
            candidates.append(
                result
            )
        candidates.sort(
            key=lambda x: x.total_score,
            reverse=True,
        )
        return ThreatAttributionResult(
            actors=candidates[:TOP_K]
        )
engine = ThreatAttributionEngine()
