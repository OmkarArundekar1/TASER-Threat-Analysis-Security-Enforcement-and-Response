from dataclasses import dataclass
from typing import List

from attribution_context import context
from attribution_similarity import AttributionSimilarity
from attribution_models import ThreatAttributionResult
from threat_actor_context import ThreatActorContext


TOP_K = 5

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
