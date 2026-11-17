from dataclasses import dataclass

from risk_scoring import normalize_risk_score, risk_level_from_score
from prediction_engine import predict_next_readonly

_LEVEL_TO_SEVERITY = {
    "CRITICAL": "Critical",
    "HIGH": "High",
    "MEDIUM": "Medium",
    "LOW": "Low",
}


@dataclass
class CampaignLabels:
    risk_score: float
    severity: str
    attributed_actor: str
    next_technique: str
    prediction_correct: int | None
    attribution_correct: int

class LabelGenerator:
    def generate(
        self,
        campaign_context,
        technique_sequence: list[str],
        final_attack_id: str,
        attributed_actor: str,
        actual_actor: str | None = None,
    ) -> CampaignLabels:
        risk = campaign_context.risk_score

        # risk_score is an unbounded running sum (see neo4j_client.create_attack_event),
        # not already on a 0-100 scale — it must be normalized before applying
        # the same 80/60/35 thresholds dashboard_api.py uses for the live risk
        # display, or nearly every real campaign trivially reads as "Critical".
        severity = _LEVEL_TO_SEVERITY[risk_level_from_score(normalize_risk_score(risk))]

        # next_technique / prediction_correct ground truth (Phase 18 fix):
        # the only within-campaign transition we can ever validate a
        # prediction against is technique_sequence[-2] -> technique_sequence[-1]
        # (the campaign's last observed step). Comparing against the live
        # campaign's own stored predicted_next is meaningless: the live loop
        # (chain_updater.py/realtime_socgraph.py) always overwrites that
        # value with a fresh prediction made FROM the most-recently-observed
        # technique, so once a campaign stops producing events there is no
        # further real event left to check that final prediction against —
        # it was compared against the very technique it was conditioned on.
        # Recomputing read-only from the second-to-last technique instead
        # asks the right question: "given what the model knew right before
        # the last transition happened, would it have called it correctly?"
        if len(technique_sequence) < 2:
            next_technique = ""  # NOT_APPLICABLE: no transition observed in this campaign
            prediction_correct = None  # NOT_APPLICABLE
        else:
            previous_technique = technique_sequence[-2]
            next_technique = final_attack_id
            predicted = predict_next_readonly(previous_technique)
            prediction_correct = (
                None if predicted is None  # NOT_APPLICABLE: no learned transition to check against
                else int(predicted["predicted"] == final_attack_id)
            )

        if actual_actor is None:
            attribution_correct = 1
        else:
            attribution_correct = int(
                attributed_actor == actual_actor
            )

        return CampaignLabels(
            risk_score=risk,
            severity=severity,
            attributed_actor=attributed_actor,
            next_technique=next_technique,
            prediction_correct=prediction_correct,
            attribution_correct=attribution_correct
        )

generator = LabelGenerator()
