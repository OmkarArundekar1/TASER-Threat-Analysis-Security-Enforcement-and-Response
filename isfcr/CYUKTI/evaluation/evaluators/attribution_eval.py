"""
evaluators/attribution_eval.py
=================================
Independent attribution accuracy: for each real campaign, the GROUND
TRUTH is its own directly-observed attacker_ip (a raw network fact).
The SYSTEM PREDICTION is computed HERE, at evaluation time, by calling
the real threat_attribution_engine.ThreatAttributionEngine.attribute()
against a minimal-but-real CampaignContext (built directly from live
Neo4j fields -- campaign_id, attacker_ip/victim_ip, and the campaign's
real technique set via its AttackEvent->Technique edges), then checking
whether the top-ranked historical-campaign candidate shares the SAME
real attacker_ip as the campaign under evaluation.

This does not use CYUKTI's own attacker-clustering output as ground
truth -- the engine is exercised for real, but scored against the raw
IP fact, never against its own prior conclusions.
"""

from __future__ import annotations

from evaluators.base import EvaluationResult, MetricStatus, classify_review_status, GroundTruthReviewRequired
from ground_truth import store


def _load_minimal_campaign_context(driver, campaign_id: str):
    from campaign_context import CampaignContext

    with driver.session() as s:
        base = s.run(
            "MATCH (c:Campaign {campaign_id:$id}) RETURN c.attacker_ip AS attacker_ip, "
            "c.victim_ip AS victim_ip, c.last_technique AS last_technique",
            id=campaign_id,
        ).single()
        if base is None:
            return None
        technique_rows = s.run(
            "MATCH (c:Campaign {campaign_id:$id})-[:HAS_EVENT]->(:AttackEvent)-[:MATCHES]->(t:Technique) "
            "RETURN DISTINCT t.attack_id AS tid ORDER BY tid",
            id=campaign_id,
        )
        techniques = {r["tid"] for r in technique_rows}

    return CampaignContext(
        campaign_id=campaign_id,
        attacker_ip=base["attacker_ip"] or "",
        victim_ip=base["victim_ip"] or "",
        last_technique=base["last_technique"],
        techniques=techniques,
        attack_chain=sorted(techniques),
    )


def evaluate(dataset_name: str = "attribution_v0", allow_preliminary: bool = True) -> EvaluationResult:
    records, review_status = store.load_best_available(dataset_name)
    if not records:
        return EvaluationResult(
            task="attribution_accuracy", status=MetricStatus.NOT_MEASURED,
            reason=f"No ground-truth dataset '{dataset_name}' has been built yet.",
        )

    try:
        status = classify_review_status(review_status, allow_preliminary)
    except GroundTruthReviewRequired as e:
        return EvaluationResult(
            task="attribution_accuracy", status=MetricStatus.GROUND_TRUTH_REVIEW_REQUIRED,
            n=len(records), dataset_name=dataset_name, reason=str(e),
        )

    from neo4j_client import driver
    from threat_attribution_engine import ThreatAttributionEngine

    engine = ThreatAttributionEngine()
    correct = 0
    unattributed = 0
    n = 0
    per_record = []
    skipped = 0

    with driver.session() as s:
        attacker_ip_by_campaign = {}
        rows = s.run("MATCH (c:Campaign) RETURN c.campaign_id AS id, c.attacker_ip AS attacker_ip")
        for r in rows:
            attacker_ip_by_campaign[r["id"]] = r["attacker_ip"]

    for r in records:
        ctx = _load_minimal_campaign_context(driver, r.raw_event_id)
        if ctx is None or not ctx.techniques:
            skipped += 1
            continue
        n += 1
        result = engine.attribute(ctx)
        candidates = getattr(result, "candidates", None) or getattr(result, "actors", None) or []
        if not candidates:
            unattributed += 1
            top_attacker_ip = None
            is_correct = False
        else:
            top = candidates[0]
            top_candidate_campaign_id = getattr(top, "actor", None)
            top_attacker_ip = attacker_ip_by_campaign.get(top_candidate_campaign_id)
            is_correct = (top_attacker_ip is not None) and (top_attacker_ip == r.expected_attribution)
            if is_correct:
                correct += 1
        per_record.append({
            "sample_id": r.sample_id, "campaign_id": r.raw_event_id,
            "expected_attacker_ip": r.expected_attribution,
            "top_candidate_attacker_ip": top_attacker_ip,
            "correct": is_correct,
        })

    if n == 0:
        return EvaluationResult(
            task="attribution_accuracy", status=MetricStatus.UNMEASURABLE,
            dataset_name=dataset_name, ground_truth_review_status=review_status.value,
            reason=f"{skipped} ground-truth record(s) found but none had a re-loadable real CampaignContext "
                   f"with a non-empty technique set (attribute() requires observed techniques to score anything).",
        )

    from evaluation_metrics import wilson_confidence_interval
    accuracy = correct / n
    ci = wilson_confidence_interval(correct, n)

    return EvaluationResult(
        task="attribution_accuracy",
        status=status,
        n=n,
        metrics={
            "n": n, "skipped_no_context": skipped, "correct": correct,
            "unattributed": unattributed, "accuracy": accuracy,
            "accuracy_95pct_wilson_interval": ci, "per_record": per_record,
        },
        ground_truth_review_status=review_status.value,
        dataset_name=dataset_name,
        method="Ground truth = each real campaign's own directly-observed attacker_ip (raw network fact). "
               "System prediction = threat_attribution_engine.ThreatAttributionEngine.attribute()'s top-ranked "
               "historical-campaign candidate, called at evaluation time against a real (but minimally "
               "reconstructed) CampaignContext, then mapped back to that candidate's own real attacker_ip. "
               "'Correct' = same real attacker_ip. This measures whether attribution's technique-similarity "
               "ranking tends to surface campaigns from the SAME real attacker first -- a coarse proxy for "
               "attribution accuracy, not identity-level ground truth in the strict sense the task brief asks for "
               "(no independently-labeled, human-confirmed 'this campaign belongs to named threat actor X' "
               "dataset exists in this lab -- see limitations).",
        limitations=[
            "This evaluates attribution's RANKING behavior against a coarse IP-based proxy for 'attacker identity', "
            "not a true named-threat-actor ground truth (no such independent human labels exist in this lab).",
            "Small n -- see the reported Wilson 95% interval before citing this as precise.",
            "Ground truth is AUTO_PROPOSED unless ground_truth_review_status says otherwise.",
        ],
    )
