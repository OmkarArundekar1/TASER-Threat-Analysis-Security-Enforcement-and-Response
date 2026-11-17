from neo4j_client import driver
from campaign_manager import campaign_manager
from config import (
    MIN_PREDICTION_CONFIDENCE,
    MIN_TRANSITION_OBSERVATIONS
)

def predict_next_readonly(current_technique):
    """Pure, side-effect-free lookup of what the learned NEXT_TECHNIQUE
    graph predicts for current_technique, using the same thresholds as
    predict_next() below -- but performs no Neo4j write and never touches
    campaign_manager state.

    Needed because predict_next() has a side effect (it overwrites the
    campaign's live LIKELY_NEXT/predicted_next with a prediction made FROM
    current_technique). That's correct for the live investigation loop, but
    it makes retrospective evaluation ("was the model's prediction right?")
    impossible: by the time a campaign is closed, its stored predicted_next
    always reflects a prediction made from the campaign's own last observed
    technique, which by definition never got a further same-campaign event
    to validate against (see Phase 18 investigation in ml/label_generator.py).
    Offline evaluation instead calls this function directly on a historical
    (technique[i], technique[i+1]) pair without disturbing live state.
    """
    try:
        with driver.session() as session:

            result = session.run(
                """
                MATCH (t:Technique {attack_id: $current})
                      -[r:NEXT_TECHNIQUE]->
                      (n:Technique)

                RETURN
                    n.attack_id AS next,
                    r.count AS count

                ORDER BY r.count DESC
                """,
                current=current_technique
            )

            rows = list(result)

            if not rows:
                print(
                    f"[PREDICTION] No learned transitions for {current_technique}"
                )
                return None

            total = sum(
                row["count"]
                for row in rows
            )

            if total < MIN_TRANSITION_OBSERVATIONS:
                print(
                    f"[PREDICTION] Insufficient observations "
                    f"({total} < {MIN_TRANSITION_OBSERVATIONS})"
                )
                return None
            best = rows[0]
            confidence = round(
                (best["count"] / total) * 100,
                2
            )

            if confidence < MIN_PREDICTION_CONFIDENCE:
                print(
                    f"[PREDICTION] Confidence below threshold "
                    f"({confidence}% < {MIN_PREDICTION_CONFIDENCE}%)"
                )
                return None

            return {
                "current": current_technique,
                "predicted": best["next"],
                "confidence": confidence,
                "transition_count": best["count"],
                "total_transitions": total
            }
    except Exception as e:
        print(f"[PREDICTION ERROR] {e}")
        return None


def predict_next(
    campaign_id,
    current_technique
):
    result = predict_next_readonly(current_technique)

    if result is None:
        return None

    campaign_manager.update_prediction(
        campaign_id,
        result["predicted"],
        result["confidence"]
    )

    print("\n[PREDICTION]")
    print(f"Current      : {result['current']}")
    print(f"Predicted    : {result['predicted']}")
    print(f"Transitions  : {result['transition_count']}")
    print(f"Total Seen   : {result['total_transitions']}")
    print(f"Confidence   : {result['confidence']}%")

    return result