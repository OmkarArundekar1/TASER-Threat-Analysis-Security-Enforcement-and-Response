"""
evaluators/prediction_eval.py
================================
Independent evaluation of CYUKTI's actual prediction target: the
learned NEXT_TECHNIQUE graph consumed by
prediction_engine.predict_next_readonly(). Per Phase 0's audit
(review/evaluation_implementation_audit.md), there is no "0/0
hits/misses" persisted tracking schema anywhere in the live codebase --
that figure was an ad hoc query against a property that does not exist
on any live node. The real, historical measurement of this exact
target already exists (Phase 18's real dataset rebuild: 4/12 correct,
33.3%, verdict INSUFFICIENT_FOR_SUPERVISED_ML) and is reused here
rather than re-derived, per this task's own "do not invent a
prediction target" instruction.

This evaluator additionally checks, LIVE, whether the current NEXT_TECHNIQUE
graph has grown enough since Phase 18 to generate NEW held-out prediction
opportunities beyond the frozen 60-row dataset.
"""

from __future__ import annotations

from evaluators.base import EvaluationResult, MetricStatus


PHASE_18_RESULT = {
    "evaluable_predictions": 12,
    "correct": 4,
    "incorrect": 8,
    "accuracy": 4 / 12,
    "verdict": "INSUFFICIENT_FOR_SUPERVISED_ML",
    "source": "scripts/phase18_next_technique_diagnosis.py, frozen ml/datasets/campaign_dataset.csv (60 rows)",
}


def evaluate() -> EvaluationResult:
    from neo4j_client import driver
    from evaluation_metrics import wilson_confidence_interval

    with driver.session() as s:
        edge_count = s.run("MATCH ()-[r:NEXT_TECHNIQUE]->() RETURN count(r) AS n").single()["n"]
        edges = list(s.run(
            "MATCH (a:Technique)-[r:NEXT_TECHNIQUE]->(b:Technique) "
            "RETURN a.attack_id AS src, b.attack_id AS dst, r.count AS count, r.confidence AS confidence"
        ))

    ci = wilson_confidence_interval(PHASE_18_RESULT["correct"], PHASE_18_RESULT["evaluable_predictions"])

    if edge_count <= 3:
        # Unchanged from the historical 3-edge graph Phase 18 evaluated --
        # no new held-out opportunity exists beyond what's already measured.
        return EvaluationResult(
            task="prediction_accuracy",
            status=MetricStatus.UNMEASURABLE,
            n=PHASE_18_RESULT["evaluable_predictions"],
            metrics={
                "live_next_technique_edge_count": edge_count,
                "live_edges": [dict(e) for e in edges],
                "historical_phase18_result": PHASE_18_RESULT,
                "historical_accuracy_95pct_wilson_interval": ci,
            },
            dataset_name="NEXT_TECHNIQUE (live Neo4j graph)",
            method="predict_next_readonly()'s real prediction target (learned NEXT_TECHNIQUE transitions) is "
                   "unchanged in scale since Phase 18's real dataset rebuild (still <= 3 edges) -- there is no "
                   "new held-out sequential data to generate a fresh prediction opportunity from, so the honest "
                   "result is UNMEASURABLE for NEW data, reporting the historical real measurement instead of "
                   "fabricating a new one.",
            reason=f"Live NEXT_TECHNIQUE graph has {edge_count} edges (same order of magnitude as Phase 18's 3); "
                   f"no new attack sessions have been ingested that would create additional distinct technique "
                   f"transitions to retrospectively score. Historical real result reused: "
                   f"{PHASE_18_RESULT['correct']}/{PHASE_18_RESULT['evaluable_predictions']} "
                   f"({PHASE_18_RESULT['accuracy']:.1%}), verdict {PHASE_18_RESULT['verdict']}.",
            limitations=[
                "No live retrospective re-scoring was performed this session -- the reused Phase 18 number is "
                "the last real measurement of this exact target, not re-verified against fresh data because no "
                "fresh data exists to re-verify it against.",
            ],
        )

    return EvaluationResult(
        task="prediction_accuracy",
        status=MetricStatus.NOT_MEASURED,
        metrics={"live_next_technique_edge_count": edge_count, "live_edges": [dict(e) for e in edges]},
        dataset_name="NEXT_TECHNIQUE (live Neo4j graph)",
        reason=f"The live NEXT_TECHNIQUE graph has grown to {edge_count} edges (more than Phase 18's 3) -- "
               f"new held-out prediction opportunities may now exist, but this evaluator has not yet been run "
               f"in retrospective-scoring mode against them (that requires rebuilding the real transition dataset "
               f"the way scripts/phase18_next_technique_diagnosis.py did, which was out of scope to re-run this pass).",
    )
