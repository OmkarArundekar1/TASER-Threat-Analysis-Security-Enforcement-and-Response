"""
evaluators/rag_eval.py
=========================
Per-source RAG retrieval evaluation (MITRE semantic, campaign
narrative, GNN topology retrievers -- scored SEPARATELY, never
blended, per journal_ready_data.md Section 21's documented
architecture). Needs a query set with INDEPENDENTLY-judged relevance
(evaluation/queries/rag_queries.json -- query_id, query_text,
relevant_ids, graded relevance, reviewer, review_status) -- one does
not exist yet in this project (no independent human reviewer has
judged retrieval relevance for any query). This evaluator is real,
tested (against synthetic fixture query sets), and ready to run the
moment such a query set exists; it reports NOT_MEASURED honestly
rather than fabricating query judgments.
"""

from __future__ import annotations

import json
import os

from evaluators.base import EvaluationResult, MetricStatus

_QUERY_SET_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "queries")


def load_query_set(source: str) -> list[dict] | None:
    path = os.path.join(_QUERY_SET_DIR, f"rag_queries_{source}.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def evaluate(source: str, retrieve_fn=None) -> EvaluationResult:
    """`source`: one of "mitre_semantic", "campaign_narrative", "gnn_topology".
    `retrieve_fn(query_text) -> list[item_id]` (ranked). If None, the
    real retriever for `source` is imported and called."""
    queries = load_query_set(source)
    if not queries:
        return EvaluationResult(
            task=f"rag_retrieval_{source}",
            status=MetricStatus.NOT_MEASURED,
            reason=f"No independently-judged query set exists at evaluation/queries/rag_queries_{source}.json. "
                   f"Building one requires a human reviewer to judge relevant documents for a real incident "
                   f"context per query (see evaluation/review/export_review_queue.py for the mechanism to "
                   f"produce this) -- fabricating relevance judgments here would defeat the purpose of an "
                   f"independent evaluation, so this is honestly reported as not yet measured rather than guessed.",
        )

    unreviewed = [q for q in queries if q.get("review_status") != "HUMAN_REVIEWED"]
    if unreviewed:
        return EvaluationResult(
            task=f"rag_retrieval_{source}",
            status=MetricStatus.GROUND_TRUTH_REVIEW_REQUIRED,
            n=len(queries),
            reason=f"{len(unreviewed)}/{len(queries)} queries in this set are not yet HUMAN_REVIEWED.",
        )

    if retrieve_fn is None:
        retrieve_fn = _real_retriever_for(source)

    from evaluation_metrics import recall_at_k, precision_at_k, mean_reciprocal_rank, ndcg_at_k

    relevant_per_query, ranked_per_query = [], []
    per_query_report = []
    for q in queries:
        ranked = retrieve_fn(q["query_text"])
        relevant = set(q["relevant_ids"])
        relevant_per_query.append(relevant)
        ranked_per_query.append(ranked)
        graded = q.get("graded_relevance", {item: 1 for item in relevant})
        per_query_report.append({
            "query_id": q["query_id"],
            "recall_at_5": recall_at_k(relevant, ranked, 5),
            "precision_at_5": precision_at_k(relevant, ranked, 5),
            "ndcg_at_5": ndcg_at_k(graded, ranked, 5),
        })

    mrr = mean_reciprocal_rank(relevant_per_query, ranked_per_query)
    n = len(queries)
    avg_recall_5 = sum(r["recall_at_5"] for r in per_query_report) / n
    avg_precision_5 = sum(r["precision_at_5"] for r in per_query_report) / n
    avg_ndcg_5 = sum(r["ndcg_at_5"] for r in per_query_report) / n

    return EvaluationResult(
        task=f"rag_retrieval_{source}",
        status=MetricStatus.MEASURED,
        n=n,
        metrics={
            "n": n, "mrr": mrr, "mean_recall_at_5": avg_recall_5,
            "mean_precision_at_5": avg_precision_5, "mean_ndcg_at_5": avg_ndcg_5,
            "per_query": per_query_report,
        },
        dataset_name=f"rag_queries_{source}",
        method=f"Independently human-reviewed query set (evaluation/queries/rag_queries_{source}.json) scored "
               f"against the real {source} retriever, called at evaluation time.",
    )


def _real_retriever_for(source: str):
    if source == "mitre_semantic":
        from rag.mitre_retriever import mitre_retriever
        return lambda q: [d.doc_id for d in mitre_retriever.query(q)]
    if source == "campaign_narrative":
        from rag.campaign_retriever import CampaignNarrativeRetriever
        retriever = CampaignNarrativeRetriever()
        return lambda q: [d.doc_id for d in retriever.query(q)]
    if source == "gnn_topology":
        from rag.gnn_topology_retriever import gnn_topology_retriever
        return lambda q: [d.doc_id for d in gnn_topology_retriever.query(q)]
    raise ValueError(f"Unknown RAG source: {source}")
