import json

import pytest

from evaluators import rag_eval
from evaluators.base import MetricStatus


@pytest.fixture(autouse=True)
def _isolate_query_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(rag_eval, "_QUERY_SET_DIR", str(tmp_path))
    yield tmp_path


def test_no_query_set_reports_not_measured():
    result = rag_eval.evaluate("mitre_semantic")
    assert result.status == MetricStatus.NOT_MEASURED


def test_unreviewed_queries_require_review(tmp_path):
    queries = [{"query_id": "Q1", "query_text": "x", "relevant_ids": ["a"], "review_status": "AUTO_PROPOSED"}]
    (tmp_path / "rag_queries_mitre_semantic.json").write_text(json.dumps(queries))
    result = rag_eval.evaluate("mitre_semantic")
    assert result.status == MetricStatus.GROUND_TRUTH_REVIEW_REQUIRED


def test_reviewed_query_set_computes_real_metrics(tmp_path):
    queries = [
        {"query_id": "Q1", "query_text": "brute force", "relevant_ids": ["doc_a", "doc_b"],
         "review_status": "HUMAN_REVIEWED"},
        {"query_id": "Q2", "query_text": "recon scan", "relevant_ids": ["doc_c"],
         "review_status": "HUMAN_REVIEWED"},
    ]
    (tmp_path / "rag_queries_mitre_semantic.json").write_text(json.dumps(queries))

    def fake_retriever(query_text):
        return {"brute force": ["doc_a", "doc_x", "doc_b"], "recon scan": ["doc_z", "doc_c"]}[query_text]

    result = rag_eval.evaluate("mitre_semantic", retrieve_fn=fake_retriever)
    assert result.status == MetricStatus.MEASURED
    assert result.n == 2
    assert result.metrics["mean_recall_at_5"] == 1.0  # both relevant docs retrieved within top 5 for both queries
