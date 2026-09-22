"""
Tests for ml/gnn/retrieval_evaluation.py -- the topology-aware
retrieval/correlation evaluation (see ../../GNN_RETRIEVAL_EVALUATION.md).
Mocks only the database boundary; the two full-LOGO live-data functions
(fold_safe_full_population_embeddings, identity_preserving_embeddings)
are expensive (train a fresh autoencoder per fold / load a real
artifact) and are exercised via the module's own __main__ run rather
than in the routine test suite, per this repository's convention of
keeping tests fast -- the pure metric/formula/diagnostic functions
below carry the real regression coverage.
"""

import torch

from ml.gnn.retrieval_evaluation import (
    FoldSafeEmbeddings,
    get_campaign_hosts,
    get_campaign_techniques,
    identity_retrieval_evaluation,
    near_duplicate_diagnostic,
    reciprocal_rank,
    recall_at_k,
    similar_to_formula_score,
    structural_diagnostic,
    technique_jaccard,
)
from ml.gnn.campaign_graphs import CampaignGraphSample
from ml.gnn.graph_encoder import encode_graph
import networkx as nx


# ---------------------------------------------------------------------------
# technique_jaccard / similar_to_formula_score — never used as labels,
# only as one of several representations being compared
# ---------------------------------------------------------------------------

def test_technique_jaccard_identical_sets():
    assert technique_jaccard({"T1078", "T1110"}, {"T1078", "T1110"}) == 1.0


def test_technique_jaccard_disjoint_sets():
    assert technique_jaccard({"T1078"}, {"T1595"}) == 0.0


def test_technique_jaccard_both_empty():
    assert technique_jaccard(set(), set()) == 0.0


def test_technique_jaccard_partial_overlap():
    result = technique_jaccard({"T1", "T2", "T3"}, {"T2", "T3", "T4"})
    assert result == 2 / 4  # intersection=2, union=4


def test_similar_to_formula_score_matches_real_neo4j_client_weights():
    """Mirrors neo4j_client.update_campaign_similarity's real 60/20/20
    weighting (verified by direct code read in
    GNN_REPRESENTATION_DESIGN.md Section 9) -- this test is the
    tripwire if that formula and this comparison representation ever
    silently drift apart."""
    score = similar_to_formula_score(
        techniques_a={"T1", "T2"}, techniques_b={"T1", "T2"},
        attacker_a="1.2.3.4", attacker_b="1.2.3.4",
        host_a="5.6.7.8", host_b="9.9.9.9",
    )
    # technique_jaccard=1.0*60 + shared attacker 20 + different host 0
    assert score == 80.0


def test_similar_to_formula_score_zero_for_fully_disjoint_pair():
    score = similar_to_formula_score(
        techniques_a={"T1"}, techniques_b={"T2"},
        attacker_a="1.1.1.1", attacker_b="2.2.2.2",
        host_a="3.3.3.3", host_b="4.4.4.4",
    )
    assert score == 0.0


# ---------------------------------------------------------------------------
# recall_at_k / reciprocal_rank — pure retrieval metrics
# ---------------------------------------------------------------------------

def _embeddings(vectors: dict[str, list[float]]) -> dict[str, torch.Tensor]:
    return {k: torch.tensor(v, dtype=torch.float32) for k, v in vectors.items()}


def test_reciprocal_rank_correct_answer_is_nearest_neighbor():
    embeddings = _embeddings({
        "query": [1.0, 0.0],
        "correct": [0.99, 0.01],   # nearly identical direction -> rank 1
        "distractor": [0.0, 1.0],  # orthogonal -> far
    })
    rr = reciprocal_rank("query", relevant_ids={"correct"}, embeddings=embeddings)
    assert rr == 1.0


def test_reciprocal_rank_correct_answer_is_second_nearest():
    embeddings = _embeddings({
        "query": [1.0, 0.0],
        "closer_but_wrong": [0.99, 0.01],
        "correct": [0.9, 0.1],
        "farthest": [0.0, 1.0],
    })
    rr = reciprocal_rank("query", relevant_ids={"correct"}, embeddings=embeddings)
    assert rr == 0.5  # rank 2 -> 1/2


def test_reciprocal_rank_no_relevant_ids_is_nan():
    embeddings = _embeddings({"query": [1.0, 0.0], "other": [0.0, 1.0]})
    rr = reciprocal_rank("query", relevant_ids=set(), embeddings=embeddings)
    assert rr != rr  # NaN != NaN


def test_recall_at_k_counts_hits_within_top_k():
    embeddings = _embeddings({
        "query": [1.0, 0.0, 0.0],
        "rel-1": [0.99, 0.01, 0.0],
        "rel-2": [0.0, 0.0, 1.0],  # far -> not in top 1
        "distractor": [0.98, 0.02, 0.0],  # closer than rel-2
    })
    recall = recall_at_k("query", relevant_ids={"rel-1", "rel-2"}, embeddings=embeddings, k=1)
    assert recall == 0.5  # only rel-1 makes the top-1


def test_recall_at_k_full_recall_when_k_covers_all_relevant():
    embeddings = _embeddings({
        "query": [1.0, 0.0],
        "rel-1": [0.9, 0.1],
        "rel-2": [0.5, 0.5],
        "distractor": [0.0, 1.0],
    })
    recall = recall_at_k("query", relevant_ids={"rel-1", "rel-2"}, embeddings=embeddings, k=3)
    assert recall == 1.0


# ---------------------------------------------------------------------------
# identity_retrieval_evaluation — group-based retrieval over synthetic embeddings
# ---------------------------------------------------------------------------

def test_identity_retrieval_evaluation_rewards_tight_clustering():
    embeddings = _embeddings({
        "a1": [1.0, 0.0], "a2": [0.99, 0.01],  # group "attackerA", close together
        "b1": [0.0, 1.0], "b2": [0.01, 0.99],  # group "attackerB", close together
    })
    group_of = {"a1": "attackerA", "a2": "attackerA", "b1": "attackerB", "b2": "attackerB"}
    result = identity_retrieval_evaluation(embeddings, group_of, k=1)
    assert result["campaigns_evaluated"] == 4
    assert result["mean_reciprocal_rank"] == 1.0  # every campaign's nearest neighbor is its own group


def test_identity_retrieval_evaluation_skips_campaigns_with_no_group_peers():
    embeddings = _embeddings({"solo": [1.0, 0.0], "other": [0.0, 1.0]})
    group_of = {"solo": "group-of-one", "other": "group-of-one-too"}
    result = identity_retrieval_evaluation(embeddings, group_of, k=1)
    assert result["campaigns_evaluated"] == 0  # neither campaign has a peer in its own group


# ---------------------------------------------------------------------------
# structural_diagnostic / near_duplicate_diagnostic — diagnostic-only,
# verified with small synthetic fixtures
# ---------------------------------------------------------------------------

def _sample_with_shape(campaign_id: str, num_techniques: int) -> CampaignGraphSample:
    G = nx.DiGraph()
    G.add_node("a", labels=["Attacker"])
    G.add_node("c", labels=["Campaign"])
    G.add_node("e", labels=["AttackEvent"])
    G.add_edge("a", "c", relationship="LAUNCHED")
    G.add_edge("c", "e", relationship="HAS_EVENT")
    for i in range(num_techniques):
        G.add_node(f"t{i}", labels=["Technique"])
        G.add_edge("e", f"t{i}", relationship="MATCHES")
    return CampaignGraphSample(
        campaign_id=campaign_id, graph=encode_graph(G), severity="Low", risk_score=0.0, num_events=1,
    )


def test_structural_diagnostic_finds_same_shape_different_technique_pair():
    samples = {
        "camp-a": _sample_with_shape("camp-a", num_techniques=1),
        "camp-b": _sample_with_shape("camp-b", num_techniques=1),  # same shape
    }
    techniques = {"camp-a": {"T1078"}, "camp-b": {"T1595"}}  # disjoint techniques
    embeddings = _embeddings({"camp-a": [1.0, 0.0], "camp-b": [0.999, 0.001]})
    fold_safe = FoldSafeEmbeddings(embeddings=embeddings, fold_of_campaign={})

    result = structural_diagnostic(fold_safe, samples, techniques)
    assert result["same_shape_different_technique_pairs_found"] == 1
    assert result["same_shape_different_technique_examples"][0]["technique_jaccard"] == 0.0


def test_near_duplicate_diagnostic_categorizes_real_causes():
    samples = {
        "camp-a": _sample_with_shape("camp-a", num_techniques=1),
        "camp-b": _sample_with_shape("camp-b", num_techniques=1),
    }
    techniques = {"camp-a": {"T1078"}, "camp-b": {"T1078"}}  # identical
    embeddings = _embeddings({"camp-a": [1.0, 0.0], "camp-b": [0.9999, 0.0001]})  # near-duplicate
    attacker_of = {"camp-a": "1.1.1.1", "camp-b": "1.1.1.1"}  # same attacker
    host_of = {"camp-a": "2.2.2.2", "camp-b": "3.3.3.3"}  # different host

    result = near_duplicate_diagnostic(embeddings, samples, techniques, attacker_of, host_of, threshold=0.01)
    assert result["near_duplicate_pairs"] == 1
    assert result["same_attacker"] == 1
    assert result["same_host"] == 0
    assert result["identical_technique_set"] == 1
    assert result["same_node_type_shape"] == 1


# ---------------------------------------------------------------------------
# get_campaign_techniques / get_campaign_hosts — DB boundary only
# ---------------------------------------------------------------------------

class _FakeSession:
    def __init__(self, rows):
        self._rows = rows

    def run(self, query, **kwargs):
        return iter(self._rows)

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class _FakeDriver:
    def __init__(self, rows):
        self._rows = rows

    def session(self):
        return _FakeSession(self._rows)


def test_get_campaign_techniques_fills_in_empty_set_for_missing_campaigns(monkeypatch):
    monkeypatch.setattr(
        "neo4j_client.driver",
        _FakeDriver([{"id": "c1", "techniques": ["T1078", "T1595"]}]),
    )
    result = get_campaign_techniques(["c1", "c2"])
    assert result == {"c1": {"T1078", "T1595"}, "c2": set()}


def test_get_campaign_hosts_maps_campaign_to_ip(monkeypatch):
    monkeypatch.setattr(
        "neo4j_client.driver",
        _FakeDriver([{"id": "c1", "ip": "10.0.0.1"}]),
    )
    result = get_campaign_hosts(["c1"])
    assert result == {"c1": "10.0.0.1"}
