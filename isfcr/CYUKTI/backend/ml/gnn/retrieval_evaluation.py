"""
ml/gnn/retrieval_evaluation.py
==================================
Evaluates whether the GNN's learned graph embedding z_G provides
topology-aware information useful for retrieval/correlation -- a
narrower, independent hypothesis from the severity ablation
(xgboost_ablation.py, which produced a clean null result). See
../../GNN_RETRIEVAL_EVALUATION.md for the full write-up.

Hard constraint, enforced by construction, not just discipline: this
module never reads SIMILAR_TO or RESEMBLES from Neo4j anywhere. Where
SIMILAR_TO's *formula* is used as one of several representations being
compared (similar_to_formula_score below), it is recomputed fresh in
Python from raw technique/attacker/host facts -- never read as a
stored edge, and never used as a label/ground truth for anything. The
only ground truths used anywhere in this module are: (a) "this later
snapshot belongs to the same real campaign" (an identity fact, not a
similarity judgment), and (b) "these two campaigns share the same real
attacker/host" (also a raw identity fact, not a derived score) --
exactly the two non-circular criteria GNN_REPRESENTATION_DESIGN.md
Section 9 and GNN_REPRESENTATION_IMPLEMENTATION.md Section 12 already
established as legitimate.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

import torch

from autoencoder_model import GraphAutoencoder
from campaign_graphs import CampaignGraphSample, build_real_campaign_dataset
from graph_encoder import NODE_TYPES
from temporal_graphs import TemporalSnapshot, build_all_temporal_snapshots
from train_autoencoder import embed_samples, fit_autoencoder, get_campaign_attacker_ips


# ---------------------------------------------------------------------------
# Real, raw identity/content facts (never SIMILAR_TO/RESEMBLES)
# ---------------------------------------------------------------------------

def get_campaign_techniques(campaign_ids: list[str]) -> dict[str, set[str]]:
    from neo4j_client import driver

    with driver.session() as session:
        result = session.run("""
            MATCH (c:Campaign)-[:HAS_EVENT]->(:AttackEvent)-[:MATCHES]->(t:Technique)
            WHERE c.campaign_id IN $ids
            RETURN c.campaign_id AS id, collect(DISTINCT t.attack_id) AS techniques
        """, ids=campaign_ids)
        techniques = {r["id"]: set(r["techniques"]) for r in result}
    return {cid: techniques.get(cid, set()) for cid in campaign_ids}


def get_campaign_hosts(campaign_ids: list[str]) -> dict[str, str]:
    from neo4j_client import driver

    with driver.session() as session:
        result = session.run("""
            MATCH (c:Campaign)-[:TARGETS]->(h:Host)
            WHERE c.campaign_id IN $ids
            RETURN c.campaign_id AS id, h.ip AS ip
        """, ids=campaign_ids)
        return {r["id"]: r["ip"] for r in result}


def technique_jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 0.0
    union = a | b
    return len(a & b) / len(union) if union else 0.0


def similar_to_formula_score(
    techniques_a: set[str], techniques_b: set[str],
    attacker_a: str | None, attacker_b: str | None,
    host_a: str | None, host_b: str | None,
) -> float:
    """Recomputes neo4j_client.update_campaign_similarity's real formula
    (60% technique-Jaccard + 20% shared-attacker + 20% shared-host,
    verified by direct code read in GNN_REPRESENTATION_DESIGN.md
    Section 9) as a CONTINUOUS score for any pair -- not read from a
    stored SIMILAR_TO edge, not gated by the production >=75 threshold
    (a threshold is for deciding whether to persist an edge, not
    relevant to ranking pairs by distance here). Used only as one of
    several REPRESENTATIONS being compared against non-circular ground
    truth -- never as a label."""
    tech_score = technique_jaccard(techniques_a, techniques_b) * 60.0
    attacker_score = 20.0 if (attacker_a is not None and attacker_a == attacker_b) else 0.0
    host_score = 20.0 if (host_a is not None and host_a == host_b) else 0.0
    return tech_score + attacker_score + host_score


# ---------------------------------------------------------------------------
# Embedding generation: fold-safe (LOGO, generalization) vs single-model
# (identity-preservation only, weaker, disclosed)
# ---------------------------------------------------------------------------

def build_full_population_attacker_groups(campaign_ids: list[str]) -> dict[str, list[str]]:
    attacker_of = get_campaign_attacker_ips(campaign_ids)
    groups: dict[str, list[str]] = defaultdict(list)
    for cid in campaign_ids:
        groups[attacker_of.get(cid, f"__no_attacker__{cid}")].append(cid)
    return dict(groups)


@dataclass
class FoldSafeEmbeddings:
    embeddings: dict[str, torch.Tensor]
    fold_of_campaign: dict[str, str]  # campaign_id -> held-out group name (which fold produced its embedding)
    models_by_fold: dict[str, tuple[GraphAutoencoder, torch.Tensor, torch.Tensor]] = field(repr=False, default_factory=dict)


def fold_safe_full_population_embeddings(seed: int = 42) -> tuple[FoldSafeEmbeddings, dict[str, CampaignGraphSample]]:
    """Leave-One-Attacker-Group-Out over ALL 71 real campaigns (not
    just the 60 with persisted XGBoost features -- retrieval needs only
    graphs, so the full population is used here, unlike
    xgboost_ablation.py). Every campaign's embedding in the returned
    dict comes from a GraphAutoencoder that never saw that campaign's
    graph during training -- the "structural/generalization" embedding
    set this phase's retrieval experiments use as primary evidence."""
    samples = {s.campaign_id: s for s in build_real_campaign_dataset()}
    campaign_ids = list(samples.keys())
    groups = build_full_population_attacker_groups(campaign_ids)

    embeddings: dict[str, torch.Tensor] = {}
    fold_of_campaign: dict[str, str] = {}
    models_by_fold: dict[str, tuple[GraphAutoencoder, torch.Tensor, torch.Tensor]] = {}

    for held_out_group in sorted(groups):
        test_ids = groups[held_out_group]
        train_ids = [c for c in campaign_ids if c not in test_ids]
        train_samples = [samples[c] for c in train_ids]
        test_samples = [samples[c] for c in test_ids]

        model, feature_mean, feature_std, _pw, _hist = fit_autoencoder(
            train_samples, hidden_dim=16, embedding_dim=8, num_layers=2, epochs=200, lr=0.01, seed=seed,
        )
        fold_embeddings = embed_samples(model, test_samples, feature_mean, feature_std)
        embeddings.update(fold_embeddings)
        for cid in test_ids:
            fold_of_campaign[cid] = held_out_group
        models_by_fold[held_out_group] = (model, feature_mean, feature_std)

    return FoldSafeEmbeddings(embeddings=embeddings, fold_of_campaign=fold_of_campaign, models_by_fold=models_by_fold), samples


def identity_preserving_embeddings() -> dict[str, torch.Tensor]:
    """Loads the SINGLE model trained on all 71 campaigns together
    (ml/models/gnn_autoencoder.pt, produced by
    `python -m ml.gnn.train_autoencoder` in a prior phase -- reused
    here, NOT retrained, per this phase's explicit instruction).
    Embeddings for the 59 campaigns that model trained on are NOT
    held-out in any sense (the model saw their exact graphs) --
    disclosed explicitly as the weaker, memorization-permitting check,
    used only for the identity-preservation experiments this phase
    keeps separate from the fold-safe generalization experiments."""
    from train_autoencoder import load_autoencoder

    model, checkpoint = load_autoencoder("ml/models/gnn_autoencoder.pt")
    samples = build_real_campaign_dataset()
    return embed_samples(model, samples, checkpoint["feature_mean"], checkpoint["feature_std"])


# ---------------------------------------------------------------------------
# Retrieval metrics — only where ground truth is legitimate
# ---------------------------------------------------------------------------

def _rank_by_distance(query_id: str, embeddings: dict[str, torch.Tensor], exclude: set[str]) -> list[str]:
    query_vec = embeddings[query_id]
    candidates = [cid for cid in embeddings if cid != query_id and cid not in exclude]
    dists = []
    for cid in candidates:
        cos_sim = torch.dot(query_vec, embeddings[cid]) / (
            query_vec.norm() * embeddings[cid].norm()
        ).clamp(min=1e-9)
        dists.append((float(1.0 - cos_sim), cid))
    dists.sort(key=lambda t: t[0])
    return [cid for _, cid in dists]


def recall_at_k(query_id: str, relevant_ids: set[str], embeddings: dict[str, torch.Tensor], k: int) -> float:
    if not relevant_ids:
        return float("nan")
    ranked = _rank_by_distance(query_id, embeddings, exclude=set())
    top_k = set(ranked[:k])
    hit = len(top_k & relevant_ids)
    return hit / len(relevant_ids)


def reciprocal_rank(query_id: str, relevant_ids: set[str], embeddings: dict[str, torch.Tensor]) -> float:
    if not relevant_ids:
        return float("nan")
    ranked = _rank_by_distance(query_id, embeddings, exclude=set())
    for position, cid in enumerate(ranked, start=1):
        if cid in relevant_ids:
            return 1.0 / position
    return 0.0


# ---------------------------------------------------------------------------
# Task A: temporal retrieval (fold-safe) -- the cleanest non-circular task
# ---------------------------------------------------------------------------

def temporal_retrieval_evaluation(fold_safe: FoldSafeEmbeddings, samples: dict[str, CampaignGraphSample], k: int = 5) -> dict:
    """Query = an EARLY snapshot's embedding (from the fold-safe model
    for that campaign's held-out attacker group -- the snapshot's own
    campaign was never in that model's training data, so this is
    fold-safe for BOTH the temporal and the attacker dimensions
    simultaneously). Candidate pool = every other real campaign's
    fold-safe final-state embedding PLUS the query campaign's own later
    snapshot. Relevant = only the query campaign's own later
    snapshot(s) -- an identity fact ("this is the same real campaign
    later in its own history"), not a derived similarity judgment."""
    all_temporal = build_all_temporal_snapshots()
    results_per_campaign = {}

    for campaign_id, snapshots in all_temporal.items():
        if len(snapshots) < 2 or campaign_id not in fold_safe.fold_of_campaign:
            continue
        held_out_group = fold_safe.fold_of_campaign[campaign_id]
        model, feature_mean, feature_std = fold_safe.models_by_fold[held_out_group]

        from autoencoder_model import NUMERIC_FEATURE_OFFSET
        query_snap = snapshots[0]
        later_snap = snapshots[-1]

        def _embed_snapshot(snap: TemporalSnapshot) -> torch.Tensor | None:
            if snap.graph.num_nodes == 0:
                return None
            x = snap.graph.x.clone()
            x[:, NUMERIC_FEATURE_OFFSET:] = (x[:, NUMERIC_FEATURE_OFFSET:] - feature_mean) / feature_std
            return model.embed_graph(x, snap.graph.edge_index)

        z_query = _embed_snapshot(query_snap)
        z_later = _embed_snapshot(later_snap)
        if z_query is None or z_later is None:
            continue

        # candidate pool: fold-safe embeddings of every OTHER real campaign
        # (all from their own respective fold-safe models) + this campaign's
        # own later snapshot (the one relevant/correct answer)
        pool = dict(fold_safe.embeddings)
        pool.pop(campaign_id, None)  # remove this campaign's own final-state embedding to avoid trivial duplication
        pool[f"{campaign_id}__later_snapshot"] = z_later
        pool[f"{campaign_id}__query"] = z_query  # for _rank_by_distance's self-exclusion

        relevant = {f"{campaign_id}__later_snapshot"}
        rank_list = _rank_by_distance(f"{campaign_id}__query", pool, exclude=set())
        rr = 0.0
        for position, cid in enumerate(rank_list, start=1):
            if cid in relevant:
                rr = 1.0 / position
                break
        recall = 1.0 if f"{campaign_id}__later_snapshot" in rank_list[:k] else 0.0

        results_per_campaign[campaign_id] = {
            "num_snapshots": len(snapshots),
            "pool_size": len(rank_list),
            "reciprocal_rank": rr,
            f"recall_at_{k}": recall,
            "rank_of_own_later_snapshot": rank_list.index(f"{campaign_id}__later_snapshot") + 1,
        }

    if not results_per_campaign:
        return {"campaigns_evaluated": 0}

    mrr = sum(v["reciprocal_rank"] for v in results_per_campaign.values()) / len(results_per_campaign)
    recall_key = f"recall_at_{k}"
    mean_recall = sum(v[recall_key] for v in results_per_campaign.values()) / len(results_per_campaign)
    return {
        "campaigns_evaluated": len(results_per_campaign),
        "mean_reciprocal_rank": mrr,
        f"mean_recall_at_{k}": mean_recall,
        "per_campaign": results_per_campaign,
        "ground_truth": "own later snapshot of the SAME real campaign (identity fact, not SIMILAR_TO)",
    }


# ---------------------------------------------------------------------------
# Task B/C: attacker/host retrieval, fold-safe (generalization) vs
# single-model (identity-preservation, weaker, disclosed)
# ---------------------------------------------------------------------------

def identity_retrieval_evaluation(embeddings: dict[str, torch.Tensor], group_of: dict[str, str], k: int = 5) -> dict:
    groups: dict[str, set[str]] = defaultdict(set)
    for cid, g in group_of.items():
        groups[g].add(cid)

    per_campaign = {}
    for cid in embeddings:
        g = group_of.get(cid)
        if g is None:
            continue
        relevant = groups[g] - {cid}
        if not relevant:
            continue
        per_campaign[cid] = {
            "reciprocal_rank": reciprocal_rank(cid, relevant, embeddings),
            f"recall_at_{k}": recall_at_k(cid, relevant, embeddings, k),
        }

    if not per_campaign:
        return {"campaigns_evaluated": 0}

    mrr = sum(v["reciprocal_rank"] for v in per_campaign.values()) / len(per_campaign)
    recall_key = f"recall_at_{k}"
    mean_recall = sum(v[recall_key] for v in per_campaign.values()) / len(per_campaign)
    return {"campaigns_evaluated": len(per_campaign), "mean_reciprocal_rank": mrr, f"mean_recall_at_{k}": mean_recall}


# ---------------------------------------------------------------------------
# Structural diagnostic: real pairs where technique overlap and topology
# disagree (Section 6/D) -- diagnostic only, no label invented
# ---------------------------------------------------------------------------

def structural_diagnostic(
    fold_safe: FoldSafeEmbeddings, samples: dict[str, CampaignGraphSample],
    techniques: dict[str, set[str]], top_n: int = 5,
) -> dict:
    """Finds real campaign pairs at the extremes of
    (technique_jaccard, topology_similarity) disagreement, using only
    measurable graph statistics (node/edge counts, node-type
    multiset) as the topology proxy -- no invented label. Reports GNN
    embedding distance alongside technique-Jaccard distance for each,
    as a diagnostic, not a validated claim."""
    def _shape(cid: str) -> tuple:
        g = samples[cid].graph
        from collections import Counter
        return tuple(sorted(Counter(g.node_types).items()))

    ids = [cid for cid in fold_safe.embeddings if cid in techniques]
    same_shape_diff_technique = []
    same_technique_diff_shape = []

    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            a, b = ids[i], ids[j]
            shape_a, shape_b = _shape(a), _shape(b)
            tech_jaccard = technique_jaccard(techniques[a], techniques[b])
            emb_a, emb_b = fold_safe.embeddings[a], fold_safe.embeddings[b]
            cos_sim = torch.dot(emb_a, emb_b) / (emb_a.norm() * emb_b.norm()).clamp(min=1e-9)
            emb_dist = float(1.0 - cos_sim)

            if shape_a == shape_b and tech_jaccard < 0.2 and (techniques[a] or techniques[b]):
                same_shape_diff_technique.append((a, b, tech_jaccard, emb_dist))
            if tech_jaccard > 0.8 and shape_a != shape_b:
                same_technique_diff_shape.append((a, b, tech_jaccard, emb_dist))

    same_shape_diff_technique.sort(key=lambda t: t[3])
    same_technique_diff_shape.sort(key=lambda t: -t[3])

    return {
        "same_shape_different_technique_pairs_found": len(same_shape_diff_technique),
        "same_shape_different_technique_examples": [
            {"campaign_a": a, "campaign_b": b, "technique_jaccard": tj, "embedding_cosine_distance": ed}
            for a, b, tj, ed in same_shape_diff_technique[:top_n]
        ],
        "same_technique_different_shape_pairs_found": len(same_technique_diff_shape),
        "same_technique_different_shape_examples": [
            {"campaign_a": a, "campaign_b": b, "technique_jaccard": tj, "embedding_cosine_distance": ed}
            for a, b, tj, ed in same_technique_diff_shape[:top_n]
        ],
    }


# ---------------------------------------------------------------------------
# Near-duplicate diagnostic (Section 12, previously deferred)
# ---------------------------------------------------------------------------

def near_duplicate_diagnostic(
    embeddings: dict[str, torch.Tensor], samples: dict[str, CampaignGraphSample],
    techniques: dict[str, set[str]], attacker_of: dict[str, str], host_of: dict[str, str],
    threshold: float = 0.01,
) -> dict:
    from collections import Counter

    def _shape(cid: str) -> tuple:
        g = samples[cid].graph
        return tuple(sorted(Counter(g.node_types).items()))

    ids = list(embeddings.keys())
    n = len(ids)
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            a, b = ids[i], ids[j]
            ea, eb = embeddings[a], embeddings[b]
            cos_sim = torch.dot(ea, eb) / (ea.norm() * eb.norm()).clamp(min=1e-9)
            dist = float(1.0 - cos_sim)
            if dist < threshold:
                pairs.append((a, b))

    same_shape = same_attacker = same_host = same_technique = 0
    for a, b in pairs:
        if _shape(a) == _shape(b):
            same_shape += 1
        if attacker_of.get(a) is not None and attacker_of.get(a) == attacker_of.get(b):
            same_attacker += 1
        if host_of.get(a) is not None and host_of.get(a) == host_of.get(b):
            same_host += 1
        if a in techniques and b in techniques and techniques[a] == techniques[b] and techniques[a]:
            same_technique += 1

    total = len(pairs)
    return {
        "near_duplicate_pairs": total,
        "same_node_type_shape": same_shape,
        "same_node_type_shape_fraction": same_shape / total if total else float("nan"),
        "same_attacker": same_attacker,
        "same_attacker_fraction": same_attacker / total if total else float("nan"),
        "same_host": same_host,
        "same_host_fraction": same_host / total if total else float("nan"),
        "identical_technique_set": same_technique,
        "identical_technique_set_fraction": same_technique / total if total else float("nan"),
    }


if __name__ == "__main__":
    import json

    fold_safe, samples = fold_safe_full_population_embeddings()
    campaign_ids = list(samples.keys())
    techniques = get_campaign_techniques(campaign_ids)
    attacker_of = get_campaign_attacker_ips(campaign_ids)
    host_of = get_campaign_hosts(campaign_ids)

    temporal_result = temporal_retrieval_evaluation(fold_safe, samples, k=5)
    attacker_result = identity_retrieval_evaluation(fold_safe.embeddings, attacker_of, k=5)
    host_result = identity_retrieval_evaluation(fold_safe.embeddings, host_of, k=5)
    structural = structural_diagnostic(fold_safe, samples, techniques)

    id_preserving = identity_preserving_embeddings()
    attacker_result_weak = identity_retrieval_evaluation(id_preserving, attacker_of, k=5)
    near_dup = near_duplicate_diagnostic(id_preserving, samples, techniques, attacker_of, host_of)

    report = {
        "temporal_retrieval_fold_safe": {k: v for k, v in temporal_result.items() if k != "per_campaign"},
        "attacker_retrieval_fold_safe_generalization": attacker_result,
        "host_retrieval_fold_safe_generalization": host_result,
        "attacker_retrieval_single_model_identity_preservation_only": attacker_result_weak,
        "structural_diagnostic": structural,
        "near_duplicate_diagnostic_single_model": near_dup,
    }
    print(json.dumps(report, indent=2, default=str))
