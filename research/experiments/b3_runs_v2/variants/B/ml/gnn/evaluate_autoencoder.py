"""
ml/gnn/evaluate_autoencoder.py
==================================
Non-circular evaluation of the trained GraphAutoencoder
(train_autoencoder.py), per GNN_REPRESENTATION_DESIGN.md Section 9.
Never reads or reasons about SIMILAR_TO/RESEMBLES. Every evaluation
here is built from raw, independently-observable graph facts (attacker/
host identity via LAUNCHED/TARGETS, real event timestamps) or from
already-existing, already-tested scalar features
(graph_feature_engine.GraphAnalytics) -- nothing here is a claim that
the GNN "understands attack campaigns" (Phase K's explicit caution);
each check states plainly what it does and does not show.
"""

from __future__ import annotations

import math
from collections import defaultdict

import torch

from autoencoder_model import NUMERIC_FEATURE_OFFSET, GraphAutoencoder
from campaign_graphs import CampaignGraphSample
from train_autoencoder import get_campaign_attacker_ips


def compute_embeddings(
    model: GraphAutoencoder,
    samples: list[CampaignGraphSample],
    feature_mean: torch.Tensor,
    feature_std: torch.Tensor,
) -> dict[str, torch.Tensor]:
    embeddings = {}
    model.eval()
    for sample in samples:
        if sample.graph.num_nodes == 0:
            continue
        x = sample.graph.x.clone()
        x[:, NUMERIC_FEATURE_OFFSET:] = (x[:, NUMERIC_FEATURE_OFFSET:] - feature_mean) / feature_std
        embeddings[sample.campaign_id] = model.embed_graph(x, sample.graph.edge_index)
    return embeddings


def check_determinism(model: GraphAutoencoder, sample: CampaignGraphSample,
                       feature_mean: torch.Tensor, feature_std: torch.Tensor) -> bool:
    """Phase G: same input -> same output, twice, in eval mode."""
    model.eval()
    x = sample.graph.x.clone()
    x[:, NUMERIC_FEATURE_OFFSET:] = (x[:, NUMERIC_FEATURE_OFFSET:] - feature_mean) / feature_std
    z1 = model.embed_graph(x, sample.graph.edge_index)
    z2 = model.embed_graph(x, sample.graph.edge_index)
    return bool(torch.allclose(z1, z2, atol=1e-7))


def check_finite(embeddings: dict[str, torch.Tensor]) -> dict:
    all_finite = all(bool(torch.isfinite(z).all()) for z in embeddings.values())
    return {"all_finite": all_finite, "count": len(embeddings)}


def check_collapse(embeddings: dict[str, torch.Tensor]) -> dict:
    """Phase G: is every z_G ~ identical (representation collapse), or
    do structurally different graphs produce distinguishable vectors?
    Reports pairwise cosine-distance distribution and embedding-norm
    distribution -- does not by itself judge semantic quality (Phase G's
    explicit caution)."""
    ids = list(embeddings.keys())
    vectors = torch.stack([embeddings[i] for i in ids])
    norms = vectors.norm(dim=1)

    normed = vectors / vectors.norm(dim=1, keepdim=True).clamp(min=1e-9)
    cos_sim = normed @ normed.T
    n = len(ids)
    off_diag = cos_sim[~torch.eye(n, dtype=torch.bool)]
    cos_dist = 1.0 - off_diag

    near_duplicate_threshold = 0.01
    near_duplicate_pairs = int((cos_dist < near_duplicate_threshold).sum().item()) // 2

    return {
        "num_embeddings": n,
        "embedding_norm_mean": float(norms.mean()),
        "embedding_norm_std": float(norms.std()),
        "embedding_norm_min": float(norms.min()),
        "embedding_norm_max": float(norms.max()),
        "pairwise_cosine_distance_mean": float(cos_dist.mean()),
        "pairwise_cosine_distance_std": float(cos_dist.std()),
        "pairwise_cosine_distance_min": float(cos_dist.min()),
        "pairwise_cosine_distance_max": float(cos_dist.max()),
        "near_duplicate_pairs_below_0.01_cosine_distance": near_duplicate_pairs,
        "total_pairs": n * (n - 1) // 2,
    }


def identity_retrieval_eval(embeddings: dict[str, torch.Tensor], attacker_of: dict[str, str],
                             host_of: dict[str, str]) -> dict:
    """Phase K, evaluation #2: does the nearest neighbor by embedding
    cosine similarity share the SAME attacker/host -- a raw,
    independently-observable identity fact (LAUNCHED/TARGETS), never a
    derived similarity score. Explicitly a low bar (an embedding that
    just re-encodes the attacker/host one-hot would pass trivially);
    reported against the chance baseline implied by the real (skewed)
    group-size distribution, not against an arbitrary 50%."""
    ids = list(embeddings.keys())
    n = len(ids)
    vectors = torch.stack([embeddings[i] for i in ids])
    normed = vectors / vectors.norm(dim=1, keepdim=True).clamp(min=1e-9)
    cos_sim = normed @ normed.T
    cos_sim.fill_diagonal_(-2.0)  # exclude self

    same_attacker_hits, same_host_hits, evaluable = 0, 0, 0
    for idx, cid in enumerate(ids):
        nn_idx = int(cos_sim[idx].argmax().item())
        nn_cid = ids[nn_idx]
        if cid not in attacker_of or nn_cid not in attacker_of:
            continue
        evaluable += 1
        if attacker_of[cid] == attacker_of[nn_cid]:
            same_attacker_hits += 1
        if host_of.get(cid) is not None and host_of.get(cid) == host_of.get(nn_cid):
            same_host_hits += 1

    # Chance baseline: probability a uniformly random OTHER campaign
    # shares the same attacker group, averaged over all campaigns
    # (accounts for the real, skewed group sizes rather than assuming 1/n).
    group_sizes = defaultdict(int)
    for cid in ids:
        if cid in attacker_of:
            group_sizes[attacker_of[cid]] += 1
    chance_hits = 0.0
    for cid in ids:
        if cid not in attacker_of:
            continue
        g = group_sizes[attacker_of[cid]]
        chance_hits += (g - 1) / (n - 1) if n > 1 else 0.0
    chance_baseline = chance_hits / evaluable if evaluable else float("nan")

    return {
        "evaluable_campaigns": evaluable,
        "same_attacker_nearest_neighbor_rate": same_attacker_hits / evaluable if evaluable else float("nan"),
        "same_attacker_chance_baseline": chance_baseline,
        "same_host_nearest_neighbor_rate": same_host_hits / evaluable if evaluable else float("nan"),
        "caveat": (
            "Low bar: same-attacker/host is a raw identity fact, and "
            "the real attacker-group distribution is highly skewed "
            "(2 of 7 attackers cover 83% of campaigns), so the chance "
            "baseline itself is already high. Passing this does not "
            "show the GNN understands attack campaigns."
        ),
    }


def temporal_self_consistency(
    model: GraphAutoencoder,
    temporal_snapshots_by_campaign: dict[str, list],
    all_embeddings: dict[str, torch.Tensor],
    feature_mean: torch.Tensor,
    feature_std: torch.Tensor,
) -> dict:
    """Phase J/K #1: for campaigns with real, distinctly-timestamped
    snapshots, is an earlier snapshot's embedding closer to a LATER
    snapshot of the SAME campaign than to other campaigns' embeddings?
    Uses only real, already-stored timestamps -- no SIMILAR_TO. Not
    presented as proof of semantic similarity (Phase J's caution)."""
    model.eval()
    other_ids = list(all_embeddings.keys())
    other_vectors = torch.stack([all_embeddings[i] for i in other_ids]) if other_ids else None

    per_campaign = {}
    for campaign_id, snapshots in temporal_snapshots_by_campaign.items():
        if len(snapshots) < 2:
            continue
        snap_embeddings = []
        for snap in snapshots:
            if snap.graph.num_nodes == 0:
                continue
            x = snap.graph.x.clone()
            x[:, NUMERIC_FEATURE_OFFSET:] = (x[:, NUMERIC_FEATURE_OFFSET:] - feature_mean) / feature_std
            snap_embeddings.append(model.embed_graph(x, snap.graph.edge_index))
        if len(snap_embeddings) < 2:
            continue

        consecutive_cos_dists = []
        for a, b in zip(snap_embeddings[:-1], snap_embeddings[1:]):
            cos_sim = torch.dot(a, b) / (a.norm() * b.norm()).clamp(min=1e-9)
            consecutive_cos_dists.append(float(1.0 - cos_sim))

        first_snap = snap_embeddings[0]
        cross_campaign_dists = []
        if other_vectors is not None:
            for other_id, other_vec in zip(other_ids, other_vectors):
                if other_id == campaign_id:
                    continue
                cos_sim = torch.dot(first_snap, other_vec) / (
                    first_snap.norm() * other_vec.norm()
                ).clamp(min=1e-9)
                cross_campaign_dists.append(float(1.0 - cos_sim))

        per_campaign[campaign_id] = {
            "num_snapshots": len(snap_embeddings),
            "mean_consecutive_snapshot_cosine_distance": sum(consecutive_cos_dists) / len(consecutive_cos_dists),
            "mean_cross_campaign_cosine_distance": (
                sum(cross_campaign_dists) / len(cross_campaign_dists) if cross_campaign_dists else float("nan")
            ),
        }

    if not per_campaign:
        return {"campaigns_evaluated": 0, "note": "No campaign had 2+ non-empty snapshots to compare."}

    within_campaign_closer = sum(
        1 for v in per_campaign.values()
        if v["mean_consecutive_snapshot_cosine_distance"] < v["mean_cross_campaign_cosine_distance"]
    )
    return {
        "campaigns_evaluated": len(per_campaign),
        "campaigns_where_own_history_is_closer_than_other_campaigns": within_campaign_closer,
        "per_campaign": per_campaign,
        "caveat": (
            "Demonstrates whether the representation preserves campaign "
            "identity while structure grows over real time -- not proof "
            "of semantic similarity between different campaigns."
        ),
    }


def trivial_node_type_pair_baseline(samples: list[CampaignGraphSample]) -> dict:
    """Phase L honesty check: a zero-parameter rule -- predict an edge
    exists iff the two nodes' types are one of the schema's
    always-connected pairs (Attacker-Campaign, Campaign-AttackEvent,
    AttackEvent-Technique, Campaign-Host). If the trained model's
    edge-existence reconstruction barely beats this, the model has not
    demonstrated it learned anything beyond what the node-type one-hot
    (already a direct input feature) trivially implies."""
    canonical_pairs = {
        frozenset({"Attacker", "Campaign"}),
        frozenset({"Campaign", "AttackEvent"}),
        frozenset({"AttackEvent", "Technique"}),
        frozenset({"Campaign", "Host"}),
    }

    tp = fp = tn = fn = 0
    for sample in samples:
        n = sample.graph.num_nodes
        if n < 2:
            continue
        types = sample.graph.node_types
        src, dst = sample.graph.edge_index
        true_adj = torch.zeros((n, n), dtype=torch.bool)
        for k in range(src.shape[0]):
            true_adj[int(src[k]), int(dst[k])] = True

        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                predicted = frozenset({types[i], types[j]}) in canonical_pairs
                actual = bool(true_adj[i, j])
                if predicted and actual:
                    tp += 1
                elif predicted and not actual:
                    fp += 1
                elif not predicted and actual:
                    fn += 1
                else:
                    tn += 1

    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    recall = tp / (tp + fn) if (tp + fn) else float("nan")
    f1 = 2 * precision * recall / (precision + recall) if precision and recall and (precision + recall) else float("nan")
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) else float("nan")
    return {"precision": precision, "recall": recall, "f1": f1, "accuracy": accuracy,
            "tp": tp, "fp": fp, "tn": tn, "fn": fn}


def model_edge_existence_at_threshold(
    model: GraphAutoencoder,
    samples: list[CampaignGraphSample],
    feature_mean: torch.Tensor,
    feature_std: torch.Tensor,
    threshold: float = 0.5,
) -> dict:
    """Same precision/recall/F1/accuracy shape as
    trivial_node_type_pair_baseline, at a plain 0.5 sigmoid threshold --
    for a fair apples-to-apples comparison (AUC and a hard 0/1 rule are
    not directly comparable; this is)."""
    model.eval()
    tp = fp = tn = fn = 0
    with torch.no_grad():
        for sample in samples:
            n = sample.graph.num_nodes
            if n < 2:
                continue
            x = sample.graph.x.clone()
            x[:, NUMERIC_FEATURE_OFFSET:] = (x[:, NUMERIC_FEATURE_OFFSET:] - feature_mean) / feature_std
            out = model(x, sample.graph.edge_index)
            probs = torch.sigmoid(out.edge_existence_logits)
            src, dst = sample.graph.edge_index
            true_adj = torch.zeros((n, n), dtype=torch.bool)
            for k in range(src.shape[0]):
                true_adj[int(src[k]), int(dst[k])] = True
            mask = ~torch.eye(n, dtype=torch.bool)
            predicted = (probs[mask] >= threshold)
            actual = true_adj[mask]
            tp += int((predicted & actual).sum().item())
            fp += int((predicted & ~actual).sum().item())
            tn += int((~predicted & ~actual).sum().item())
            fn += int((~predicted & actual).sum().item())

    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    recall = tp / (tp + fn) if (tp + fn) else float("nan")
    f1 = 2 * precision * recall / (precision + recall) if precision and recall and (precision + recall) else float("nan")
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) else float("nan")
    return {"precision": precision, "recall": recall, "f1": f1, "accuracy": accuracy,
            "tp": tp, "fp": fp, "tn": tn, "fn": fn}


def compare_with_scalar_features(embeddings: dict[str, torch.Tensor]) -> dict:
    """Phase L: does pairwise embedding distance correlate with pairwise
    distance in XGBoost's existing 19 GraphFeatures scalars, or does it
    carry different information? High correlation would suggest the
    embedding mostly reproduces what the scalars already capture; lower
    correlation is evidence (not proof) of additional structural
    variation. Reuses graph_analytics.extract_features unchanged (the
    same production function XGBoost's own features already call) --
    does not duplicate or reimplement it."""
    from graph_feature_engine import graph_analytics

    scalar_vectors = {}
    for campaign_id in embeddings:
        try:
            features = graph_analytics.extract_features(campaign_id)
        except Exception:
            continue
        scalar_vectors[campaign_id] = torch.tensor([
            features.graph_density, features.graph_connectivity, features.average_degree,
            features.attacker_degree, features.victim_degree, features.technique_degree,
            features.attack_chain_depth, features.average_path_length, features.graph_diameter,
            features.branching_factor, features.average_clustering, features.average_betweenness,
            features.average_closeness, features.community_count, features.largest_community,
            features.campaign_complexity, features.structural_risk, features.evolution_rate,
            float(features.node_count),
        ], dtype=torch.float32)

    common_ids = [cid for cid in embeddings if cid in scalar_vectors]
    if len(common_ids) < 3:
        return {"pairs_compared": 0, "note": "Too few campaigns with extractable scalar features."}

    scalar_mat = torch.stack([scalar_vectors[cid] for cid in common_ids])
    scalar_std = scalar_mat.std(dim=0).clamp(min=1e-6)
    scalar_norm = (scalar_mat - scalar_mat.mean(dim=0)) / scalar_std

    embed_mat = torch.stack([embeddings[cid] for cid in common_ids])

    embed_dists, scalar_dists = [], []
    n = len(common_ids)
    for i in range(n):
        for j in range(i + 1, n):
            embed_dists.append(float(torch.norm(embed_mat[i] - embed_mat[j])))
            scalar_dists.append(float(torch.norm(scalar_norm[i] - scalar_norm[j])))

    ed = torch.tensor(embed_dists)
    sd = torch.tensor(scalar_dists)
    if ed.std() < 1e-9 or sd.std() < 1e-9:
        correlation = float("nan")
    else:
        correlation = float(torch.corrcoef(torch.stack([ed, sd]))[0, 1])

    return {
        "pairs_compared": len(embed_dists),
        "pearson_correlation_embedding_dist_vs_scalar_feature_dist": correlation,
        "interpretation": (
            "Correlation close to 1.0 would suggest the embedding mostly "
            "reproduces the 19-scalar summary; materially below 1.0 is "
            "evidence (not proof) the embedding varies along dimensions "
            "the scalars collapse. No claim of superiority is made from "
            "this number alone."
        ),
    }


if __name__ == "__main__":
    import json

    from campaign_graphs import build_real_campaign_dataset
    from neo4j_client import driver
    from temporal_graphs import build_all_temporal_snapshots
    from train_autoencoder import load_autoencoder

    samples = build_real_campaign_dataset()
    model, checkpoint = load_autoencoder("ml/models/gnn_autoencoder.pt")
    feature_mean, feature_std = checkpoint["feature_mean"], checkpoint["feature_std"]

    embeddings = compute_embeddings(model, samples, feature_mean, feature_std)
    campaign_ids = [s.campaign_id for s in samples]
    attacker_of = get_campaign_attacker_ips(campaign_ids)
    with driver.session() as session:
        host_of = {
            r["id"]: r["ip"] for r in session.run(
                "MATCH (c:Campaign)-[:TARGETS]->(h:Host) WHERE c.campaign_id IN $ids "
                "RETURN c.campaign_id AS id, h.ip AS ip", ids=campaign_ids,
            )
        }

    determinism_sample = next((s for s in samples if s.graph.num_nodes > 0), None)
    temporal_snapshots = build_all_temporal_snapshots()

    report = {
        "num_embeddings": len(embeddings),
        "determinism_check": (
            check_determinism(model, determinism_sample, feature_mean, feature_std)
            if determinism_sample else None
        ),
        "finiteness_check": check_finite(embeddings),
        "collapse_check": check_collapse(embeddings),
        "identity_retrieval": identity_retrieval_eval(embeddings, attacker_of, host_of),
        "temporal_self_consistency": temporal_self_consistency(
            model, temporal_snapshots, embeddings, feature_mean, feature_std,
        ),
        "trivial_node_type_pair_baseline": trivial_node_type_pair_baseline(samples),
        "model_edge_existence_at_0.5_threshold": model_edge_existence_at_threshold(
            model, samples, feature_mean, feature_std,
        ),
        "comparison_with_19_scalar_features": compare_with_scalar_features(embeddings),
    }
    print(json.dumps(report, indent=2, default=str))
