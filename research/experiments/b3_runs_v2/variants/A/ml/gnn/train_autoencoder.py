"""
ml/gnn/train_autoencoder.py
==============================
Trains GraphAutoencoder (autoencoder_model.py) on real campaign graphs
(ml.gnn.campaign_graphs.build_real_campaign_dataset — the same,
unmodified, already-tested extraction path the severity-classification
GNN work used; GraphSnapshotLoader is not touched by this module).

Objective A from GNN_REPRESENTATION_DESIGN.md Section 6: graph
autoencoding, chosen because it requires no label at all -- sidesteps
every circularity risk documented there (SIMILAR_TO/RESEMBLES are never
read by this module).

Split: by attacker-identity group (LAUNCHED, a raw, independently
observable fact -- not a derived similarity score), never by
SIMILAR_TO/RESEMBLES/technique overlap. See split_by_attacker_group's
docstring for why a group split, not a random split, is used.
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn.functional as F

from autoencoder_model import NUMERIC_FEATURE_OFFSET, GraphAutoencoder
from campaign_graphs import CampaignGraphSample
from graph_encoder import EDGE_FEATURE_DIM, NUMERIC_PROPS

MIN_TRAINING_GRAPHS = 30  # same guard/rationale as train_gnn.py, ml/dataset_utils.py


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


def get_campaign_attacker_ips(campaign_ids: list[str]) -> dict[str, str]:
    """Real, raw LAUNCHED-relationship identity fact -- not a derived
    similarity score. Used only to build a leakage-safe split
    (Phase E), never fed into the model as a feature."""
    from neo4j_client import driver

    if not campaign_ids:
        return {}
    with driver.session() as session:
        result = session.run(
            "MATCH (a:Attacker)-[:LAUNCHED]->(c:Campaign) "
            "WHERE c.campaign_id IN $ids RETURN c.campaign_id AS id, a.ip AS ip",
            ids=campaign_ids,
        )
        return {record["id"]: record["ip"] for record in result}


def split_by_attacker_group(
    campaign_ids: list[str],
    attacker_of: dict[str, str],
    val_fraction: float = 0.2,
) -> tuple[list[str], list[str]]:
    """Group campaigns by attacker_ip so no attacker's identity/reputation
    feature vector appears in both train and val (GNN_REPRESENTATION_DESIGN.md
    Section 2's leakage concern). The real distribution (verified live
    this phase) is highly skewed: 7 attackers with campaign counts
    [33, 26, 5, 4, 1, 1, 1]. A random per-campaign split would very
    likely place some of a shared attacker's campaigns on each side; a
    group split greedily assigns whole attacker groups (smallest first,
    for a deterministic, non-random tie-break) to validation until the
    target fraction is reached, then the rest to training -- so held-out
    graphs are guaranteed to involve attackers the model never saw
    during training.
    """
    groups: dict[str, list[str]] = defaultdict(list)
    for cid in campaign_ids:
        groups[attacker_of.get(cid, f"__no_attacker__{cid}")].append(cid)

    ordered_groups = sorted(groups.values(), key=lambda g: (len(g), g[0]))
    target_val = max(1, round(len(campaign_ids) * val_fraction))

    val_ids: list[str] = []
    train_ids: list[str] = []
    for group in ordered_groups:
        # Greedily add the smallest remaining group to val only if doing
        # so lands closer to target_val than leaving it in train --
        # without this, a lumpy real distribution (this repository's
        # real attacker-group sizes are [1, 1, 1, 4, 5, 26, 33]) can
        # overshoot massively the moment a large group is considered
        # (12 -> 38 by blindly adding the 26-group), inverting the
        # intended small-validation-holdout design.
        if len(val_ids) >= target_val:
            train_ids.extend(group)
            continue
        distance_if_added = abs(len(val_ids) + len(group) - target_val)
        distance_if_skipped = abs(len(val_ids) - target_val)
        if distance_if_added <= distance_if_skipped:
            val_ids.extend(group)
        else:
            train_ids.extend(group)
    return train_ids, val_ids


def _build_targets(sample: CampaignGraphSample) -> tuple[torch.Tensor, torch.Tensor]:
    """adj[i,j] in {0,1} (off-diagonal only); edge_type_idx[i,j] in
    [0, EDGE_FEATURE_DIM) where an edge exists, -1 elsewhere (ignored by
    the edge-type loss)."""
    n = sample.graph.num_nodes
    adj = torch.zeros((n, n))
    edge_type_idx = torch.full((n, n), -1, dtype=torch.long)
    src, dst = sample.graph.edge_index
    for k in range(src.shape[0]):
        i, j = int(src[k]), int(dst[k])
        adj[i, j] = 1.0
        edge_type_idx[i, j] = int(sample.graph.edge_attr[k].argmax().item())
    return adj, edge_type_idx


def _feature_stats(samples: list[CampaignGraphSample]) -> tuple[torch.Tensor, torch.Tensor]:
    """Mean/std of the numeric-property columns, computed over the
    TRAINING split only (standard practice -- avoids leaking
    validation-set statistics into the standardization). Stored in the
    saved artifact so inference can apply the identical transform."""
    rows = []
    for s in samples:
        numeric_cols = s.graph.x[:, NUMERIC_FEATURE_OFFSET:]
        rows.append(numeric_cols)
    all_rows = torch.cat(rows, dim=0) if rows else torch.zeros((0, len(NUMERIC_PROPS)))
    mean = all_rows.mean(dim=0) if all_rows.shape[0] > 0 else torch.zeros(len(NUMERIC_PROPS))
    std = all_rows.std(dim=0).clamp(min=1e-3) if all_rows.shape[0] > 0 else torch.ones(len(NUMERIC_PROPS))
    return mean, std


def _compute_pos_weight(samples: list[CampaignGraphSample]) -> float:
    """Global (not per-graph) positive-class weight for the edge-existence
    BCE loss, computed once over the training split -- per-graph weights
    would be undefined/unstable on graphs with zero negatives."""
    total_pos, total_pairs = 0, 0
    for s in samples:
        n = s.graph.num_nodes
        if n < 2:
            continue
        adj, _ = _build_targets(s)
        mask = ~torch.eye(n, dtype=torch.bool)
        total_pos += int(adj[mask].sum().item())
        total_pairs += int(mask.sum().item())
    total_neg = total_pairs - total_pos
    return float(total_neg / max(total_pos, 1))


@dataclass
class LossBreakdown:
    existence: float
    edge_type: float
    feature: float
    total: float


@dataclass
class AutoencoderTrainingResult:
    train_graphs: int
    val_graphs: int
    excluded_singleton_graphs: int
    train_loss_history: list[dict] = field(default_factory=list)
    val_loss_final: dict | None = None
    edge_existence_auc_train: float = float("nan")
    edge_existence_auc_val: float = float("nan")
    edge_type_accuracy_train: float = float("nan")
    edge_type_accuracy_val: float = float("nan")
    seed: int = 42
    embedding_dim: int = 8
    hidden_dim: int = 16
    epochs: int = 200
    lr: float = 0.01
    model_path: str = ""


def _epoch_pass(
    model: GraphAutoencoder,
    samples: list[CampaignGraphSample],
    feature_mean: torch.Tensor,
    feature_std: torch.Tensor,
    pos_weight: float,
    optimizer: torch.optim.Optimizer | None,
) -> LossBreakdown:
    """One pass (train if optimizer given, else eval) over all samples,
    summed loss then a single step -- full-batch training, standard and
    simplest choice for a 71-graph dataset (see
    GNN_REPRESENTATION_DESIGN.md Section 4's "small proof-of-concept
    baseline" framing, not a throughput optimization)."""
    is_train = optimizer is not None
    model.train(is_train)

    total_existence, total_type, total_feature = 0.0, 0.0, 0.0
    n_type_terms, n_graphs = 0, 0

    if is_train:
        optimizer.zero_grad()

    accum_loss = torch.tensor(0.0)
    for sample in samples:
        n = sample.graph.num_nodes
        if n < 2:
            continue  # no off-diagonal pair exists to reconstruct
        n_graphs += 1

        x = sample.graph.x.clone()
        x[:, NUMERIC_FEATURE_OFFSET:] = (x[:, NUMERIC_FEATURE_OFFSET:] - feature_mean) / feature_std

        out = model(x, sample.graph.edge_index)
        adj, edge_type_idx = _build_targets(sample)
        mask = ~torch.eye(n, dtype=torch.bool)

        existence_loss = F.binary_cross_entropy_with_logits(
            out.edge_existence_logits[mask], adj[mask],
            pos_weight=torch.tensor(pos_weight),
        )

        type_mask = edge_type_idx >= 0
        if type_mask.any():
            type_logits = out.edge_type_logits[type_mask]
            type_targets = edge_type_idx[type_mask]
            type_loss = F.cross_entropy(type_logits, type_targets)
            n_type_terms += 1
        else:
            type_loss = torch.tensor(0.0)

        target_features = (x[:, NUMERIC_FEATURE_OFFSET:])
        feature_loss = F.mse_loss(out.node_feature_recon, target_features)

        graph_loss = existence_loss + type_loss + feature_loss
        accum_loss = accum_loss + graph_loss

        total_existence += float(existence_loss.item())
        total_type += float(type_loss.item())
        total_feature += float(feature_loss.item())

    if n_graphs == 0:
        return LossBreakdown(0.0, 0.0, 0.0, 0.0)

    mean_loss = accum_loss / n_graphs
    if is_train:
        mean_loss.backward()
        optimizer.step()

    return LossBreakdown(
        existence=total_existence / n_graphs,
        edge_type=total_type / max(n_type_terms, 1),
        feature=total_feature / n_graphs,
        total=float(mean_loss.item()),
    )


def _edge_existence_auc(model: GraphAutoencoder, samples: list[CampaignGraphSample],
                         feature_mean: torch.Tensor, feature_std: torch.Tensor) -> float:
    from sklearn.metrics import roc_auc_score

    model.eval()
    all_scores, all_labels = [], []
    with torch.no_grad():
        for sample in samples:
            n = sample.graph.num_nodes
            if n < 2:
                continue
            x = sample.graph.x.clone()
            x[:, NUMERIC_FEATURE_OFFSET:] = (x[:, NUMERIC_FEATURE_OFFSET:] - feature_mean) / feature_std
            out = model(x, sample.graph.edge_index)
            adj, _ = _build_targets(sample)
            mask = ~torch.eye(n, dtype=torch.bool)
            all_scores.extend(out.edge_existence_logits[mask].tolist())
            all_labels.extend(adj[mask].tolist())
    if len(set(all_labels)) < 2:
        return float("nan")  # AUC undefined with only one class present
    return float(roc_auc_score(all_labels, all_scores))


def _edge_type_accuracy(model: GraphAutoencoder, samples: list[CampaignGraphSample],
                         feature_mean: torch.Tensor, feature_std: torch.Tensor) -> float:
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for sample in samples:
            n = sample.graph.num_nodes
            if n < 2:
                continue
            x = sample.graph.x.clone()
            x[:, NUMERIC_FEATURE_OFFSET:] = (x[:, NUMERIC_FEATURE_OFFSET:] - feature_mean) / feature_std
            out = model(x, sample.graph.edge_index)
            _, edge_type_idx = _build_targets(sample)
            type_mask = edge_type_idx >= 0
            if not type_mask.any():
                continue
            preds = out.edge_type_logits[type_mask].argmax(dim=-1)
            targets = edge_type_idx[type_mask]
            correct += int((preds == targets).sum().item())
            total += int(type_mask.sum().item())
    return float(correct / total) if total > 0 else float("nan")


def fit_autoencoder(
    train_samples: list[CampaignGraphSample],
    hidden_dim: int = 16,
    embedding_dim: int = 8,
    num_layers: int = 2,
    epochs: int = 200,
    lr: float = 0.01,
    seed: int = 42,
) -> tuple[GraphAutoencoder, torch.Tensor, torch.Tensor, float, list[dict]]:
    """Fold-safe core fit: trains a fresh GraphAutoencoder on exactly
    the samples given (no internal train/val split of its own) and
    returns (model, feature_mean, feature_std, pos_weight, loss_history)
    -- everything a caller needs to embed OTHER (e.g. held-out) samples
    consistently, without this function ever seeing them. Factored out
    of train_autoencoder() (below) so the GNN-XGBoost ablation
    (xgboost_ablation.py) can fit a fresh, fold-scoped encoder per
    cross-validation fold without duplicating the training loop --
    train_autoencoder()'s own behavior/return value is unchanged by
    this refactor (verified: its own tests pass unchanged)."""
    feature_mean, feature_std = _feature_stats(train_samples)
    pos_weight = _compute_pos_weight(train_samples)

    set_seed(seed)
    model = GraphAutoencoder(hidden_dim=hidden_dim, embedding_dim=embedding_dim, num_layers=num_layers)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history = []
    for epoch in range(epochs):
        train_loss = _epoch_pass(model, train_samples, feature_mean, feature_std, pos_weight, optimizer)
        history.append({"epoch": epoch, **train_loss.__dict__})

    return model, feature_mean, feature_std, pos_weight, history


def embed_samples(
    model: GraphAutoencoder,
    samples: list[CampaignGraphSample],
    feature_mean: torch.Tensor,
    feature_std: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Inference only -- passes each sample's graph through the
    already-trained `model` (no gradient, no parameter update, no
    statistics fitting). Safe to call with samples the model was never
    trained on (that is precisely the fold-safe embedding-generation
    step the ablation needs)."""
    model.eval()
    embeddings = {}
    for sample in samples:
        if sample.graph.num_nodes == 0:
            continue
        x = sample.graph.x.clone()
        x[:, NUMERIC_FEATURE_OFFSET:] = (x[:, NUMERIC_FEATURE_OFFSET:] - feature_mean) / feature_std
        embeddings[sample.campaign_id] = model.embed_graph(x, sample.graph.edge_index)
    return embeddings


def train_autoencoder(
    samples: list[CampaignGraphSample],
    save_dir: str = "ml/models",
    hidden_dim: int = 16,
    embedding_dim: int = 8,
    num_layers: int = 2,
    epochs: int = 200,
    lr: float = 0.01,
    val_fraction: float = 0.2,
    seed: int = 42,
) -> AutoencoderTrainingResult:
    if len(samples) < MIN_TRAINING_GRAPHS:
        raise ValueError(
            f"Not enough real campaign graphs to train the autoencoder: "
            f"found {len(samples)}, need at least {MIN_TRAINING_GRAPHS}."
        )

    excluded = [s for s in samples if s.graph.num_nodes < 2]
    usable = [s for s in samples if s.graph.num_nodes >= 2]

    campaign_ids = [s.campaign_id for s in usable]
    attacker_of = get_campaign_attacker_ips(campaign_ids)
    train_ids, val_ids = split_by_attacker_group(campaign_ids, attacker_of, val_fraction)
    train_set = {cid: s for cid, s in zip(campaign_ids, usable)}
    train_samples = [train_set[cid] for cid in train_ids]
    val_samples = [train_set[cid] for cid in val_ids]

    model, feature_mean, feature_std, pos_weight, history = fit_autoencoder(
        train_samples, hidden_dim=hidden_dim, embedding_dim=embedding_dim,
        num_layers=num_layers, epochs=epochs, lr=lr, seed=seed,
    )

    val_loss = _epoch_pass(model, val_samples, feature_mean, feature_std, pos_weight, None) if val_samples else None

    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    model_file = save_path / "gnn_autoencoder.pt"
    torch.save({
        "state_dict": model.state_dict(),
        "hidden_dim": hidden_dim,
        "embedding_dim": embedding_dim,
        "num_layers": num_layers,
        "feature_mean": feature_mean,
        "feature_std": feature_std,
        "seed": seed,
        "train_campaign_ids": train_ids,
        "val_campaign_ids": val_ids,
        "pos_weight": pos_weight,
    }, model_file)

    return AutoencoderTrainingResult(
        train_graphs=len(train_samples),
        val_graphs=len(val_samples),
        excluded_singleton_graphs=len(excluded),
        train_loss_history=history,
        val_loss_final=val_loss.__dict__ if val_loss else None,
        edge_existence_auc_train=_edge_existence_auc(model, train_samples, feature_mean, feature_std),
        edge_existence_auc_val=_edge_existence_auc(model, val_samples, feature_mean, feature_std) if val_samples else float("nan"),
        edge_type_accuracy_train=_edge_type_accuracy(model, train_samples, feature_mean, feature_std),
        edge_type_accuracy_val=_edge_type_accuracy(model, val_samples, feature_mean, feature_std) if val_samples else float("nan"),
        seed=seed,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        epochs=epochs,
        lr=lr,
        model_path=str(model_file),
    )


def load_autoencoder(model_path: str) -> tuple[GraphAutoencoder, dict]:
    checkpoint = torch.load(model_path, weights_only=False)
    model = GraphAutoencoder(
        hidden_dim=checkpoint["hidden_dim"],
        embedding_dim=checkpoint["embedding_dim"],
        num_layers=checkpoint["num_layers"],
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, checkpoint


if __name__ == "__main__":
    from campaign_graphs import build_real_campaign_dataset

    samples = build_real_campaign_dataset()
    result = train_autoencoder(samples)
    report = {k: v for k, v in result.__dict__.items() if k != "train_loss_history"}
    report["final_train_loss"] = result.train_loss_history[-1] if result.train_loss_history else None
    report["first_train_loss"] = result.train_loss_history[0] if result.train_loss_history else None
    print(json.dumps(report, indent=2, default=str))
