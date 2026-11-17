"""
ml/gnn/train_gnn.py
======================
Trains CampaignGNN on a list of (networkx.DiGraph, severity_label) pairs
— real campaign subgraphs once CYUKTI has accumulated enough resolved
campaigns (via GraphSnapshotLoader + GraphBuilder against live Neo4j),
or the synthetic fixture generator for pipeline validation.

Refuses to train below MIN_TRAINING_GRAPHS, same principle as
ml/train_xgboost.py's row-count guard: silently training on too little
data is worse than refusing to.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import networkx as nx
import torch
import torch.nn as nn

from graph_encoder import encode_graph
from model import CampaignGNN, batch_graphs

MIN_TRAINING_GRAPHS = 30


@dataclass
class GNNTrainingResult:
    classes: list[str]
    train_graphs: int
    val_graphs: int
    val_accuracy: float
    val_macro_f1: float
    final_train_loss: float
    model_path: str


def train_gnn(
    graphs_and_labels: list[tuple[nx.DiGraph, str]],
    save_dir: str = "models",
    hidden_dim: int = 32,
    num_layers: int = 2,
    epochs: int = 60,
    lr: float = 0.01,
    val_fraction: float = 0.2,
    seed: int = 42,
) -> GNNTrainingResult:
    from sklearn.metrics import accuracy_score, f1_score
    from sklearn.model_selection import train_test_split

    if len(graphs_and_labels) < MIN_TRAINING_GRAPHS:
        raise ValueError(
            f"Not enough labeled campaign graphs to train the GNN: "
            f"found {len(graphs_and_labels)}, need at least {MIN_TRAINING_GRAPHS}. "
            "Real graphs accumulate as campaigns resolve; for pipeline "
            "development use synthetic_graphs.generate_synthetic_dataset()."
        )

    graphs, labels = zip(*graphs_and_labels)
    classes = sorted(set(labels))
    class_to_idx = {c: i for i, c in enumerate(classes)}
    if len(classes) < 2:
        raise ValueError(f"Only one class present ({classes}) — nothing to classify.")

    y = torch.tensor([class_to_idx[label] for label in labels], dtype=torch.long)
    encoded = [encode_graph(g) for g in graphs]

    train_idx, val_idx = train_test_split(
        range(len(encoded)), test_size=val_fraction, random_state=seed, stratify=y.tolist(),
    )

    train_graphs = [encoded[i] for i in train_idx]
    val_graphs = [encoded[i] for i in val_idx]
    y_train = y[train_idx]
    y_val = y[val_idx]

    torch.manual_seed(seed)
    model = CampaignGNN(hidden_dim=hidden_dim, num_classes=len(classes), num_layers=num_layers)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    x_train, edge_index_train, batch_train = batch_graphs(train_graphs)
    x_val, edge_index_val, batch_val = batch_graphs(val_graphs)

    model.train()
    final_loss = float("nan")
    for _ in range(epochs):
        optimizer.zero_grad()
        logits, _ = model(x_train, edge_index_train, batch_train, num_graphs=len(train_graphs))
        loss = loss_fn(logits, y_train)
        loss.backward()
        optimizer.step()
        final_loss = float(loss.item())

    model.eval()
    with torch.no_grad():
        val_logits, _ = model(x_val, edge_index_val, batch_val, num_graphs=len(val_graphs))
        val_pred = val_logits.argmax(dim=1)

    val_accuracy = accuracy_score(y_val.tolist(), val_pred.tolist())
    val_macro_f1 = f1_score(y_val.tolist(), val_pred.tolist(), average="macro", zero_division=0)

    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    model_file = save_path / "gnn_severity.pt"
    torch.save({
        "state_dict": model.state_dict(),
        "classes": classes,
        "hidden_dim": hidden_dim,
        "num_layers": num_layers,
    }, model_file)

    return GNNTrainingResult(
        classes=classes,
        train_graphs=len(train_graphs),
        val_graphs=len(val_graphs),
        val_accuracy=float(val_accuracy),
        val_macro_f1=float(val_macro_f1),
        final_train_loss=final_loss,
        model_path=str(model_file),
    )


def load_gnn(model_path: str) -> CampaignGNN:
    checkpoint = torch.load(model_path, weights_only=False)
    model = CampaignGNN(hidden_dim=checkpoint["hidden_dim"], num_classes=len(checkpoint["classes"]),
                         num_layers=checkpoint["num_layers"])
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    model.classes_ = checkpoint["classes"]
    return model


if __name__ == "__main__":
    from synthetic_graphs import generate_synthetic_dataset

    data = generate_synthetic_dataset(300)
    result = train_gnn(data)
    print(json.dumps(result.__dict__, indent=2))
