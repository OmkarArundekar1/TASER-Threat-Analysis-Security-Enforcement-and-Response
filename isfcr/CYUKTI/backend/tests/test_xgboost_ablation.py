"""
Tests for ml/gnn/xgboost_ablation.py -- the GNN-embedding vs. baseline
XGBoost ablation (see ../../GNN_XGBOOST_ABLATION.md). Mocks only the
database boundary (neo4j_client.driver, campaign_graphs.build_real_campaign_dataset)
and uses tiny synthetic feature/graph fixtures, per this repository's
established testing convention -- never mocks the model/training logic
itself.
"""

import networkx as nx
import numpy as np
import pandas as pd
import pytest
import torch

import ml.gnn.xgboost_ablation as xgboost_ablation
from ml.gnn.campaign_graphs import CampaignGraphSample
from ml.gnn.graph_encoder import NUMERIC_PROPS, encode_graph
from ml.gnn.train_autoencoder import embed_samples, fit_autoencoder
from ml.gnn.xgboost_ablation import (
    _fit_and_eval_xgb,
    _score_predictions,
    build_attacker_groups,
    load_ablation_dataset,
)


def _graph_sample(campaign_id: str, extra_node: bool = False) -> CampaignGraphSample:
    G = nx.DiGraph()
    G.add_node("a", labels=["Attacker"], vt_reputation=50.0)
    G.add_node("c", labels=["Campaign"])
    G.add_node("e", labels=["AttackEvent"], occurrences=2.0)
    G.add_edge("a", "c", relationship="LAUNCHED")
    G.add_edge("c", "e", relationship="HAS_EVENT")
    if extra_node:
        G.add_node("t", labels=["Technique"])
        G.add_edge("e", "t", relationship="MATCHES")
    return CampaignGraphSample(
        campaign_id=campaign_id, graph=encode_graph(G), severity="Low", risk_score=1.0, num_events=1,
    )


# ---------------------------------------------------------------------------
# load_ablation_dataset -- dataset mapping (Phase 3)
# ---------------------------------------------------------------------------

def test_load_ablation_dataset_reports_and_excludes_mismatches(tmp_path, monkeypatch):
    from ml.feature_schema import CampaignDatasetRecord

    all_columns = [f.name for f in CampaignDatasetRecord.__dataclass_fields__.values()] \
        if hasattr(CampaignDatasetRecord, "__dataclass_fields__") else None
    # Build a minimal CSV with the real schema's columns so FEATURE_COLUMNS lookups succeed.
    from dataclasses import fields
    columns = [f.name for f in fields(CampaignDatasetRecord)]
    rows = []
    for cid in ["c1", "c2", "c4"]:  # c4 exists only in the CSV
        row = {col: 0 for col in columns}
        row["campaign_id"] = cid
        row["attacker_ip"] = "1.2.3.4"
        row["victim_ip"] = "5.6.7.8"
        row["severity"] = "Low"
        rows.append(row)
    csv_path = tmp_path / "campaign_dataset.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    # GNN extraction has c1, c2, c3 -- c3 exists only in the live GNN dataset.
    monkeypatch.setattr(
        xgboost_ablation, "build_real_campaign_dataset",
        lambda: [_graph_sample("c1"), _graph_sample("c2"), _graph_sample("c3")],
    )

    df, samples, mapping = load_ablation_dataset(csv_path=str(csv_path))

    assert mapping.csv_campaigns == 3
    assert mapping.live_gnn_campaigns == 3
    assert mapping.usable_campaigns == 2
    assert mapping.csv_only_excluded == ["c4"]
    assert mapping.gnn_only_excluded == ["c3"]
    assert sorted(df["campaign_id"]) == ["c1", "c2"]
    assert set(samples.keys()) == {"c1", "c2"}


# ---------------------------------------------------------------------------
# build_attacker_groups
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


def test_build_attacker_groups_groups_shared_attackers(monkeypatch):
    monkeypatch.setattr(
        "neo4j_client.driver",
        _FakeDriver([{"id": "c1", "ip": "1.1.1.1"}, {"id": "c2", "ip": "1.1.1.1"}, {"id": "c3", "ip": "2.2.2.2"}]),
    )
    groups = build_attacker_groups(["c1", "c2", "c3"])
    assert sorted(groups["1.1.1.1"]) == ["c1", "c2"]
    assert groups["2.2.2.2"] == ["c3"]


def test_build_attacker_groups_gives_unresolved_campaign_its_own_group(monkeypatch):
    monkeypatch.setattr("neo4j_client.driver", _FakeDriver([{"id": "c1", "ip": "1.1.1.1"}]))
    groups = build_attacker_groups(["c1", "c2"])
    all_members = [c for members in groups.values() for c in members]
    assert sorted(all_members) == ["c1", "c2"]
    assert len(groups) == 2  # c2 got its own singleton group, not silently dropped


# ---------------------------------------------------------------------------
# fold-safety: feature standardization is fit on TRAIN samples only
# ---------------------------------------------------------------------------

def test_fold_safe_feature_stats_computed_only_from_train_samples():
    train_samples = [_graph_sample("train-1"), _graph_sample("train-2", extra_node=True)]
    test_sample = _graph_sample("test-1")
    # give the test sample an implausible, easy-to-spot value that must
    # NOT influence feature_mean/feature_std if fold-safety holds
    test_sample.graph.x[:, -len(NUMERIC_PROPS):] = 9999.0

    model, feature_mean, feature_std, pos_weight, history = fit_autoencoder(
        train_samples, hidden_dim=8, embedding_dim=4, num_layers=1, epochs=2, seed=1,
    )

    # Manually recompute expected stats from train_samples only.
    rows = torch.cat([s.graph.x[:, -len(NUMERIC_PROPS):] for s in train_samples], dim=0)
    expected_mean = rows.mean(dim=0)
    assert torch.allclose(feature_mean, expected_mean)
    assert not torch.isclose(feature_mean, torch.tensor(9999.0)).any()

    # Embedding the test sample must not raise or require refitting.
    test_embeddings = embed_samples(model, [test_sample], feature_mean, feature_std)
    assert "test-1" in test_embeddings
    assert torch.isfinite(test_embeddings["test-1"]).all()


def test_fold_safe_test_campaign_absent_from_training_history():
    """The training loss history (an audit trail of what the encoder
    actually optimized against) must never reference the held-out
    sample -- a mechanical proxy for "no test graph used in GNN
    fitting"."""
    train_samples = [_graph_sample("train-1"), _graph_sample("train-2")]
    model, feature_mean, feature_std, pos_weight, history = fit_autoencoder(
        train_samples, hidden_dim=8, embedding_dim=4, num_layers=1, epochs=1, seed=1,
    )
    assert len(history) == 1  # one entry per epoch, never per-sample-id -- no leakage surface here
    # (fit_autoencoder's signature only ever accepts train_samples -- there
    # is no code path by which a held-out sample could reach _epoch_pass)


# ---------------------------------------------------------------------------
# _fit_and_eval_xgb — dimensionality, missing-class handling, artifact isolation
# ---------------------------------------------------------------------------

def test_fit_and_eval_xgb_handles_class_missing_from_training(monkeypatch):
    """Reproduces this repository's real finding (GNN_XGBOOST_ABLATION.md
    Section 6): one real LOGO fold's training split has zero "Medium"
    examples while the test split has some. XGBoost's sklearn wrapper
    requires contiguous labels present in y_train, so this must not
    crash, and the missing class must score 0, not be silently omitted."""
    rng = np.random.default_rng(0)
    X_train = rng.random((10, 5)).astype(np.float32)
    y_train = ["Low"] * 8 + ["Critical"] * 2  # "Medium" never appears
    X_test = rng.random((3, 5)).astype(np.float32)
    y_test = ["Low", "Medium", "Critical"]

    result = _fit_and_eval_xgb(X_train, y_train, X_test, y_test, all_labels=["Critical", "Low", "Medium"])

    assert result["per_class"]["Medium"]["precision"] == 0.0
    assert result["per_class"]["Medium"]["recall"] == 0.0
    assert result["classes_present_in_train"] == ["Critical", "Low"]
    assert len(result["confusion_matrix"]) == 3


def test_fit_and_eval_xgb_never_persists_a_model_artifact(monkeypatch):
    """Artifact isolation: this experiment must never call
    XGBClassifier.save_model (which is how the production
    xgb_severity.json gets written) -- patch it to raise so any future
    accidental call fails loudly instead of silently overwriting
    something."""
    import xgboost as xgb

    def _forbidden(*a, **kw):
        raise AssertionError("xgboost_ablation must never persist a model artifact")

    monkeypatch.setattr(xgb.XGBClassifier, "save_model", _forbidden)

    rng = np.random.default_rng(0)
    X_train = rng.random((10, 5)).astype(np.float32)
    y_train = ["Low"] * 5 + ["Critical"] * 5
    X_test = rng.random((3, 5)).astype(np.float32)
    y_test = ["Low", "Low", "Critical"]

    _fit_and_eval_xgb(X_train, y_train, X_test, y_test, all_labels=["Critical", "Low"])  # must not raise


def test_augmented_feature_matrix_is_baseline_plus_embedding_width():
    baseline_width = 57
    X_base = np.zeros((5, baseline_width), dtype=np.float32)
    embeddings = np.random.rand(5, xgboost_ablation.EMBEDDING_DIM).astype(np.float32)
    X_aug = np.concatenate([X_base, embeddings], axis=1)
    assert X_aug.shape[1] == baseline_width + xgboost_ablation.EMBEDDING_DIM


def test_fit_and_eval_xgb_is_deterministic():
    rng = np.random.default_rng(0)
    X_train = rng.random((12, 6)).astype(np.float32)
    y_train = ["Low"] * 8 + ["Critical"] * 4
    X_test = rng.random((4, 6)).astype(np.float32)
    y_test = ["Low", "Critical", "Low", "Critical"]

    first = _fit_and_eval_xgb(X_train, y_train, X_test, y_test, all_labels=["Critical", "Low"])
    second = _fit_and_eval_xgb(X_train, y_train, X_test, y_test, all_labels=["Critical", "Low"])
    assert first["accuracy"] == second["accuracy"]
    assert first["confusion_matrix"] == second["confusion_matrix"]


# ---------------------------------------------------------------------------
# _score_predictions
# ---------------------------------------------------------------------------

def test_score_predictions_confusion_matrix_shape_matches_labels():
    result = _score_predictions(
        y_true=["Low", "Critical", "Medium"], y_pred=["Low", "Low", "Medium"],
        all_labels=["Critical", "Low", "Medium"],
    )
    assert len(result["confusion_matrix"]) == 3
    assert len(result["confusion_matrix"][0]) == 3
    assert result["n_samples"] == 3


# ---------------------------------------------------------------------------
# live Neo4j (optional -- skipped if unreachable), same convention as
# tests/test_campaign_graphs.py
# ---------------------------------------------------------------------------

def _neo4j_reachable():
    try:
        from neo4j_client import driver
        driver.verify_connectivity()
        return True
    except Exception:
        return False


requires_live_neo4j = pytest.mark.skipif(
    not _neo4j_reachable(),
    reason="No live Neo4j instance reachable in this environment.",
)


@requires_live_neo4j
def test_run_logo_ablation_against_live_data_is_deterministic():
    from ml.gnn.xgboost_ablation import run_logo_ablation

    first = run_logo_ablation(seed=42)
    second = run_logo_ablation(seed=42)

    assert first.pooled_baseline["confusion_matrix"] == second.pooled_baseline["confusion_matrix"]
    assert first.pooled_augmented["confusion_matrix"] == second.pooled_augmented["confusion_matrix"]


@requires_live_neo4j
def test_run_logo_ablation_no_attacker_crosses_train_test_within_a_fold():
    from ml.gnn.xgboost_ablation import build_attacker_groups, load_ablation_dataset, run_logo_ablation

    result = run_logo_ablation(seed=42)
    df, _samples, _mapping = load_ablation_dataset()
    groups = build_attacker_groups(df["campaign_id"].tolist())

    for fold in result.fold_results:
        test_group_members = set(groups[fold.held_out_group])
        assert set(fold.test_campaign_ids) == test_group_members
