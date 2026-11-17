"""
ml/gnn/inference.py
=======================
Production GNN inference path. See ../../GNN_PRODUCTION_INTEGRATION.md
for the full integration writeup.

Hard guarantees, by construction, not just discipline:

  - Never trains. Only ever loads a frozen artifact
    (train_autoencoder.py's `load_autoencoder`, unmodified) and calls
    the already-existing `GraphAutoencoder.embed_graph` (which already
    wraps in `torch.no_grad()`); this module additionally forces
    `model.eval()` on load for defense in depth.
  - Deterministic: the same campaign_id, on the same artifact, always
    produces the same embedding (verified by test and by this phase's
    live re-run).
  - Fails safe, everywhere: every public method catches its own
    failures (missing artifact, corrupt artifact, malformed graph,
    empty graph, non-finite output, dimension mismatch) and returns
    `None`, logging the reason -- it never raises into a caller.
    `GNN_ENABLED=false` (config.py) short-circuits to `None` before
    touching the filesystem or importing torch at all, so the rest of
    CYUKTI's pipeline is byte-for-byte unaffected when GNN is off.
  - Never a single point of failure: nothing in this module can crash
    Wazuh alert processing, campaign management, correlation,
    attribution, investigation, or the dashboard API -- every call site
    that uses this service treats `None` as "no topology evidence this
    time", the same way `default_model_predictor` already treats a
    missing XGBoost artifact (investigation/loop.py, the pattern this
    module's fail-safe design mirrors).
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GNNModelMetadata:
    model_type: str
    architecture: str
    hidden_dim: int
    embedding_dim: int
    num_layers: int
    node_feature_schema: list[str] = field(default_factory=list)
    edge_feature_schema: list[str] = field(default_factory=list)
    normalization: str = ""
    training_seed: int = -1
    model_version: str = ""
    training_dataset_description: str = ""
    artifact_scope: str = ""

    def to_dict(self) -> dict:
        return {
            "model_type": self.model_type,
            "architecture": self.architecture,
            "hidden_dim": self.hidden_dim,
            "embedding_dim": self.embedding_dim,
            "num_layers": self.num_layers,
            "node_feature_schema": list(self.node_feature_schema),
            "edge_feature_schema": list(self.edge_feature_schema),
            "normalization": self.normalization,
            "training_seed": self.training_seed,
            "model_version": self.model_version,
            "training_dataset_description": self.training_dataset_description,
            "artifact_scope": self.artifact_scope,
        }


def _model_version_from_path(model_path: str) -> str:
    """A stable, human-inspectable version string derived from the
    artifact file itself (mtime + size) -- not a fabricated semantic
    version, since this research-stage artifact has no release process.
    Changes automatically if the artifact file is ever replaced."""
    try:
        stat = os.stat(model_path)
        return f"{os.path.basename(model_path)}@{int(stat.st_mtime)}-{stat.st_size}b"
    except OSError:
        return "unknown"


class GNNInferenceService:
    """One instance per process (see module-level `gnn_inference_service`
    singleton below). Lazily loads the artifact on first use, not at
    import time, so importing this module has zero cost when
    GNN_ENABLED=false."""

    def __init__(self, model_path: str | None = None) -> None:
        self._model_path_override = model_path
        self._loaded = False
        self._model = None
        self._feature_mean = None
        self._feature_std = None
        self._metadata: GNNModelMetadata | None = None
        self._embedding_cache: dict[str, "object"] = {}

    def _resolve_model_path(self) -> str:
        if self._model_path_override is not None:
            return self._model_path_override
        import config
        return config.GNN_MODEL_PATH

    def _ensure_loaded(self) -> bool:
        if self._loaded:
            return self._model is not None
        self._loaded = True

        import config
        if not config.GNN_ENABLED:
            logger.info("GNN inference disabled (GNN_ENABLED=false) -- skipping model load.")
            return False

        model_path = self._resolve_model_path()
        if not os.path.exists(model_path):
            logger.warning(
                "GNN model artifact not found at %s -- GNN inference unavailable for this "
                "process; the rest of CYUKTI's pipeline is unaffected.", model_path,
            )
            return False

        try:
            from graph_encoder import EDGE_TYPES, NODE_TYPES, NUMERIC_PROPS
            from train_autoencoder import load_autoencoder

            model, checkpoint = load_autoencoder(model_path)

            required_keys = {"hidden_dim", "embedding_dim", "num_layers", "feature_mean", "feature_std"}
            missing = required_keys - set(checkpoint.keys())
            if missing:
                logger.error(
                    "GNN model artifact at %s is missing required metadata keys %s -- "
                    "treating as corrupt, GNN inference unavailable.", model_path, sorted(missing),
                )
                return False

            model.eval()  # defense in depth -- load_autoencoder already does this
            self._model = model
            self._feature_mean = checkpoint["feature_mean"]
            self._feature_std = checkpoint["feature_std"]
            self._metadata = GNNModelMetadata(
                model_type="graph_autoencoder",
                architecture="2-layer SAGEConv encoder, mean-pool + Linear+tanh bottleneck",
                hidden_dim=checkpoint["hidden_dim"],
                embedding_dim=checkpoint["embedding_dim"],
                num_layers=checkpoint["num_layers"],
                node_feature_schema=list(NODE_TYPES) + list(NUMERIC_PROPS),
                edge_feature_schema=list(EDGE_TYPES),
                normalization="z-score on the 10 numeric node properties, fit on the training split only",
                training_seed=checkpoint.get("seed", -1),
                model_version=_model_version_from_path(model_path),
                training_dataset_description=(
                    f"{len(checkpoint.get('train_campaign_ids', []))} training + "
                    f"{len(checkpoint.get('val_campaign_ids', []))} validation real campaigns "
                    "from a SINGLE all-population split -- this is NOT the same as the "
                    "fold-safe (Leave-One-Attacker-Group-Out) cross-validation embeddings "
                    "reported in GNN_XGBOOST_ABLATION.md / GNN_RETRIEVAL_EVALUATION.md; those "
                    "were computed by separately-trained fold-specific models never exposed "
                    "here. See GNN_PRODUCTION_INTEGRATION.md Section 3."
                ),
                artifact_scope="single all-population split (not cross-validated)",
            )
            logger.info(
                "GNN model artifact loaded from %s (version=%s, embedding_dim=%d).",
                model_path, self._metadata.model_version, self._metadata.embedding_dim,
            )
            return True
        except Exception:
            logger.exception(
                "Failed to load GNN model artifact at %s -- GNN inference unavailable for "
                "this process; the rest of CYUKTI's pipeline is unaffected.", model_path,
            )
            self._model = None
            self._metadata = None
            return False

    @property
    def available(self) -> bool:
        return self._ensure_loaded()

    @property
    def metadata(self) -> GNNModelMetadata | None:
        self._ensure_loaded()
        return self._metadata

    def embed_campaign(self, campaign_id: str, use_cache: bool = True):
        """Returns a torch.Tensor embedding, or None on any failure
        (GNN disabled, no artifact, campaign not found, empty/malformed
        graph, non-finite output). Never raises."""
        if not self._ensure_loaded():
            return None

        if use_cache and campaign_id in self._embedding_cache:
            return self._embedding_cache[campaign_id]

        try:
            from campaign_graphs import build_campaign_graph_sample

            sample = build_campaign_graph_sample(campaign_id)
        except Exception:
            logger.exception("GNN: failed to build graph sample for campaign %s.", campaign_id)
            return None

        if sample is None:
            logger.info("GNN: campaign %s not found or has no risk_score on record.", campaign_id)
            return None
        if sample.graph.num_nodes == 0:
            logger.info("GNN: campaign %s produced an empty graph (0 nodes) -- no embedding to compute.", campaign_id)
            return None

        embedding = self._embed_encoded_graph(sample.graph)
        if embedding is not None and use_cache:
            self._embedding_cache[campaign_id] = embedding
        return embedding

    def _embed_encoded_graph(self, encoded_graph):
        try:
            import torch

            from autoencoder_model import NUMERIC_FEATURE_OFFSET

            self._model.eval()
            x = encoded_graph.x.clone()
            x[:, NUMERIC_FEATURE_OFFSET:] = (x[:, NUMERIC_FEATURE_OFFSET:] - self._feature_mean) / self._feature_std

            embedding = self._model.embed_graph(x, encoded_graph.edge_index)

            if not torch.isfinite(embedding).all():
                logger.error("GNN produced a non-finite embedding -- discarding, treating as unavailable.")
                return None
            if embedding.shape[0] != self._metadata.embedding_dim:
                logger.error(
                    "GNN embedding dimension mismatch: model reports embedding_dim=%d, "
                    "produced shape=%s -- discarding.", self._metadata.embedding_dim, tuple(embedding.shape),
                )
                return None
            return embedding
        except Exception:
            logger.exception("GNN embedding computation failed -- treating as unavailable.")
            return None

    def invalidate_cache(self, campaign_id: str | None = None) -> None:
        if campaign_id is None:
            self._embedding_cache.clear()
        else:
            self._embedding_cache.pop(campaign_id, None)


gnn_inference_service = GNNInferenceService()
