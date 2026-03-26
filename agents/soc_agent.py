"""
agents/soc_agent.py
====================
SOCAgent — the main pipeline orchestrator.

This is the central "Detector Agent" that integrates all pipeline modules:
    1. Anomaly detection (Autoencoder + MAD scoring)
    2. Temporal smoothing
    3. Feature attribution
    4. Attack classification (LightGBM)
    5. MITRE ATT&CK / Kill-Chain mapping
    6. Playbook generation

Designed for:
    - LangChain tool wrapping (each stage can be a LangChain Tool)
    - Direct Python API: agent.detect(window_or_batch)
    - Structured JSON output (ThreatIntelReport)
    - SSFT pseudo-label export

No torch tensors or numpy objects in output — all JSON-serialisable.
"""

from __future__ import annotations

import datetime
import logging
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from soc_engine.model import Autoencoder, load_autoencoder, get_device
from soc_engine.scorer import SeverityScorer
from soc_engine.temporal_engine import TemporalSmoother
from soc_engine.explainer import FeatureExplainer
from classifiers.attack_stage_mapper import AttackStageMapper
from response.playbook_generator import PlaybookGenerator
from agents.output_schema import ThreatIntelReport

logger = logging.getLogger(__name__)

_CONFIG_PATH = "config.yaml"


def _load_config(path: str = _CONFIG_PATH) -> dict[str, Any]:
    try:
        with open(path) as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        return {}


class SOCAgent:
    """
    Single-entry point for the full SOC detection + response pipeline.

    Usage (minimal — uses pre-built window_features.npy)::

        agent = SOCAgent.from_config()
        reports = agent.detect_batch(windows)   # np.ndarray (N, 312)

    Usage (with classifier)::

        agent = SOCAgent.from_config(load_classifier=True)
        report = agent.detect_single(window)    # np.ndarray (312,)

    Output::

        ThreatIntelReport  (Pydantic model, .to_dict() for JSON)
    """

    def __init__(
        self,
        model: Autoencoder,
        scorer: SeverityScorer,
        smoother: TemporalSmoother,
        explainer: FeatureExplainer,
        mapper: AttackStageMapper,
        playbook_gen: PlaybookGenerator,
        classifier: Any = None,   # optional AttackClassifier
        device: torch.device | None = None,
        feature_names: list[str] | None = None,
    ) -> None:
        self.model = model
        self.scorer = scorer
        self.smoother = smoother
        self.explainer = explainer
        self.mapper = mapper
        self.playbook_gen = playbook_gen
        self.classifier = classifier
        self.device = device or get_device("auto")
        self.feature_names = feature_names
        self.model.to(self.device).eval()

    @classmethod
    def from_config(
        cls,
        config_path: str = _CONFIG_PATH,
        load_classifier: bool = False,
        feature_names: list[str] | None = None,
    ) -> "SOCAgent":
        """
        Build SOCAgent from config.yaml.
        Loads existing model weights if available; builds fresh model otherwise.
        """
        cfg = _load_config(config_path)
        se_cfg = cfg.get("soc_engine", {})
        sv_cfg = cfg.get("severity_engine", {})
        te_cfg = cfg.get("temporal_engine", {})
        ex_cfg = cfg.get("explainer", {})

        device = get_device(se_cfg.get("device", "auto"))

        # Model
        weights = Path(cfg.get("paths", {}).get("autoencoder_weights", "models/autoencoder_best.pth"))
        model = Autoencoder(
            input_dim=se_cfg.get("input_dim", 312),
            hidden_dims=se_cfg.get("hidden_dims", [128, 64]),
            latent_dim=se_cfg.get("latent_dim", 32),
            dropout=se_cfg.get("dropout", 0.1),
        )
        if weights.exists():
            state = torch.load(weights, map_location=device, weights_only=True)
            model.load_state_dict(state)
            logger.info("Loaded weights from %s", weights)
        else:
            logger.warning("No weights found at %s — using untrained model", weights)
        model.to(device).eval()

        # Scorer — fit on existing window_features if available
        scorer = SeverityScorer(
            thresholds={
                "HIGH": sv_cfg.get("severity_thresholds", {}).get("HIGH", 8.0),
                "MEDIUM": sv_cfg.get("severity_thresholds", {}).get("MEDIUM", 4.0),
                "LOW": sv_cfg.get("severity_thresholds", {}).get("LOW", 2.0),
            },
            clip_percentile=sv_cfg.get("clip_percentile", 99.9),
        )

        wf_path = Path(cfg.get("paths", {}).get("window_features", "ml_pipeline/window_features.npy"))
        if wf_path.exists():
            data = np.load(wf_path)
            # Compute calibration errors from existing data
            t = torch.tensor(data, dtype=torch.float32).to(device)
            errors = model.reconstruction_errors(t).cpu().numpy()
            scorer.fit(errors)
            logger.info("Scorer fitted on %d windows from %s", len(errors), wf_path)
        else:
            logger.warning("window_features.npy not found — scorer not calibrated")

        # Temporal smoother
        rules = te_cfg.get("rules", {})
        smoother = TemporalSmoother(
            smooth_window=te_cfg.get("smooth_window", 5),
            high_min_count=rules.get("HIGH_min_count", 1),
            medium_min_count=rules.get("MEDIUM_min_count", 2),
            low_min_count=rules.get("LOW_min_count", 3),
        )

        explainer = FeatureExplainer(
            feature_names=feature_names,
            top_k=ex_cfg.get("top_k_features", 5),
        )

        classifier = None
        if load_classifier:
            try:
                from classifiers.lightgbm_classifier import AttackClassifier
                clf_path = cfg.get("paths", {}).get("lgbm_model", "models/lgbm_classifier.pkl")
                if Path(clf_path).exists():
                    classifier = AttackClassifier.load(clf_path)
                    logger.info("AttackClassifier loaded from %s", clf_path)
                else:
                    logger.warning("LightGBM model not found at %s", clf_path)
            except ImportError:
                logger.warning("lightgbm not installed — classifier disabled")

        return cls(
            model=model,
            scorer=scorer,
            smoother=smoother,
            explainer=explainer,
            mapper=AttackStageMapper(),
            playbook_gen=PlaybookGenerator(),
            classifier=classifier,
            device=device,
            feature_names=feature_names,
        )

    # ── Main API ──────────────────────────────────────────────────────────────

    def detect_single(
        self,
        window: np.ndarray,
        window_index: int = 0,
        sensor: str = "cicids",
        attacker_ip: str = "unknown",  # audit only
    ) -> ThreatIntelReport:
        """
        Analyse a single window feature vector.

        Args:
            window:       1D array of shape (D,) — NO IP features.
            window_index: Index for temporal linkage.
            sensor:       Data source label.
            attacker_ip:  Audit-only, NOT used in features.

        Returns:
            ThreatIntelReport (call .to_dict() for JSON).
        """
        window = np.asarray(window, dtype=np.float32).ravel()[np.newaxis, :]  # (1, D)
        reports = self.detect_batch(window, start_index=window_index, sensor=sensor, attacker_ip=attacker_ip)
        return reports[0]

    def detect_batch(
        self,
        windows: np.ndarray,
        start_index: int = 0,
        sensor: str = "cicids",
        attacker_ip: str = "unknown",  # audit only
    ) -> list[ThreatIntelReport]:
        """
        Analyse a batch of window feature vectors.

        Args:
            windows:      2D array (N, D) — NO IP features.
            start_index:  Window index offset.
            sensor:       Data source label.
            attacker_ip:  Audit-only, NOT used in features.

        Returns:
            List of ThreatIntelReport dicts.
        """
        windows = np.nan_to_num(np.asarray(windows, dtype=np.float32), nan=0.0, posinf=1e6)
        t = torch.tensor(windows, dtype=torch.float32).to(self.device)

        # ── 1. Reconstruction errors ──────────────────────────────────────
        errors = self.model.reconstruction_errors(t).cpu().numpy()  # (N,)

        # ── 2. Scorer — severity labels ──────────────────────────────────
        if not self.scorer._fitted:
            self.scorer.fit(errors)
        score_results = self.scorer.score(errors)

        # ── 3. Explainability ─────────────────────────────────────────────
        explain_results = self.explainer.explain(self.model, t)

        # ── 4. Classifier (optional) ─────────────────────────────────────
        if self.classifier is not None:
            clf_results = self.classifier.classify_batch(windows)
        else:
            clf_results = [{"label": "BENIGN", "confidence": 0.0, "probabilities": {}}] * len(windows)

        # ── 5. Assemble reports ───────────────────────────────────────────
        reports = []
        for i in range(len(windows)):
            w_idx = start_index + i
            sc = score_results[i]
            ex = explain_results[i]
            clf = clf_results[i]

            # ── Temporal smoothing ────────────────────────────────────────
            temporal = self.smoother.update(sc["severity_score"], sc["severity_label"])

            # ── MITRE / Kill-Chain mapping ────────────────────────────────
            stage_info = self.mapper.map(clf["label"])

            # ── Playbook ─────────────────────────────────────────────────
            pb = self.playbook_gen.generate({
                "severity_label": sc["severity_label"],
                "attack_label": clf["label"],
                "kill_chain_phase": stage_info["kill_chain_phase"],
                "recommended_priority": self._priority(sc["severity_score"]),
                "attacker_ip": attacker_ip,
            })

            # ── SSFT pseudo-label ─────────────────────────────────────────
            ssft_label = 1 if sc["severity_label"] == "HIGH" else (0 if sc["severity_label"] == "NORMAL" else -1)

            report = ThreatIntelReport(
                report_id=str(uuid.uuid4()),
                timestamp=datetime.datetime.utcnow().isoformat() + "Z",
                sensor=sensor,
                window_index=w_idx,
                severity_score=sc["severity_score"],
                severity_label=sc["severity_label"],
                temporal_label=temporal["temporal_label"],
                smoothed_score=temporal["smoothed_score"],
                is_anomaly=sc["is_anomaly"],
                confidence=sc["confidence"],
                top_features=ex["top_features"],
                attack_label=clf["label"],
                attack_confidence=clf["confidence"],
                mitre_tactic=stage_info["mitre_tactic"],
                kill_chain_phase=stage_info["kill_chain_phase"],
                risk_level=stage_info["risk_level"],
                is_active_campaign=False,
                progression_score=0.0,
                recommended_priority=self._priority(sc["severity_score"]),
                playbook_id=pb["playbook_id"],
                recommended_actions=pb["actions"][:3],  # top 3 actions in report
                ssft_label=ssft_label,
            )
            reports.append(report)

        return reports

    # ── SSFT Export ───────────────────────────────────────────────────────────

    def export_ssft_dataset(
        self,
        windows: np.ndarray,
        reports: list[ThreatIntelReport],
        save_path: str = "models/ssft_dataset.npz",
    ) -> dict[str, Any]:
        """
        Export pseudo-labelled dataset for semi-supervised fine-tuning.

        Labels:
            HIGH → 1 (attack)
            NORMAL → 0 (benign)
            LOW/MEDIUM → excluded (ambiguous)
        """
        include_mask = np.array([r.ssft_label in (0, 1) for r in reports])
        X_ssft = windows[include_mask]
        y_ssft = np.array([r.ssft_label for r in reports])[include_mask]
        scores = np.array([r.severity_score for r in reports])[include_mask]

        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        np.savez(save_path, X=X_ssft, y=y_ssft, severity_scores=scores)

        logger.info("SSFT dataset saved: %d samples (%d attack, %d benign)",
                    len(y_ssft), int(y_ssft.sum()), int((y_ssft == 0).sum()))
        return {
            "path": save_path,
            "total": int(len(y_ssft)),
            "attack": int(y_ssft.sum()),
            "benign": int((y_ssft == 0).sum()),
            "excluded_ambiguous": int(include_mask.size - include_mask.sum()),
        }

    @staticmethod
    def _priority(severity_score: float) -> str:
        if severity_score > 8:
            return "P1_CRITICAL"
        if severity_score > 4:
            return "P2_HIGH"
        if severity_score > 2:
            return "P3_MEDIUM"
        return "P4_LOW"
