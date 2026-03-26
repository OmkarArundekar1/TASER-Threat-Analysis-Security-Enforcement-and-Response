"""soc_engine/__init__.py — SOC Anomaly Detection Engine package."""
from .model import Autoencoder, load_autoencoder
from .scorer import SeverityScorer, compute_scores
from .temporal_engine import TemporalSmoother
from .explainer import FeatureExplainer
from .threshold import AdaptiveThreshold
from .trainer import AutoencoderTrainer

__all__ = [
    "Autoencoder", "load_autoencoder",
    "SeverityScorer", "compute_scores",
    "TemporalSmoother",
    "FeatureExplainer",
    "AdaptiveThreshold",
    "AutoencoderTrainer",
]
