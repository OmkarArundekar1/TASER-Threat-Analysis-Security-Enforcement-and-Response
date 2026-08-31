"""
soc_engine/model.py
====================
Improved Autoencoder for network anomaly detection.

Design decisions:
- Batch Normalization after each encoder layer → stable training on network flow data
- Dropout(0.1) in encoder → regularization, reduces overfitting to benign patterns
- Residual projection in bottleneck → preserves global structure, improves reconstruction
- Decoder mirrors encoder without Dropout → full reconstruction capacity

Input: 312-dim window feature vector (mean/std/min/max × 78 CICIDS features, NO IP columns)
Bottleneck: 32-dim latent space
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class Autoencoder(nn.Module):
    """
    Production-grade Autoencoder for SOC anomaly detection.

    Args:
        input_dim:   Dimension of the input feature vector (default 312).
        hidden_dims: List of hidden layer sizes encoder-side (default [128, 64]).
        latent_dim:  Bottleneck dimension (default 32).
        dropout:     Dropout probability in encoder layers (default 0.1).
    """

    def __init__(
        self,
        input_dim: int = 312,
        hidden_dims: list[int] | None = None,
        latent_dim: int = 32,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [128, 64]

        self.input_dim = input_dim
        self.latent_dim = latent_dim

        # ── Encoder ───────────────────────────────────────────────────────
        enc_layers: list[nn.Module] = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            enc_layers += [
                nn.Linear(prev_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout),
            ]
            prev_dim = h_dim

        enc_layers.append(nn.Linear(prev_dim, latent_dim))
        self.encoder = nn.Sequential(*enc_layers)

        # ── Residual projection: input → latent (bottleneck skip) ─────────
        self.residual_proj = nn.Linear(input_dim, latent_dim, bias=False)

        # ── Decoder ───────────────────────────────────────────────────────
        dec_layers: list[nn.Module] = []
        rev_dims = list(reversed(hidden_dims))
        prev_dim = latent_dim
        for h_dim in rev_dims:
            dec_layers += [
                nn.Linear(prev_dim, h_dim),
                nn.ReLU(inplace=True),
            ]
            prev_dim = h_dim

        dec_layers.append(nn.Linear(prev_dim, input_dim))
        self.decoder = nn.Sequential(*dec_layers)

    # ── Forward ───────────────────────────────────────────────────────────
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        latent = self.encoder(x) + self.residual_proj(x)
        return self.decoder(latent)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Return latent representation (for downstream tasks / GNN input)."""
        return self.encoder(x) + self.residual_proj(x)

    def reconstruction_errors(self, x: torch.Tensor) -> torch.Tensor:
        """Per-sample mean squared reconstruction error (no gradient)."""
        self.eval()
        with torch.no_grad():
            recon = self.forward(x)
            return torch.mean((x - recon) ** 2, dim=1)

    def per_feature_errors(self, x: torch.Tensor) -> torch.Tensor:
        """Per-feature absolute reconstruction error — used for explainability."""
        self.eval()
        with torch.no_grad():
            recon = self.forward(x)
            return torch.abs(x - recon)


# ── Utility helpers ────────────────────────────────────────────────────────────

def load_autoencoder(
    weights_path: str | Path,
    input_dim: int = 312,
    hidden_dims: list[int] | None = None,
    latent_dim: int = 32,
    dropout: float = 0.1,
    device: str | torch.device = "cpu",
) -> Autoencoder:
    """
    Load a saved Autoencoder from disk.

    Args:
        weights_path: Path to `.pth` checkpoint (state_dict).
        device:       Target device.

    Returns:
        Autoencoder in eval mode on *device*.
    """
    model = Autoencoder(
        input_dim=input_dim,
        hidden_dims=hidden_dims or [128, 64],
        latent_dim=latent_dim,
        dropout=dropout,
    )
    state = torch.load(weights_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    logger.info("Autoencoder loaded from %s on %s", weights_path, device)
    return model


def get_device(preference: str = "auto") -> torch.device:
    """Resolve device string: 'auto' → cuda if available, else cpu."""
    if preference == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(preference)
