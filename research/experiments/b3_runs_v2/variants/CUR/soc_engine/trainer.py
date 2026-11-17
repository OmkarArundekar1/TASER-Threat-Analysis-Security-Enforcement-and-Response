"""
soc_engine/trainer.py
======================
Production training loop for the SOC Autoencoder.

Features:
    - GPU-accelerated (auto-detects CUDA)
    - Early stopping (patience-based, on validation loss)
    - ReduceLROnPlateau scheduler
    - Saves best model checkpoint automatically
    - Structured training log
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from .model import Autoencoder, get_device

logger = logging.getLogger(__name__)


class AutoencoderTrainer:
    """
    Train the Autoencoder with early stopping and LR scheduling.

    Args:
        model:       Autoencoder instance to train.
        device:      'auto', 'cuda', or 'cpu'.
        lr:          Learning rate (default 5e-4).
        weight_decay:L2 regularisation (default 1e-5).
        batch_size:  Mini-batch size (default 256).
        epochs:      Maximum training epochs (default 30).
        patience:    Early stopping patience in epochs (default 5).
        save_path:   Where to save best model weights (.pth).
    """

    def __init__(
        self,
        model: Autoencoder,
        device: str = "auto",
        lr: float = 5e-4,
        weight_decay: float = 1e-5,
        batch_size: int = 256,
        epochs: int = 30,
        patience: int = 5,
        save_path: str | Path = "models/autoencoder_best.pth",
    ) -> None:
        self.device = get_device(device)
        self.model = model.to(self.device)
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.save_path = Path(save_path)
        self.save_path.parent.mkdir(parents=True, exist_ok=True)

        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=3, min_lr=1e-6
        )
        self.history: list[dict[str, float]] = []

    def fit(self, data_scaled: np.ndarray, val_fraction: float = 0.1) -> dict[str, Any]:
        """
        Train on scaled numpy array. Splits off val_fraction for validation.

        Args:
            data_scaled:  Scaled feature matrix of shape (N, D).
            val_fraction: Fraction of data to use as validation set.

        Returns:
            Training summary dict.
        """
        n = len(data_scaled)
        val_n = max(1, int(n * val_fraction))
        idx = np.random.permutation(n)
        val_idx, train_idx = idx[:val_n], idx[val_n:]

        train_tensor = torch.tensor(data_scaled[train_idx], dtype=torch.float32)
        val_tensor = torch.tensor(data_scaled[val_idx], dtype=torch.float32)

        train_loader = DataLoader(
            TensorDataset(train_tensor), batch_size=self.batch_size, shuffle=True
        )
        val_loader = DataLoader(
            TensorDataset(val_tensor), batch_size=self.batch_size
        )

        best_val_loss = float("inf")
        patience_counter = 0
        start_time = time.time()

        logger.info("Training on %d samples, validating on %d | device=%s",
                    len(train_idx), len(val_idx), self.device)

        for epoch in range(1, self.epochs + 1):
            # ── Train ────────────────────────────────────────────────────
            self.model.train()
            train_loss = 0.0
            for (batch,) in train_loader:
                batch = batch.to(self.device)
                out = self.model(batch)
                loss = self.criterion(out, batch)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()
            train_loss /= len(train_loader)

            # ── Validate ─────────────────────────────────────────────────
            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for (batch,) in val_loader:
                    batch = batch.to(self.device)
                    out = self.model(batch)
                    val_loss += self.criterion(out, batch).item()
            val_loss /= len(val_loader)

            self.scheduler.step(val_loss)
            self.history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})

            logger.info("Epoch [%d/%d]  train=%.6f  val=%.6f  lr=%.2e",
                        epoch, self.epochs, train_loss, val_loss,
                        self.optimizer.param_groups[0]["lr"])

            # ── Early stopping ───────────────────────────────────────────
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(self.model.state_dict(), self.save_path)
                logger.info("  ✓ Best model saved (val_loss=%.6f)", best_val_loss)
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    logger.info("Early stopping at epoch %d", epoch)
                    break

        elapsed = time.time() - start_time
        logger.info("Training complete in %.1fs | best_val_loss=%.6f", elapsed, best_val_loss)

        return {
            "best_val_loss": best_val_loss,
            "epochs_trained": len(self.history),
            "elapsed_seconds": elapsed,
            "model_path": str(self.save_path),
            "history": self.history,
        }
