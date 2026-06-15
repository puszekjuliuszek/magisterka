"""
Training loop for the Chagas disease ECG classifier.

Features:
  - Class-weighted BCE loss (handles 2.7% Chagas prevalence)
  - AdamW optimiser with cosine annealing
  - Early stopping on validation Challenge Score
  - Checkpoint saving / resuming
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from src.metrics import compute_metrics, EvalResult

logger = logging.getLogger(__name__)

class EarlyStopping:
    """Stop training when the monitored metric stops improving."""

    def __init__(self, patience: int = 10, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score: Optional[float] = None

    def __call__(self, score: float) -> bool:
        if self.best_score is None or score > self.best_score + self.min_delta:
            self.best_score = score
            self.counter = 0
            return False
        self.counter += 1
        return self.counter >= self.patience

class Trainer:
    """Encapsulates the full train / validate / checkpoint loop."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        cfg: dict,
        device: torch.device,
        output_dir: str = "checkpoints",
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        pos_weight_val = cfg.get("pos_weight", 10.0)
        self.criterion = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([pos_weight_val], device=device),
        )

        lr = cfg.get("lr", 1e-3)
        weight_decay = cfg.get("weight_decay", 1e-4)
        self.optimiser = AdamW(
            self.model.parameters(), lr=lr, weight_decay=weight_decay
        )

        self.epochs = cfg.get("epochs", 50)
        self.scheduler = CosineAnnealingLR(self.optimiser, T_max=self.epochs)

        patience = cfg.get("patience", 10)
        self.early_stopping = EarlyStopping(patience=patience)

        self.best_score = -1.0
        self.history: list[dict] = []

    def train(self) -> list[dict]:
        """Run the full training loop. Returns training history."""
        for epoch in range(1, self.epochs + 1):
            t0 = time.time()

            train_loss = self._train_one_epoch()
            val_result, val_loss = self._validate()

            elapsed = time.time() - t0
            lr = self.optimiser.param_groups[0]["lr"]

            record = {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_auroc": val_result.auroc,
                "val_auprc": val_result.auprc,
                "val_challenge_score": val_result.challenge_score,
                "val_f1": val_result.f1,
                "val_sensitivity": val_result.sensitivity,
                "val_specificity": val_result.specificity,
                "lr": lr,
                "time_s": elapsed,
            }
            self.history.append(record)

            logger.info(
                "Epoch %3d/%d | loss=%.4f | val_loss=%.4f | "
                "AUROC=%.4f  AUPRC=%.4f  CS=%.4f | F1=%.4f | "
                "sens=%.4f  spec=%.4f | lr=%.2e | %.1fs",
                epoch,
                self.epochs,
                train_loss,
                val_loss,
                val_result.auroc,
                val_result.auprc,
                val_result.challenge_score,
                val_result.f1,
                val_result.sensitivity,
                val_result.specificity,
                lr,
                elapsed,
            )

            if val_result.challenge_score > self.best_score:
                self.best_score = val_result.challenge_score
                self._save_checkpoint(epoch, val_result)
                logger.info("New best model (CS=%.4f)", self.best_score)

            self.scheduler.step()

            if self.early_stopping(val_result.challenge_score):
                logger.info("Early stopping at epoch %d", epoch)
                break

        return self.history

    def _train_one_epoch(self) -> float:
        self.model.train()
        running_loss = 0.0
        n_batches = 0

        for signals, labels in self.train_loader:
            signals = signals.to(self.device)
            labels = labels.to(self.device)

            self.optimiser.zero_grad()
            logits = self.model(signals)
            loss = self.criterion(logits, labels)
            loss.backward()

            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimiser.step()

            running_loss += loss.item()
            n_batches += 1

        return running_loss / max(n_batches, 1)

    @torch.no_grad()
    def _validate(self) -> tuple[EvalResult, float]:
        self.model.eval()
        all_labels: list[int] = []
        all_probs: list[float] = []
        running_loss = 0.0
        n_batches = 0

        for signals, labels in self.val_loader:
            signals = signals.to(self.device)
            labels = labels.to(self.device)

            logits = self.model(signals)
            loss = self.criterion(logits, labels)
            running_loss += loss.item()
            n_batches += 1

            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.extend(probs.tolist())
            all_labels.extend(labels.cpu().numpy().astype(int).tolist())

        val_loss = running_loss / max(n_batches, 1)
        result = compute_metrics(np.array(all_labels), np.array(all_probs))
        return result, val_loss

    def _save_checkpoint(self, epoch: int, result: EvalResult):
        path = self.output_dir / "best_model.pt"
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimiser_state_dict": self.optimiser.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "challenge_score": result.challenge_score,
                "auroc": result.auroc,
                "auprc": result.auprc,
            },
            path,
        )

    def load_best_model(self):
        path = self.output_dir / "best_model.pt"
        if path.exists():
            ckpt = torch.load(path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(ckpt["model_state_dict"])
            logger.info(
                "Loaded best model from epoch %d (CS=%.4f)",
                ckpt["epoch"],
                ckpt["challenge_score"],
            )
        else:
            logger.warning("No checkpoint found at %s", path)
