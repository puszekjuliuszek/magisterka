"""
PyTorch Dataset for the PhysioNet/CinC 2025 Chagas disease ECG data.

Each item yields a (signal, label) pair where *signal* has shape
(12, 5000)  (leads-first, 10 s @ 500 Hz) and *label* is 0 or 1.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from src.preprocessing import find_records, load_record, preprocess_ecg

logger = logging.getLogger(__name__)

class ChagasECGDataset(Dataset):
    """Lazy-loading ECG dataset backed by WFDB files on disk."""

    def __init__(
        self,
        data_dir: str,
        target_fs: float = 500.0,
        target_duration_s: float = 10.0,
        filter_low: float = 0.5,
        filter_high: float = 47.0,
        records: Optional[list[str]] = None,
        transform=None,
    ):
        self.data_dir = str(data_dir)
        self.target_fs = target_fs
        self.target_duration_s = target_duration_s
        self.filter_low = filter_low
        self.filter_high = filter_high
        self.transform = transform

        if records is not None:
            self.records = records
        else:
            self.records = find_records(self.data_dir)

        self.labels: list[int] = []
        self.ages: list[float] = []
        self.sexes: list[int] = []
        self._valid_indices: list[int] = []

        self._scan_metadata()

    def _scan_metadata(self):
        """Pre-scan headers for labels so we can build a sampler quickly."""
        valid = []
        for idx, rec in enumerate(self.records):
            try:
                _, meta = load_record(self.data_dir, rec)
                label = meta["chagas"]
                if label not in (0, 1):
                    continue
                self.labels.append(label)
                self.ages.append(meta["age"])
                self.sexes.append(meta["sex"])
                valid.append(idx)
            except Exception as e:
                logger.warning("Skipping record %s: %s", rec, e)
        self._valid_indices = valid
        logger.info(
            "Dataset %s: %d valid records (pos=%d, neg=%d)",
            self.data_dir,
            len(valid),
            sum(self.labels),
            len(valid) - sum(self.labels),
        )

    def __len__(self) -> int:
        return len(self._valid_indices)

    def __getitem__(self, idx: int):
        rec_idx = self._valid_indices[idx]
        rec_name = self.records[rec_idx]

        signal, meta = load_record(self.data_dir, rec_name)
        signal = preprocess_ecg(
            signal,
            fs=meta["fs"],
            target_fs=self.target_fs,
            target_duration_s=self.target_duration_s,
            filter_low=self.filter_low,
            filter_high=self.filter_high,
        )
        signal = signal.T
        signal = torch.from_numpy(signal).float()

        label = torch.tensor(self.labels[idx], dtype=torch.float32)

        if self.transform is not None:
            signal = self.transform(signal)

        return signal, label

def build_weighted_sampler(dataset: ChagasECGDataset) -> WeightedRandomSampler:
    """Build a WeightedRandomSampler that oversamples the minority class."""
    labels = np.array(dataset.labels)
    class_counts = np.bincount(labels.astype(int), minlength=2)
    weights_per_class = 1.0 / class_counts
    sample_weights = weights_per_class[labels.astype(int)]
    return WeightedRandomSampler(
        weights=torch.from_numpy(sample_weights).double(),
        num_samples=len(dataset),
        replacement=True,
    )

def create_dataloaders(
    train_dir: str,
    val_dir: str,
    batch_size: int = 32,
    num_workers: int = 4,
    target_fs: float = 500.0,
    target_duration_s: float = 10.0,
) -> tuple[DataLoader, DataLoader]:
    """Convenience factory returning (train_loader, val_loader)."""
    train_ds = ChagasECGDataset(
        train_dir,
        target_fs=target_fs,
        target_duration_s=target_duration_s,
    )
    val_ds = ChagasECGDataset(
        val_dir,
        target_fs=target_fs,
        target_duration_s=target_duration_s,
    )

    sampler = build_weighted_sampler(train_ds)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader
