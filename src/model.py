"""
1-D ResNet for 12-lead ECG classification (Chagas disease detection).

Architecture adapted from Ribeiro et al. (2020),
"Automatic diagnosis of the 12-lead ECG using a deep neural network",
for binary classification.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock1D(nn.Module):
    """A single residual block with two convolution layers.

    Structure: BN, ReLU, Conv, BN, ReLU, Dropout, Conv, plus a skip
    connection (with optional 1x1 conv for dimension matching).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 7,
        stride: int = 1,
        dropout: float = 0.2,
    ):
        super().__init__()
        padding = kernel_size // 2

        self.bn1 = nn.BatchNorm1d(in_channels)
        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, bias=False,
        )
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size,
            stride=1, padding=padding, bias=False,
        )

        self.skip = nn.Identity()
        if in_channels != out_channels or stride != 1:
            self.skip = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)

        out = self.bn1(x)
        out = F.relu(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = F.relu(out)
        out = self.dropout(out)
        out = self.conv2(out)

        return out + residual

class ECGResNet(nn.Module):
    """1-D ResNet for 12-lead ECG binary classification.

    Parameters
    ----------
    in_channels : int
        Number of ECG leads (default 12).
    base_filters : int
        Number of filters in the first conv layer; doubles every block group.
    num_blocks_per_group : int
        Residual blocks in each of the 4 groups.
    kernel_size : int
        Convolution kernel width in residual blocks.
    dropout : float
        Dropout probability inside residual blocks.
    """

    def __init__(
        self,
        in_channels: int = 12,
        base_filters: int = 64,
        num_blocks_per_group: int = 2,
        kernel_size: int = 7,
        dropout: float = 0.3,
    ):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, base_filters, kernel_size=15, padding=7, bias=False),
            nn.BatchNorm1d(base_filters),
            nn.ReLU(inplace=True),
        )

        channels = [base_filters, base_filters * 2, base_filters * 4, base_filters * 8]
        self.groups = nn.ModuleList()
        in_ch = base_filters
        for i, out_ch in enumerate(channels):
            blocks = []
            for j in range(num_blocks_per_group):
                stride = 2 if (j == 0 and i > 0) else 1
                blocks.append(
                    ResidualBlock1D(
                        in_ch if j == 0 else out_ch,
                        out_ch,
                        kernel_size=kernel_size,
                        stride=stride,
                        dropout=dropout,
                    )
                )
            self.groups.append(nn.Sequential(*blocks))
            in_ch = out_ch

        self.final_bn = nn.BatchNorm1d(channels[-1])
        self.classifier = nn.Sequential(
            nn.Linear(channels[-1], 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor, shape (batch, 12, seq_len)

        Returns
        -------
        logits : Tensor, shape (batch,)
        """
        out = self.stem(x)
        for group in self.groups:
            out = group(out)

        out = self.final_bn(out)
        out = F.relu(out)
        out = out.mean(dim=-1)

        logits = self.classifier(out).squeeze(-1)
        return logits

def build_model(cfg: dict) -> ECGResNet:
    """Construct an ECGResNet from a config dictionary."""
    return ECGResNet(
        in_channels=cfg.get("in_channels", 12),
        base_filters=cfg.get("base_filters", 64),
        num_blocks_per_group=cfg.get("num_blocks_per_group", 2),
        kernel_size=cfg.get("kernel_size", 7),
        dropout=cfg.get("dropout", 0.3),
    )
