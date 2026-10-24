"""
losses.py
==========
Loss functions for tonal segmentation. Tonal pixels are a tiny fraction of
the spectrogram (typically < 2%), so plain BCE alone gets dominated by the
background class. We combine Dice (region overlap, imbalance-robust) with
BCE (stable pixel-wise gradients) as specified in the proposal's training
pipeline: "Dice Loss + Binary Cross Entropy".
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


def dice_loss(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    probs = probs.flatten(1)
    target = target.flatten(1)
    intersection = (probs * target).sum(dim=1)
    union = probs.sum(dim=1) + target.sum(dim=1)
    dice = (2 * intersection + eps) / (union + eps)
    return 1 - dice.mean()


class DiceBCELoss(nn.Module):
    def __init__(self, bce_weight: float = 0.5, pos_weight: float | None = 8.0):
        super().__init__()
        self.bce_weight = bce_weight
        pw = torch.tensor(pos_weight) if pos_weight is not None else None
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pw)

    def forward(self, logits, target):
        bce = self.bce(logits, target)
        dice = dice_loss(logits, target)
        return self.bce_weight * bce + (1 - self.bce_weight) * dice, {
            "bce": bce.item(), "dice": dice.item()
        }
