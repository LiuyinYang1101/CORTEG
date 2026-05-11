"""Learning rate schedulers for ECoG training."""
from __future__ import annotations

import math

import torch


class WarmupCosineLR(torch.optim.lr_scheduler._LRScheduler):
    """Cosine annealing schedule with a linear warmup phase.

    During the first ``warmup_epochs`` epochs the LR increases linearly from 0
    to ``base_lr``. After that it follows a cosine decay down to ``min_lr``.

    Args:
        optimizer: The wrapped optimizer.
        warmup_epochs: Number of linear-warmup epochs.
        max_epochs: Total number of training epochs.
        min_lr: Floor LR at the end of cosine decay.
        last_epoch: Index of the last epoch (default: -1).
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_epochs: int,
        max_epochs: int,
        min_lr: float = 1e-6,
        last_epoch: int = -1,
    ):
        self.warmup_epochs = int(warmup_epochs)
        self.max_epochs = int(max_epochs)
        self.min_lr = float(min_lr)
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = self.last_epoch + 1
        if step <= self.warmup_epochs:
            scale = step / max(1, self.warmup_epochs)
            return [base_lr * scale for base_lr in self.base_lrs]
        t = step - self.warmup_epochs
        T = max(1, self.max_epochs - self.warmup_epochs)
        cos = 0.5 * (1.0 + math.cos(math.pi * t / T))
        return [self.min_lr + (base_lr - self.min_lr) * cos for base_lr in self.base_lrs]
