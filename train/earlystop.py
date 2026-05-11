
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict
import torch
import torch.nn as nn


@dataclass
class EarlyStopper:
    patience: int = 100
    min_delta: float = 1e-3
    best: float = -1e18
    bad_epochs: int = 0
    best_state: Optional[Dict[str, torch.Tensor]] = None

    def step(self, score: float, model: nn.Module) -> bool:
        improved = score > (self.best + self.min_delta)
        if improved:
            self.best = float(score)
            self.bad_epochs = 0
            self.best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            self.bad_epochs += 1
        return self.bad_epochs >= self.patience

    def restore(self, model: nn.Module):
        if self.best_state is not None:
            model.load_state_dict(self.best_state, strict=True)
