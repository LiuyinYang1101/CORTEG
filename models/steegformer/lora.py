"""LoRA (Low-Rank Adaptation) injection utilities for STEEGFormer transformer blocks."""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """
    Minimal LoRA for nn.Linear: y = Wx + (B(A(drop(x))) * alpha/r)
    """
    def __init__(self, base: nn.Linear, r: int = 8, alpha: int = 16, dropout: float = 0.0):
        super().__init__()
        if not isinstance(base, nn.Linear):
            raise TypeError("LoRALinear expects a nn.Linear module")

        self.base = base
        self.r = int(r)
        self.alpha = int(alpha)
        self.scaling = self.alpha / max(self.r, 1)
        self.drop = nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity()

        in_f = base.in_features
        out_f = base.out_features
        dev = base.weight.device
        dt = base.weight.dtype

        self.A = nn.Linear(in_f, self.r, bias=False, device=dev, dtype=dt)
        self.B = nn.Linear(self.r, out_f, bias=False, device=dev, dtype=dt)

        nn.init.kaiming_uniform_(self.A.weight, a=np.sqrt(5))
        nn.init.zeros_(self.B.weight)

        for p in self.base.parameters():
            p.requires_grad = False

    @property
    def weight(self):
        """Expose base weight for code that accesses module.weight directly."""
        return self.base.weight

    @property
    def bias(self):
        """Expose base bias for code that accesses module.bias directly."""
        return self.base.bias

    @property
    def in_features(self):
        return self.base.in_features

    @property
    def out_features(self):
        return self.base.out_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + self.B(self.A(self.drop(x))) * self.scaling


def _replace_linear_with_lora(module: nn.Module, targets=("qkv", "proj", "fc1", "fc2"), r=8, alpha=16, dropout=0.0):
    """Recursively replace target Linear layers inside *module* with LoRALinear wrappers."""
    for name, child in list(module.named_children()):
        _replace_linear_with_lora(child, targets=targets, r=r, alpha=alpha, dropout=dropout)
        if isinstance(child, nn.Linear) and any(t in name for t in targets):
            setattr(module, name, LoRALinear(child, r=r, alpha=alpha, dropout=dropout))


def inject_lora_lastn_steegformer(model: nn.Module, n_last: int, r: int, alpha: int, dropout: float, targets=("qkv","proj","fc1","fc2")):
    """Inject LoRA adapters into the last *n_last* blocks of a STEEGFormer backbone."""
    backbone = getattr(model, "backbone", model)
    if not hasattr(backbone, "blocks"):
        raise ValueError("Expected backbone.blocks (timm ViT).")
    blocks = backbone.blocks
    n = len(blocks)
    n_last = int(n_last)
    start = max(0, n - n_last)
    for bi in range(start, n):
        _replace_linear_with_lora(blocks[bi], targets=targets, r=r, alpha=alpha, dropout=dropout)


# ---------------------------------------------------------------------------
# Task-specific LoRA: independent adapter pairs per task (per-subject LoRA on BrainTreebank)
# ---------------------------------------------------------------------------

class TaskLoRALinear(nn.Module):
    """LoRA wrapper with N independent adapter pairs, one per task.

    The active adapter is selected via ``self.active_task`` (int index).
    Base weight is frozen and shared across all tasks.
    """

    def __init__(self, base: nn.Linear, r: int = 8, alpha: int = 16,
                 dropout: float = 0.0, n_tasks: int = 2):
        super().__init__()
        if not isinstance(base, nn.Linear):
            raise TypeError("TaskLoRALinear expects a nn.Linear module")

        self.base = base
        self.r = int(r)
        self.alpha = int(alpha)
        self.scaling = self.alpha / max(self.r, 1)
        self.n_tasks = n_tasks
        self.drop = nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity()

        in_f = base.in_features
        out_f = base.out_features
        dev = base.weight.device
        dt = base.weight.dtype

        self.A = nn.ModuleList([
            nn.Linear(in_f, self.r, bias=False, device=dev, dtype=dt)
            for _ in range(n_tasks)
        ])
        self.B = nn.ModuleList([
            nn.Linear(self.r, out_f, bias=False, device=dev, dtype=dt)
            for _ in range(n_tasks)
        ])
        for i in range(n_tasks):
            nn.init.kaiming_uniform_(self.A[i].weight, a=np.sqrt(5))
            nn.init.zeros_(self.B[i].weight)

        for p in self.base.parameters():
            p.requires_grad = False

        self.active_task: int = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + self.B[self.active_task](
            self.A[self.active_task](self.drop(x))) * self.scaling


def set_active_task(model: nn.Module, task: int):
    """Set the active task on all TaskLoRALinear modules in *model*."""
    for m in model.modules():
        if isinstance(m, TaskLoRALinear):
            m.active_task = task
