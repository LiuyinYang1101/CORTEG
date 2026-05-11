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
# Dual-LoRA: two independent adapter pairs per Linear, selected by stream ID
# ---------------------------------------------------------------------------

class DualLoRALinear(nn.Module):
    """LoRA wrapper with two independent adapter pairs (lo / hi).

    The active adapter is selected via ``self.stream`` (0 = lo, 1 = hi).
    Base weight is frozen and shared between both streams.
    """

    def __init__(self, base: nn.Linear, r: int = 8, alpha: int = 16, dropout: float = 0.0):
        super().__init__()
        if not isinstance(base, nn.Linear):
            raise TypeError("DualLoRALinear expects a nn.Linear module")

        self.base = base
        self.r = int(r)
        self.alpha = int(alpha)
        self.scaling = self.alpha / max(self.r, 1)
        self.drop = nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity()

        in_f = base.in_features
        out_f = base.out_features
        dev = base.weight.device
        dt = base.weight.dtype

        # Lo stream adapters
        self.A_lo = nn.Linear(in_f, self.r, bias=False, device=dev, dtype=dt)
        self.B_lo = nn.Linear(self.r, out_f, bias=False, device=dev, dtype=dt)
        nn.init.kaiming_uniform_(self.A_lo.weight, a=np.sqrt(5))
        nn.init.zeros_(self.B_lo.weight)

        # Hi stream adapters
        self.A_hi = nn.Linear(in_f, self.r, bias=False, device=dev, dtype=dt)
        self.B_hi = nn.Linear(self.r, out_f, bias=False, device=dev, dtype=dt)
        nn.init.kaiming_uniform_(self.A_hi.weight, a=np.sqrt(5))
        nn.init.zeros_(self.B_hi.weight)

        for p in self.base.parameters():
            p.requires_grad = False

        self.stream: int = 0  # 0 = lo, 1 = hi

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        if self.stream == 0:
            return base_out + self.B_lo(self.A_lo(self.drop(x))) * self.scaling
        else:
            return base_out + self.B_hi(self.A_hi(self.drop(x))) * self.scaling


def _replace_linear_with_dual_lora(module: nn.Module, targets=("qkv", "proj", "fc1", "fc2"), r=8, alpha=16, dropout=0.0):
    """Recursively replace target Linear layers with DualLoRALinear wrappers."""
    for name, child in list(module.named_children()):
        _replace_linear_with_dual_lora(child, targets=targets, r=r, alpha=alpha, dropout=dropout)
        if isinstance(child, nn.Linear) and any(t in name for t in targets):
            setattr(module, name, DualLoRALinear(child, r=r, alpha=alpha, dropout=dropout))


def set_block_stream(block: nn.Module, stream: int):
    """Set the active stream (0=lo, 1=hi) on all DualLoRALinear modules in *block*."""
    for m in block.modules():
        if isinstance(m, DualLoRALinear):
            m.stream = stream


def inject_dual_lora_blocks(model: nn.Module, block_indices, r: int, alpha: int, dropout: float, targets=("qkv", "proj", "fc1", "fc2")):
    """Inject DualLoRALinear adapters into specific transformer block indices."""
    backbone = getattr(model, "backbone", model)
    if not hasattr(backbone, "blocks"):
        raise ValueError("Expected backbone.blocks (timm ViT).")
    blocks = backbone.blocks
    for bi in block_indices:
        _replace_linear_with_dual_lora(blocks[bi], targets=targets, r=r, alpha=alpha, dropout=dropout)


# ---------------------------------------------------------------------------
# Switchable LoRA: adapter active only when self.enabled = True
# ---------------------------------------------------------------------------

class SwitchableLoRALinear(nn.Module):
    """LoRA wrapper that can be toggled on/off at runtime.

    enabled=False  →  base(x)                          (pure frozen pass-through)
    enabled=True   →  base(x) + B(A(drop(x))) * scale  (frozen + LoRA)

    Use case: early blocks process lo tokens with enabled=False (no adapter)
    and hi tokens with enabled=True (LoRA adaptation).
    """

    def __init__(self, base: nn.Linear, r: int = 8, alpha: int = 16, dropout: float = 0.0):
        super().__init__()
        if not isinstance(base, nn.Linear):
            raise TypeError("SwitchableLoRALinear expects a nn.Linear module")

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

        self.enabled: bool = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        if self.enabled:
            out = out + self.B(self.A(self.drop(x))) * self.scaling
        return out


def set_block_lora_enabled(block: nn.Module, enabled: bool):
    """Toggle all SwitchableLoRALinear modules in *block*."""
    for m in block.modules():
        if isinstance(m, SwitchableLoRALinear):
            m.enabled = enabled


def _replace_linear_with_switchable_lora(module: nn.Module, targets=("qkv", "proj", "fc1", "fc2"), r=8, alpha=16, dropout=0.0):
    """Recursively replace target Linear layers with SwitchableLoRALinear wrappers."""
    for name, child in list(module.named_children()):
        _replace_linear_with_switchable_lora(child, targets=targets, r=r, alpha=alpha, dropout=dropout)
        if isinstance(child, nn.Linear) and any(t in name for t in targets):
            setattr(module, name, SwitchableLoRALinear(child, r=r, alpha=alpha, dropout=dropout))


def inject_switchable_lora_blocks(model: nn.Module, block_indices, r: int, alpha: int, dropout: float, targets=("qkv", "proj", "fc1", "fc2")):
    """Inject SwitchableLoRALinear adapters into specific transformer block indices."""
    backbone = getattr(model, "backbone", model)
    if not hasattr(backbone, "blocks"):
        raise ValueError("Expected backbone.blocks (timm ViT).")
    blocks = backbone.blocks
    for bi in block_indices:
        _replace_linear_with_switchable_lora(blocks[bi], targets=targets, r=r, alpha=alpha, dropout=dropout)


# ---------------------------------------------------------------------------
# Task-specific LoRA: independent adapter pairs per task (for multi-task)
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


def _replace_linear_with_task_lora(module: nn.Module, targets=("qkv", "proj", "fc1", "fc2"),
                                    r=8, alpha=16, dropout=0.0, n_tasks=2):
    """Recursively replace target Linear layers with TaskLoRALinear wrappers."""
    for name, child in list(module.named_children()):
        _replace_linear_with_task_lora(child, targets=targets, r=r, alpha=alpha,
                                        dropout=dropout, n_tasks=n_tasks)
        if isinstance(child, nn.Linear) and any(t in name for t in targets):
            setattr(module, name, TaskLoRALinear(child, r=r, alpha=alpha,
                                                 dropout=dropout, n_tasks=n_tasks))


def inject_task_lora_lastn(model: nn.Module, n_last: int, r: int, alpha: int,
                           dropout: float, n_tasks: int = 2,
                           targets=("qkv", "proj", "fc1", "fc2")):
    """Inject TaskLoRALinear adapters into the last *n_last* blocks."""
    backbone = getattr(model, "backbone", model)
    if not hasattr(backbone, "blocks"):
        raise ValueError("Expected backbone.blocks (timm ViT).")
    blocks = backbone.blocks
    n = len(blocks)
    start = max(0, n - n_last)
    for bi in range(start, n):
        _replace_linear_with_task_lora(blocks[bi], targets=targets, r=r, alpha=alpha,
                                        dropout=dropout, n_tasks=n_tasks)


def inject_hybrid_lora_lastn(model: nn.Module, n_last: int, n_shared: int,
                              r: int, alpha: int, dropout: float,
                              n_tasks: int = 2,
                              targets=("qkv", "proj", "fc1", "fc2")):
    """Inject shared LoRA in early adapted blocks, task-specific in late blocks.

    Example with n_last=4, n_shared=2 on 8-block model:
      Blocks 0-3: frozen (no LoRA)
      Blocks 4-5: shared LoRA (both tasks learn together)
      Blocks 6-7: task-specific LoRA (per-task specialization)
    """
    backbone = getattr(model, "backbone", model)
    if not hasattr(backbone, "blocks"):
        raise ValueError("Expected backbone.blocks (timm ViT).")
    blocks = backbone.blocks
    n = len(blocks)
    start = max(0, n - n_last)
    shared_end = start + n_shared

    # Early adapted blocks: shared LoRA
    for bi in range(start, min(shared_end, n)):
        _replace_linear_with_lora(blocks[bi], targets=targets, r=r, alpha=alpha, dropout=dropout)

    # Late adapted blocks: task-specific LoRA
    for bi in range(shared_end, n):
        _replace_linear_with_task_lora(blocks[bi], targets=targets, r=r, alpha=alpha,
                                        dropout=dropout, n_tasks=n_tasks)
