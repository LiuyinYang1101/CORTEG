"""Parameter-freezing and probe-configuration utilities for STEEGFormer fine-tuning."""
from __future__ import annotations
import torch
import torch.nn as nn
from .lora import inject_lora_lastn_steegformer, inject_dual_lora_blocks, _replace_linear_with_lora


def set_requires_grad(module: nn.Module, requires: bool):
    """Set ``requires_grad`` on all parameters of *module*."""
    for p in module.parameters():
        p.requires_grad = requires


def unfreeze_ecog_fuser_if_present(model: nn.Module):
    """Unfreeze the ECoG-to-EEG fuser inside *model.backbone*, if present."""
    b = getattr(model, "backbone", None)
    if b is None:
        return
    if hasattr(b, "ecog_fuser") and (b.ecog_fuser is not None):
        for p in b.ecog_fuser.parameters():
            p.requires_grad = True


def unfreeze_hi_components_if_present(model: nn.Module):
    """Unfreeze hi-frequency patch embed, router, and hi_scale inside *model.backbone*, if present."""
    b = getattr(model, "backbone", None)
    if b is None:
        return
    if hasattr(b, "patch_embed_hi") and (b.patch_embed_hi is not None):
        for p in b.patch_embed_hi.parameters():
            p.requires_grad = True
    if hasattr(b, "router") and (b.router is not None):
        for p in b.router.parameters():
            p.requires_grad = True
    if hasattr(b, "hi_scale") and isinstance(getattr(b, "hi_scale"), torch.nn.Parameter):
        b.hi_scale.requires_grad = True


def configure_linear_probe(model: nn.Module):
    """Freeze all parameters except the regression head (and any ECoG-specific adapters)."""
    set_requires_grad(model, False)
    # common wrapper uses .head as reg head
    if hasattr(model, "head"):
        set_requires_grad(model.head, True)
    unfreeze_ecog_fuser_if_present(model)
    unfreeze_hi_components_if_present(model)


def configure_full_finetune(model: nn.Module):
    """Unfreeze every parameter for full fine-tuning."""
    set_requires_grad(model, True)


def configure_lora_lastn_probe(
    model: nn.Module,
    *,
    n_last: int,
    r: int,
    alpha: int,
    dropout: float,
    targets=("qkv","proj","fc1","fc2"),
):
    """Freeze all params, inject LoRA into the last *n_last* transformer blocks, unfreeze head."""
    set_requires_grad(model, False)
    if hasattr(model, "head"):
        set_requires_grad(model.head, True)

    inject_lora_lastn_steegformer(model, n_last=n_last, r=r, alpha=alpha, dropout=dropout, targets=targets)

    # unfreeze LayerNorms in last n blocks
    backbone = getattr(model, "backbone", None)
    if backbone is not None and hasattr(backbone, "blocks"):
        blocks = backbone.blocks
        start = max(0, len(blocks) - int(n_last))
        for bi in range(start, len(blocks)):
            for m in blocks[bi].modules():
                if isinstance(m, nn.LayerNorm):
                    for p in m.parameters():
                        p.requires_grad = True

    unfreeze_ecog_fuser_if_present(model)
    unfreeze_hi_components_if_present(model)


def configure_dual_lora_probe(
    model: nn.Module,
    *,
    split_k: int,
    n_late_lora: int,
    r: int,
    alpha: int,
    dropout: float,
    targets=("qkv", "proj", "fc1", "fc2"),
):
    """Freeze all, inject DualLoRA in early blocks [0..split_k), standard LoRA in late blocks, unfreeze head.

    Args:
        split_k: block index where streams merge. Blocks [0..split_k) get DualLoRA.
        n_late_lora: how many of the late blocks [split_k..end) get standard LoRA.
                     Blocks without any LoRA remain frozen.
    """
    set_requires_grad(model, False)
    if hasattr(model, "head"):
        set_requires_grad(model.head, True)

    backbone = getattr(model, "backbone", model)
    if not hasattr(backbone, "blocks"):
        raise ValueError("Expected backbone.blocks (timm ViT).")
    blocks = backbone.blocks
    depth = len(blocks)

    # Early blocks: DualLoRA
    early_indices = list(range(min(split_k, depth)))
    inject_dual_lora_blocks(model, early_indices, r=r, alpha=alpha, dropout=dropout, targets=targets)

    # Late blocks: standard LoRA
    late_start = max(split_k, depth - n_late_lora)
    for bi in range(late_start, depth):
        _replace_linear_with_lora(blocks[bi], targets=targets, r=r, alpha=alpha, dropout=dropout)

    # Unfreeze LayerNorms in all LoRA-injected blocks
    for bi in list(early_indices) + list(range(late_start, depth)):
        for m in blocks[bi].modules():
            if isinstance(m, nn.LayerNorm):
                for p in m.parameters():
                    p.requires_grad = True

    unfreeze_ecog_fuser_if_present(model)
    unfreeze_hi_components_if_present(model)
