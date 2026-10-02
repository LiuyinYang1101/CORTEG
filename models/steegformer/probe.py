"""Parameter-freezing and probe-configuration utilities for STEEGFormer fine-tuning."""
from __future__ import annotations
import torch.nn as nn
from .lora import inject_lora_lastn_steegformer


def set_requires_grad(module: nn.Module, requires: bool):
    """Set ``requires_grad`` on all parameters of *module*."""
    for p in module.parameters():
        p.requires_grad = requires


def unfreeze_hi_components_if_present(model: nn.Module):
    """Unfreeze the hi-frequency patch embed inside *model.backbone*, if present."""
    b = getattr(model, "backbone", None)
    if b is None:
        return
    if hasattr(b, "patch_embed_hi") and (b.patch_embed_hi is not None):
        for p in b.patch_embed_hi.parameters():
            p.requires_grad = True


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

    unfreeze_hi_components_if_present(model)
