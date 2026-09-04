#!/usr/bin/env python3
# experiments/run_regression_hilo_clean.py
"""Runner for Hi-Lo merge-strategy experiments.

Supports MSE loss with optional SPVAE auxiliary loss (KL + agreement).
Merge strategies: average, hi_lora, learned_router, cross_attn, spvae_router.
"""
from __future__ import annotations

import os, argparse, time, json
from typing import Dict, Any, Optional
from paths import get_data_root, resolve_pretrained_path

import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset

from data.io import load_subject
from data.scalers import compute_ea_whitening, apply_ea
from data.splits import TailSplit
from data.scalers import (
    fit_zscore_2d, apply_zscore_2d,
    fit_zscore_3d_per_channel, apply_zscore_3d_per_channel,
)
from data.datasets import HiLoAddDataset
from data.collate import make_collate_fn, SubjectXYZBank

from train.engine import EngineConfig, train_one_epoch, evaluate
from train.earlystop import EarlyStopper
from train.lr_schedule import WarmupCosineLR
from train.utils import log_detailed_trainable

from experiments.common import (
    set_seed, seed_worker,
    format_subject_table,
    evaluate_multi_loader,
    safe_parse_model_kwargs,
)
from train.sampling import SubjectInterleavedSampler

from models.steegformer.pretrained import load_pretrained_with_report
from models.steegformer.steegformer_hilo_clean import (
    hilo_clean_small, hilo_clean_base, hilo_clean_large,
    HiLoCleanBackbone, HiLoCleanRegressor,
)
from models.steegformer.probe import configure_lora_lastn_probe, set_requires_grad
from models.steegformer.lora import inject_switchable_lora_blocks, SwitchableLoRALinear, LoRALinear


# ============================================================
# Step function — pure MSE
# ============================================================

def make_step_fn(
    stream: str = "both",
    sp_weight: float = 1.0,
    input_noise_std: float = 0.0,
    channel_drop_prob: float = 0.0,
):
    """Create step function.

    stream:
        "both"    — default, lo + hi fusion
        "lo_only" — no hi tokens (x_hi=None), measures lo ceiling
        "hi_only" — hi fed as x_raw through lo pathway, no fusion.
    sp_weight:
        Weight for SPVAE auxiliary loss (KL + agreement).  0 = no SPVAE loss.
    input_noise_std:
        Std of Gaussian noise added to inputs during training (0 = off).
    channel_drop_prob:
        Probability of dropping entire ECoG channels during training (0 = off).
    """
    def step_fn(model: nn.Module, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        y = batch["y"]
        xyz = batch.get("ecog_xyz", None)
        x_lo = batch["x_raw"]
        x_hi = batch.get("x_hi", None)

        # --- Input augmentation (training only) ---
        if model.training:
            if input_noise_std > 0:
                x_lo = x_lo + input_noise_std * torch.randn_like(x_lo)
                if x_hi is not None:
                    x_hi = x_hi + input_noise_std * torch.randn_like(x_hi)
            if channel_drop_prob > 0:
                B, C, T = x_lo.shape
                mask = (torch.rand(B, C, 1, device=x_lo.device) > channel_drop_prob).float()
                x_lo = x_lo * mask
                if x_hi is not None:
                    x_hi = x_hi * mask

        if stream == "lo_only":
            out = model(x_lo, x_hi=None, ecog_xyz=xyz, return_losses=True)
        elif stream == "hi_only":
            out = model(x_hi, x_hi=None, ecog_xyz=xyz, return_losses=True)
        else:
            out = model(x_lo, x_hi=x_hi, ecog_xyz=xyz, return_losses=True)
        y_hat = out["y_hat"]
        loss = torch.mean((y_hat - y) ** 2)

        # Add SPVAE auxiliary loss if present
        backbone = getattr(model, "backbone", model)
        loss_spvae = backbone.last_aux.get("loss_spvae", None) if hasattr(backbone, "last_aux") else None
        if loss_spvae is not None and sp_weight > 0:
            loss = loss + sp_weight * loss_spvae

        return {"y_hat": y_hat, "loss": loss}
    return step_fn


# ============================================================
# Interpolate pretrained lo patch embed → hi patch embed
# ============================================================

def _init_hi_embed_from_lo(backbone: nn.Module):
    """Warm-start hi patch embed by interpolating pretrained lo weights.

    lo proj: Linear(16, 512) → weight (512, 16)
    hi proj: Linear(25, 512) → weight (512, 25)

    Interpolate along the input dimension (patch_size) using linear interp.
    """
    lo_w = backbone.patch_embed.proj.weight.data    # (D, 16)
    hi_proj = backbone.patch_embed_hi.proj
    D, ps_lo = lo_w.shape
    ps_hi = hi_proj.weight.shape[1]

    if ps_lo == ps_hi:
        # Same size — just copy
        hi_proj.weight.data.copy_(lo_w)
    else:
        # Interpolate: (D, ps_lo) → (1, D, ps_lo) → interpolate → (D, ps_hi)
        lo_w_3d = lo_w.unsqueeze(0)                 # (1, D, ps_lo)
        hi_w_3d = torch.nn.functional.interpolate(
            lo_w_3d, size=ps_hi, mode="linear", align_corners=False,
        )
        hi_proj.weight.data.copy_(hi_w_3d.squeeze(0))

    # Copy bias if both have it
    lo_b = backbone.patch_embed.proj.bias
    if lo_b is not None and hi_proj.bias is not None:
        hi_proj.bias.data.copy_(lo_b.data)

    print(
        f"  hi patch embed warm-started from lo: "
        f"({D},{ps_lo}) → interpolate → ({D},{ps_hi})",
        flush=True,
    )


# ============================================================
# Model construction
# ============================================================

def build_model(args, C_in: int, T_in: int, ecog_xyz_m: Optional[np.ndarray] = None, d_out: int = 5) -> nn.Module:
    mk = safe_parse_model_kwargs(args.model_kwargs_json)
    pretrained = mk.pop("pretrained", None)
    backbone_kwargs = mk.pop("backbone_kwargs", {}) or {}
    mk.pop("include_cls", None)
    mk.pop("head_dropout", None)

    variant = str(args.steegformer_variant).lower()

    # Variant → default architecture params
    arch_defaults = {
        "small": dict(patch_size=16, embed_dim=512, depth=8, num_heads=8),
        "base":  dict(patch_size=16, embed_dim=768, depth=12, num_heads=12),
        "large": dict(patch_size=16, embed_dim=1024, depth=24, num_heads=16),
    }[variant]

    # hi_only: model processes x_hi as lo input, so override patch_size to match
    if getattr(args, "stream", "both") == "hi_only":
        hi_ps = int(args.hi_patch_size)
        arch_defaults["patch_size"] = hi_ps
        print(f"  hi_only mode: overriding lo patch_size → {hi_ps}", flush=True)

    # If using full codebook (e.g. HBN 256 slots), set max_ch_idx=256
    _max_ch_idx = 256 if bool(getattr(args, "use_full_codebook", False)) else 145
    backbone = HiLoCleanBackbone(
        **arch_defaults,
        expect_num_chans=C_in,
        merge_strategy=args.merge_strategy,
        hi_inject_last_n=int(args.hi_inject_last_n),
        hi_patch_size=int(args.hi_patch_size),
        layerwise_gate_bottleneck=int(getattr(args, "layerwise_gate_bottleneck", 16)),
        layerwise_gate_act=str(getattr(args, "layerwise_gate_act", "tanh")),
        layerwise_gate_share_blocks=bool(getattr(args, "layerwise_gate_share_blocks", False)),
        max_ch_idx=_max_ch_idx,
        **backbone_kwargs,
    )

    # Load MAE pretrained weights.
    #
    # A missing `pretrained` block used to fall through here silently, training
    # a randomly-initialised backbone under the CORTEG name -- which is the
    # random-init ablation, not the method. Refuse instead: pass
    #   --model_kwargs_json configs/steegformer_<variant>.json
    # to load the backbone, or --no_pretrained to ask for random init on purpose.
    skip_pretrained = getattr(args, "no_pretrained", False)
    ckpt_path = str((pretrained or {}).get("path", "")).strip()
    if skip_pretrained:
        print("  [pretrained] --no_pretrained: backbone left at random init", flush=True)
    else:
        if not ckpt_path:
            raise SystemExit(
                "No pretrained backbone configured.\n"
                "  CORTEG loads ST-EEGFormer weights via --model_kwargs_json; without it the\n"
                "  backbone stays randomly initialised, which is the Table 2 'random init'\n"
                "  ablation rather than CORTEG.\n"
                "  Fix:  --model_kwargs_json configs/steegformer_%s.json\n"
                "  This is required even with --finetune_from: the released CORTEG\n"
                "  adapter holds only the trainable parameters, not the frozen backbone.\n"
                "  Or, to request random init deliberately:  --no_pretrained"
                % str(getattr(args, "steegformer_variant", "small")).lower()
            )
        ckpt_path = resolve_pretrained_path(ckpt_path)
        load_pretrained_with_report(
            backbone, ckpt_path,
            ckpt_key=pretrained.get("ckpt_key", "model"),
            strict=pretrained.get("strict", False),
            strip_prefix=pretrained.get("strip_prefix", ""),
            trust_checkpoint=True,
        )

    # Channel embedding controls
    ch_mode = getattr(args, "channel_embed_mode", "pretrained_learnable")
    if ch_mode.startswith("random"):
        # Reset to default zero init (same as unpretrained backbone)
        nn.init.zeros_(backbone.enc_channel.emb.weight)
        print(f"  [channel_embed] Reset to zeros — no pretrained spatial info (mode={ch_mode})")
    if ch_mode.endswith("frozen"):
        backbone.enc_channel.emb.weight.requires_grad = False
        print(f"  [channel_embed] Frozen (mode={ch_mode})")

    if getattr(args, "shuffle_channel_embed", False):
        with torch.no_grad():
            w = backbone.enc_channel.emb.weight.data
            perm = torch.randperm(w.shape[0])
            backbone.enc_channel.emb.weight.data = w[perm]
        print(f"  [channel_embed] Shuffled rows (spatial mapping destroyed)")
    if getattr(args, "synthetic_codebook", False):
        with torch.no_grad():
            w = backbone.enc_channel.emb.weight.data
            norms = w.norm(dim=1, keepdim=True)
            rand_dirs = torch.randn_like(w)
            rand_dirs = rand_dirs / (rand_dirs.norm(dim=1, keepdim=True) + 1e-8)
            backbone.enc_channel.emb.weight.data = rand_dirs * norms
        print(f"  [channel_embed] Synthetic codebook (matched norms, random directions)")

    # Layer-wise transfer ablation
    reinit = getattr(args, "reinit_components", "none")
    if reinit != "none":
        merge_k = backbone.merge_block_idx
        def _reinit_block(blk):
            for m in blk.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                elif isinstance(m, nn.LayerNorm):
                    nn.init.ones_(m.weight)
                    nn.init.zeros_(m.bias)

        if reinit in ("early_blocks", "all_blocks", "all_except_channel"):
            for blk in backbone.blocks[:merge_k]:
                _reinit_block(blk)
            print(f"  [reinit] Early blocks 0-{merge_k-1} reinitialized")
        if reinit in ("late_blocks", "all_blocks", "all_except_channel"):
            for blk in backbone.blocks[merge_k:]:
                _reinit_block(blk)
            print(f"  [reinit] Late blocks {merge_k}-{len(backbone.blocks)-1} reinitialized")
        if reinit in ("patch_embed", "all_except_channel"):
            _reinit_block(backbone.patch_embed)
            print(f"  [reinit] patch_embed reinitialized")
        if reinit == "all_except_channel":
            if hasattr(backbone, 'norm'):
                nn.init.ones_(backbone.norm.weight)
                nn.init.zeros_(backbone.norm.bias)
            print(f"  [reinit] all_except_channel: everything except enc_channel reinitialized")

    # Warm-start hi patch embed by interpolating pretrained lo weights
    # (skip for hi_only: lo patch_embed has different size, nothing to copy from)
    if getattr(args, "stream", "both") != "hi_only":
        _init_hi_embed_from_lo(backbone)

    # Attach channel adapter (must be after pretrained load)
    adapter_type = getattr(args, "channel_adapter", "none")
    if adapter_type != "none":
        # KNN adapters need electrode positions at construction time
        _xyz_m_tensor = None
        if ecog_xyz_m is not None:
            _xyz_m_tensor = torch.from_numpy(ecog_xyz_m / 1000.0).float()  # mm → meters
        backbone.attach_channel_adapter(
            adapter_type, M=int(args.M_EEG), hidden=int(args.fuser_hidden),
            ecog_xyz_m=_xyz_m_tensor,
            knn_k=int(getattr(args, "knn_k", 8)),
            knn_sigma=getattr(args, "knn_sigma", None),
            use_full_table=bool(getattr(args, "use_full_codebook", False)),
        )
        print(f"  channel_adapter={adapter_type} attached (M={args.M_EEG}, hidden={args.fuser_hidden})", flush=True)

        # Adapter branch ablation: disable one branch after construction
        if args.adapter_branch != "both" and hasattr(backbone, "channel_adapter"):
            ca = backbone.channel_adapter
            # KNNSoftFourierAdapter has .soft (KNNSoftAdapter) and .residual (FourierMLPAdapter)
            if hasattr(ca, "soft") and hasattr(ca, "residual"):
                if args.adapter_branch == "soft_only":
                    for p in ca.residual.parameters():
                        p.data.zero_()
                        p.requires_grad = False
                    print(f"  [adapter_branch=soft_only] Fourier residual zeroed & frozen")
                elif args.adapter_branch == "fourier_only":
                    ca.soft.scale.data.zero_()
                    ca.soft.scale.requires_grad = False
                    for p in ca.soft.net.parameters():
                        p.requires_grad = False
                    print(f"  [adapter_branch=fourier_only] Soft branch zeroed & frozen")

    elif getattr(args, "use_ecog_fuser", False):
        # Legacy path: use old ecog_fuser directly
        backbone.attach_ecog_fuser_from_channel_embed(
            M=int(args.M_EEG), hidden=int(args.fuser_hidden),
        )
        print(f"  ecog_fuser attached (M={args.M_EEG}, hidden={args.fuser_hidden})", flush=True)

    # Attach SPVAE latent router (must be after pretrained load)
    if getattr(args, "use_spvae", False):
        backbone.attach_spvae(
            z_shared_dim=int(args.spvae_z_shared_dim),
            z_private_dim=int(args.spvae_z_private_dim),
            hidden=int(args.spvae_hidden),
            beta_s=float(args.spvae_beta_s),
            beta_p=float(args.spvae_beta_p),
            lambda_agree=float(args.spvae_lambda_agree),
        )
        print(f"  SPVAE attached (zs={args.spvae_z_shared_dim}, zp={args.spvae_z_private_dim}, "
              f"beta_s={args.spvae_beta_s}, beta_p={args.spvae_beta_p}, "
              f"agree={args.spvae_lambda_agree})", flush=True)

    model = HiLoCleanRegressor(
        backbone=backbone,
        d_out=d_out,
        token_mode="mean",
        include_cls=True,
        head_dropout=float(args.head_dropout),
        head_hidden=int(getattr(args, "head_hidden", 0)),
    )
    return model


# ============================================================
# Unfreeze merge-specific parameters
# ============================================================

def unfreeze_merge_params(model: nn.Module):
    """Unfreeze the merge module that matches the active strategy."""
    b = getattr(model, "backbone", None)
    if b is None:
        return
    strategy = getattr(b, "merge_strategy", "average")
    # Map strategy -> attribute names to unfreeze
    names = {
        "learned_router": ["learned_router"],
        "hi_lora_router": ["learned_router"],
        "cross_attn": ["cross_attn_layer"],
        "spvae_router": ["enc_s_lo", "enc_s_hi", "enc_p_lo", "enc_p_hi"],
        "layerwise_gate": ["layerwise_gate"],
    }.get(strategy, [])
    for name in names:
        mod = getattr(b, name, None)
        if mod is not None:
            for p in mod.parameters():
                p.requires_grad = True
    # Always unfreeze channel adapter / ecog fuser if present
    for attr in ("channel_adapter", "ecog_fuser"):
        mod = getattr(b, attr, None)
        if mod is not None:
            for p in mod.parameters():
                p.requires_grad = True
    # Always unfreeze SPVAE encoders if attached (even if merge != spvae_router,
    # the SPVAE loss still trains them)
    if getattr(b, "use_spvae", False):
        for enc_name in ("enc_s_lo", "enc_s_hi", "enc_p_lo", "enc_p_hi"):
            mod = getattr(b, enc_name, None)
            if mod is not None:
                for p in mod.parameters():
                    p.requires_grad = True


def _is_hi_param(name: str, backbone) -> bool:
    """Check if a named parameter belongs to the hi-freq pathway."""
    if "patch_embed_hi" in name:
        return True
    # SwitchableLoRA adapters in early blocks (blocks 0..merge_k-1)
    if hasattr(backbone, "merge_block_idx"):
        merge_k = backbone.merge_block_idx
        for bi in range(merge_k):
            prefix = f"backbone.blocks.{bi}."
            if name.startswith(prefix) and (".A." in name or ".B." in name):
                return True
    # Merge modules (router, cross-attn)
    if ("learned_router" in name or "cross_attn_layer" in name
            or "layerwise_gate" in name):
        return True
    # ECoG fuser / channel adapter adapts to hi-freq too
    if "ecog_fuser" in name or "channel_adapter" in name:
        return True
    return False


def configure_finetune_modules(model: nn.Module, modules: list, lora_last_n: int,
                               lora_targets: tuple = ("qkv", "proj", "fc1", "fc2")):
    """Freeze all, then selectively unfreeze specified module groups for stage-2 finetuning.

    modules: list of strings from {"head", "ln", "lora", "adapter"}.
        head    — regression head
        ln      — LayerNorms in LoRA blocks
        lora    — LoRA A/B weights + LayerNorms in LoRA blocks
        adapter — channel adapter (e.g. KNN Fourier residual)
    lora_targets: which LoRA modules to unfreeze (e.g. ("qkv",) for minimal adaptation).
        Only LoRALinear modules whose attribute name matches a target are unfrozen.
    """
    set_requires_grad(model, False)

    backbone = getattr(model, "backbone", model)
    blocks = backbone.blocks if hasattr(backbone, "blocks") else []
    depth = len(blocks)
    start = max(0, depth - lora_last_n)

    if "head" in modules:
        if hasattr(model, "head"):
            set_requires_grad(model.head, True)

    need_ln = "ln" in modules or "lora" in modules
    if need_ln:
        for bi in range(start, depth):
            for m in blocks[bi].modules():
                if isinstance(m, nn.LayerNorm):
                    for p in m.parameters():
                        p.requires_grad = True

    if "lora" in modules:
        for bi in range(start, depth):
            for name, m in blocks[bi].named_modules():
                if isinstance(m, LoRALinear) and any(t in name for t in lora_targets):
                    m.A.weight.requires_grad = True
                    m.B.weight.requires_grad = True

    if "adapter" in modules:
        for attr in ("channel_adapter", "ecog_fuser"):
            mod = getattr(backbone, attr, None)
            if mod is not None:
                set_requires_grad(mod, True)

    if "hi" in modules:
        for attr in ("patch_embed_hi", "hi_scale"):
            mod = getattr(backbone, attr, None)
            if mod is not None:
                if isinstance(mod, nn.Parameter):
                    mod.requires_grad = True
                else:
                    set_requires_grad(mod, True)


def reinit_lora(model: nn.Module, lora_last_n: int):
    """Re-initialize LoRA A/B weights to their fresh state (B=0, A=kaiming).

    Use after loading a pooled checkpoint so LoRA starts fresh for per-subject
    training, while the backbone base weights retain the population prior.
    """
    backbone = getattr(model, "backbone", model)
    blocks = backbone.blocks if hasattr(backbone, "blocks") else []
    depth = len(blocks)
    start = max(0, depth - lora_last_n)
    count = 0
    for bi in range(start, depth):
        for m in blocks[bi].modules():
            if isinstance(m, LoRALinear):
                nn.init.kaiming_uniform_(m.A.weight, a=np.sqrt(5))
                nn.init.zeros_(m.B.weight)
                count += 1
    print(f"  Re-initialized {count} LoRA modules to fresh state", flush=True)


def compute_prior_reg(model: nn.Module, prior_state: dict) -> torch.Tensor:
    """Compute L2 penalty toward pooled prior weights (Empirical Bayes MAP).

    Returns: scalar tensor  Σ_i ‖θ_i - θ_i^pool‖²  over all trainable params
    that exist in prior_state.
    """
    penalty = torch.tensor(0.0, device=next(model.parameters()).device)
    for name, param in model.named_parameters():
        if param.requires_grad and name in prior_state:
            penalty = penalty + torch.sum((param - prior_state[name]) ** 2)
    return penalty


def inject_hi_lora(model: nn.Module, args):
    """For hi_lora strategy: inject SwitchableLoRA into early blocks.

    SwitchableLoRA wraps each target Linear so that:
      - enabled=False → pure frozen base(x)       (used for lo tokens)
      - enabled=True  → base(x) + LoRA(x)         (used for hi tokens)

    Only the LoRA adapter params (A, B) are trainable. The base weights stay frozen.
    """
    b = getattr(model, "backbone", None)
    if b is None or not hasattr(b, "blocks"):
        return
    merge_k = b.merge_block_idx
    if merge_k <= 0:
        return
    targets = tuple(t.strip() for t in args.lora_targets.split(",") if t.strip())
    early_indices = list(range(merge_k))
    inject_switchable_lora_blocks(
        model, early_indices,
        r=int(args.lora_r), alpha=int(args.lora_alpha),
        dropout=float(args.lora_dropout), targets=targets,
    )
    # Unfreeze LayerNorms in early blocks so they can adapt to hi input
    for bi in early_indices:
        for m in b.blocks[bi].modules():
            if isinstance(m, nn.LayerNorm):
                for p in m.parameters():
                    p.requires_grad = True


# ============================================================
# Prediction / representation extraction
# ============================================================

def save_predictions(model, loader, save_dir, device, use_amp=True, stream="both"):
    """Extract and save predictions + mean-pooled representations.

    Handles stream modes: for hi_only, swaps x_raw ← x_hi (matching step_fn).
    For lo_only, sets x_hi=None.
    """
    os.makedirs(save_dir, exist_ok=True)
    model.eval()
    all_true, all_pred, all_mean = [], [], []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
            x_main = batch["x_raw"]
            x_hi = batch.get("x_hi")
            if stream == "hi_only":
                x_main = x_hi
                x_hi = None
            elif stream == "lo_only":
                x_hi = None
            with torch.amp.autocast("cuda", enabled=use_amp):
                tokens = model.backbone.forward_tokens(
                    x_main, x_hi=x_hi, ecog_xyz=batch.get("ecog_xyz"))
                y_hat = model.head(tokens)
                h = tokens.mean(dim=1)
            all_true.append(batch["y"].cpu().numpy())
            all_pred.append(y_hat.detach().float().cpu().numpy())
            all_mean.append(h.detach().float().cpu().numpy())
    np.save(os.path.join(save_dir, "y_true.npy"), np.concatenate(all_true))
    np.save(os.path.join(save_dir, "y_pred.npy"), np.concatenate(all_pred))
    np.save(os.path.join(save_dir, "z_mean.npy"), np.concatenate(all_mean))


# ============================================================
# Main
# ============================================================

def main():
    p = argparse.ArgumentParser("Hi-Lo Clean Merge Experiments")

    # Data
    p.add_argument("--dataset", type=str, default="Stanford",
                    choices=["Stanford", "Ghent"])
    p.add_argument("--step_ms", type=int, default=0,
                    help="Training window step in ms for Ghent (0=use default: 200 Ghent)")
    p.add_argument("--eval_step_ms", type=int, default=50,
                    help="Evaluation window step in ms for Ghent (default 50=20Hz)")
    p.add_argument("--target_mode", type=str, default="endpoint",
                    choices=["endpoint", "mean"],
                    help="Ghent target: endpoint (last sample) or mean (window avg)")
    p.add_argument("--data_root", type=str, default="")
    p.add_argument("--val_ratio", type=float, default=0.1)
    p.add_argument("--train_mode", type=str, default="per_subject",
                    choices=["per_subject", "pooled", "finetune"])

    # Training
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--early_stop_patience", type=int, default=190)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--weight_decay", type=float, default=5e-3)
    p.add_argument("--accum_iter", type=int, default=1)
    p.add_argument("--max_norm", type=float, default=1.0)
    p.add_argument("--use_amp", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--sched", type=str, default="cosine", choices=["none", "cosine"])
    p.add_argument("--warmup_epochs", type=int, default=10)
    p.add_argument("--min_lr", type=float, default=1e-5)

    # Model
    p.add_argument("--steegformer_variant", type=str, default="small")
    p.add_argument("--model_kwargs_json", type=str, default="")
    p.add_argument("--head_dropout", type=float, default=0.0)
    p.add_argument("--head_hidden", type=int, default=0,
                    help="Hidden dim for 2-layer MLP head (0 = single linear)")

    # Input regularization
    p.add_argument("--input_noise_std", type=float, default=0.0,
                    help="Gaussian noise std added to inputs during training")
    p.add_argument("--channel_drop_prob", type=float, default=0.0,
                    help="Probability of dropping entire ECoG channels during training")

    # LoRA
    p.add_argument("--lora_last_n", type=int, default=4)
    p.add_argument("--lora_r", type=int, default=4)
    p.add_argument("--lora_alpha", type=int, default=16)
    p.add_argument("--lora_dropout", type=float, default=0.2)
    p.add_argument("--lora_targets", type=str, default="qkv,proj,fc1,fc2")

    # Hi-Lo
    p.add_argument("--hi_patch_size", type=int, default=25)
    p.add_argument("--hi_inject_last_n", type=int, default=4)
    p.add_argument("--merge_strategy", type=str, default="average",
                    choices=["average", "hi_lora", "learned_router", "cross_attn",
                             "spvae_router", "hi_lora_router", "layerwise_gate"])
    p.add_argument("--layerwise_gate_bottleneck", type=int, default=16,
                    help="Hidden width of the layer-wise gate MLP")
    p.add_argument("--layerwise_gate_act", type=str, default="tanh",
                    choices=["tanh", "sigmoid", "none"],
                    help="Gate activation; tanh starts at 0 (lo-only) and can suppress hi")
    p.add_argument("--layerwise_gate_share_blocks", action="store_true",
                    help="Also push the hi stream through each block (shared weights)")
    p.add_argument("--stream", type=str, default="both",
                    choices=["both", "lo_only", "hi_only"],
                    help="Which stream(s) to use: both (default), lo_only, hi_only")
    p.add_argument("--hi_lr_mult", type=float, default=1.0,
                    help="LR multiplier for hi-freq params (patch_embed_hi, "
                         "SwitchableLoRA in early blocks, merge modules)")

    # ECoG channel adapter
    p.add_argument("--use_ecog_fuser", action="store_true")
    p.add_argument("--channel_adapter", type=str, default="none",
                    choices=["none", "original", "zero_mlp", "additive_mlp",
                             "fourier_add", "subspace", "soft_lookup_add", "coord_pe",
                             "knn_hard", "knn_fourier", "knn_soft", "knn_soft_fourier",
                             "gp_hard", "gp_fourier"],
                    help="Channel adaptation method (requires --use_ecog_fuser for xyz data)")
    p.add_argument("--M_EEG", type=int, default=145)
    p.add_argument("--fuser_hidden", type=int, default=128)
    p.add_argument("--knn_k", type=int, default=8, help="Number of nearest EEG neighbours for KNN adapters")
    p.add_argument("--use_full_codebook", action="store_true",
                    help="Use the full EEG embedding table (e.g. 256 slots for HBN) in KNNSoft adapter, instead of capping at 142 positioned channels")
    p.add_argument("--knn_sigma", type=float, default=None, help="Gaussian bandwidth for KNN (None=auto)")

    # SPVAE latent router
    p.add_argument("--use_spvae", action="store_true", help="Attach SPVAE encoders for precision-gated merge")
    p.add_argument("--spvae_z_shared_dim", type=int, default=128)
    p.add_argument("--spvae_z_private_dim", type=int, default=128)
    p.add_argument("--spvae_hidden", type=int, default=256)
    p.add_argument("--spvae_beta_s", type=float, default=1e-4, help="KL weight for shared latent")
    p.add_argument("--spvae_beta_p", type=float, default=1e-3, help="KL weight for private latent")
    p.add_argument("--spvae_lambda_agree", type=float, default=1e-3, help="Agreement loss weight")
    p.add_argument("--sp_weight", type=float, default=1.0, help="Weight for total SPVAE aux loss")

    # Transfer ablation controls
    p.add_argument("--no_pretrained", action="store_true",
                    help="Skip pretrained weight loading (random init)")
    p.add_argument("--full_finetune", action="store_true",
                    help="Train ALL parameters (no freeze, no LoRA) — fair scratch baseline")
    p.add_argument("--channel_embed_mode", type=str, default="pretrained_learnable",
                    choices=["pretrained_learnable", "pretrained_frozen",
                             "random_learnable", "random_frozen"],
                    help="Channel embedding ablation: pretrained/random x learnable/frozen")
    p.add_argument("--shuffle_channel_embed", action="store_true",
                    help="Randomly permute channel embedding rows (breaks spatial mapping)")
    p.add_argument("--synthetic_codebook", action="store_true",
                    help="Replace channel embed with norm-matched random vectors")
    p.add_argument("--reinit_components", type=str, default="none",
                    choices=["none", "early_blocks", "late_blocks", "all_blocks",
                             "patch_embed", "all_except_channel"],
                    help="Reinitialize specific pretrained components to isolate transfer")
    p.add_argument("--train_fraction", type=float, default=1.0,
                    help="Fraction of training data to use (for low-data curves)")
    p.add_argument("--xyz_mode", type=str, default="real",
                    choices=["real", "shuffled", "zero", "random"],
                    help="Geometry ablation: real=actual XYZ, shuffled=permute within subject, "
                         "zero=all (0,0,0), random=uniform random per electrode")
    p.add_argument("--adapter_branch", type=str, default="both",
                    choices=["both", "soft_only", "fourier_only"],
                    help="Adapter branch ablation: both=full adapter, soft_only=disable Fourier "
                         "residual, fourier_only=disable soft lookup")

    # Euclidean Alignment preprocessing
    p.add_argument("--use_ea", action="store_true",
                    help="Apply Euclidean Alignment (covariance whitening) before z-score")
    p.add_argument("--ea_shrinkage", type=float, default=0.1,
                    help="EA shrinkage toward identity (0=pure whitening, 1=identity)")

    # Stage-2 finetuning (Empirical Bayes: pooled LOO → per-subject adaptation)
    p.add_argument("--exclude_subjects", type=str, default="",
                    help="Comma-separated subjects to EXCLUDE from pooled training (LOO for stage-2)")
    p.add_argument("--finetune_from", type=str, default="",
                    help="Path to pooled checkpoint (.pt) for stage-2 finetuning")
    p.add_argument("--finetune_modules", type=str, default="head",
                    help="Comma-separated modules to unfreeze: head,ln,lora,adapter")
    p.add_argument("--finetune_subjects", type=str, default="",
                    help="Comma-separated subjects for finetuning (empty = all)")
    p.add_argument("--finetune_lora_targets", type=str, default="qkv,proj,fc1,fc2",
                    help="Which LoRA modules to unfreeze in stage-2 (e.g. 'qkv' for minimal)")
    p.add_argument("--prior_reg_weight", type=float, default=0.0,
                    help="L2 regularization toward pooled prior (Empirical Bayes MAP). 0=off")
    p.add_argument("--reinit_lora", action="store_true",
                    help="Re-initialize LoRA A/B to fresh state after loading checkpoint")
    p.add_argument("--finetune_lr_lora", type=float, default=0.0,
                    help="Separate LR for LoRA params in stage-2 (0=use main --lr)")
    p.add_argument("--finetune_lr_adapter", type=float, default=0.0,
                    help="Separate LR for adapter params in stage-2 (0=use main --lr)")
    p.add_argument("--finetune_token_mode", type=str, default="",
                    choices=["", "mean", "flatten", "cls"],
                    help="Override token pooling mode for stage-2 head (empty=keep original)")

    # Misc
    p.add_argument("--save_root", type=str,
                    default=os.path.join(
                        os.environ.get("CORTEG_OUTPUT_ROOT",
                                       os.path.expanduser("~/workspace/outputs/corteg")),
                        "default_run"))
    p.add_argument("--save_ckpt_path", type=str, default="",
                    help="Save best model state_dict to this path after training")

    args = p.parse_args()
    os.makedirs(args.save_root, exist_ok=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    need_xyz = bool(args.use_ecog_fuser) or (args.channel_adapter != "none")
    print(
        f"device={device} dataset={args.dataset} stream={args.stream} "
        f"merge_strategy={args.merge_strategy} "
        f"hi_inject_last_n={args.hi_inject_last_n} "
        f"lora_last_n={args.lora_last_n} lora_r={args.lora_r} "
        f"hi_lr_mult={args.hi_lr_mult} channel_adapter={args.channel_adapter}",
        flush=True,
    )

    step_fn = make_step_fn(
        stream=args.stream,
        sp_weight=float(args.sp_weight),
        input_noise_std=float(getattr(args, "input_noise_std", 0.0)),
        channel_drop_prob=float(getattr(args, "channel_drop_prob", 0.0)),
    )

    # ---- Dataset-specific data loading ----
    if args.dataset == "Ghent":
        # Ghent speech envelope: on-the-fly HDF5 windows, runtime z-score normalized
        from data.ghent_loader import load_ghent_datasets, get_ghent_subjects

        ghent_data_root = args.data_root if args.data_root else ""
        ghent_subjects = get_ghent_subjects(data_root=ghent_data_root)
        step_ms = args.step_ms if args.step_ms > 0 else 200  # default 200ms train step
        eval_step_ms = args.eval_step_ms  # default 50ms = 20 Hz eval
        tr_list, va_list, te_list, shapes_by_sid, xyz_bank, collate_fn, d_out, xyz_mm = \
            load_ghent_datasets(
                subjects=ghent_subjects,
                data_root=args.data_root if args.data_root else "",
                step_ms=step_ms,
                eval_step_ms=eval_step_ms,
                target_mode=args.target_mode,
                need_xyz=need_xyz,
            )
        subjects = ghent_subjects

        # Geometry ablation for Ghent: transform XYZ after loading
        if args.xyz_mode != "real" and need_xyz and xyz_mm is not None:
            rng_xyz = np.random.RandomState(args.seed)
            for i in range(len(xyz_mm)):
                if xyz_mm[i] is None:
                    continue
                C = xyz_mm[i].shape[0]
                if args.xyz_mode == "shuffled":
                    perm = rng_xyz.permutation(C)
                    xyz_mm[i] = xyz_mm[i][perm]
                elif args.xyz_mode == "zero":
                    xyz_mm[i] = np.zeros_like(xyz_mm[i])
                elif args.xyz_mode == "random":
                    lo = xyz_mm[i].min(axis=0)
                    hi = xyz_mm[i].max(axis=0)
                    xyz_mm[i] = rng_xyz.uniform(lo, hi, size=(C, 3)).astype(np.float32)
            # Rebuild xyz_bank and collate_fn with transformed coordinates
            xyz_bank = SubjectXYZBank.from_mm(xyz_mm)
            collate_fn = make_collate_fn(xyz_bank)
            print(f"  [xyz_mode={args.xyz_mode}] Ghent XYZ coordinates transformed")
    else:
        # Stanford finger trajectory: pre-epoched pickle format
        d_out = 5
        datasets_meta = {
            "Stanford": {
                "subjects": ["bp", "cc", "ht", "jc", "jp", "mv", "wc", "wm", "zt"],
                "path": get_data_root(),
            },
        }
        subjects = datasets_meta[args.dataset]["subjects"]
        file_root = args.data_root if args.data_root else datasets_meta[args.dataset]["path"]

        split = TailSplit(val_ratio=float(args.val_ratio))

        subj_data, xyz_mm = [], []
        for sub in subjects:
            sd = load_subject(file_root, sub, require_xyz=need_xyz)
            subj_data.append(sd)
            xyz_mm.append(sd.ecog_xyz_mm)

        # Geometry ablation: transform XYZ before building the adapter
        if args.xyz_mode != "real" and need_xyz:
            rng_xyz = np.random.RandomState(args.seed)
            for i in range(len(xyz_mm)):
                if xyz_mm[i] is None:
                    continue
                C = xyz_mm[i].shape[0]
                if args.xyz_mode == "shuffled":
                    perm = rng_xyz.permutation(C)
                    xyz_mm[i] = xyz_mm[i][perm]
                elif args.xyz_mode == "zero":
                    xyz_mm[i] = np.zeros_like(xyz_mm[i])
                elif args.xyz_mode == "random":
                    lo = xyz_mm[i].min(axis=0)
                    hi = xyz_mm[i].max(axis=0)
                    xyz_mm[i] = rng_xyz.uniform(lo, hi, size=(C, 3)).astype(np.float32)
            print(f"  [xyz_mode={args.xyz_mode}] XYZ coordinates transformed")

        xyz_bank = SubjectXYZBank.from_mm(xyz_mm) if need_xyz else None
        collate_fn = make_collate_fn(xyz_bank)

        tr_list, va_list, te_list, shapes_by_sid = [], [], [], []
        for sid, sd in enumerate(subj_data):
            n = int(sd.y_tr.shape[0])
            idx_tr, idx_val = split.split(n)

            # Low-data ablation: subsample training indices
            train_frac = getattr(args, "train_fraction", 1.0)
            if train_frac < 1.0:
                rng = np.random.RandomState(args.seed)
                n_keep = max(1, int(len(idx_tr) * train_frac))
                idx_tr = rng.choice(idx_tr, size=n_keep, replace=False)
                idx_tr.sort()

            y_stats = fit_zscore_2d(sd.y_tr[idx_tr])
            y_tr = apply_zscore_2d(sd.y_tr[idx_tr], y_stats)
            y_val = apply_zscore_2d(sd.y_tr[idx_val], y_stats)
            y_te = apply_zscore_2d(sd.y_te, y_stats)

            x_raw_tr0, x_raw_val0 = sd.X_raw_tr[idx_tr], sd.X_raw_tr[idx_val]
            x_hi_tr0, x_hi_val0 = sd.X_feat_tr[idx_tr][..., 0], sd.X_feat_tr[idx_val][..., 0]

            # Euclidean Alignment: apply BEFORE z-score normalization
            if getattr(args, "use_ea", False):
                if sid == 0:
                    print(f"  EA enabled: shrinkage={args.ea_shrinkage}", flush=True)
                W_raw = compute_ea_whitening(x_raw_tr0, shrinkage=float(args.ea_shrinkage))
                W_hi = compute_ea_whitening(x_hi_tr0, shrinkage=float(args.ea_shrinkage))
                x_raw_tr0 = apply_ea(x_raw_tr0, W_raw)
                x_raw_val0 = apply_ea(x_raw_val0, W_raw)
                x_hi_tr0 = apply_ea(x_hi_tr0, W_hi)
                x_hi_val0 = apply_ea(x_hi_val0, W_hi)
                sd_X_raw_te = apply_ea(sd.X_raw_te, W_raw)
                sd_X_feat_te_hi = apply_ea(sd.X_feat_te[..., 0], W_hi)
            else:
                sd_X_raw_te = sd.X_raw_te
                sd_X_feat_te_hi = sd.X_feat_te[..., 0]

            raw_stats = fit_zscore_3d_per_channel(x_raw_tr0)
            hi_stats = fit_zscore_3d_per_channel(x_hi_tr0)

            x_tr = apply_zscore_3d_per_channel(x_raw_tr0, raw_stats)
            x_val = apply_zscore_3d_per_channel(x_raw_val0, raw_stats)
            x_te = apply_zscore_3d_per_channel(sd_X_raw_te, raw_stats)
            h_tr = apply_zscore_3d_per_channel(x_hi_tr0, hi_stats)
            h_val = apply_zscore_3d_per_channel(x_hi_val0, hi_stats)
            h_te = apply_zscore_3d_per_channel(sd_X_feat_te_hi, hi_stats)

            tr_list.append(HiLoAddDataset(x_tr, h_tr, y_tr, sid))
            va_list.append(HiLoAddDataset(x_val, h_val, y_val, sid))
            te_list.append(HiLoAddDataset(x_te, h_te, y_te, sid))
            # For hi_only: model input is x_hi, so track hi temporal dim
            if args.stream == "hi_only":
                shapes_by_sid.append((x_tr.shape[1], h_tr.shape[2]))
            else:
                shapes_by_sid.append((x_tr.shape[1], x_tr.shape[2]))

        # Free the raw float64 SubjectData records now that the float32 dataset
        # tensors (tr_list/va_list/te_list) are built. subj_data is never read
        # again, but otherwise stays alive in this frame for the whole run and
        # holds ~27 GB (all subjects, float64) -> drove the OOM that froze the
        # box (2026-06-19, 2026-06-24). Frees ~27 GB; numerically a no-op.
        del subj_data
        import gc
        gc.collect()

    # ---- Pooled Training ----
    if args.train_mode == "pooled":
        set_seed(args.seed)
        g = torch.Generator().manual_seed(args.seed)
        max_C = max(s[0] for s in shapes_by_sid)
        T0 = shapes_by_sid[0][1]

        # LOO: exclude specified subjects from training (for stage-2 finetuning)
        exclude = set(s.strip() for s in args.exclude_subjects.split(",") if s.strip())
        train_sids = [i for i, s in enumerate(subjects) if s not in exclude]

        if exclude:
            print(f"\n  LOO: excluding {sorted(exclude)} from training "
                  f"({len(train_sids)}/{len(subjects)} subjects)", flush=True)

        # For pooled KNN: use first subject's xyz for model init (runtime computes per-batch)
        init_xyz = xyz_mm[0] if need_xyz else None
        model = build_model(args, max_C, T0, ecog_xyz_m=init_xyz, d_out=d_out).to(device)

        if args.full_finetune:
            set_requires_grad(model, True)
            # Re-apply adapter branch freezing (set_requires_grad unfreezes everything)
            if args.adapter_branch != "both" and hasattr(model.backbone, "channel_adapter"):
                ca = model.backbone.channel_adapter
                if hasattr(ca, "soft") and hasattr(ca, "residual"):
                    if args.adapter_branch == "soft_only":
                        for p in ca.residual.parameters():
                            p.requires_grad = False
                    elif args.adapter_branch == "fourier_only":
                        ca.soft.scale.requires_grad = False
                        for p in ca.soft.net.parameters():
                            p.requires_grad = False
            print(f"  [full_finetune] {sum(p.numel() for p in model.parameters() if p.requires_grad):,} "
                  f"params trainable", flush=True)
        else:
            configure_lora_lastn_probe(
                model,
                n_last=int(args.lora_last_n),
                r=int(args.lora_r),
                alpha=int(args.lora_alpha),
                dropout=float(args.lora_dropout),
                targets=tuple(t.strip() for t in args.lora_targets.split(",") if t.strip()),
            )
            if args.merge_strategy in ("hi_lora", "hi_lora_router"):
                inject_hi_lora(model, args)
            unfreeze_merge_params(model)
        log_detailed_trainable(model)

        opt = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=args.lr, weight_decay=args.weight_decay,
        )
        scheduler = (
            WarmupCosineLR(opt, warmup_epochs=args.warmup_epochs, max_epochs=args.epochs, min_lr=args.min_lr)
            if args.sched == "cosine" else None
        )
        eng_cfg = EngineConfig(use_amp=bool(args.use_amp), accum_iter=int(args.accum_iter), max_norm=float(args.max_norm))
        early = EarlyStopper(patience=args.early_stop_patience)

        # Training: only included subjects
        pool_tr = [tr_list[i] for i in train_sids]
        tr_sizes = [len(ds) for ds in pool_tr]
        tr_sampler = SubjectInterleavedSampler(tr_sizes, args.batch_size, shuffle=True)
        train_loader = DataLoader(
            ConcatDataset(pool_tr), batch_sampler=tr_sampler,
            num_workers=0, pin_memory=False, worker_init_fn=seed_worker, collate_fn=collate_fn,
        )
        # Validation/test: all subjects (monitor held-out performance)
        val_loaders = [DataLoader(ds, batch_size=64, collate_fn=collate_fn) for ds in va_list]
        test_loaders = [DataLoader(ds, batch_size=64, collate_fn=collate_fn) for ds in te_list]

        print(f"\nPooled Training: {len(train_sids)}/{len(subjects)} subjects "
              f"(excluded: {sorted(exclude) if exclude else 'none'}), max_C={max_C}", flush=True)
        scaler = torch.cuda.amp.GradScaler(enabled=(args.use_amp and device.type == "cuda"))
        t0_train = time.time()
        for ep in range(args.epochs):
            tr = train_one_epoch(model, train_loader, opt, device, cfg=eng_cfg, step_fn=step_fn, scaler=scaler)
            if scheduler:
                scheduler.step()
            val = evaluate_multi_loader(model, val_loaders, device, step_fn, bool(args.use_amp))
            test = evaluate_multi_loader(model, test_loaders, device, step_fn, bool(args.use_amp))
            if ep % 20 == 0 or ep == args.epochs - 1:
                gate_str = ""
                backbone = getattr(model, "backbone", model)
                gate_mean = getattr(backbone, "_last_gate_mean", None)
                if gate_mean is not None:
                    gate_str = f" gate={gate_mean:.3f}"
                print(
                    f"[ep {ep+1:03d}] train_loss={tr['loss']:.4f} val_score={val['score']:.4f} "
                    f"test_score={test['score']:.4f}{gate_str}",
                    flush=True,
                )
                print(format_subject_table(test, subjects), flush=True)
            if early.step(val["score"], model):
                break

        early.restore(model)
        test = evaluate_multi_loader(model, test_loaders, device, step_fn, bool(args.use_amp))
        elapsed = time.time() - t0_train
        print("\n[FINAL TEST]", flush=True)
        print(format_subject_table(test, subjects), flush=True)

        # Save results JSON
        results = {
            "variant": "pooled",
            "train_mode": "pooled",
            "score": test["score"],
            "score_mse": test.get("score_mse", None),
            "elapsed_s": round(elapsed, 1),
            "per_subject": {},
            "args": {k: str(v) if not isinstance(v, (int, float, bool, type(None))) else v
                     for k, v in vars(args).items()},
        }
        for sid in sorted(test["by_sid"].keys()):
            rec = test["by_sid"][sid]
            sub_name = subjects[sid] if sid < len(subjects) else str(sid)
            results["per_subject"][sub_name] = {
                "corr_mean": rec["corr_mean"],
                "corr": [float(c) for c in rec["corr"]],
                "mse": rec["mse"],
                "n": rec["n"],
            }
        results_path = os.path.join(args.save_root, "results_pooled.json")
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved: {results_path}", flush=True)
        print(f"SCORE = {test['score']:.4f}", flush=True)

        # Save per-subject predictions and representations for manifold analysis
        for sid_i, ds_te in enumerate(te_list):
            if sid_i not in train_sids and train_sids:
                continue
            sub_name = subjects[sid_i] if sid_i < len(subjects) else str(sid_i)
            pred_dir = os.path.join(args.save_root, "predictions", sub_name)
            te_ld = DataLoader(ds_te, batch_size=64, collate_fn=collate_fn)
            save_predictions(model, te_ld, pred_dir, device, args.use_amp, stream=args.stream)
        print(f"Predictions saved: {os.path.join(args.save_root, 'predictions/')}", flush=True)

        # Save trainable weights (LoRA + head + adapter — not frozen backbone)
        trainable_names = {n for n, p in model.named_parameters() if p.requires_grad}
        trainable_state = {k: v for k, v in model.state_dict().items() if k in trainable_names}
        ckpt_dir = os.path.join(args.save_root, "checkpoints")
        os.makedirs(ckpt_dir, exist_ok=True)
        torch.save(trainable_state, os.path.join(ckpt_dir, "trainable_weights.pt"))
        print(f"Trainable weights saved: {ckpt_dir}/trainable_weights.pt "
              f"({len(trainable_state)} params)", flush=True)

        # Save full checkpoint if explicitly requested
        ckpt_path = args.save_ckpt_path
        if ckpt_path:
            os.makedirs(os.path.dirname(ckpt_path) or ".", exist_ok=True)
            torch.save(model.state_dict(), ckpt_path)
            print(f"  Saved full checkpoint → {ckpt_path}", flush=True)
        return

    # ---- Stage-2 Finetune: load pooled checkpoint, per-subject adaptation ----
    if args.train_mode == "finetune":
        ft_modules = [m.strip() for m in args.finetune_modules.split(",") if m.strip()]
        ft_subjects = (
            [s.strip() for s in args.finetune_subjects.split(",") if s.strip()]
            if args.finetune_subjects else subjects
        )
        max_C = max(s[0] for s in shapes_by_sid)
        T0 = shapes_by_sid[0][1]
        init_xyz = xyz_mm[0] if need_xyz else None

        print(f"\n{'='*60}", flush=True)
        print(f"STAGE 2 FINETUNE: modules={ft_modules} subjects={ft_subjects}", flush=True)
        print(f"  checkpoint: {args.finetune_from}", flush=True)
        print(f"{'='*60}", flush=True)

        for sub_name in ft_subjects:
            sid = subjects.index(sub_name)
            set_seed(args.seed)
            g = torch.Generator().manual_seed(args.seed)

            print(f"\n===== Finetune Subject {sid} ({sub_name}) =====", flush=True)

            # Build model with max_C (matching pooled architecture)
            model = build_model(args, max_C, T0, ecog_xyz_m=init_xyz, d_out=d_out).to(device)

            # Inject LoRA (must match pooled structure for state_dict compatibility)
            configure_lora_lastn_probe(
                model, n_last=int(args.lora_last_n),
                r=int(args.lora_r), alpha=int(args.lora_alpha),
                dropout=float(args.lora_dropout),
                targets=tuple(t.strip() for t in args.lora_targets.split(",") if t.strip()),
            )
            if args.merge_strategy in ("hi_lora", "hi_lora_router"):
                inject_hi_lora(model, args)
            unfreeze_merge_params(model)

            # Materialize lazy head before loading checkpoint
            # (TokenRegressor.head is None until first forward; infer dim from checkpoint)
            ckpt = torch.load(args.finetune_from, map_location=device)
            if model.head.head is None:
                # Find first weight to infer input dim (works for both Linear and Sequential heads)
                head_w_key = next(
                    (k for k in ckpt if k.startswith("head.head.") and k.endswith(".weight")),
                    None,
                )
                if head_w_key is not None:
                    in_dim = ckpt[head_w_key].shape[-1]
                    model.head.head = model.head._build_head(in_dim, device)

            # Filter out shape-mismatched keys (e.g. head when d_out differs)
            model_sd = model.state_dict()
            mismatched = [k for k in list(ckpt.keys())
                          if k in model_sd and ckpt[k].shape != model_sd[k].shape]
            if mismatched:
                for k in mismatched:
                    ckpt.pop(k)
                print(f"  Skipped shape-mismatched keys: {mismatched}", flush=True)
            model.load_state_dict(ckpt, strict=False)
            print(f"  Loaded pooled checkpoint (strict=False, {len(mismatched)} skipped)",
                  flush=True)

            # Override token pooling mode for stage-2 (e.g. mean → flatten)
            if args.finetune_token_mode:
                old_mode = model.head.token_mode
                model.head.token_mode = args.finetune_token_mode
                # Materialize new head immediately: run a dummy forward to get token shape
                model.eval()
                sample_batch = next(iter(DataLoader(tr_list[sid], batch_size=2, collate_fn=collate_fn)))
                with torch.no_grad():
                    sample_x = sample_batch["x_raw"].to(device)
                    fwd_kw = {}
                    if "x_hi" in sample_batch:
                        fwd_kw["x_hi"] = sample_batch["x_hi"].to(device)
                    if "ecog_xyz" in sample_batch:
                        fwd_kw["ecog_xyz"] = sample_batch["ecog_xyz"].to(device)
                    tokens = model.backbone.forward_tokens(sample_x, **fwd_kw)
                    if model.head.include_cls:
                        feat = tokens
                    else:
                        feat = tokens[:, 1:, :] if tokens.shape[1] > 1 else tokens
                    if args.finetune_token_mode in ("cls", "mean"):
                        in_dim = feat.shape[2]
                    else:
                        in_dim = feat.shape[1] * feat.shape[2]
                model.head.head = model.head._build_head(in_dim, device)
                print(f"  Token mode: {old_mode} → {args.finetune_token_mode} (head re-init: {in_dim} → {model.head.d_out})", flush=True)

            # Plan B: optionally re-initialize LoRA to fresh state
            if args.reinit_lora:
                reinit_lora(model, int(args.lora_last_n))

            # Freeze all, then selectively unfreeze
            ft_lora_targets = tuple(
                t.strip() for t in args.finetune_lora_targets.split(",") if t.strip()
            )
            configure_finetune_modules(model, ft_modules, int(args.lora_last_n),
                                       lora_targets=ft_lora_targets)
            log_detailed_trainable(model)

            # Plan A: store prior state for Empirical Bayes MAP regularization
            prior_reg_weight = float(args.prior_reg_weight)
            prior_state = {}
            if prior_reg_weight > 0:
                for name, param in model.named_parameters():
                    if param.requires_grad:
                        prior_state[name] = param.detach().clone()
                print(f"  Prior reg: λ={prior_reg_weight}, {len(prior_state)} param tensors", flush=True)

            # Wrap step_fn to add prior regularization
            if prior_reg_weight > 0:
                _base_step_fn = step_fn
                def ft_step_fn(model, batch, _bsf=_base_step_fn, _ps=prior_state, _lam=prior_reg_weight):
                    out = _bsf(model, batch)
                    reg = compute_prior_reg(model, _ps)
                    loss = out["loss"] + _lam * reg
                    return {**out, "loss": loss, "loss_main": out["loss"], "loss_prior_reg": _lam * reg}
            else:
                ft_step_fn = step_fn

            # Zero-shot: evaluate pooled checkpoint on this subject before any finetuning
            va_loader = DataLoader(va_list[sid], batch_size=64, collate_fn=collate_fn)
            te_loader = DataLoader(te_list[sid], batch_size=64, collate_fn=collate_fn)
            zs_val = evaluate(model, va_loader, device, step_fn=step_fn, use_amp=args.use_amp)
            zs_test = evaluate(model, te_loader, device, step_fn=step_fn, use_amp=args.use_amp)
            print(f"\n[ZERO-SHOT] subject={sub_name} val={zs_val['score']:.4f} test={zs_test['score']:.4f}", flush=True)
            print(format_subject_table(zs_test, subjects), flush=True)

            # Plan C: differential learning rates per module group
            lr_lora = float(args.finetune_lr_lora) if args.finetune_lr_lora > 0 else args.lr
            lr_adapter = float(args.finetune_lr_adapter) if args.finetune_lr_adapter > 0 else args.lr
            backbone = getattr(model, "backbone", model)

            lora_params, adapter_params, other_params = [], [], []
            for name, param in model.named_parameters():
                if not param.requires_grad:
                    continue
                if ".A.weight" in name or ".B.weight" in name:
                    lora_params.append(param)
                elif "channel_adapter" in name or "ecog_fuser" in name:
                    adapter_params.append(param)
                else:
                    other_params.append(param)

            param_groups = []
            if other_params:
                param_groups.append({"params": other_params, "lr": args.lr})
            if lora_params:
                param_groups.append({"params": lora_params, "lr": lr_lora})
            if adapter_params:
                param_groups.append({"params": adapter_params, "lr": lr_adapter})

            if lr_lora != args.lr or lr_adapter != args.lr:
                print(f"  Differential LR: head/LN={args.lr:.1e}, LoRA={lr_lora:.1e}, adapter={lr_adapter:.1e}", flush=True)

            opt = torch.optim.AdamW(param_groups, weight_decay=args.weight_decay)
            scheduler = (
                WarmupCosineLR(opt, warmup_epochs=args.warmup_epochs, max_epochs=args.epochs, min_lr=args.min_lr)
                if args.sched == "cosine" else None
            )
            eng_cfg = EngineConfig(use_amp=bool(args.use_amp), accum_iter=int(args.accum_iter), max_norm=float(args.max_norm))
            early = EarlyStopper(patience=args.early_stop_patience)

            # Apply train_fraction if specified (for low-data experiments)
            ft_train_ds = tr_list[sid]
            train_frac = getattr(args, "train_fraction", 1.0)
            if train_frac < 1.0:
                n_total = len(ft_train_ds)
                n_keep = max(1, int(n_total * train_frac))
                ft_train_ds = torch.utils.data.Subset(ft_train_ds, list(range(n_keep)))
                print(f"  train_fraction={train_frac}: {n_keep}/{n_total} samples", flush=True)

            tr_loader = DataLoader(
                ft_train_ds, batch_size=args.batch_size, shuffle=True,
                collate_fn=collate_fn, generator=g, worker_init_fn=seed_worker,
            )
            scaler = torch.cuda.amp.GradScaler(enabled=args.use_amp)

            for ep in range(args.epochs):
                tr = train_one_epoch(model, tr_loader, opt, device, cfg=eng_cfg, step_fn=ft_step_fn, scaler=scaler)
                if scheduler:
                    scheduler.step()
                val = evaluate(model, va_loader, device, step_fn=step_fn, use_amp=args.use_amp)
                if ep % 20 == 0:
                    reg_str = f" reg={tr.get('loss_prior_reg', 0):.4f}" if prior_reg_weight > 0 else ""
                    print(f"[ep {ep+1}] mse={tr.get('loss_main', tr['loss']):.4f}{reg_str} | val={val['score']:.4f}", flush=True)
                if early.step(val["score"], model):
                    break

            early.restore(model)
            test_result = evaluate(model, te_loader, device, step_fn=step_fn)
            print(f"\n[FINAL TEST] subject={sub_name} modules={ft_modules}", flush=True)
            print(format_subject_table(test_result, subjects))

            # Save per-subject finetune results
            rec = list(test_result["by_sid"].values())[0]
            results = {
                "variant": "finetune",
                "subject": sub_name,
                "modules": ft_modules,
                "train_fraction": getattr(args, "train_fraction", 1.0),
                "score": float(test_result["score"]),
                "per_finger": [float(c) for c in rec["corr"]],
                "mse": float(rec["mse"]),
            }
            os.makedirs(args.save_root, exist_ok=True)
            results_path = os.path.join(args.save_root, "results_persub.json")
            with open(results_path, "w") as f:
                json.dump(results, f, indent=2)
            print(f"\nResults saved: {results_path}", flush=True)
        return

    # ---- Per-Subject Training ----
    all_per_subject = {}
    t0_all = time.time()
    for sid in range(len(subjects)):
        set_seed(args.seed)
        g = torch.Generator().manual_seed(args.seed)
        C, T = shapes_by_sid[sid]
        print(f"\n===== Subject {sid} ({subjects[sid]}) =====", flush=True)

        model = build_model(args, C, T, ecog_xyz_m=xyz_mm[sid], d_out=d_out).to(device)

        # Probe: freeze all, inject LoRA into last N blocks, unfreeze head
        configure_lora_lastn_probe(
            model,
            n_last=int(args.lora_last_n),
            r=int(args.lora_r),
            alpha=int(args.lora_alpha),
            dropout=float(args.lora_dropout),
            targets=tuple(
                t.strip() for t in args.lora_targets.split(",") if t.strip()
            ),
        )
        # hi_lora: also inject DualLoRA into the early (frozen) blocks
        if args.merge_strategy in ("hi_lora", "hi_lora_router"):
            inject_hi_lora(model, args)
        # Unfreeze merge-specific modules (router, cross-attn)
        unfreeze_merge_params(model)
        log_detailed_trainable(model)

        # Log strategy info at start
        if args.merge_strategy in ("learned_router", "hi_lora_router"):
            print(f"  router gate init = 0.5000 (zero-init)", flush=True)

        # Per-modality LR: give hi-freq params a higher learning rate
        hi_mult = float(args.hi_lr_mult)
        if hi_mult != 1.0:
            hi_params, other_params = [], []
            for name, param in model.named_parameters():
                if not param.requires_grad:
                    continue
                if _is_hi_param(name, model.backbone):
                    hi_params.append(param)
                else:
                    other_params.append(param)
            param_groups = [
                {"params": other_params, "lr": args.lr},
                {"params": hi_params, "lr": args.lr * hi_mult},
            ]
            print(
                f"  Per-modality LR: {len(hi_params)} hi params @ "
                f"{args.lr * hi_mult:.1e}, {len(other_params)} other @ "
                f"{args.lr:.1e}", flush=True,
            )
        else:
            param_groups = [p for p in model.parameters() if p.requires_grad]

        opt = torch.optim.AdamW(
            param_groups,
            lr=args.lr, weight_decay=args.weight_decay,
        )
        scheduler = (
            WarmupCosineLR(
                opt, warmup_epochs=args.warmup_epochs,
                max_epochs=args.epochs, min_lr=args.min_lr,
            )
            if args.sched == "cosine" else None
        )
        eng_cfg = EngineConfig(
            use_amp=bool(args.use_amp),
            accum_iter=int(args.accum_iter),
            max_norm=float(args.max_norm),
        )
        early = EarlyStopper(patience=args.early_stop_patience)
        tr_loader = DataLoader(
            tr_list[sid], batch_size=args.batch_size, shuffle=True,
            collate_fn=collate_fn, generator=g, worker_init_fn=seed_worker,
        )
        va_loader = DataLoader(va_list[sid], batch_size=64, collate_fn=collate_fn)
        te_loader = DataLoader(te_list[sid], batch_size=64, collate_fn=collate_fn)
        scaler = torch.cuda.amp.GradScaler(enabled=args.use_amp)

        for ep in range(args.epochs):
            tr = train_one_epoch(
                model, tr_loader, opt, device,
                cfg=eng_cfg, step_fn=step_fn, scaler=scaler,
            )
            if scheduler:
                scheduler.step()
            val = evaluate(model, va_loader, device, step_fn=step_fn, use_amp=args.use_amp)

            if ep % 20 == 0:
                extra = ""
                if args.merge_strategy in ("learned_router", "hi_lora_router"):
                    gate = model.backbone.get_router_gate_mean()
                    if gate is not None:
                        extra = f" gate={gate:.4f}"
                print(
                    f"[ep {ep+1}] mse={tr['loss']:.4f} | "
                    f"val={val['score']:.4f}{extra}",
                    flush=True,
                )
            if early.step(val["score"], model):
                break

        early.restore(model)

        # Final gate log
        if args.merge_strategy in ("learned_router", "hi_lora_router"):
            gate = model.backbone.get_router_gate_mean()
            if gate is not None:
                print(f"  router gate final = {gate:.4f}", flush=True)

        test_result = evaluate(model, te_loader, device, step_fn=step_fn)
        print(format_subject_table(test_result, subjects))

        sub_name = subjects[sid]
        rec = test_result["by_sid"][sid]
        all_per_subject[sub_name] = {
            "corr_mean": rec["corr_mean"],
            "corr": [float(c) for c in rec["corr"]],
            "mse": rec["mse"],
            "n": rec["n"],
        }

        # Save predictions and representations for manifold analysis
        pred_dir = os.path.join(args.save_root, "predictions", sub_name)
        save_predictions(model, te_loader, pred_dir, device, args.use_amp, stream=args.stream)
        print(f"  Predictions saved: {pred_dir}", flush=True)

        # Save trainable weights per subject
        trainable_names = {n for n, p in model.named_parameters() if p.requires_grad}
        trainable_state = {k: v for k, v in model.state_dict().items() if k in trainable_names}
        ckpt_dir = os.path.join(args.save_root, "checkpoints", sub_name)
        os.makedirs(ckpt_dir, exist_ok=True)
        torch.save(trainable_state, os.path.join(ckpt_dir, "trainable_weights.pt"))

    # Save aggregated per-subject results
    if all_per_subject:
        elapsed_all = time.time() - t0_all
        mean_score = float(np.mean([v["corr_mean"] for v in all_per_subject.values()]))
        results = {
            "variant": "per_subject",
            "train_mode": "per_subject",
            "score": mean_score,
            "elapsed_s": round(elapsed_all, 1),
            "per_subject": all_per_subject,
            "args": {k: str(v) if not isinstance(v, (int, float, bool, type(None))) else v
                     for k, v in vars(args).items()},
        }
        results_path = os.path.join(args.save_root, "results_persub.json")
        os.makedirs(args.save_root, exist_ok=True)
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved: {results_path}", flush=True)
        print(f"SCORE = {mean_score:.4f}", flush=True)


if __name__ == "__main__":
    main()
