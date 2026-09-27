#!/usr/bin/env python3
"""LaBraM competing FM baseline — uses IDENTICAL pipeline to run_regression_hilo_clean.py.

Only differences from STEEGFormer:
1. Backbone: LaBraM NeuralTransformer instead of HiLoCleanBackbone
2. Input: hi-gamma features only (C, 200) → (C, 1, 200) for LaBraM TemporalConv
   STEEGFormer uses dual-stream (raw 128Hz + hi-gamma 200Hz); LaBraM uses single-stream
3. Channel embedding: LaBraM uses pos_embed (129, 200) instead of enc_channel (145, embed_dim)
4. Fine-tuning: unfreeze last N blocks (same philosophy as LoRA, but no adapter injection)

Everything else is identical: data loading, splits, Z-score, EA, collate, optimizer,
scheduler, early stopping, evaluation, metrics.
"""
from __future__ import annotations

import json
import os
import argparse
import time
from typing import Dict, Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data.io import load_subject
from data.splits import TailSplit
from data.scalers import (
    fit_zscore_2d, apply_zscore_2d,
    compute_ea_whitening, apply_ea,
)
from data.datasets import HiLoAddDataset
from data.collate import make_collate_fn

from train.engine import EngineConfig, train_one_epoch
from train.earlystop import EarlyStopper
from train.lr_schedule import WarmupCosineLR
from train.sampling import SubjectInterleavedSampler

from experiments.common import (
    set_seed, seed_worker,
    evaluate_multi_loader,
    STANFORD_SUBJECTS,
)
from paths import get_data_root, get_output_root
from models.labram_backbone import NeuralTransformer


# ============================================================
# LaBraM Regressor (wraps backbone + regression head)
# ============================================================

class LaBraMRegressor(nn.Module):
    def __init__(self, backbone: NeuralTransformer, d_out: int = 5, head_dropout: float = 0.0):
        super().__init__()
        self.backbone = backbone
        self.channel_adapter = None  # optionally set after construction
        self.adapter_proj = None     # projection if adapter dim != embed_dim
        self.head = nn.Sequential(
            nn.Dropout(head_dropout),
            nn.Linear(backbone.embed_dim, d_out),
        )

    def attach_adapter(self, adapter, adapter_dim):
        """Attach spatial adapter with proper projection layer."""
        self.channel_adapter = adapter
        if adapter_dim != self.backbone.embed_dim:
            self.adapter_proj = nn.Linear(adapter_dim, self.backbone.embed_dim)

    def forward(self, x_lo, x_hi=None, ecog_xyz=None, sid=None, return_losses=False):
        B, C, T_lo = x_lo.shape

        if x_hi is not None:
            # Pad x_lo to match x_hi length if needed (x_lo=128, x_hi=200)
            if T_lo < x_hi.shape[2]:
                x_lo = torch.nn.functional.pad(x_lo, (0, x_hi.shape[2] - T_lo))
            x = torch.stack([x_hi, x_lo], dim=2)  # (B, C, 2, T)
        else:
            if T_lo < 200:
                x_lo = torch.nn.functional.pad(x_lo, (0, 200 - T_lo))
            x = x_lo.unsqueeze(2)  # (B, C, 1, 200)

        if not hasattr(self, '_input_chans') or self._input_chans.shape[0] != C + 1:
            self._input_chans = torch.arange(C + 1, device=x.device)

        features = self.backbone.forward_features(
            x, input_chans=self._input_chans.to(x.device), return_patch_tokens=False
        )

        # If adapter is attached, add adapter output as residual to the pooled features
        if self.channel_adapter is not None and ecog_xyz is not None:
            adapter_emb = self.channel_adapter(ecog_xyz)  # (B, C, adapter_dim)
            if self.adapter_proj is not None:
                adapter_emb = self.adapter_proj(adapter_emb)  # (B, C, embed_dim)
            # Mean-pool adapter embeddings across channels and add to features
            adapter_pooled = adapter_emb.mean(dim=1)  # (B, embed_dim)
            features = features + adapter_pooled

        y_hat = self.head(features)

        if return_losses:
            return {"y_hat": y_hat}
        return y_hat


# ============================================================
# Build LaBraM model
# ============================================================

def build_labram(args, d_out: int = 5) -> nn.Module:
    """Build LaBraM-Base and load pretrained weights."""
    backbone = NeuralTransformer(
        EEG_size=200, patch_size=200, in_chans=1, out_chans=8,
        num_classes=0, embed_dim=200, depth=12, num_heads=10,
        mlp_ratio=4., qkv_bias=True, qk_norm=nn.LayerNorm,
        drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.1,
        init_values=1e-4, use_abs_pos_emb=True, use_mean_pooling=True,
        init_scale=0.001,
    )

    skip_pretrained = getattr(args, "no_pretrained", False)
    pretrained_path = args.pretrained_path

    if not skip_pretrained and pretrained_path and os.path.exists(pretrained_path):
        print(f"  Loading LaBraM pretrained: {pretrained_path}")
        ckpt = torch.load(pretrained_path, map_location='cpu', weights_only=False)
        sd = ckpt.get('model', ckpt)

        # Strip "student." prefix
        sd_clean = {}
        for k, v in sd.items():
            if k.startswith("student."):
                sd_clean[k[len("student."):]] = v

        # Drop heads
        for k in [k for k in sd_clean if k.startswith(("lm_head", "head"))]:
            del sd_clean[k]
        sd_clean.pop("mask_token", None)

        if not sd_clean:
            raise SystemExit(
                f"{pretrained_path} yielded no usable tensors. This loader keeps only\n"
                "  keys prefixed 'student.'; a checkpoint saved without that prefix\n"
                "  loads nothing while still reporting success.")
        msg = backbone.load_state_dict(sd_clean, strict=False)
        print(f"  Loaded: missing={len(msg.missing_keys)}, unexpected={len(msg.unexpected_keys)}")
    elif skip_pretrained:
        print("  LaBraM: random init (--no_pretrained)")
    else:
        raise SystemExit(
            f"LaBraM weights not found at {pretrained_path!r}.\n"
            "  Without them the backbone is randomly initialised, but the run would\n"
            "  still be labelled 'labram_pretrained' -- a random-init number reported\n"
            "  as a pretrained one.\n"
            "  Fix:  --pretrained_path /path/to/labram-base.pth   (source: README.md)\n"
            "  Or, to request random init deliberately:  --no_pretrained")

    # Channel embedding controls (same as STEEGFormer)
    ch_mode = getattr(args, "channel_embed_mode", "pretrained_learnable")
    if ch_mode.startswith("random"):
        nn.init.zeros_(backbone.pos_embed)
        print(f"  [channel_embed] pos_embed reinitialized to zeros (mode={ch_mode})")
    if ch_mode.endswith("frozen"):
        backbone.pos_embed.requires_grad = False
        print(f"  [channel_embed] pos_embed frozen (mode={ch_mode})")

    if getattr(args, "shuffle_channel_embed", False):
        with torch.no_grad():
            # Shuffle channel positions (skip CLS at index 0)
            w = backbone.pos_embed.data[0, 1:, :]  # (128, 200)
            perm = torch.randperm(w.shape[0])
            backbone.pos_embed.data[0, 1:, :] = w[perm]
        print(f"  [channel_embed] pos_embed shuffled")

    if getattr(args, "synthetic_codebook", False):
        with torch.no_grad():
            w = backbone.pos_embed.data[0, 1:, :]
            norms = w.norm(dim=1, keepdim=True)
            rand_dirs = torch.randn_like(w)
            rand_dirs = rand_dirs / (rand_dirs.norm(dim=1, keepdim=True) + 1e-8)
            backbone.pos_embed.data[0, 1:, :] = rand_dirs * norms
        print(f"  [channel_embed] pos_embed synthetic codebook")

    # Fine-tuning: freeze all, then selectively unfreeze
    lora_last_n = int(args.unfreeze_last_n)
    for p in backbone.parameters():
        p.requires_grad = False

    if getattr(args, 'use_lora', False):
        # LoRA mode: inject LoRA into last N blocks (matches CORTEG)
        from models.steegformer.lora import inject_lora_lastn_steegformer
        inject_lora_lastn_steegformer(
            backbone, n_last=lora_last_n, r=args.lora_r,
            alpha=args.lora_alpha, dropout=0.1,
        )
        # Unfreeze LayerNorms in LoRA blocks
        for blk in backbone.blocks[-lora_last_n:]:
            for m in blk.modules():
                if isinstance(m, nn.LayerNorm):
                    for p in m.parameters():
                        p.requires_grad = True
        print(f"  LaBraM: LoRA r={args.lora_r} on last {lora_last_n} blocks")
    else:
        # Full unfreeze mode (original)
        for blk in backbone.blocks[-lora_last_n:]:
            for p in blk.parameters():
                p.requires_grad = True

    # Unfreeze final norm
    if backbone.fc_norm is not None:
        for p in backbone.fc_norm.parameters():
            p.requires_grad = True

    # Unfreeze channel embed if not frozen
    if not ch_mode.endswith("frozen"):
        backbone.pos_embed.requires_grad = True

    # Unfreeze time embed
    if backbone.time_embed is not None:
        backbone.time_embed.requires_grad = True

    n_train = sum(p.numel() for p in backbone.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in backbone.parameters())
    print(f"  LaBraM: {n_train}/{n_total} trainable ({100*n_train/n_total:.1f}%)")

    model = LaBraMRegressor(backbone, d_out=d_out, head_dropout=float(args.head_dropout))
    return model


# ============================================================
# Step function — IDENTICAL to STEEGFormer's
# ============================================================

def make_step_fn(pass_sid: bool = False):
    def step_fn(model: nn.Module, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        y = batch["y"]
        xyz = batch.get("ecog_xyz", None)
        x_lo = batch["x_raw"]
        x_hi = batch.get("x_hi", None)
        sid = batch.get("sid", None) if pass_sid else None

        out = model(x_lo, x_hi=x_hi, ecog_xyz=xyz, sid=sid, return_losses=True)
        y_hat = out["y_hat"]
        loss = torch.mean((y_hat - y) ** 2)
        return {"y_hat": y_hat, "loss": loss}
    return step_fn


# ============================================================
# Main — mirrors run_regression_hilo_clean.py exactly
# ============================================================

def main():
    p = argparse.ArgumentParser("LaBraM Baseline (same pipeline as STEEGFormer)")

    # Data
    p.add_argument("--data_root", type=str, default="")
    p.add_argument("--val_ratio", type=float, default=0.1)

    # Training
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--early_stop_patience", type=int, default=90)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--weight_decay", type=float, default=5e-3)
    p.add_argument("--use_amp", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--warmup_epochs", type=int, default=10)
    p.add_argument("--min_lr", type=float, default=1e-5)

    # Model
    p.add_argument("--head_dropout", type=float, default=0.0)
    p.add_argument("--unfreeze_last_n", type=int, default=4)
    p.add_argument("--use_lora", action="store_true",
                    help="Use LoRA instead of full block unfreezing (fair control vs CORTEG)")
    p.add_argument("--lora_r", type=int, default=4)
    p.add_argument("--lora_alpha", type=int, default=16)
    p.add_argument("--pretrained_path", type=str,
                    default=os.path.expanduser("~/workspace/datasets/pretrained_eeg_fms/labram/labram-base.pth"))

    # EA — IDENTICAL args
    p.add_argument("--use_ea", action="store_true")
    p.add_argument("--ea_shrinkage", type=float, default=0.1)
    p.add_argument("--ea_streams", type=str, default="both", choices=["both", "raw", "hi"])

    # Dataset
    p.add_argument("--dataset", type=str, default="Stanford",
                    choices=["Stanford"])
    p.add_argument("--target_mode", type=str, default="endpoint")
    p.add_argument("--step_ms", type=int, default=200)
    p.add_argument("--eval_step_ms", type=int, default=50)

    # Spatial adapter (optional — test if our adapter design transfers to LaBraM)
    p.add_argument("--use_adapter", action="store_true",
                    help="Attach KNNSoftFourier adapter to LaBraM for spatial adaptation")
    p.add_argument("--knn_k", type=int, default=8)

    # Transfer ablation — IDENTICAL args
    p.add_argument("--no_pretrained", action="store_true")
    p.add_argument("--channel_embed_mode", type=str, default="pretrained_learnable",
                    choices=["pretrained_learnable", "pretrained_frozen", "random_learnable", "random_frozen"])
    p.add_argument("--train_fraction", type=float, default=1.0)
    p.add_argument("--shuffle_channel_embed", action="store_true")
    p.add_argument("--synthetic_codebook", action="store_true")

    # Training mode & output
    p.add_argument("--train_mode", type=str, default="per_subject",
                    choices=["per_subject", "pooled"])
    p.add_argument("--save_root", type=str, default="")
    p.add_argument("--exp_name", type=str, default="labram_baseline")
    p.add_argument("--subjects", type=str, default="")

    args = p.parse_args()
    if not args.save_root:
        args.save_root = os.path.join(get_output_root(), args.exp_name)
    os.makedirs(args.save_root, exist_ok=True)

    variant_name = "labram_pretrained" if not args.no_pretrained else "labram_random"

    if args.subjects:
        subjects = [s.strip() for s in args.subjects.split(",") if s.strip()]
    else:
        subjects = STANFORD_SUBJECTS
    d_out = 5
    file_root = args.data_root if args.data_root else get_data_root()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print(f"{'='*60}")
    print(f"LaBraM Experiment: {variant_name} ({args.dataset})")
    print(f"  device={device}")
    print(f"  no_pretrained={args.no_pretrained}")
    print(f"  channel_embed_mode={args.channel_embed_mode}")
    print(f"  train_fraction={args.train_fraction}")
    print(f"  unfreeze_last_n={args.unfreeze_last_n}")
    print(f"{'='*60}\n")

    set_seed(args.seed)
    split = TailSplit(val_ratio=float(args.val_ratio))
    step_fn = make_step_fn(pass_sid=False)
    collate_fn = make_collate_fn(None)

    # ============================================================
    # Data preparation — IDENTICAL to run_regression_hilo_clean.py
    # ============================================================
    def prepare_subject(sd, sid):
        n = int(sd.y_tr.shape[0])
        idx_tr, idx_val = split.split(n)

        train_frac = args.train_fraction
        if train_frac < 1.0:
            rng = np.random.RandomState(args.seed)
            n_keep = max(1, int(len(idx_tr) * train_frac))
            idx_tr = rng.choice(idx_tr, size=n_keep, replace=False)
            idx_tr.sort()

        # Z-score targets (same as STEEGFormer)
        y_stats = fit_zscore_2d(sd.y_tr[idx_tr])
        y_tr = apply_zscore_2d(sd.y_tr[idx_tr], y_stats)
        y_val = apply_zscore_2d(sd.y_tr[idx_val], y_stats)
        y_te = apply_zscore_2d(sd.y_te, y_stats)

        # LaBraM uses BOTH feature streams at 200Hz (not raw 128Hz):
        #   x_lo = X_feat[..., 1] (low-freq at 200Hz, shape C x 200)
        #   x_hi = X_feat[..., 0] (hi-gamma at 200Hz, shape C x 200)
        # NO per-channel Z-score — LaBraM's TemporalConv has GroupNorm
        x_lo_tr = sd.X_feat_tr[idx_tr][..., 1]   # (N, C, 200)
        x_lo_val = sd.X_feat_tr[idx_val][..., 1]
        x_lo_te = sd.X_feat_te[..., 1]

        x_hi_tr = sd.X_feat_tr[idx_tr][..., 0]   # (N, C, 200)
        x_hi_val = sd.X_feat_tr[idx_val][..., 0]
        x_hi_te = sd.X_feat_te[..., 0]

        # EA on feature streams (optional, same as STEEGFormer)
        if args.use_ea:
            ea_streams = args.ea_streams
            if ea_streams in ("both", "raw"):
                W_lo = compute_ea_whitening(x_lo_tr, shrinkage=args.ea_shrinkage)
                x_lo_tr = apply_ea(x_lo_tr, W_lo)
                x_lo_val = apply_ea(x_lo_val, W_lo)
                x_lo_te = apply_ea(x_lo_te, W_lo)
            if ea_streams in ("both", "hi"):
                W_hi = compute_ea_whitening(x_hi_tr, shrinkage=args.ea_shrinkage)
                x_hi_tr = apply_ea(x_hi_tr, W_hi)
                x_hi_val = apply_ea(x_hi_val, W_hi)
                x_hi_te = apply_ea(x_hi_te, W_hi)

        # Use HiLoAddDataset: x_raw slot = lo-freq (200), x_hi slot = hi-gamma (200)
        # LaBraM model stacks them as (B, C, 2, 200) in forward()
        ds_tr = HiLoAddDataset(x_lo_tr, x_hi_tr, y_tr, sid)
        ds_va = HiLoAddDataset(x_lo_val, x_hi_val, y_val, sid)
        ds_te = HiLoAddDataset(x_lo_te, x_hi_te, y_te, sid)
        C_in = x_lo_tr.shape[1]
        T_in = x_lo_tr.shape[2]
        return ds_tr, ds_va, ds_te, C_in, T_in

    # ============================================================
    # Pooled training: one model for all subjects
    # ============================================================
    if args.train_mode == "pooled":
        from torch.utils.data import ConcatDataset
        tr_list, va_list, te_list = [], [], []

        for sid, sub in enumerate(subjects):
            sd = load_subject(file_root, sub, require_xyz=False)
            ds_tr, ds_va, ds_te, _, _ = prepare_subject(sd, sid)
            tr_list.append(ds_tr); va_list.append(ds_va); te_list.append(ds_te)

        model = build_labram(args, d_out=d_out).to(device)
        for p_param in model.head.parameters():
            p_param.requires_grad = True

        # Optionally attach our KNNSoftFourier adapter
        if getattr(args, "use_adapter", False):
            from models.steegformer.steegformer_hilo_clean import KNNSoftFourierAdapter
            # Need initial XYZ for adapter construction — use first subject with valid XYZ
            init_xyz = None
            sd0 = load_subject(file_root, subjects[0], require_xyz=True)
            if sd0.ecog_xyz_mm is not None:
                init_xyz = torch.tensor(sd0.ecog_xyz_mm / 1000.0, dtype=torch.float32)

            if init_xyz is not None:
                # Use LaBraM's channel positions from its montage for kNN spatial lookup
                import mne
                labram_channels = [
                    'FP1','FPZ','FP2','AF9','AF7','AF5','AF3','AF1','AFZ','AF2','AF4','AF6','AF8','AF10',
                    'F9','F7','F5','F3','F1','FZ','F2','F4','F6','F8','F10',
                    'FT9','FT7','FC5','FC3','FC1','FCZ','FC2','FC4','FC6','FT8','FT10',
                    'T9','T7','C5','C3','C1','CZ','C2','C4','C6','T8','T10',
                    'TP9','TP7','CP5','CP3','CP1','CPZ','CP2','CP4','CP6','TP8','TP10',
                    'P9','P7','P5','P3','P1','PZ','P2','P4','P6','P8','P10',
                    'PO9','PO7','PO5','PO3','PO1','POZ','PO2','PO4','PO6','PO8','PO10',
                    'O1','OZ','O2','O9','CB1','CB2','IZ','O10',
                    'T3','T5','T4','T6','M1','M2','A1','A2',
                    'CFC1','CFC2','CFC3','CFC4','CFC5','CFC6','CFC7','CFC8',
                    'CCP1','CCP2','CCP3','CCP4','CCP5','CCP6','CCP7','CCP8',
                    'T1','T2','FTT9h','TTP7h','TPP9h','FTT10h','TPP8h','TPP10h',
                ]
                # Get XYZ for positioned channels from MNE standard_1005
                montage = mne.channels.make_standard_montage('standard_1005')
                montage_pos = montage.get_positions()['ch_pos']
                montage_names_upper = {n.upper(): n for n in montage_pos}
                labram_xyz = []
                labram_emb_indices = []
                for i, ch in enumerate(labram_channels):
                    ch_up = ch.upper()
                    if ch_up in montage_names_upper:
                        labram_xyz.append(montage_pos[montage_names_upper[ch_up]])
                        labram_emb_indices.append(i)
                labram_xyz_t = torch.tensor(np.array(labram_xyz), dtype=torch.float32)
                # Use the positioned subset of pos_embed as codebook
                eeg_codebook = model.backbone.pos_embed.data[0, 1:, :].clone()  # (128, 200)
                eeg_codebook_positioned = eeg_codebook[labram_emb_indices]  # (N_pos, 200)
                print(f"  [adapter] LaBraM: {len(labram_emb_indices)}/{len(labram_channels)} channels have MNE positions")

                # Override _EEG_XYZ_142 with LaBraM positions for the adapter
                from models.steegformer.steegformer_hilo_clean import KNNSoftFourierAdapter
                adapter = KNNSoftFourierAdapter(eeg_codebook_positioned, init_xyz, k=args.knn_k)
                adapter_dim = eeg_codebook_positioned.shape[1]  # 200 for LaBraM
                model.attach_adapter(adapter, adapter_dim)
                # Move adapter + projection to the same device as the model
                model.channel_adapter.to(device)
                if model.adapter_proj is not None:
                    model.adapter_proj.to(device)
                # Need xyz_bank for collate
                from data.collate import SubjectXYZBank
                xyz_list = [load_subject(file_root, s, require_xyz=True).ecog_xyz_mm for s in subjects]
                xyz_bank = SubjectXYZBank.from_mm(xyz_list)
                collate_fn = make_collate_fn(xyz_bank)
                print(f"  [adapter] KNNSoftFourier attached (k={args.knn_k})")

        opt = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=args.lr, weight_decay=args.weight_decay,
        )
        scheduler = WarmupCosineLR(opt, warmup_epochs=args.warmup_epochs,
                                   max_epochs=args.epochs, min_lr=args.min_lr)
        eng_cfg = EngineConfig(use_amp=args.use_amp, max_norm=1.0)
        early = EarlyStopper(patience=args.early_stop_patience)

        tr_sizes = [len(ds) for ds in tr_list]
        tr_sampler = SubjectInterleavedSampler(tr_sizes, args.batch_size, shuffle=True)
        train_loader = DataLoader(ConcatDataset(tr_list), batch_sampler=tr_sampler,
                                  num_workers=0, pin_memory=True, collate_fn=collate_fn)
        val_loaders = [DataLoader(ds, batch_size=64, collate_fn=collate_fn) for ds in va_list]
        test_loaders = [DataLoader(ds, batch_size=64, collate_fn=collate_fn) for ds in te_list]

        scaler = torch.amp.GradScaler("cuda", enabled=(args.use_amp and device.type == "cuda"))
        t0 = time.time()
        for ep in range(args.epochs):
            tr = train_one_epoch(model, train_loader, opt, device, cfg=eng_cfg, step_fn=step_fn, scaler=scaler)
            scheduler.step()
            val = evaluate_multi_loader(model, val_loaders, device, step_fn, args.use_amp)
            if ep % 20 == 0 or ep == args.epochs - 1:
                test = evaluate_multi_loader(model, test_loaders, device, step_fn, args.use_amp)
                print(f"  [pooled ep {ep+1:03d}] loss={tr['loss']:.4f} val={val['score']:.4f} "
                      f"test={test['score']:.4f} ({time.time()-t0:.0f}s)", flush=True)
            if early.step(val["score"], model):
                break

        early.restore(model)
        test = evaluate_multi_loader(model, test_loaders, device, step_fn, args.use_amp)
        total_elapsed = time.time() - t0

        results = {
            "variant": variant_name, "backbone": "labram_base",
            "train_mode": "pooled", "score": test["score"],
            "elapsed_s": total_elapsed, "per_subject": {},
            "args": vars(args),
        }
        for sid in sorted(test["by_sid"].keys()):
            rec = test["by_sid"][sid]
            sub_name = subjects[sid] if sid < len(subjects) else str(sid)
            results["per_subject"][sub_name] = {
                "corr_mean": rec["corr_mean"],
                "corr": [float(c) for c in rec["corr"]],
                "mse": rec["mse"], "n": rec["n"],
            }

        print(f"\n{'='*60}")
        print(f"FINAL RESULTS ({variant_name}, pooled)")
        print(f"{'='*60}")
        from experiments.common import format_subject_table
        print(format_subject_table(test, subjects))
        print(f"  MEAN SCORE = {test['score']:.4f}  (total {total_elapsed:.0f}s)")

        results_path = os.path.join(args.save_root, f"results_{variant_name}.json")
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved: {results_path}")
        return

    # ============================================================
    # Per-subject training
    # ============================================================
    all_results = {}
    all_scores = []
    total_t0 = time.time()

    for si, sub_name in enumerate(subjects):
        print(f"\n{'='*50}")
        print(f"  Subject {si}: {sub_name}")
        print(f"{'='*50}")
        set_seed(args.seed)

        sd = load_subject(file_root, sub_name, require_xyz=False)
        ds_tr, ds_va, ds_te, C_in, T_in = prepare_subject(sd, sid=0)

        model = build_labram(args).to(device)

        # Unfreeze head
        for p_param in model.head.parameters():
            p_param.requires_grad = True

        if si == 0:
            n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"  Total trainable (incl head): {n_trainable}")

        opt = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=args.lr, weight_decay=args.weight_decay,
        )
        scheduler = WarmupCosineLR(opt, warmup_epochs=args.warmup_epochs,
                                   max_epochs=args.epochs, min_lr=args.min_lr)
        eng_cfg = EngineConfig(use_amp=bool(args.use_amp), max_norm=1.0)
        early = EarlyStopper(patience=args.early_stop_patience)

        tr_sampler = SubjectInterleavedSampler([len(ds_tr)], args.batch_size, shuffle=True)
        train_loader = DataLoader(
            ds_tr, batch_sampler=tr_sampler,
            num_workers=0, pin_memory=True, worker_init_fn=seed_worker, collate_fn=collate_fn,
        )
        val_loaders = [DataLoader(ds_va, batch_size=64, collate_fn=collate_fn)]
        test_loaders = [DataLoader(ds_te, batch_size=64, collate_fn=collate_fn)]

        scaler = torch.amp.GradScaler("cuda", enabled=(args.use_amp and device.type == "cuda"))
        t0 = time.time()
        for ep in range(args.epochs):
            tr = train_one_epoch(model, train_loader, opt, device, cfg=eng_cfg, step_fn=step_fn, scaler=scaler)
            scheduler.step()
            val = evaluate_multi_loader(model, val_loaders, device, step_fn, bool(args.use_amp))
            if ep % 20 == 0 or ep == args.epochs - 1:
                test = evaluate_multi_loader(model, test_loaders, device, step_fn, bool(args.use_amp))
                print(f"  [{sub_name} ep {ep+1:03d}] loss={tr['loss']:.4f} val={val['score']:.4f} "
                      f"test={test['score']:.4f} ({time.time()-t0:.0f}s)", flush=True)
            if early.step(val["score"], model):
                break

        early.restore(model)
        test = evaluate_multi_loader(model, test_loaders, device, step_fn, bool(args.use_amp))
        elapsed = time.time() - t0

        rec = test["by_sid"][0]
        all_results[sub_name] = {
            "corr_mean": rec["corr_mean"],
            "corr": [float(c) for c in rec["corr"]],
            "mse": rec["mse"],
            "n": rec["n"],
            "elapsed_s": elapsed,
        }
        all_scores.append(rec["corr_mean"])
        print(f"  {sub_name}: corr={rec['corr_mean']:.4f} mse={rec['mse']:.5f} ({elapsed:.0f}s)", flush=True)

        del sd, ds_tr, ds_va, ds_te, model
        torch.cuda.empty_cache()

    mean_score = float(np.nanmean(all_scores))
    total_elapsed = time.time() - total_t0

    print(f"\n{'='*60}")
    print(f"FINAL RESULTS ({variant_name})")
    print(f"{'='*60}")
    for sub_name in subjects:
        r = all_results[sub_name]
        corr_str = "[" + ",".join(f"{c:.3f}" for c in r["corr"]) + "]"
        print(f"  {sub_name}: corr_mean={r['corr_mean']:.4f}  {corr_str}  mse={r['mse']:.5f}")
    print(f"  MEAN SCORE = {mean_score:.4f}  (total {total_elapsed:.0f}s)")

    results = {
        "variant": variant_name,
        "backbone": "labram_base",
        "train_mode": "per_subject",
        "score": mean_score,
        "elapsed_s": total_elapsed,
        "per_subject": all_results,
        "args": vars(args),
    }

    results_path = os.path.join(args.save_root, f"results_{variant_name}.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved: {results_path}")


if __name__ == "__main__":
    main()
