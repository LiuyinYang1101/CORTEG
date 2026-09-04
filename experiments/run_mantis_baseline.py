#!/usr/bin/env python3
"""MantisV2 generic time-series FM baseline — CONTROL for EEG-specific pretraining.

Tests whether ANY pretrained TS FM transfers to ECoG, or specifically EEG-pretrained FMs.
MantisV2 (ICLR 2025 workshop) is pretrained contrastively on ~7M generic time series
from UCR archive + synthetic data — NOT brain signals.

Design choices (all from external review):
1. Two UNTIED MantisV2 encoders (both init from same pretrained checkpoint)
   - Lo encoder: X_raw (B, C, 128) at 128 Hz
   - Hi encoder: X_feat[...,0] (B, C, 200) → antialiased resample to 192
2. Per-channel processing, then lightweight attention pooling over channels
   (mean pool discards spatial selectivity)
3. Full fine-tuning with SPLIT LRs: backbone 3e-4, fusion/head 3e-3
4. Anti-aliased resampling via scipy.signal.resample_poly (polyphase FIR)
5. Combined output (cls + mean tokens) → 512-dim per stream
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
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from scipy.signal import resample_poly

from data.io import load_subject
from data.splits import TailSplit
from data.scalers import fit_zscore_2d, apply_zscore_2d
from data.collate import make_collate_fn
from data.datasets import HiLoAddDataset

from train.engine import EngineConfig, train_one_epoch
from train.earlystop import EarlyStopper
from train.lr_schedule import WarmupCosineLR
from train.sampling import SubjectInterleavedSampler

from experiments.common import (
    set_seed, seed_worker, evaluate_multi_loader, STANFORD_SUBJECTS,
)
from paths import get_data_root, get_output_root

from mantis.architecture import MantisV2


# ============================================================
# MantisV2 wrapper with dual untied encoders + attention pooling
# ============================================================

class MantisDualRegressor(nn.Module):
    """Two untied MantisV2 encoders + attention pooling over channels."""

    def __init__(self, pretrained_path: str, no_pretrained: bool = False,
                 d_out: int = 5, head_dropout: float = 0.1, device: str = "cuda",
                 pretrained_state: dict = None, freeze_backbone: bool = False):
        """pretrained_state: cached state_dict to avoid reloading from disk per subject.
        freeze_backbone: if True, encoders stay in eval mode even when model.train() is called.
        """
        super().__init__()
        self.freeze_backbone = freeze_backbone

        def _build_encoder():
            # Always build with identical architecture kwargs (pretrained and random have same arch)
            enc = MantisV2(
                hidden_dim=256, num_patches=32, kernel_size=41,
                transf_depth=6, transf_num_heads=8, transf_mlp_dim=512,
                transf_dim_head=32, transf_dropout=0.1,
                output_token='combined',  # cls + mean → 512-dim
                device=device, pre_training=False,
            )
            if not no_pretrained and pretrained_state is not None:
                enc.load_state_dict(pretrained_state, strict=False)
            return enc

        self.encoder_lo = _build_encoder()
        self.encoder_hi = _build_encoder()

        d_model = 2 * 256  # combined output (cls + mean)
        fused_dim = 2 * d_model  # concat lo + hi

        # Lightweight attention pooling over channels
        self.channel_attn = nn.Linear(fused_dim, 1)
        self.head = nn.Sequential(
            nn.Dropout(head_dropout),
            nn.Linear(fused_dim, d_out),
        )

        self.fused_dim = fused_dim

    def train(self, mode: bool = True):
        """Override train() to keep encoders in eval mode when frozen."""
        super().train(mode)
        if self.freeze_backbone:
            self.encoder_lo.eval()
            self.encoder_hi.eval()
        return self

    def forward(self, x_lo, x_hi=None, ecog_xyz=None, sid=None, return_losses=False):
        """
        x_lo: (B, C, 128) — raw ECoG at 128 Hz
        x_hi: (B, C, 192) — hi-gamma resampled from 200 to 192
        """
        B, C, T_lo = x_lo.shape
        _, _, T_hi = x_hi.shape

        # Per-channel processing: flatten (B, C, T) → (B*C, 1, T)
        x_lo_flat = x_lo.reshape(B * C, 1, T_lo)
        x_hi_flat = x_hi.reshape(B * C, 1, T_hi)

        # Two untied encoders
        emb_lo = self.encoder_lo(x_lo_flat)  # (B*C, 512)
        emb_hi = self.encoder_hi(x_hi_flat)  # (B*C, 512)

        # Concat streams per channel
        emb = torch.cat([emb_lo, emb_hi], dim=-1)  # (B*C, 1024)
        emb = emb.reshape(B, C, self.fused_dim)   # (B, C, 1024)

        # Attention pooling over channels (bmm avoids (B,C,D) broadcast allocation)
        scores = self.channel_attn(emb)                                   # (B, C, 1)
        alpha = torch.softmax(scores, dim=1).transpose(1, 2)              # (B, 1, C)
        pooled = torch.bmm(alpha, emb).squeeze(1)                         # (B, 1024)

        y_hat = self.head(pooled)

        if return_losses:
            return {"y_hat": y_hat}
        return y_hat


def get_param_groups(model: MantisDualRegressor, backbone_lr: float, head_lr: float):
    """Split params: backbones vs fusion+head with different LRs. Skips frozen backbones."""
    backbone_params = [p for p in list(model.encoder_lo.parameters()) + list(model.encoder_hi.parameters())
                       if p.requires_grad]
    head_params = list(model.channel_attn.parameters()) + list(model.head.parameters())
    groups = [{"params": head_params, "lr": head_lr}]
    if backbone_params:
        groups.insert(0, {"params": backbone_params, "lr": backbone_lr})
    return groups


# ============================================================
# Step function
# ============================================================

def step_fn(model: nn.Module, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    y = batch["y"]
    x_lo = batch["x_raw"]  # 128 samples @ 128 Hz
    x_hi = batch["x_hi"]   # 192 samples, antialiased from 200
    out = model(x_lo, x_hi=x_hi, return_losses=True)
    y_hat = out["y_hat"]
    loss = torch.mean((y_hat - y) ** 2)
    return {"y_hat": y_hat, "loss": loss}


# ============================================================
# Main
# ============================================================

def main():
    p = argparse.ArgumentParser("MantisV2 Generic TS FM Baseline")
    p.add_argument("--data_root", type=str, default="")
    p.add_argument("--val_ratio", type=float, default=0.1)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--early_stop_patience", type=int, default=90)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--max_effective_batch", type=int, default=1024,
                    help="Cap B*C for Mantis (controls VRAM). batch_size adapts per subject.")
    p.add_argument("--backbone_lr", type=float, default=3e-4)
    p.add_argument("--head_lr", type=float, default=3e-3)
    p.add_argument("--weight_decay", type=float, default=5e-3)
    p.add_argument("--use_amp", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--warmup_epochs", type=int, default=10)
    p.add_argument("--min_lr", type=float, default=1e-5)
    p.add_argument("--head_dropout", type=float, default=0.1)
    p.add_argument("--pretrained_path", type=str,
                    default=os.path.expanduser(
                        os.environ.get("MANTIS_PRETRAINED_PATH",
                                       "~/workspace/datasets/pretrained_tsfm/mantis_v2")))
    p.add_argument("--no_pretrained", action="store_true")
    p.add_argument("--train_fraction", type=float, default=1.0)
    # Dataset
    p.add_argument("--dataset", type=str, default="Stanford",
                    choices=["Stanford"])
    p.add_argument("--target_mode", type=str, default="endpoint")
    p.add_argument("--step_ms", type=int, default=200)
    p.add_argument("--eval_step_ms", type=int, default=50)
    p.add_argument("--train_mode", type=str, default="per_subject",
                    choices=["per_subject", "pooled"])
    p.add_argument("--long_context", action="store_true",
                    help="Resample both streams to 512 samples (Mantis's native pretraining length)")
    p.add_argument("--freeze_backbone", action="store_true",
                    help="Freeze both encoders, only train attention pool + head (linear probe)")
    p.add_argument("--save_root", type=str, default="")
    p.add_argument("--exp_name", type=str, default="mantis_baseline")
    p.add_argument("--subjects", type=str, default="")
    args = p.parse_args()

    if not args.save_root:
        args.save_root = os.path.join(get_output_root(), args.exp_name)
    os.makedirs(args.save_root, exist_ok=True)

    variant_name = "mantis_pretrained" if not args.no_pretrained else "mantis_random"

    subjects = [s.strip() for s in args.subjects.split(",") if s.strip()] if args.subjects else STANFORD_SUBJECTS
    d_out = 5
    file_root = args.data_root if args.data_root else get_data_root()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print(f"{'='*60}")
    print(f"MantisV2 Experiment: {variant_name} ({args.dataset})")
    print(f"  device={device}, train_mode={args.train_mode}")
    print(f"  no_pretrained={args.no_pretrained}")
    print(f"  backbone_lr={args.backbone_lr}, head_lr={args.head_lr}")
    print(f"  long_context={args.long_context}, freeze_backbone={args.freeze_backbone}")
    if args.long_context:
        print(f"  Design: dual untied encoders, both streams resampled to 512 (Mantis native)")
    else:
        print(f"  Design: dual untied encoders, lo=128 hi=192 (antialiased)")
    print(f"{'='*60}\n")

    set_seed(args.seed)
    split = TailSplit(val_ratio=args.val_ratio)
    collate_fn = make_collate_fn(None)

    def prepare_subject(sd, sid):
        n = int(sd.y_tr.shape[0])
        idx_tr, idx_val = split.split(n)

        if args.train_fraction < 1.0:
            rng = np.random.RandomState(args.seed)
            n_keep = max(1, int(len(idx_tr) * args.train_fraction))
            idx_tr = rng.choice(idx_tr, size=n_keep, replace=False)
            idx_tr.sort()

        y_stats = fit_zscore_2d(sd.y_tr[idx_tr])
        y_tr = apply_zscore_2d(sd.y_tr[idx_tr], y_stats)
        y_val = apply_zscore_2d(sd.y_tr[idx_val], y_stats)
        y_te = apply_zscore_2d(sd.y_te, y_stats)

        x_lo_tr_raw = sd.X_raw_tr[idx_tr]           # (N, C, 128)
        x_lo_val_raw = sd.X_raw_tr[idx_val]
        x_lo_te_raw = sd.X_raw_te

        x_hi_tr_200 = sd.X_feat_tr[idx_tr][..., 0]      # (N, C, 200)
        x_hi_val_200 = sd.X_feat_tr[idx_val][..., 0]
        x_hi_te_200 = sd.X_feat_te[..., 0]

        if args.long_context:
            # Resample both streams to 512 (Mantis's pretrained length)
            x_lo_tr = resample_poly(x_lo_tr_raw, up=512, down=128, axis=-1).astype(np.float32)
            x_lo_val = resample_poly(x_lo_val_raw, up=512, down=128, axis=-1).astype(np.float32)
            x_lo_te = resample_poly(x_lo_te_raw, up=512, down=128, axis=-1).astype(np.float32)
            x_hi_tr = resample_poly(x_hi_tr_200, up=512, down=200, axis=-1).astype(np.float32)
            x_hi_val = resample_poly(x_hi_val_200, up=512, down=200, axis=-1).astype(np.float32)
            x_hi_te = resample_poly(x_hi_te_200, up=512, down=200, axis=-1).astype(np.float32)
        else:
            # Lo stream: raw ECoG at 128 samples (128 Hz)
            x_lo_tr = x_lo_tr_raw.astype(np.float32)
            x_lo_val = x_lo_val_raw.astype(np.float32)
            x_lo_te = x_lo_te_raw.astype(np.float32)
            # Hi stream: antialiased resample 200 → 192 (32×6)
            x_hi_tr = resample_poly(x_hi_tr_200, up=192, down=200, axis=-1).astype(np.float32)
            x_hi_val = resample_poly(x_hi_val_200, up=192, down=200, axis=-1).astype(np.float32)
            x_hi_te = resample_poly(x_hi_te_200, up=192, down=200, axis=-1).astype(np.float32)

        # No external z-score: Mantis has internal ts_scaler
        ds_tr = HiLoAddDataset(x_lo_tr, x_hi_tr, y_tr, sid)
        ds_va = HiLoAddDataset(x_lo_val, x_hi_val, y_val, sid)
        ds_te = HiLoAddDataset(x_lo_te, x_hi_te, y_te, sid)
        return ds_tr, ds_va, ds_te

    # Cache pretrained state_dict once (avoids 2×N_subjects disk loads)
    _cached_state = None
    if not args.no_pretrained and args.pretrained_path and os.path.exists(args.pretrained_path):
        tmp = MantisV2(
            hidden_dim=256, num_patches=32, kernel_size=41,
            transf_depth=6, transf_num_heads=8, transf_mlp_dim=512,
            transf_dim_head=32, transf_dropout=0.1,
            output_token='combined', device='cpu', pre_training=False,
        )
        tmp = tmp.from_pretrained(args.pretrained_path, local_files_only=True)
        _cached_state = {k: v.cpu() for k, v in tmp.state_dict().items()}
        del tmp
        print(f"  Cached pretrained state_dict ({len(_cached_state)} tensors)")

    def build_model():
        model = MantisDualRegressor(
            pretrained_path=args.pretrained_path,
            no_pretrained=args.no_pretrained,
            d_out=d_out,
            head_dropout=args.head_dropout,
            device=str(device),
            pretrained_state=_cached_state,
            freeze_backbone=args.freeze_backbone,
        ).to(device)
        if args.freeze_backbone:
            for p_param in model.encoder_lo.parameters():
                p_param.requires_grad = False
            for p_param in model.encoder_hi.parameters():
                p_param.requires_grad = False
            model.encoder_lo.eval()
            model.encoder_hi.eval()
        return model

    # ============================================================
    # Pooled mode
    # ============================================================
    if args.train_mode == "pooled":
        from experiments.common import format_subject_table
        tr_list, va_list, te_list = [], [], []
        max_C = 0

        for sid, sub in enumerate(subjects):
            sd = load_subject(file_root, sub, require_xyz=False)
            ds_tr, ds_va, ds_te = prepare_subject(sd, sid)
            tr_list.append(ds_tr); va_list.append(ds_va); te_list.append(ds_te)
            max_C = max(max_C, int(sd.X_raw_tr.shape[1]))

        adaptive_bs = max(1, min(args.batch_size, args.max_effective_batch // max_C))
        if adaptive_bs < args.batch_size:
            print(f"  [adaptive bs pooled] max_C={max_C}, batch_size {args.batch_size}→{adaptive_bs}")

        model = build_model()
        n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  Trainable params: {n_train:,}")

        opt = torch.optim.AdamW(
            get_param_groups(model, args.backbone_lr, args.head_lr),
            weight_decay=args.weight_decay,
        )
        scheduler = WarmupCosineLR(opt, warmup_epochs=args.warmup_epochs,
                                   max_epochs=args.epochs, min_lr=args.min_lr)
        eng_cfg = EngineConfig(use_amp=args.use_amp, max_norm=1.0)
        early = EarlyStopper(patience=args.early_stop_patience)

        tr_sizes = [len(ds) for ds in tr_list]
        tr_sampler = SubjectInterleavedSampler(tr_sizes, adaptive_bs, shuffle=True)
        train_loader = DataLoader(ConcatDataset(tr_list), batch_sampler=tr_sampler,
                                  num_workers=2, persistent_workers=True,
                                  pin_memory=True, collate_fn=collate_fn)
        val_loaders = [DataLoader(ds, batch_size=adaptive_bs, collate_fn=collate_fn, pin_memory=True) for ds in va_list]
        test_loaders = [DataLoader(ds, batch_size=adaptive_bs, collate_fn=collate_fn, pin_memory=True) for ds in te_list]

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
            "variant": variant_name, "backbone": "mantis_v2",
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
        print(format_subject_table(test, subjects))
        print(f"  MEAN SCORE = {test['score']:.4f}  (total {total_elapsed:.0f}s)")

        results_path = os.path.join(args.save_root, f"results_{variant_name}.json")
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved: {results_path}")
        return

    # ============================================================
    # Per-subject mode
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
        ds_tr, ds_va, ds_te = prepare_subject(sd, sid=0)

        # Adaptive batch size: cap B*C to control VRAM
        C_sub = int(sd.X_raw_tr.shape[1])
        adaptive_bs = max(1, min(args.batch_size, args.max_effective_batch // C_sub))
        if adaptive_bs < args.batch_size:
            print(f"  [adaptive bs] C={C_sub}, batch_size {args.batch_size}→{adaptive_bs} (effective B*C={adaptive_bs*C_sub})")

        model = build_model()
        if si == 0:
            n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"  Trainable params: {n_train:,}")

        opt = torch.optim.AdamW(
            get_param_groups(model, args.backbone_lr, args.head_lr),
            weight_decay=args.weight_decay,
        )
        scheduler = WarmupCosineLR(opt, warmup_epochs=args.warmup_epochs,
                                   max_epochs=args.epochs, min_lr=args.min_lr)
        eng_cfg = EngineConfig(use_amp=args.use_amp, max_norm=1.0)
        early = EarlyStopper(patience=args.early_stop_patience)

        tr_sampler = SubjectInterleavedSampler([len(ds_tr)], adaptive_bs, shuffle=True)
        train_loader = DataLoader(ds_tr, batch_sampler=tr_sampler,
                                  num_workers=2, persistent_workers=True,
                                  pin_memory=True, worker_init_fn=seed_worker, collate_fn=collate_fn)
        val_loaders = [DataLoader(ds_va, batch_size=adaptive_bs, collate_fn=collate_fn, pin_memory=True)]
        test_loaders = [DataLoader(ds_te, batch_size=adaptive_bs, collate_fn=collate_fn, pin_memory=True)]

        scaler = torch.amp.GradScaler("cuda", enabled=(args.use_amp and device.type == "cuda"))
        t0 = time.time()
        for ep in range(args.epochs):
            tr = train_one_epoch(model, train_loader, opt, device, cfg=eng_cfg, step_fn=step_fn, scaler=scaler)
            scheduler.step()
            val = evaluate_multi_loader(model, val_loaders, device, step_fn, args.use_amp)
            if ep % 20 == 0 or ep == args.epochs - 1:
                test = evaluate_multi_loader(model, test_loaders, device, step_fn, args.use_amp)
                print(f"  [{sub_name} ep {ep+1:03d}] loss={tr['loss']:.4f} val={val['score']:.4f} "
                      f"test={test['score']:.4f} ({time.time()-t0:.0f}s)", flush=True)
            if early.step(val["score"], model):
                break

        early.restore(model)
        test = evaluate_multi_loader(model, test_loaders, device, step_fn, args.use_amp)
        elapsed = time.time() - t0

        rec = test["by_sid"][0]
        all_results[sub_name] = {
            "corr_mean": rec["corr_mean"],
            "corr": [float(c) for c in rec["corr"]],
            "mse": rec["mse"], "n": rec["n"], "elapsed_s": elapsed,
        }
        all_scores.append(rec["corr_mean"])
        print(f"  {sub_name}: corr={rec['corr_mean']:.4f} mse={rec['mse']:.5f} ({elapsed:.0f}s)", flush=True)

        # Incremental save: overwrite partial results after each subject
        partial = {
            "variant": variant_name, "backbone": "mantis_v2",
            "train_mode": "per_subject", "status": "partial",
            "subjects_done": si + 1, "subjects_total": len(subjects),
            "running_mean_score": float(np.nanmean(all_scores)),
            "elapsed_s": time.time() - total_t0,
            "per_subject": all_results,
            "args": vars(args),
        }
        partial_path = os.path.join(args.save_root, f"results_{variant_name}_partial.json")
        with open(partial_path, "w") as f:
            json.dump(partial, f, indent=2)

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
        "variant": variant_name, "backbone": "mantis_v2",
        "train_mode": "per_subject", "score": mean_score,
        "elapsed_s": total_elapsed, "per_subject": all_results,
        "args": vars(args),
    }

    results_path = os.path.join(args.save_root, f"results_{variant_name}.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    # Remove partial file since full results are now saved
    partial_path = os.path.join(args.save_root, f"results_{variant_name}_partial.json")
    if os.path.exists(partial_path):
        os.remove(partial_path)
    print(f"\nResults saved: {results_path}")


if __name__ == "__main__":
    main()
