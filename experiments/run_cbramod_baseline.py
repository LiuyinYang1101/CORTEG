#!/usr/bin/env python3
"""CBraMod competing FM baseline — IDENTICAL pipeline to LaBraM/STEEGFormer.

CBraMod (ICLR 2025): Criss-Cross Brain Foundation Model
- Separate spatial + temporal attention (d_model=200, 12 layers)
- Dual input: time-domain conv + spectral FFT branch
- Conv-based positional encoding (no explicit channel embedding table)
- Input: (B, C, num_patches, 200)

Differences from STEEGFormer/LaBraM:
- No channel embedding table (uses conv-based spatial PE)
- Internal FFT branch → no external standardization needed
- Criss-cross attention vs standard self-attention
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

from models.cbramod_backbone import CBraMod


# ============================================================
# CBraMod Regressor
# ============================================================

class CBraModRegressor(nn.Module):
    def __init__(self, backbone: CBraMod, d_out: int = 5, head_dropout: float = 0.0):
        super().__init__()
        self.backbone = backbone
        d_model = backbone.proj_out[0].in_features  # derive from backbone
        self.head = nn.Sequential(
            nn.Dropout(head_dropout),
            nn.Linear(d_model, d_out),
        )

    def forward(self, x_lo, x_hi=None, ecog_xyz=None, sid=None, return_losses=False):
        """Match STEEGFormer's forward signature.

        Dual-stream: stack lo (200) + hi (200) as 2 patches → (B, C, 2, 200).
        CBraMod's criss-cross attention handles spatial and temporal dims.
        """
        B, C, T_lo = x_lo.shape

        if x_hi is not None:
            T_hi = x_hi.shape[2]
            if T_lo < T_hi:
                x_lo = torch.nn.functional.pad(x_lo, (0, T_hi - T_lo))
            x = torch.stack([x_hi, x_lo], dim=2)  # (B, C, 2, 200)
        else:
            if T_lo < 200:
                x_lo = torch.nn.functional.pad(x_lo, (0, 200 - T_lo))
            x = x_lo.unsqueeze(2)  # (B, C, 1, 200)

        # CBraMod forward: (B, C, num_patches, 200) → (B, C, num_patches, d_model)
        features = self.backbone(x)  # (B, C, num_patches, 200)

        # Global average pooling over channels and patches
        features = features.mean(dim=(1, 2))  # (B, 200)
        y_hat = self.head(features)

        if return_losses:
            return {"y_hat": y_hat}
        return y_hat


# ============================================================
# Build CBraMod
# ============================================================

def build_cbramod(args, d_out: int = 5) -> nn.Module:
    backbone = CBraMod(
        in_dim=200, out_dim=200, d_model=200,
        dim_feedforward=800, seq_len=30, n_layer=12, nhead=8,
    )

    skip_pretrained = getattr(args, "no_pretrained", False)
    pretrained_path = args.pretrained_path

    if not skip_pretrained and pretrained_path and os.path.exists(pretrained_path):
        print(f"  Loading CBraMod pretrained: {pretrained_path}")
        ckpt = torch.load(pretrained_path, map_location='cpu', weights_only=False)
        sd = ckpt.get('model', ckpt.get('state_dict', ckpt))

        # Drop projection head if present
        sd_clean = {k: v for k, v in sd.items() if not k.startswith("head")}

        msg = backbone.load_state_dict(sd_clean, strict=False)
        print(f"  Loaded: missing={len(msg.missing_keys)}, unexpected={len(msg.unexpected_keys)}")
        if msg.missing_keys:
            print(f"  Missing: {msg.missing_keys[:5]}")
    else:
        print(f"  CBraMod: random init")

    # Fine-tuning: freeze all, then selectively unfreeze
    lora_last_n = int(args.unfreeze_last_n)
    for p in backbone.parameters():
        p.requires_grad = False

    if getattr(args, 'use_lora', False):
        # LoRA mode: inject LoRA into last N encoder layers
        from models.steegformer.lora import _replace_linear_with_lora
        n_layers = len(backbone.encoder.layers)
        start = max(0, n_layers - lora_last_n)
        for li in range(start, n_layers):
            _replace_linear_with_lora(
                backbone.encoder.layers[li],
                targets=("linear1", "linear2", "self_attn"),
                r=args.lora_r, alpha=args.lora_alpha, dropout=0.1,
            )
        # Unfreeze LayerNorms in LoRA layers
        for li in range(start, n_layers):
            for m_ in backbone.encoder.layers[li].modules():
                if isinstance(m_, nn.LayerNorm):
                    for p in m_.parameters():
                        p.requires_grad = True
        print(f"  CBraMod: LoRA r={args.lora_r} on last {lora_last_n} layers")
    else:
        # Full unfreeze mode (original)
        for layer in backbone.encoder.layers[-lora_last_n:]:
            for p in layer.parameters():
                p.requires_grad = True

    # Unfreeze proj_out
    for p in backbone.proj_out.parameters():
        p.requires_grad = True

    n_train = sum(p.numel() for p in backbone.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in backbone.parameters())
    print(f"  CBraMod: {n_train}/{n_total} trainable ({100*n_train/n_total:.1f}%)")

    model = CBraModRegressor(backbone, d_out=d_out, head_dropout=float(args.head_dropout))
    return model


# ============================================================
# Step function — IDENTICAL
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
# Main — mirrors LaBraM/STEEGFormer pipeline
# ============================================================

def main():
    p = argparse.ArgumentParser("CBraMod Baseline")
    p.add_argument("--data_root", type=str, default="")
    p.add_argument("--val_ratio", type=float, default=0.1)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--early_stop_patience", type=int, default=90)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--weight_decay", type=float, default=5e-3)
    p.add_argument("--use_amp", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--warmup_epochs", type=int, default=10)
    p.add_argument("--min_lr", type=float, default=1e-5)
    p.add_argument("--head_dropout", type=float, default=0.0)
    p.add_argument("--unfreeze_last_n", type=int, default=4)
    p.add_argument("--use_lora", action="store_true",
                    help="Use LoRA instead of full block unfreezing (fair control vs CORTEG)")
    p.add_argument("--lora_r", type=int, default=4)
    p.add_argument("--lora_alpha", type=int, default=16)
    p.add_argument("--pretrained_path", type=str,
                    default=os.path.expanduser("~/workspace/datasets/pretrained_eeg_fms/cbramod/pretrained_weights.pth"))
    p.add_argument("--use_ea", action="store_true")
    p.add_argument("--ea_shrinkage", type=float, default=0.1)
    p.add_argument("--ea_streams", type=str, default="both", choices=["both", "raw", "hi"])
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
    p.add_argument("--save_root", type=str, default="")
    p.add_argument("--exp_name", type=str, default="cbramod_baseline")
    p.add_argument("--subjects", type=str, default="")
    args = p.parse_args()

    if not args.save_root:
        args.save_root = os.path.join(get_output_root(), args.exp_name)
    os.makedirs(args.save_root, exist_ok=True)

    variant_name = "cbramod_pretrained" if not args.no_pretrained else "cbramod_random"

    subjects = [s.strip() for s in args.subjects.split(",") if s.strip()] if args.subjects else STANFORD_SUBJECTS
    d_out = 5
    file_root = args.data_root if args.data_root else get_data_root()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print(f"{'='*60}")
    print(f"CBraMod Experiment: {variant_name} ({args.dataset})")
    print(f"  device={device}, no_pretrained={args.no_pretrained}")
    print(f"{'='*60}\n")

    set_seed(args.seed)
    split = TailSplit(val_ratio=args.val_ratio)
    step_fn = make_step_fn(pass_sid=False)
    collate_fn = make_collate_fn(None)

    # Data preparation — same as LaBraM (lo+hi at 200Hz, no Z-score on features)
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

        # Both streams at 200Hz, no Z-score (CBraMod has GroupNorm + FFT internally)
        x_lo_tr = sd.X_feat_tr[idx_tr][..., 1]
        x_lo_val = sd.X_feat_tr[idx_val][..., 1]
        x_lo_te = sd.X_feat_te[..., 1]
        x_hi_tr = sd.X_feat_tr[idx_tr][..., 0]
        x_hi_val = sd.X_feat_tr[idx_val][..., 0]
        x_hi_te = sd.X_feat_te[..., 0]

        if args.use_ea:
            if args.ea_streams in ("both", "raw"):
                W_lo = compute_ea_whitening(x_lo_tr, shrinkage=args.ea_shrinkage)
                x_lo_tr = apply_ea(x_lo_tr, W_lo)
                x_lo_val = apply_ea(x_lo_val, W_lo)
                x_lo_te = apply_ea(x_lo_te, W_lo)
            if args.ea_streams in ("both", "hi"):
                W_hi = compute_ea_whitening(x_hi_tr, shrinkage=args.ea_shrinkage)
                x_hi_tr = apply_ea(x_hi_tr, W_hi)
                x_hi_val = apply_ea(x_hi_val, W_hi)
                x_hi_te = apply_ea(x_hi_te, W_hi)

        ds_tr = HiLoAddDataset(x_lo_tr, x_hi_tr, y_tr, sid)
        ds_va = HiLoAddDataset(x_lo_val, x_hi_val, y_val, sid)
        ds_te = HiLoAddDataset(x_lo_te, x_hi_te, y_te, sid)
        return ds_tr, ds_va, ds_te

    # Pooled training
    if args.train_mode == "pooled":
        from torch.utils.data import ConcatDataset
        from experiments.common import format_subject_table
        tr_list, va_list, te_list = [], [], []

        for sid, sub in enumerate(subjects):
            sd = load_subject(file_root, sub, require_xyz=False)
            ds_tr, ds_va, ds_te = prepare_subject(sd, sid)
            tr_list.append(ds_tr); va_list.append(ds_va); te_list.append(ds_te)

        model = build_cbramod(args, d_out=d_out).to(device)
        for p_param in model.head.parameters():
            p_param.requires_grad = True

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
            "variant": variant_name, "backbone": "cbramod",
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

    # Per-subject training
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

        model = build_cbramod(args).to(device)
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
        train_loader = DataLoader(ds_tr, batch_sampler=tr_sampler, num_workers=0,
                                  pin_memory=True, worker_init_fn=seed_worker, collate_fn=collate_fn)
        val_loaders = [DataLoader(ds_va, batch_size=64, collate_fn=collate_fn)]
        test_loaders = [DataLoader(ds_te, batch_size=64, collate_fn=collate_fn)]

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
        "backbone": "cbramod",
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
