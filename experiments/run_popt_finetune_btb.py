"""Trained PopT arms on BrainTreebank: LoRA, full fine-tune, head-only.

The paper's PopT row in Tables 3 and 9 (0.600 on Task A, 0.779 on Task B, the
best Task B entry) is NOT a frozen probe. It is ``--mode lora`` here: PopT
adapted per subject with LoRA. The other two modes give the remaining trained
PopT rows of Table 20:

    --mode lora       LoRA r=4, alpha=16 on the last 4 of PopT's 6 blocks, on
                      CORTEG's four target roles (qkv, proj, fc1, fc2), which in
                      PopT are in_proj, out_proj, linear1 and linear2
    --mode full_ft    every PopT parameter trainable
    --mode head_only  PopT frozen; only the standardiser's affine and the linear
                      head train

The frozen-probe row ("PopT, frozen probe") is ``run_ieeg_fm_baselines --fm popt``.
The BrainBERT and Brant head-only / LoRA / full fine-tune rows of Table 20 are
not released.

What is trained, and on what. BrainBERT below PopT stays frozen: PopT is defined
as a population model over frozen per-electrode BrainBERT embeddings, and its
released checkpoint was trained that way. Every mode therefore reads the SAME
per-electrode BrainBERT embeddings the frozen arms read, from the cache that
``run_ieeg_fm_baselines --fm brainbert`` writes::

    <output root>/braintreebank/cache/
        fm_brainbert_{endpoint}_{subj}_{trial}_win{w}_pre{p}_n{max_per_class}_s{event_seed}.npz

If that cache is missing it is built first, by the same code. The electrodes
are the voxel-table set the frozen FM arms use, and the coordinates are PopT's
native integer L/I/P voxel indices, re-derived for the cached electrode list.

Protocol. Causal forward chaining, 4 folds, the 7 s embargo, and a 15% causal
validation block carved as fit | embargo | val | embargo | test; early stopping
on validation AUROC only; BCE-with-logits on the 0/1 labels. One model per
subject and fold. Defaults are the recorded configuration of the paper runs:
lr 1e-4 (PopT), 1e-3 (readout), weight decay 5e-3, 60 epochs, patience 8,
validation every 2 epochs, batch 32, AMP on CUDA, seed 42.

Disclosed differences from CORTEG's LoRA. The adapters are applied in weight
space (a parametrization on the weight tensor, W -> W + B A * alpha/r), because
PyTorch's attention and encoder-layer fast paths read several of these weights
as raw tensors and never call the module, which silently bypasses a wrapped
nn.Linear. A weight-space adapter has no input activations to drop, so these
adapters are dropout-free: --lora_dropout is recorded for parity with the paper
runs' configuration but has no effect.

Result files. ``popt_<endpoint>_<mode>_seed<seed>.json`` under --save_root
(default ``<output root>/braintreebank/fm_runs``); a setting that differs from
the paper run's default adds a tag (``..._seed42_epochs1_sub_9.json``), so a
smoke or partial run never overwrites a full one.

Example:

    python -m experiments.run_popt_finetune_btb --endpoint word_nonword --mode lora
"""

from __future__ import annotations

import argparse
import json
import os
import time
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn

import ieeg_fm
from data.braintreebank import (
    EMBARGO_SEC,
    assert_valid_times,
    btb_output_root,
    btb_root,
    forward_chaining_split,
    load_localization,
    split_report,
)
from experiments.run_ieeg_fm_baselines import (
    ENDPOINTS,
    NAME_NEUTRAL,
    load_subject,
    result_name,
    result_tags,
)

MODES = ("lora", "full_ft", "head_only")

# Settings left out of the result file name. Unlike the frozen runner's forward
# batch, --batch_size here is the TRAINING batch and changes the result, so it
# is tagged like any other recipe change.
NAME_NEUTRAL_FT = NAME_NEUTRAL - {"batch_size"}


# --------------------------------------------------------------------------- #
# LoRA, in weight space
# --------------------------------------------------------------------------- #
def add_weight_lora(module, attr: str, r: int, alpha: int):
    """Register weight-space LoRA on `module.<attr>`, and return the added parameter count.

    Why weight space for every role, not nn.Linear substitution: wrapping an
    nn.Linear only works if the consumer CALLS the module. Inside
    nn.MultiheadAttention / nn.TransformerEncoderLayer several weights are read
    as raw tensors instead:

      * nn.MultiheadAttention.forward passes ``self.out_proj.weight`` into
        F.multi_head_attention_forward and never calls ``self.out_proj(...)``;
        a wrapper that exposes the base ``.weight`` is silently inert there.
      * the encoder-layer fast path hands ``linear1.weight`` / ``linear2.weight``
        to a fused kernel, so a wrapper active in train() can be bypassed in
        eval(): the model is then evaluated with weights it was not trained with.

    A parametrization is evaluated on attribute access, so it applies through
    module calls, functional calls and fused fast paths alike. Same delta and
    init as LoRALinear ((W + BA)x = Wx + B(Ax), kaiming_uniform(A), zeros(B)),
    so it starts as an exact no-op. It has no input activations, so no dropout.
    """
    import torch.nn.utils.parametrize as P
    W = getattr(module, attr)
    if W is None:
        return 0
    d_out, d_in = W.shape
    P.register_parametrization(module, attr,
                               _QKVLoRA(d_out, d_in, r, alpha, device=W.device, dtype=W.dtype))
    return r * d_in + d_out * r


class _QKVLoRA(nn.Module):
    """Weight-space LoRA delta for a single weight matrix: W -> W + (B@A) * alpha/r.

    CORTEG's LoRA targets include "qkv". In PopT the equivalent projection is
    ``self_attn.in_proj_weight``, a single (3*d, d) Parameter rather than an
    nn.Linear, so a name-based ``isinstance(child, nn.Linear)`` injection would
    silently skip it and adapt 3 of the 4 roles. Registered as a parametrization
    so the base weight stays frozen and autograd flows only into A/B.
    """

    def __init__(self, d_packed: int, d_in: int, r: int, alpha: int,
                 device=None, dtype=None):
        super().__init__()
        # device/dtype must match the weight being parametrized:
        # register_parametrization evaluates the parametrization once at
        # registration, before any later .to(device) could move A/B, so creating
        # them on the default device fails when the model is already on the GPU.
        self.A = nn.Parameter(torch.empty(r, d_in, device=device, dtype=dtype))
        self.B = nn.Parameter(torch.zeros(d_packed, r, device=device, dtype=dtype))
        nn.init.kaiming_uniform_(self.A, a=float(np.sqrt(5)))
        self.scaling = alpha / max(r, 1)

    def forward(self, W):
        return W + (self.B @ self.A) * self.scaling


def inject_popt_lora(model, n_last: int, r: int, alpha: int, dropout: float):
    """LoRA the last `n_last` PopT encoder layers on CORTEG's four target roles.

    `dropout` is accepted for parity with CORTEG's LoRA and ignored: weight-space
    adapters have no activations to drop (see add_weight_lora).

    Returns (n_lora_params, per_block_counts, targeted_roles).
    """
    layers = model.transformer_encoder.layers
    idxs = list(range(len(layers)))[-int(n_last):]
    per_block, roles = {}, set()
    for bi in idxs:
        blk = layers[bi]
        before = sum(p.numel() for p in blk.parameters() if p.requires_grad)
        attn = blk.self_attn
        targets = (("qkv", attn, "in_proj_weight"),
                   ("proj", attn.out_proj, "weight"),
                   ("fc1", blk.linear1, "weight"),
                   ("fc2", blk.linear2, "weight"))
        for role, mod, attr in targets:
            if mod is not None and getattr(mod, attr, None) is not None:
                add_weight_lora(mod, attr, r, alpha)
                roles.add(role)
        after = sum(p.numel() for p in blk.parameters() if p.requires_grad)
        per_block[int(bi)] = int(after - before)
    total = sum(per_block.values())
    return total, per_block, sorted(roles)


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
class StandardizeAffine(nn.Module):
    """Per-feature standardisation with fit-set statistics, plus a learnable affine.

    This is ``StandardScaler().fit(X_fit)`` -- the transform the frozen arm
    applies before its logistic probe -- up to the variance estimator (torch's
    std is unbiased, StandardScaler's is not: a factor sqrt(n/(n-1)), which the
    affine absorbs), so ``head_only`` is an apples-to-apples linear probe.
    PopT's [CLS] features have per-dimension std spanning 0.000 to 0.151, which
    an unnormalised linear head cannot fit at a sane learning rate. Statistics
    come from FIT indices only (never val or test). The affine starts at
    identity, so the trainable modes begin from pure standardisation. LayerNorm
    is the wrong tool here: it normalises each sample across dimensions, and
    the problem is per-dimension scale across samples.
    """

    def __init__(self, d: int):
        super().__init__()
        self.register_buffer("mean", torch.zeros(d))
        self.register_buffer("std", torch.ones(d))
        self.weight = nn.Parameter(torch.ones(d))
        self.bias = nn.Parameter(torch.zeros(d))
        self.register_buffer("fitted", torch.zeros(1))

    @torch.no_grad()
    def fit(self, feats: torch.Tensor):
        m = feats.mean(0)
        s = feats.std(0)
        # zero-variance dims exist; leave them at 1.0 so they map to 0 rather
        # than exploding, as StandardScaler's own zero guard does.
        s = torch.where(s < 1e-8, torch.ones_like(s), s)
        self.mean.copy_(m)
        self.std.copy_(s)
        self.fitted.fill_(1.0)

    def forward(self, x):
        return ((x - self.mean) / self.std) * self.weight + self.bias


class PopTClassifier(nn.Module):
    """PopT trunk + standardiser + linear head on the [CLS] token.

    Input is cached per-electrode BrainBERT embeddings, (B, C, 768).
    """

    def __init__(self, popt, n_elec: int, coords_lip: np.ndarray, device: str):
        super().__init__()
        self.popt = popt
        self.norm = StandardizeAffine(ieeg_fm.POPT_HIDDEN_DIM)
        self.head = nn.Linear(ieeg_fm.POPT_HIDDEN_DIM, 1)
        self.register_buffer("coords", torch.as_tensor(coords_lip, dtype=torch.long).unsqueeze(0))
        self.n_elec = int(n_elec)
        self.dev = device

    def forward(self, emb):                              # emb: (B, C, 768)
        b = emb.shape[0]
        cls = torch.ones(b, 1, ieeg_fm.POPT_INPUT_DIM, dtype=emb.dtype, device=emb.device)
        inputs = torch.cat([cls, emb], dim=1)            # (B, 1+C, 768)
        pad = torch.zeros(b, 1 + self.n_elec, dtype=torch.bool, device=emb.device)
        coords = self.coords.expand(b, self.n_elec, 3)
        seq_id = torch.zeros(b, self.n_elec, dtype=torch.long, device=emb.device)
        rep = self.popt.forward(inputs, pad, (coords, seq_id), intermediate_rep=True)
        return self.head(self.norm(self.pool(rep))).squeeze(-1)  # (B,)

    def pool(self, rep):
        return rep[:, 0, :]                              # [CLS]


def build(mode, coords, n_elec, args, device):
    """A fresh PopT classifier for one fold, with `mode` deciding what trains.

    The checkpoint is re-read for every fold. That is deliberate beyond giving
    each fold untouched weights: building PopT draws from the torch RNG before
    the head and the LoRA factors are initialised, so reusing a loaded model
    would change their initial values relative to the paper runs.
    """
    popt = ieeg_fm.load_popt_model(device=device)
    model = PopTClassifier(popt, n_elec, coords, device).to(device)
    meta = {"mode": mode}
    if mode == "head_only":
        for p in model.popt.parameters():
            p.requires_grad = False
    elif mode == "lora":
        for p in model.popt.parameters():
            p.requires_grad = False
        n, per_blk, roles = inject_popt_lora(model.popt, args.lora_last_n, args.lora_r,
                                             args.lora_alpha, args.lora_dropout)
        model.to(device)
        meta.update({"lora_params": n, "lora_per_block": per_blk, "lora_targets": roles,
                     "lora_r": args.lora_r, "lora_alpha": args.lora_alpha,
                     "lora_last_n": args.lora_last_n})
        if len(roles) != 4:
            raise RuntimeError(f"expected 4 LoRA roles (qkv/proj/fc1/fc2), got {roles} -- "
                               f"a silently skipped target makes this arm NOT matched")
    elif mode == "full_ft":
        for p in model.popt.parameters():
            p.requires_grad = True
    else:
        raise ValueError(mode)
    tot = sum(p.numel() for p in model.parameters())
    tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    meta.update({"total_params": int(tot), "trainable_params": int(tr),
                 "trainable_pct": round(100.0 * tr / tot, 4)})
    return model, meta


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def load_cached(subj, args):
    """Per-electrode BrainBERT embeddings, labels, event times and PopT coords.

    Reads the frozen BrainBERT arm's cache (building it if absent), then
    re-derives PopT's integer L/I/P coordinates for the cached electrodes in
    cache column order, so coordinates and embeddings cannot be misaligned.
    Returns (emb (N, C, 768), y, event_times, lip (C, 3), trial).
    """
    ns = SimpleNamespace(fm="brainbert", endpoint=args.endpoint, trial=args.trial,
                         max_per_class=args.max_per_class, event_seed=args.event_seed,
                         event_chunk=args.event_chunk, batch_size=None,
                         device=args.device, no_cache=False)
    d = load_subject(subj, ns)
    emb = np.asarray(d["emb"], dtype=np.float32)          # (N, C, 768)
    if emb.ndim != 3 or emb.shape[1] != len(d["electrodes"]):
        raise RuntimeError(f"{subj}: cached embeddings {emb.shape} do not match "
                           f"{len(d['electrodes'])} cached electrode names")
    loc = load_localization(btb_root(), subj)
    lip = np.clip(np.rint(np.array([loc[n] for n in d["electrodes"]], dtype=np.float64)),
                  0, ieeg_fm.POPT_PE_MAX_LEN - 1).astype(np.int64)
    y = np.asarray(d["y"]).astype(np.float32)
    ev = np.asarray(d["event_times"], dtype=np.float64)
    return emb, y, ev, lip, d["trial"]


# --------------------------------------------------------------------------- #
# Train / eval one fold
# --------------------------------------------------------------------------- #
def run_fold(model, emb, y, fit, val, test, args, device):
    """Train on `fit`, early-stop on `val` AUROC, return (test AUROC, best val AUROC)."""
    from sklearn.metrics import roc_auc_score

    use_amp = bool(args.use_amp) and device.startswith("cuda")
    # Fit the per-feature standardiser on FIT INDICES ONLY.
    model.eval()
    with torch.no_grad():
        _f = []
        for i in range(0, len(fit), args.eval_batch_size):
            b = fit[i:i + args.eval_batch_size]
            xb = torch.as_tensor(emb[b], device=device)
            cls = torch.ones(xb.shape[0], 1, ieeg_fm.POPT_INPUT_DIM, dtype=xb.dtype,
                             device=device)
            inp = torch.cat([cls, xb], dim=1)
            pad = torch.zeros(xb.shape[0], 1 + model.n_elec, dtype=torch.bool, device=device)
            rep = model.popt.forward(inp, pad,
                                     (model.coords.expand(xb.shape[0], model.n_elec, 3),
                                      torch.zeros(xb.shape[0], model.n_elec, dtype=torch.long,
                                                  device=device)),
                                     intermediate_rep=True)
            _f.append(model.pool(rep).float())
        model.norm.fit(torch.cat(_f, dim=0))
        del _f

    # A separate lr for the readout: the small backbone lr that protects the
    # pretrained weights leaves a from-scratch linear head underfitted.
    _head = {id(p) for p in list(model.norm.parameters()) + list(model.head.parameters())}
    _groups = [
        {"params": [p for p in model.parameters() if p.requires_grad and id(p) in _head],
         "lr": args.head_lr},
        {"params": [p for p in model.parameters() if p.requires_grad and id(p) not in _head],
         "lr": args.lr},
    ]
    opt = torch.optim.AdamW([g for g in _groups if g["params"]], weight_decay=args.wd)
    # Both endpoints are binary classification with balanced 0/1 labels, so the
    # matched loss is cross-entropy; outputs carry no sigmoid, hence with-logits.
    lossf = nn.BCEWithLogitsLoss() if args.loss == "bce" else nn.MSELoss()
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    Yt = torch.as_tensor(y, device=device)
    best, best_state, bad = -np.inf, None, 0

    def _scores(idx):
        model.eval()
        out = []
        with torch.no_grad():
            for i in range(0, len(idx), args.eval_batch_size):
                b = idx[i:i + args.eval_batch_size]
                xb = torch.as_tensor(emb[b], device=device)
                with torch.amp.autocast("cuda", enabled=use_amp):
                    out.append(model(xb).float().cpu().numpy())
        return np.concatenate(out) if out else np.zeros(0)

    for ep in range(args.epochs):
        model.train()
        # head_only must see the same frozen features the frozen probe sees.
        # PopT's trunk carries dropout (p=0.1), so leaving it in train() would
        # fit the head on stochastic features and score it on deterministic
        # ones. The adapted modes keep train(): there trunk dropout is a
        # legitimate regulariser, as it is for CORTEG.
        if args.mode == "head_only":
            model.popt.eval()
        perm = np.random.RandomState(args.seed + ep).permutation(fit)
        for i in range(0, len(perm), args.batch_size):
            b = perm[i:i + args.batch_size]
            xb = torch.as_tensor(emb[b], device=device)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                loss = lossf(model(xb), Yt[b])
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],
                                           args.max_norm)
            scaler.step(opt)
            scaler.update()
        if (ep + 1) % args.eval_every == 0 or ep == args.epochs - 1:
            s = _scores(val)
            va = roc_auc_score(y[val], s) if len(np.unique(y[val])) > 1 else float("nan")
            if np.isfinite(va) and va > best:
                best, bad = va, 0
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            else:
                bad += 1
                if bad >= args.patience:
                    break
    if best_state is not None:
        model.load_state_dict(best_state)
    s = _scores(test)
    te = roc_auc_score(y[test], s) if len(np.unique(y[test])) > 1 else float("nan")
    return float(te), float(best)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--endpoint", default="word_nonword",
                    choices=["sentence_onset", "word_nonword"])
    ap.add_argument("--mode", default="lora", choices=["lora", "full_ft", "head_only"])
    ap.add_argument("--subjects", nargs="+",
                    default=[f"sub_{i}" for i in range(1, 11)])
    ap.add_argument("--trial", default=None)
    ap.add_argument("--max_per_class", type=int, default=900)
    ap.add_argument("--event_seed", type=int, default=42,
                    help="selects the BrainBERT cache, i.e. which events; the paper "
                         "used 42. --seed does not change the events")
    ap.add_argument("--event_chunk", type=int, default=200,
                    help="only used if the BrainBERT cache has to be built")
    ap.add_argument("--folds", type=int, default=4)
    ap.add_argument("--val_frac", type=float, default=0.15)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--eval_every", type=int, default=2)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--eval_batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4, help="PopT lr (lora/full_ft)")
    ap.add_argument("--head_lr", type=float, default=1e-3,
                    help="lr for the standardiser affine + linear readout")
    ap.add_argument("--wd", type=float, default=5e-3)
    ap.add_argument("--max_norm", type=float, default=1.0)
    ap.add_argument("--lora_r", type=int, default=4)
    ap.add_argument("--lora_alpha", type=int, default=16)
    ap.add_argument("--lora_dropout", type=float, default=0.2,
                    help="recorded for parity with the paper runs; weight-space "
                         "LoRA has no activations to drop, so it has no effect")
    ap.add_argument("--lora_last_n", type=int, default=4)
    ap.add_argument("--use_amp", dest="use_amp", action="store_true", default=True,
                    help="mixed precision on CUDA (the default, as in the paper runs)")
    ap.add_argument("--no_amp", dest="use_amp", action="store_false")
    ap.add_argument("--loss", default="bce", choices=["bce", "mse"],
                    help="binary classification -> cross-entropy; mse only to "
                         "reproduce superseded numbers")
    ap.add_argument("--seed", type=int, default=42,
                    help="initialisation, batch order and dropout; not the events")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--save_root", default="")
    a = ap.parse_args()
    if a.val_frac <= 0:
        ap.error("--val_frac must be > 0: early stopping needs a validation block")

    if a.device.startswith("cuda") and not torch.cuda.is_available():
        print("[device] CUDA is not available; running on CPU (AMP off)", flush=True)
        a.device = "cpu"
    device = a.device
    save_root = a.save_root or os.path.join(btb_output_root(), "fm_runs")
    os.makedirs(save_root, exist_ok=True)
    win = ENDPOINTS[a.endpoint]["win_sec"]                # the arm's footprint

    rows, splits_all, metas = {}, [], {}
    print(f"[popt-ft] endpoint={a.endpoint} mode={a.mode} folds={a.folds} device={device}")

    # Preflight on the real device before touching data: a parametrization
    # device mismatch only shows up once the model is on the GPU.
    use_amp = bool(a.use_amp) and device.startswith("cuda")
    _probe, _pm = build(a.mode, np.zeros((8, 3), dtype=np.int64), 8, a, device)
    with torch.no_grad(), torch.amp.autocast("cuda", enabled=use_amp):
        _probe(torch.zeros(1, 8, ieeg_fm.POPT_INPUT_DIM, device=device))
    print(f"  [preflight] {a.mode} builds and forwards on {device}: trainable "
          f"{_pm['trainable_params']:,}/{_pm['total_params']:,} ({_pm['trainable_pct']}%)"
          + (f", LoRA targets={_pm.get('lora_targets')}" if a.mode == "lora" else ""),
          flush=True)
    del _probe
    if device.startswith("cuda"):
        torch.cuda.empty_cache()

    t_all = time.time()
    for subj in a.subjects:
        emb, y, ev, lip, trial = load_cached(subj, a)
        tv = assert_valid_times(ev, n_expected=len(y))
        folds = forward_chaining_split(tv, win_sec=win, n_folds=a.folds,
                                       embargo_sec=EMBARGO_SEC, val_frac=a.val_frac)
        aucs, t0 = [], time.time()
        for fi, (fit, val, test) in enumerate(folds):
            rep = split_report(tv, win, fit, test, val, scheme="forward_chaining")
            rep.update({"subject": subj, "fold": fi})
            if rep["overlapping_train_test_pairs"] or not rep["causal"]:
                raise AssertionError(f"{subj} fold {fi} leaks: {rep}")
            splits_all.append(rep)
            torch.manual_seed(a.seed + fi)
            model, meta = build(a.mode, lip, emb.shape[1], a, device)
            metas[subj] = meta
            te, va = run_fold(model, emb, y, fit, val, test, a, device)
            aucs.append(te)
            print(f"  {subj} fold{fi}: n_fit={len(fit)} n_val={len(val)} n_test={len(test)} "
                  f"val={va:.4f} test={te:.4f}", flush=True)
            del model
            if device.startswith("cuda"):
                torch.cuda.empty_cache()
        rows[subj] = {"trial": trial, "C": int(emb.shape[1]), "n": int(len(y)),
                      "per_fold": aucs, "mean_auroc": float(np.nanmean(aucs)),
                      "sec": round(time.time() - t0, 1)}
        print(f"  {subj}: MEAN AUROC = {rows[subj]['mean_auroc']:.4f} "
              f"({rows[subj]['sec']}s)", flush=True)
        del emb

    # The layout of the paper runs' result files (which scripts/aggregate_btb.py
    # reads): "arm", "endpoint", "seed", per_subject[s]["mean_auroc"], "splits",
    # "config"; plus the cohort SD, which the paper reports.
    vals = [r["mean_auroc"] for r in rows.values() if np.isfinite(r["mean_auroc"])]
    any_meta = next(iter(metas.values()), {})
    tags = result_tags(a, ap, NAME_NEUTRAL_FT)
    out = {"arm": f"PopT_{a.mode}", "fm": "popt", "mode": a.mode, "endpoint": a.endpoint,
           "folds": a.folds, "val_frac": a.val_frac, "embargo_sec": EMBARGO_SEC,
           "loss": a.loss, "electrode_set": "voxel", "seed": a.seed,
           "event_seed": a.event_seed,
           "amp": bool(a.use_amp) and device.startswith("cuda"),
           "mean_auroc": float(np.mean(vals)) if vals else None,
           # An SD needs two subjects; one is recorded as null, not as 0.
           "cohort_sd": float(np.std(vals, ddof=1)) if len(vals) > 1 else None,
           "per_subject": rows, "model": metas, "splits": splits_all,
           "note": "PopT trained on the same cached per-electrode BrainBERT embeddings "
                   "the frozen arms read; BrainBERT below stays frozen, which is PopT's "
                   "own design.",
           "config": vars(a), "name_tags": tags, "elapsed_s": time.time() - t_all}
    mean = "n/a" if out["mean_auroc"] is None else f"{out['mean_auroc']:.4f}"
    sd = "n/a" if out["cohort_sd"] is None else f"{out['cohort_sd']:.4f}"
    print(f"\n[PopT_{a.mode}] cohort mean AUROC = {mean} ± {sd} over {len(vals)} subjects")
    print(f"  trainable {any_meta.get('trainable_params', 0):,} / "
          f"{any_meta.get('total_params', 0):,} ({any_meta.get('trainable_pct')}%)")
    # Non-default settings (a smoke run's --epochs 1, a --subjects subset, ...)
    # are tagged onto the name, so they never overwrite the paper-setting file.
    dest = os.path.join(save_root, result_name(f"popt_{a.endpoint}_{a.mode}", a.seed, tags))
    with open(dest, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2, default=str)
    print(f"written: {dest}")


if __name__ == "__main__":
    main()
